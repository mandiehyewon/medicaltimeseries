import tensorflow as tf
import numpy as np
import pdb


class BAYESIAN_AMTL_NORMALLOGITS(object):
    def __init__(self, config):
        for name in config.__dict__:
            setattr(self,name,getattr(config,name))

        self.x = tf.placeholder(shape=[None, config.num_steps, config.num_features], dtype=tf.float32, name='data')
        self.y = tf.placeholder(shape=[None, config.num_tasks], dtype=tf.float32, name='labels')
        self.num_samples_ph = tf.placeholder(dtype=tf.int32,name='num_samples')
        self.output_keep_prob = 1
        self.state_keep_prob = 1
        self.input_keep_prob = 1
        # self.lr = config.LR

        self.build_model()


    def output(self, task_id, embed, beta_output):
        with tf.variable_scope("task_"+str(task_id)+'/output'):
            beta_att = tf.layers.dense(beta_output,self.num_hidden,activation=tf.nn.tanh,use_bias=True,name='beta_att')

            c_i = tf.reduce_mean(beta_att * embed, 1)
            logits = tf.layers.dense(c_i,1,activation=None,use_bias=True,name='output_layer')

            self.beta_att_each.append(beta_att)

            return logits

    def att(self,Q,K,V,to_task,from_task):
        with tf.variable_scope("task_"+str(to_task)+'/transferfrom_'+str(from_task)):
        # with tf.variable_scope("transferatt_", reuse=tf.AUTO_REUSE):
            Q = tf.layers.dense(Q,self.num_hidden,activation=None,use_bias=True)
            K = tf.layers.dense(K,self.num_hidden,activation=None,use_bias=True)
            K_transpose = tf.transpose(K,[0,2,1])
            att_loc = tf.matmul(Q, K_transpose)

            # att_loc = tf.layers.dense(trans_output,self.num_steps,activation=None,use_bias=True,name='att_loc')
            att_scale = tf.nn.softplus(tf.layers.dense(att_loc,self.num_steps,activation=None,use_bias=True,name='att_scale'))

            self.att_loc[to_task,from_task] = att_loc
            self.att_scale[to_task,from_task] = att_scale

            # e = tf.distributions.Normal(tf.zeros([self.num_steps,self.num_steps]),tf.ones([self.num_steps,self.num_steps])).sample([1])
            # att = att_loc + att_scale * e

            att = tf.nn.sigmoid(tf.reduce_mean(tf.distributions.Normal(att_loc,att_scale).sample(self.num_samples_ph),0))

            att = att * self.helper

            KL = (tf.square(att_loc)+tf.square(att_scale))/2 + tf.log(1/att_scale)
            KL = KL * self.helper
            self.KL[to_task][from_task] = tf.reduce_mean(tf.reduce_sum(KL,[1,2]))

            r = tf.matmul(att,V)

            # r = []
            # self.att_each[to_task,from_task] = []
            # for i in range(self.num_steps):
            #     x = att[:,i:i+1,:i+1]
            #     normalized = x
            #     # mean = tf.reduce_mean(x,2,keepdims=True)
            #     # x = x - mean
            #     # normalized = tf.math.l2_normalize(x,axis=2)
            #     r.append(tf.matmul(tf.nn.softmax(normalized,2), V[:,:i+1]))
            #     self.att_each[to_task,from_task].append(tf.nn.softmax(normalized,2))


            # r = tf.concat(r,1)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
        return r


    def G(self,F,to_task,from_task):
        # with tf.variable_scope("G_"+str(to_task)+'_'+str(from_task)):
        with tf.variable_scope("G", reuse=tf.AUTO_REUSE):
            out = tf.layers.dense(F,self.num_hidden,activation=tf.nn.tanh,use_bias=True,name="layer1")
            #out2 = tf.layers.dense(out,self.num_hidden,activation=tf.nn.tanh,use_bias=True,name="layer2")
        return out


    def build_model(self, use_lstm=True):
        print('Start building model')
        self.loss_each = []
        self.preds_each = []
        self.beta_att_each = []
        self.beta_main_each = []
        loss_task = 0
        self.att_each = {}
        self.transfer = {}
        self.test = []
        self.KL=[[0 for _ in range(self.num_tasks)] for i in range(self.num_tasks)]
        self.att_loc = {}
        self.att_scale = {}

        self.beta_output = []

        self.helper = tf.contrib.distributions.fill_triangular(tf.ones([int(self.num_steps*(self.num_steps+1))/2]))


        with tf.variable_scope("embed"):
            embed = tf.layers.dense(self.x,self.num_hidden,activation=None,use_bias=False)
        for task_id in range(self.num_tasks):
            with tf.variable_scope("task_"+str(task_id)+'/rnn'):
                def single_cell():            
                    lstm_cell = tf.contrib.rnn.LSTMCell(self.num_hidden)
                    return tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, 
                                                         output_keep_prob=self.output_keep_prob,
                                                         input_keep_prob=self.input_keep_prob,
                                                         state_keep_prob=self.state_keep_prob,
                                                         dtype=tf.float32
                                                         )

                cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(self.num_layers)])
                beta_output, _ = tf.nn.dynamic_rnn(cell,
                                                       embed,
                                                       dtype=tf.float32)
                #beta_loc = tf.layers.dense(rnn_output,self.num_hidden,activation=None,use_bias=True,name='beta_loc')
                #beta_scale = tf.nn.softplus(tf.layers.dense(rnn_output,self.num_hidden,activation=None,use_bias=True,name='beta_scale'))

                #e = tf.distributions.Normal(tf.zeros([self.num_steps,self.num_hidden]),tf.ones([self.num_steps,self.num_hidden])).sample([1])
                #beta_output = beta_loc + beta_scale * e
            self.beta_output.append(beta_output)


        for task_id in range(self.num_tasks):
            beta_output_comb = self.beta_output[task_id]

            for source_id in range(self.num_tasks):
                if source_id != task_id:
                    beta_output_transfer = self.att(self.beta_output[task_id],self.beta_output[source_id],self.beta_output[source_id],task_id,source_id)
                    beta_output_comb = beta_output_comb + self.G(beta_output_transfer,task_id,source_id)
            # self.test.append(beta_output_main)


            # attention_sum 
            logits = self.output(task_id, embed, beta_output_comb)
            preds = tf.nn.sigmoid(logits)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.y[:,task_id:task_id+1])) 
            self.loss_each.append(loss)
            self.preds_each.append(preds)

            loss_task += loss 


        #l2_losses = [tf.nn.l2_loss(v) for v in tf.trainable_variables() if ('weight' in v.name or 'kernel' in v.name)]
        #loss_l2 = self.l2_coeff*tf.reduce_sum(l2_losses)

        KL_loss = self.KL_coeff * np.sum(self.KL)

        self.loss_sum = loss_task + KL_loss #+ loss_l2
        self.loss_all = {'loss_task':loss_task, 'KL_loss':KL_loss}#, 'loss_l2': loss_l2}
        shared_variable = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='G') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='embed')
        self.optim = [tf.train.AdamOptimizer(self.lr).minimize(self.loss_each[task_id]+self.KL_coeff*np.sum(self.KL[task_id]),var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='task_'+str(task_id))+shared_variable) for task_id in range(self.num_tasks)]
        print ('Model built')

