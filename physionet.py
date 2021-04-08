import numpy as np
import pdb

# data loader
def load_data(task=0,path='/st2/HEALTHCARE_DATA/PhysioNet/processing_code_allstep/'):
    # Get data
    data = np.load(path+'x.npy')
    label = np.load(path+'y.npy')

    label = label[:,task:task+1]

    inp = np.array(data)
    label = np.array(label)

    index = range(len(label))
    # np.random.shuffle(index)
    train = int(len(label)*0.7)

    train_x = []
    train_y = []
    for i in range(train):
        train_x.append(inp[index[i]])
        train_y.append(label[index[i]])
    train_x = np.array(train_x,np.float32)
    train_y = np.array(train_y,np.float32)


    eval_x = []
    eval_y = []
    for i in range(train,len(index)):
        eval_x.append(inp[index[i]])
        eval_y.append(label[index[i]])
    eval_x = np.array(eval_x,np.float32)
    eval_y = np.array(eval_y,np.float32)


    return train_x, train_y, eval_x, eval_y, train_x.shape[2]