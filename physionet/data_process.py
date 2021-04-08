import pdb
import numpy as np

# path to raw data
path = '/st2/HEALTHCARE_DATA/PhysioNet/raw_data/'

def process_data():

    pid = []
    label1 = []
    label2 = []
    label3 = []
    label4 = []
    fid = open(path+'Outcomes-a.txt','r')
    #pdb.set_trace()
    fid.readline()
    for line in fid:
        l = line.strip().split(',')
        ID = l[0]
        days = int(l[3])
        less_3_days = (days <3)
        status = float(l[5])
        pid.append(ID)
        label1.append([status])
        label2.append([float(less_3_days)])
    fid.close()

    # Create index map for each feature
    features=['PaO2', 'DiasABP', 'HR', 'Na', 'NIDiasABP', 'Mg', 'WBC', 'pH', 'Glucose', 'SaO2', 
              'HCT', 'HCO3', 'BUN', 'Bilirubin', 'K', 'RespRate', 'Temp', 'SysABP', 'Age', 'FiO2', 
              'Creatinine', 'NIMAP', 'Lactate', 'NISysABP', 'MAP', 'Weight', 'PaCO2', 'Platelets', 
              'Urine', 'GCS', 'Height']    
    _33features = features + ['Gender','MechVent','ICUType']
    index = {}
    # indices 0,1,2,3 are for Gender and MechVent (one-hot encoding)
    for i,feature in enumerate(features):
        index[feature] = i+4

    inp = []
    for ID in pid:
        #pdb.set_trace()
        f = open(path+'set-a/'+ID+'.txt','r')
        f.readline()
        f.readline()
        seqs = []
        last_time = None
        seq = [0.0]*35
        for line in f:
            l = line.strip().split(',')
            time = l[0]
            if time != last_time:
                if last_time != None:
                    seqs.append(seq)
                last_time = time
                seq = [0.0]*35
                if l[1] in _33features:
                    if l[1] == 'Gender':
                        if l[2] == '0':
                            seq[0] = 1.0
                            seq[1] = 0.0
                        else:
                            seq[0] = 0.0
                            seq[1] = 1.0
                    elif l[1] == 'MechVent':
                        if l[2] == '0':
                            seq[2] = 1.0
                            seq[3] = 0.0
                        else:
                            seq[2] =0.0
                            seq[3] = 1.0
                    elif l[1] in index:
                        seq[index[l[1]]] = float(l[2])
                    elif l[1] == 'ICUType':
                        ICUType = int(l[2])
            else:
                if l[1] in _33features:
                    if l[1] == 'Gender':
                        if l[2] == '0':
                            seq[0] = 1.0
                            seq[1] = 0.0
                        else:
                            seq[0] = 0.0
                            seq[1] = 1.0
                    elif l[1] == 'MechVent':
                        if l[2] == '0':
                            seq[2] = 1.0
                            seq[3] = 0.0
                        else:
                            seq[2] =0.0
                            seq[3] = 1.0
                    elif l[1] in index:
                        seq[index[l[1]]] = float(l[2])
                    elif l[1] == 'ICUType':
                        ICUType = int(l[2])


        l3 = float(ICUType==2)
        l4 = float(ICUType==4)

        label3.append([l3])
        label4.append([l4])

        inp.append(seqs)
        f.close()

    label = np.concatenate([label1,label2,label3,label4],1)
    
    max_step = max([len(seqs) for seqs in inp])
    #max_step = 155
    for i in range(len(inp)):
        if len(inp[i])<max_step:
            inp[i] = [[0.0]*35 for i in range(max_step-len(inp[i]))] + inp[i]
        elif len(inp[i])>max_step:
            inp[i] = inp[i][:max_step]
     
    inp = np.array(inp)
    
    np.save('x.npy',inp)
    np.save('y.npy',label)
    pdb.set_trace()

if __name__ == '__main__':
    process_data()

