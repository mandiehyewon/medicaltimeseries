from datetime import datetime,timedelta
import numpy as np
import pdb #중간에 끊어서 볼 수 있음.
from icd_code import *
from tqdm import tqdm #python 진행바
import mmap #memory mapped file

_range = 48 # for positive instance
interval = 1 # hour
num_steps = int(_range/interval)

# interval = timedelta(seconds=interval*60*60)
_range = timedelta(seconds=_range*60*60)

def write_list(g,l):
    d = str(l[0])
    for i in range(1,len(l)):
        d = d + ',' + str(l[i])
    d = d + '\n'
    g.write(d)

## l 리스트에 있는 각 component ,로 연결 - 한 줄 띔


# Label and data info
path = '/st2/HEALTHCARE_DATA/MIMIC3/raw_data/'

tasks = ['mortality', 'respiratory_failure', 'hypotension', 'heart_failure']

features_list= ['Age', 'Gender=M', 'Gender=F', 'Heart Rate', 'FiO2', 'Temperature', 'Systolic Blood Pressure', 'Diastolic Blood Pressure',
                'PaO2', 'GCS - Verbal Response', 'GCS - Motor Response', 'GCS - Eye Opening', 'Serum Urea Nitrogen Level',
                'Sodium Level', 'White Blood Cells Count', 'Urine Output']


items_group =  [[],
                [],
                [],
                ['211','220045'],
                ['223835','3420','3422','190'],
                ['676','678','223761','223762'],
                ['51','442','455','6701','220179','220050'],
                ['8368','8440','8441','8555','220051','220180'],
                ['50821','50816'],
                ['223900'],
                ['223901'],
                ['220739'],
                ['51006'],
                ['950824'],
                ['51300','51301'],
                ['40055','43175','40069','40094','40715','40473','40085','40057','40056','40405','40428','40086','40096','40651','226559',
                    '226560','226561','226584','226563','226564','226565','226567','226557','226558','227488','227489']]



# print('Get all MICU admission')
# MICU_ADM = set([])
# f = open(path+'TRANSFERS.csv','r')
# f.readline()
# for line in f:
#     l = line.strip().split(',')
#     if l[7] == "\"MICU\"":
#         MICU_ADM.add(l[2])
# f.close()

print('Get Admissions information')
ID_ADM = {}
ADM_TIME = {}
f = open(path+'ADMISSIONS.csv','r')
f.readline()

for line in f:
    l = line.strip().split(',')
    if l[1] in ID_ADM: #subject ID
        ID_ADM[l[1]].append(l[2]) #hadm_id
    else:
        ID_ADM[l[1]] = [l[2]] #subject_ID가 NULL일 때
    ADM_TIME[l[2]] = datetime.strptime(l[3], '%Y-%m-%d %H:%M:%S')
f.close()

selectedID_ADM = {}
for ID in ID_ADM:
    if len(ID_ADM[ID]) == 1: # and ID_ADM[ID][0] in MICU_ADM:
        selectedID_ADM[ID] = ID_ADM[ID][0]


# Get labels
target = {}
for ID in selectedID_ADM:
    target[ID] = [0.0]*len(tasks)

for i,task in enumerate(tasks):
    # mortality task
    if task == 'mortality':
        f = open(path+'ADMISSIONS.csv','r')
        f.readline()
        for line in f:
            l = line.strip().split(',')
            if l[1] in selectedID_ADM and l[2]== selectedID_ADM[l[1]]:
                target[l[1]][i] = float((len(l[5])!=0))

        f.close()
    # some disease task
    else:
        f = open(path+'DIAGNOSES_ICD.csv','r')
        f.readline()
        for line in f:
            l = line.strip().split(',')
            if l[1] in selectedID_ADM and l[2] == selectedID_ADM[l[1]] and l[4][1:-1] in ICD[task]:
                target[l[1]][i] = 1.0
        f.close()

print('Total: ', len(selectedID_ADM))


ID_STATIC = {}
f = open(path+'PATIENTS.csv','r')
f.readline()
for line in f:
    l = line.strip().split(',')
    ID = l[1]
    if ID in selectedID_ADM:
        age = ADM_TIME[selectedID_ADM[ID]] - datetime.strptime(l[3], '%Y-%m-%d %H:%M:%S')
        year = age.days / 365 + 1
        if l[2] == "\"M\"":
            ID_STATIC[ID]= [year,1.0,0.0]
        elif l[2] == "\"F\"":
            ID_STATIC[ID]= [year,0.0,1.0]
        else:
            raise Exception('Missing Gender!')
f.close()


print('Get features')


features_idx = {features_list[i]:i for i in range(len(features_list))}

items = []
items_idx = {}
for i,group in enumerate(items_group):
    items += group
    for item in group:
        items_idx[item] = i

ID_data = {}



print('>>> Chartevents')
time_index = 5
item_index = 4
value_index = 9
f = open(path+'CHARTEVENTS.csv','r')
f.readline()
for line in tqdm(f, total=330712483):
    l = line.strip().split(',')
    ID = l[1]
    ADM = l[2]
    if len(l[time_index])>0 and ADM in ADM_TIME:
        time = datetime.strptime(l[time_index], '%Y-%m-%d %H:%M:%S') - ADM_TIME[ADM]
        item = l[item_index]
        if (item in items) and (ID in selectedID_ADM) and (selectedID_ADM[ID]==ADM) and (time<_range):
            value = l[value_index]
            if ID in ID_data:
                ID_data[ID].append((time,item,value))
            else:
                ID_data[ID] = [(time,item,value)]
f.close()

print('Making dataset')
x = []
y = []

for ID in ID_data:
    data = ID_data[ID]
    labels = target[ID]
    static = ID_STATIC[ID]
    data.sort(key=lambda x: x[0])
    seqs = [static+[0.0]*(len(features_list)-len(static)) for i in range(num_steps)]
    for k in data:
        td = k[0]
        days, hours, minutes = td.days, td.seconds // 3600, td.seconds // 60 % 60
        index_time = int((days*24+hours)/interval)
        index_feature = items_idx[k[1]]
        if index_time>=0:
            seqs[index_time][index_feature] = k[2]

    seqs[-1] = seqs[-1]
    x.append(seqs)
    y.append(labels)

pdb.set_trace()

print('Making dataset')
f = open('mimic3_4tasks.csv','w')
write_list(f,features_list)

for ID in ID_data:
    data = ID_data[ID]
    labels = target[ID]
    static = ID_STATIC[ID]
    data.sort(key=lambda x: x[0])
    seqs = [static+[0.0]*(len(features_list)-len(static)) for i in range(num_steps)]
    for k in data:
        td = k[0]
        days, hours, minutes = td.days, td.seconds // 3600, td.seconds // 60 % 60
        index_time = (days*24+hours)/interval
        index_feature = items_idx[k[1]]
        if index_time>=0:
            seqs[index_time][index_feature] = k[2]

    seqs[-1] = seqs[-1] + labels
    for seq in seqs:
        write_list(f,seq)

f.close()

