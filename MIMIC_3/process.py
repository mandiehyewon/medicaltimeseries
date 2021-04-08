from datetime import datetime, timedelta
import pdb
import glob
import psycopg2
import numpy as np

data = {}
mortality = {}
sepsis = {}
ADM_ID = {}
ADM_time = {}
length_of_stay = {}
heart_failure = {}
kedney_failure = {}
heart_failure_codes = ['39891','40201','40211','40291','40401','40403','40411','40413','40491','40493','4280','4281',
        				'42820','42821','42822','42823','42830','42831','42832','42833','42840','42841','42842','42843','4289']
kedney_failure_codes = []
time_interval = 1 # hours
time_range = 48 # from admision time
time_interval = timedelta(hours=time_interval)
time_range = timedelta(hours=time_range)
steps = int(time_range.total_seconds()/time_interval.total_seconds())
num_features = 9

def average(l):
	d = [[] for _ in range(num_features)]
	for k in l:
		for i in range(num_features):
			if len(k[i])!=0 and float(k[i])!=0:
				d[i].append(float(k[i]))
	for i in range(num_features):
		if len(d[i])>0:
			d[i] = np.mean(d[i])
		else:
			d[i] = 0.0
	return d

def write_list(f,l):
	s = str(l[0])
	for x in l[1:]:
		s = s + ',' + str(x)
	s = s + '\n'
	f.write(s)

path = '/st1/Nips_Sepsis/ICU-Data-Refiner.share/'

# Load data of all files and make label for mortality
print "Load data of all files and make label for mortality"
all_file_names = glob.glob(path+'result.MIMIC-All/*.csv')

for file_name in all_file_names:
	f = open(file_name,'r')
	f.readline()
	data_point = []
	for line in f:
		l = line.strip().split(',')
		l[5] = datetime.strptime(l[5], '%Y-%m-%d %H:%M:%S')
		data_point.append([l[i] for i in [5,3,4,6,7,8,9,10,11,12]])
	pid, time = l[0].split('_')
	mortality[(pid,time)] = l[1]
	data[(pid,time)] = data_point

# Make label for sepsis
print "Make label for sepsis"
for (pid,time) in data:
	sepsis[(pid,time)] = '0'
f = open(path+'result.MIMIC-All.sepsis/onset.csv','r')
f.readline()
for line in f:
	l = line.strip().split(',')
	pid, time = l[0].split('_')
	if (pid,time) in data:
		sepsis[(pid,time)] = '1'
	else:
		pdb.set_trace()
		raise Exception('Check the sepsis file')

# Make other labels
try:
    conn = psycopg2.connect("dbname='mimic' user='postgres' host='127.0.0.1' password='tuan5696'")
except:
    print "I am unable to connect to the database"
cur = conn.cursor()
cur.execute("SET search_path TO mimiciii")

count0 = 0
count2 = 0
# Length of stay label
print "Load admission info and make Length of stay label"
_1day = timedelta(1)
for pid,time in data.keys():
	count = 0
	t = datetime(int(time[0:4]),int(time[4:6]),int(time[6:8]),int(time[8:10]),int(time[10:12]),int(time[12:14]))
	cur.execute("""SELECT hadm_id,admittime,dischtime FROM admissions where subject_id=%s"""%(pid))
	rows = cur.fetchall()
	for (hadm_id,admittime,dischtime) in rows:
		# find admision time for the case here. If admittime is in within 1 day of the time they give us, it is a match
	    if admittime< t:
	    	gap = t - admittime
	    else:
	    	gap = admittime - t

	    if gap <= _1day:
	    	# this is the one
	    	ADM_ID[(pid,time)] = hadm_id
	    	ADM_time[(pid,time)] = admittime
	    	length_of_stay[(pid,time)] = (dischtime-admittime).days
	    	count += 1
	# if there is no match
	if count  == 0:
		count0 += 1
	# if there are more than 2 matches
	if count >= 2:
		count2 += 1
	# both cases, discard the data
	if count  == 0 or count >= 2:
		del data[(pid,time)]
		del mortality[(pid,time)]
		del sepsis[(pid,time)]
	if count >= 2:
		del ADM_ID[(pid,time)]
		del ADM_time[(pid,time)]
		del length_of_stay[(pid,time)]

# Make diseases labels
print "Make diseases labels"
for pid,time in data.keys():
	heart_failure[(pid,time)] = '0'
	kedney_failure[(pid,time)] = '0'
	cur.execute("""SELECT icd9_code FROM diagnoses_icd where hadm_id=%s"""%(ADM_ID[(pid,time)]))
	rows = cur.fetchall()
	for row in rows:
		icd9_code = row[0]
		if icd9_code in heart_failure_codes:
			heart_failure[(pid,time)] = '1'
		if icd9_code in kedney_failure_codes:
			kedney_failure[(pid,time)] = '1'

# Data and write to file
f = open('data_mimic.csv','w')
for pid,time in data.keys():
	data_point = [[] for _ in range(steps)]
	admittime = ADM_time[(pid,time)]
	for k in data[(pid,time)]:
		record_time = k[0]
		step = int((record_time-admittime).total_seconds()/time_interval.total_seconds())
		if step < steps:
			data_point[step].append(k[1:])
	for i in range(steps):
		data_point[i] = average(data_point[i])

	data_point[-1] = data_point[-1] + [mortality[(pid,time)],sepsis[(pid,time)],heart_failure[(pid,time)],kedney_failure[(pid,time)],length_of_stay[(pid,time)]]
	for l in data_point:
		write_list(f,l)


print "Result:"
print "Total patients: ", len(data)
num_mortality = 0
num_sepsis = 0
num_heart_failure = 0
num_kedney_failure = 0
for pid,time in data.keys():
	num_mortality += float(mortality[(pid,time)])
	num_sepsis += float(sepsis[(pid,time)])
	num_heart_failure += float(heart_failure[(pid,time)])
	num_kedney_failure += float(kedney_failure[(pid,time)])
print "Mortality: ", num_mortality
print "Sepsis: ", num_sepsis
print "Heart failure: ", num_heart_failure
print "Kedney failure: ", num_kedney_failure

pdb.set_trace()
