import sys
import numpy as np

train = sys.argv[1]
test = sys.argv[2]
Trainy = []
Trainx = []
samplecount = 0
classes = ["2", "4"]
a = np.zeros(10)
w = np.zeros(10)

def gradient_decent(margin):
	global samplecount, Trainx, a
	epoch = 0
	eta = float(0.1)
	while True:
		if (epoch == 1000) : 
			break
		wrong = 0
		for i in range(0, samplecount):
			if (np.dot(a, Trainx[i])) <= margin:
				temp = margin - np.dot(a, Trainx[i])
				temp /= np.dot(Trainx[i], Trainx[i])
				temp *= eta
				np.add(a, temp * np.array(Trainx[i]), out=a)
				wrong += 1
		if wrong == 0: 
			break
		epoch += 1

def modified_perceptron(batch_size, margin):
    global w, samplecount, trainx
    epochs = 0
    eta = float(0.0)
    while True:
    	if epochs == 1000:
    		break
        wrong = 0
        batch = 0
        x = np.zeros(10)
        for i in range(0, samplecount):
            if np.dot(w, Trainx[i]) <= margin: 
                x = np.add(x, Trainx[i])
                wrong += 1
                batch += 1
            if batch == min(batch_size, samplecount): 
                w = np.add(w, eta * x)
                x = np.zeros(10)
                batch = 0
        eta = float(wrong)/float(samplecount) 
        if wrong == 0: 
            break
        epochs += 1

# read input
with open(train) as f:
	lines = f.readlines()
	for line in lines:
		line = line.rsplit(',')
		l = [1]
		bad = False
		for i in range(1, 11):
			if (line[i] == '?'):
				bad = True
				break
			if i != 10: l.append(int(line[i]))
			else: Trainy.append(int(line[i]))

		if bad:
			continue
		if line[10] == "2\n":
			for i in range(len(l)):
				l[i] = -l[i]

		Trainx.append(l)
		samplecount += 1
f.close()

gradient_decent(10)
modified_perceptron(1, 10)

samplecount = 0
tmpvar = 0
y = ""
correct = 0
Y1 = []
Y2 = []

# generate output
with open(test) as f:
	lines = f.readlines()
	for line in lines:
		line = line.rsplit(',')
		l = [1]
		for i in range(1, 10):
			l.append(int(line[i]))		
		
		if np.dot(a, l) > 0: Y1.append(classes[1])
		else : Y1.append(classes[0])
		
		if np.dot(w, l) > 0: Y2.append(classes[1])
		else: Y2.append(classes[0]) 
f.close()

for y in Y1: 
	print y
for y in Y2:
	print y
