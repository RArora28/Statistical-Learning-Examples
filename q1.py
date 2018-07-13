import sys
import numpy as np

train = sys.argv[1]
test = sys.argv[2]
classes = ["0", "1"]
w = np.zeros(785)
samplecount = 0
trainx = []
testy = []
testx = []

def perceptron(batch_size, margin):
    global w, samplecount, trainx
    epochs = 0
    w = np.zeros(785)
    while True:
        if epochs == 1000: 
            break
        wrong = 0
        batch = 0
        x = np.zeros(785)
        for i in range(0, samplecount):
            if np.dot(w, trainx[i]) <= margin: 
                x = np.add(x, trainx[i])
                wrong += 1
                batch += 1
            if batch == min(batch_size, samplecount): 
                w = np.add(w, x)
                x = np.zeros(785)
                batch = 0
        if wrong == 0: 
            break
        epochs += 1

# TRAINING 
# Parsing the training data
with open (train) as f:
    data = f.readlines()
    for line in data: 
        templist = []
        if line[0] == "1": 
            templist.append(1)
        else:
            templist.append(-1) 
        line = line.split(',')
        for i in range(1, 785): 
            if line[0] == "1": 
                templist.append(int(line[i]))
            else :
                templist.append(-int(line[i]))
        trainx.append(templist)
        samplecount += 1
f.close()

samplecount = 0
with open(test) as f:
    data = f.readlines()
    for line in data:
        templist = []
        templist.append(1)
        line = line.split(',')
        for i in range(0, 784): 
            templist.append(int(line[i]))
        testx.append(templist)
        samplecount += 1
        
f.close()


# incremental with no margin.
perceptron(1, 0)
for i in range(0, samplecount):
  Y = 0
  if np.dot(w, testx[i]) > 0: Y = 1
  else : Y = 0
  print Y

# incremental with margin.
perceptron(1, 10)
for i in range(0, samplecount):
  Y = 0
  if np.dot(w, testx[i]) > 0: Y = 1
  else : Y = 0
  print Y

# batch without margin.
# batch size: samplecount (because implemented mini batch)
perceptron(samplecount, 0)
for i in range(0, samplecount):
  Y = 0
  if np.dot(w, testx[i]) > 0: Y = 1
  else : Y = 0
  print Y

# batch with margin.
perceptron(samplecount, 10)
for i in range(0, samplecount):
  Y = 0
  if np.dot(w, testx[i]) > 0: Y = 1
  else : Y = 0
  print Y
