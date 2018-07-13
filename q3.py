import sys
import numpy as np
from collections import *
from math import log

attributes = []
Trainx = defaultdict(list)
Trainy = []
continuous_attributes = {0 : 1, 1 : 1, 2 : 1, 3 : 1, 4 : 1, 5 : 1, 6 : 1}
tree = defaultdict(list)

# read training and testing files.
train = sys.argv[1]
test = sys.argv[2]

def entropy(sample_indices): 
	one = float(0)
	zero = float(0)
	for i in sample_indices: 
		if Trainy[i] == '1': one += float(1)
		else: zero += float(1)
	tot = float(one + zero)
	ans = float(0)
	if (one != 0): ans -= float(one * log( (one / tot), 2))
	if (zero != 0): ans -= float(zero * log ( (zero / tot), 2))
	if (tot != 0): ans /= tot 
	return ans
	
def quality_continuous (attr, sample_indices):
	M = -100000
	m = 100000
	for i in sample_indices: 
		M = max(M, float(Trainx[attr][i]))
		m = min(m, float(Trainx[attr][i]))		
	qual = 1000000
	v = 0
	step = (M - m) / 100
	if step == 0: step = 0.01
	for val in np.arange(m, M + 1, step):
		less = []
		greater_equal = []
		for i in sample_indices: 
			if float(Trainx[attr][i]) < val: less.append(i)
			else: greater_equal.append(i)
		l  = float(len(less))
		ge = float(len(greater_equal))
		t  = float(l + ge)
		q  = l * entropy(less) + ge * entropy(greater_equal)
		if (t != 0): q /= t
		if q < qual: 
			qual = q
			v = val
	ret = []
	ret.append(qual)
	ret.append(v)
	return ret

def quality_discrete (attr, sample_indices): 
	possible_attr_values = defaultdict(list)
	for i in sample_indices: 
		possible_attr_values[Trainx[attr][i]].append(i)
	quality = float(0)
	for value in possible_attr_values: 
		quality += float(len(possible_attr_values[value])) * entropy(possible_attr_values[value])
	if float(len(sample_indices)) != 0: quality /= float(len(sample_indices))
	ret = []
	ret.append(quality)
	for value in possible_attr_values: 
		ret.append(value)
	return ret

def label(sample_indices): 
	one = 0
	zero = 0
	for i in sample_indices: 
		if Trainy[i] == "0": zero += 1
		else: one += 1
	if zero >= one: return 0
	return 1

node = 0

def train_decision_tree(depth, sample_indices):
	global node
	minimum_quality = 1e5
	split = []
	selected_attr = -1
	curr_node = node
	node += 1

	for i in range(0, 9): 
		if i in continuous_attributes: 
			qual = quality_continuous(i, sample_indices) 
			if qual[0] < minimum_quality: 
				minimum_quality = qual[0]
				split = qual[1:]
				selected_attr = i
		else: 
			qual = quality_discrete(i, sample_indices)
			if qual[0] < minimum_quality: 
				minimum_quality = qual[0]
				split = qual[1:]
				selected_attr = i

	if minimum_quality != 0:
		if selected_attr not in continuous_attributes: 
			for split_val in split: 				
				curr = []
				for i in sample_indices: 
					if Trainx[selected_attr][i] == split_val: 
						curr.append(i)
				t = train_decision_tree(depth + 1, curr)
				tree[curr_node].append(["con", selected_attr, split_val, t])

		else : 
			curr1, curr2 = [], []
			for i in sample_indices: 
				if float(Trainx[selected_attr][i]) < float(split[0]): curr1.append(i)
				else: curr2.append(i)
			t = train_decision_tree(depth + 1, curr1)
			tree[curr_node].append(["con", selected_attr, 0, split[0], t])
			t = train_decision_tree(depth + 1, curr2)	
			tree[curr_node].append(["con", selected_attr, 1, split[0], t])
	
	else :
		if selected_attr in continuous_attributes:
			curr1, curr2 = [], []
			for i in sample_indices:
				if float(Trainx[selected_attr][i]) < float(split[0]): curr1.append(i)
				else: curr2.append(i)
			tree[curr_node].append(["leaf", selected_attr, 0, split[0], label(curr1)])
			tree[curr_node].append(["leaf", selected_attr, 1, split[0], label(curr2)])
			
		else :
			for split_val in split: 
				curr = []
				for i in sample_indices: 
					if Trainx[selected_attr][i] == split_val: 
						curr.append(i)
				tree[curr_node].append(["leaf", selected_attr, split_val, label(curr)])

	return curr_node

def decide(tnode, data_set):
	if (tree[tnode][0][0] == "leaf"):
		if tree[tnode][0][1] not in continuous_attributes:
			for curr in tree[tnode]:
				if curr[2] == data_set[curr[1]]: 
					return curr[3]
			cnt = [0, 0]
			for curr in tree[tnode]: 
				cnt[int(curr[3])] += 1
					
			if (cnt[0] >= cnt[1]): return 0
			else: return 1

		if float(data_set[tree[tnode][0][1]]) < tree[tnode][0][3]: 
			return tree[tnode][0][4]
		else : 
			return tree[tnode][1][4]

	if tree[tnode][0][1] not in continuous_attributes: 
		for curr in tree[tnode]: 
			if data_set[curr[1]] == curr[2]:
				return decide(curr[3], data_set)
		cnt = [0, 0]
		for curr in tree[tnode]: 
			cnt[int(decide( curr[3], data_set ))] += 1
				
		if (cnt[0] >= cnt[1]): return 0
		else: return 1

	else: 
		if float(data_set[tree[tnode][0][1]]) < tree[tnode][0][3]: 
			return decide(tree[tnode][0][4], data_set)
		else : 
			return decide(tree[tnode][1][4], data_set)

samplecount = 0
with open(train) as f:
    data = f.readlines()
    for line in data:
        line = line.split(',')
        if samplecount == 0:
        	pass 
        else:
        	ok = False
        	for i in range(0, 10): 
        		if i == 6: 
        			Trainy.append(line[i])
        			ok = True
       			else: 
       				if ok: Trainx[i-1].append(line[i])
       				else: Trainx[i].append(line[i])
        samplecount += 1
f.close()

samplecount = len(Trainx[0])
sample_indices = []

for i in range(0, samplecount):
	sample_indices.append(i)

train_decision_tree(0, sample_indices)

samplecount = 0
y = 0
x = []

with open(test) as f:
    data = f.readlines()
    for line in data:
        line = line.split(',')
        if samplecount == 0:
        	pass
        else:
        	x = []
        	for i in range(0, 9): 
       			x.append(line[i])
       		print decide(0, x)
        samplecount += 1
f.close()
samplecount -= 1
