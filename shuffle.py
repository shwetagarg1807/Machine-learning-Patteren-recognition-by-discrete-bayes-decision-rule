#create training, validation and test datasets
import random
from time import time

start_time = time()
filename = 'pr_data.txt' #data file name
random.seed(1234) #random seed

with open(filename,'r') as source:
    data = [ (random.random(), line) for line in source ]
data.sort()
total = len(data)
t1 = total/3
t2 = 2*(total/3)

with open('train_shuf.txt','w') as target:
    for _, line in data[:t1]:
        target.write(line)
with open('val_shuf.txt','w') as target:
    for _, line in data[t1:t2]:
        target.write(line)
with open('testset_shuf.txt','w') as target:
    for _, line in data[t2:]:
        target.write(line)

print '------time taken-------'
print time()-start_time