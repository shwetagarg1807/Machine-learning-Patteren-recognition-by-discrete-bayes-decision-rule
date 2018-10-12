from __future__ import division
import numpy as np
import pandas as pd
from collections import Counter
import mmap

#total number of lines in a file
def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

#linear address
from operator import mul
def get_lin_add(d,M): #d={d_1,d_2,...,d_N}
    sum = 0
    for i in range(len(d)):
        sum += X[i]*d[i]
    return sum
def get_mul_list(M):
    x = []
    for i in range(len(M) - 1):
        x.append(reduce(mul, M[i + 1:], 1))
    x.append(1)
    return x

#find volume of bin in a given linear address
def get_volume(lin_address):
    prod = 1
    for i in range(len(L)):
        tuple_quantized_value = tuples_[lin_address][i]
        if (tuple_quantized_value == (L[i] - 1)):
            b_ = 1
        else:
            b_ = all_bins[i][tuple_quantized_value]
        if (tuple_quantized_value == 0):
            a_ = 0
        else:
            a_ = all_bins[i][tuple_quantized_value - 1]
        length = b_ - a_
        prod = prod * length
    return prod
#find list of adjacent hypercubes given linear address
def get_adjacents(lin_address):
    adj = []
    for i in range(len(L)):
        tuple_quantized_value = tuples_[lin_address][i]
        if (tuple_quantized_value != (L[i] - 1)):
            current_tuple = []
            for j in range(len(L)):
                current_tuple.append(tuples_[lin_address][j])
            # current_tuple = tuples_[lin_address]
            oneplus = tuple_quantized_value+1
            current_tuple[i] = oneplus
            adj.append(get_lin_add(current_tuple, X))
        if (tuple_quantized_value != 0):
            current_tuple = []
            for j in range(len(L)):
                current_tuple.append(tuples_[lin_address][j])
            # current_tuple = tuples_[lin_address]
            oneminus = tuple_quantized_value-1
            current_tuple[i] = oneminus
            adj.append(get_lin_add(current_tuple, X))
    return adj

k_smoothing = 0 #k smoothing parameter
M = np.array([6,6,6,6,6])
filename = 'train_shuf.txt' #training data
filename_val = 'testset_shuf.txt' #validation data(to find expected gain on test dataset,
                                  #use test dataset filename here)

X = get_mul_list(M)

att = len(M) #number of attributes
k = 2 #number of classes
p_c0_given = 0.4
p_c1_given = 0.6
e = np.array([[1,-1],[-2,3]]) #economic gain matrix


##########
#TRAINING#
##########

names = ['d0','d1','d2','d3','d4','c']
data = pd.read_csv(filename, delim_whitespace=True, names=names)
print 'training data loaded'

#find c0(#class 0) and c1 (#class 1)
c_counts = data.groupby('c').size()
c0 = c_counts[0]
c1 = c_counts[1]
#find number of datapoints in the training data
count = get_num_lines(filename)

p_c0 = c0/count
p_c1 = c1/count
possible = reduce(mul,M,1)
freq = np.array([[0,0]]*possible)
#assign quantization boundaries

#website
# all_bins = [[0.02037,    0.18117,    0.56374,    0.90676,    0.98718],
#             [0.47436,    0.49150,    0.53603,    0.58948,    0.62023],
#             [0.08321,    0.15193,    0.24154,    0.60653,    0.94493],
#             [0.02697,    0.22574,    0.25494,    0.32456,    0.85316],
#             [0.06348,    0.10362,    0.19162,    0.51352,    0.60375]]

#obtained
all_bins = [[0.020396230352061586, 0.18112093576230517, 0.56371108997429864, 0.90573194987351546, 0.98713552434720719], [0.47740305825561957, 0.49158230891229621, 0.53604530753654833, 0.58945797857319193, 0.62133588387997052], [0.083527517494396325, 0.15043785729425685, 0.24212870105257039, 0.60883725653314369, 0.94429274843755207], [0.027062732204467147, 0.22586873166582544, 0.25495747557281101, 0.32423799658245084, 0.85452868021262018], [0.10338670244319639, 0.18964591655552912, 0.51164305299824786, 0.60293289381089343, 0.7600441109968824]]

L = M.tolist()
print 'bin boundaries set for quantization'
#quantize training data
for i,item in enumerate(names[:-1]):
    dq = pd.np.digitize(data[item], bins=all_bins[i],right=True)
    data['{}_quantized'.format(item)] = dq
print 'training data quantized'
#calculate frequency of d,c
counter0 = Counter(tuple(a[1:]) for a in data.values[:,5:11] if a[0]==0)
counter1 = Counter(tuple(a[1:]) for a in data.values[:,5:11] if a[0]==1)
for key in counter0.keys():
    freq[int(get_lin_add(key, X))][0] = counter0[key]
for key in counter1.keys():
    freq[int(get_lin_add(key, X))][1] = counter1[key]


##################
##################
### SMOOTHING ###
##################
##################

tuples_ = np.array([[0]*len(L)]*possible) #stores the tuples
row = 0
for i in range(L[0]):
    for j in range(L[1]):
        for k_ in range(L[2]):
            for l in range(L[3]):
                for m in range(L[4]):
                    tuples_[row] = i,j,k_,l,m
                    row = row + 1

pm = np.array([[0, 0]] * possible, dtype=float)
#calculate pm for class 0
for row in range(possible):
    #find v
    v = get_volume(row)
    #find b and v1
    b = freq[row][0]
    v1 = v
    if (b < k_smoothing):
        adjacents = get_adjacents(row) #get_adjacents(row) should return a list of linear addresses
        for liad in adjacents:
            b = b + freq[liad][0]
            v1 = v1 + get_volume(liad)
            if (b >= k_smoothing):
                break
    #find pm[row][0]
    pm[row][0] = (float(b)*v)/v1


#normalize
pm_sum0 = pm[:,0].sum()
for row in range(possible):
    if pm_sum0 != 0:
        pm[row][0] = pm[row][0]/pm_sum0


#calculate pm for class 1
for row in range(possible):
    #find v
    v = get_volume(row)
    #find b and v1
    b = freq[row][1]
    v1 = v
    if (b < k_smoothing):
        adjacents = get_adjacents(row) #get_adjacents(row) should return a list of linear addresses
        for liad in adjacents:
            b = b + freq[liad][1]
            v1 = v1 + get_volume(liad)
            if (b >= k_smoothing):
                break

    #find pm[row][1]
    pm[row][1] = (float(b)*v)/v1
#normalize
pm_sum1 = pm[:,1].sum()
for row in range(possible):
    if pm_sum1 != 0:
        pm[row][1] = pm[row][1]/pm_sum1
print 'smoothing done'

#calculate P(d,c)
p_d_c = freq/count
#calculate P(d|c)
p_d_given_c_prev = freq/count #just initialize
for row in range(possible):
    p_d_given_c_prev[row][0] = p_d_given_c_prev[row][0]/p_c0
    p_d_given_c_prev[row][1] = p_d_given_c_prev[row][1]/p_c1
#calculate P(d|c) (NOW USING SMOOTHING)
p_d_given_c = pm
#calculate P(c|d) (uses p_c0_given and p_c1_given)
# p_c_given_d = freq/count #just initialize
p_c_given_d = np.array([[0,0]]*possible, dtype=float)
for row in range(possible):
    denominator = (p_d_given_c[row][0]*p_c0_given) + (p_d_given_c[row][1]*p_c1_given)
    if (denominator!=0):
        p_c_given_d[row][0] = (p_d_given_c[row][0]*p_c0_given)/denominator
        p_c_given_d[row][1] = (p_d_given_c[row][1]*p_c1_given)/denominator
#calculate P(c,d): to be used (**not same as P(d,c) calculated before)
#(uses p_c0_given and p_c1_given)
p_c_d = freq/count #just initialize
# p_c_d = np.array([[0,0]]*possible, dtype=float)
for row in range(possible):
    p_c_d[row][0] = p_d_given_c[row][0]*p_c0_given
    p_c_d[row][1] = p_d_given_c[row][1]*p_c1_given

##find p_c_given_d_another using p_c_d##
p_c_given_d_another = np.array([[0,0]]*possible, dtype=float)
for row in range(possible):
    deno = p_c_d[row][0] + p_c_d[row][1]
    if deno!=0:
        p_c_given_d_another[row][0] = p_c_d[row][0]/deno
        p_c_given_d_another[row][1] = p_c_d[row][1]/deno
##

#calculate fd(c)
fd = np.array([[0,0]]*possible)
for row in range(possible):
    sum0 = p_c_d[row][0]*e[0,0]+p_c_d[row][1]*e[1,0]
    sum1 = p_c_d[row][0]*e[0,1]+p_c_d[row][1]*e[1,1]

    if sum0 > sum1:
        fd[row][0] = 1
    else:
        fd[row][1] = 1
print 'decision rule built'


############
#VALIDATION#
############

names = ['d0','d1','d2','d3','d4','c']
data_val = pd.read_csv(filename_val, delim_whitespace=True, names=names)
print 'validation data loaded'

#find c0(#class 0) and c1 (#class 1)
c_counts_val = data_val.groupby('c').size()
c0_val = c_counts_val[0]
c1_val = c_counts_val[1]
#find number of datapoints in the training data
count_val = get_num_lines(filename_val)

p_c0_val = c0_val/count_val
p_c1_val = c1_val/count_val
freq_val = np.array([[0,0]]*possible)
#quantize validation data
for i,item in enumerate(names[:-1]):
    dq = pd.np.digitize(data_val[item], bins=all_bins[i])
    data_val['{}_quantized'.format(item)] = dq
print 'validation data quantized'
#calculate frequency of d,c
counter0_val = Counter(tuple(a[1:]) for a in data_val.values[:,5:11] if a[0]==0)
counter1_val = Counter(tuple(a[1:]) for a in data_val.values[:,5:11] if a[0]==1)
for key in counter0_val.keys():
    freq_val[int(get_lin_add(key, X))][0] = counter0_val[key]
for key in counter1_val.keys():
    freq_val[int(get_lin_add(key, X))][1] = counter1_val[key]

#calculate P(d,c)
p_d_c_val = freq_val/count_val
#calculate P(d|c)
p_d_given_c_val = freq_val/count_val #just initialize
for row in range(possible):
    p_d_given_c_val[row][0] = p_d_given_c_val[row][0]/p_c0_val
    p_d_given_c_val[row][1] = p_d_given_c_val[row][1]/p_c1_val
#calculate P(c|d) (uses p_c0_given and p_c1_given)
p_c_given_d_val = freq_val/count_val #just initialize
for row in range(possible):
    denominator = (p_d_given_c_val[row][0]*p_c0_given) + (p_d_given_c_val[row][1]*p_c1_given)
    if (denominator!=0):
        p_c_given_d_val[row][0] = (p_d_given_c_val[row][0]*p_c0_given)/denominator
        p_c_given_d_val[row][1] = (p_d_given_c_val[row][1]*p_c1_given)/denominator
#calculate P(c,d): to be used (**not same as P(d,c) calculated before)
#(uses p_c0_given and p_c1_given)
p_c_d_val = freq_val/count_val #just initialize
for row in range(possible):
    p_c_d_val[row][0] = p_d_given_c_val[row][0]*p_c0_given
    p_c_d_val[row][1] = p_d_given_c_val[row][1]*p_c1_given

#calculate confusion matrix
conf_mat = np.array([[0,0],[0,0]], dtype=float) #initialize
for tr in range(k):
    for assigned in range(k):
        sum = 0
        for i in range(possible):
            sum = sum + p_d_c_val[i,tr]*fd[i,assigned] #p_c_d_val changed to p_d_c_val
        conf_mat[tr,assigned] = sum
#calculate economic gain
exp_gain = (e*conf_mat).sum()

print M
print e

print freq
print p_d_c
print p_d_given_c
print p_c_given_d
print p_c_d
print fd
print conf_mat
print exp_gain

print "Accuracy =", conf_mat[0,0]+conf_mat[1,1]