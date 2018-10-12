from __future__ import division
import numpy as np
import pandas as pd
import math
from collections import Counter
import mmap
from time import time
import random

start_time = time()

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

M = np.array([6,6,6,6,6])
filename = 'train_shuf.txt' #training data
filename_val = 'val_shuf.txt'

X = get_mul_list(M)

att = len(M) #number of attributes
k = 2 #number of classes
p_c0_given = 0.4 #prior probability for class 0
p_c1_given = 0.6 #prior probability for class 1
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
all_bins = []
L = M.tolist()
columns = data.columns.tolist()[:-1]
for i,item in enumerate(columns):
    index_new = 0
    next = int(math.ceil(float(count) / L[0]))
    bins = []
    for j in range(1,L[i]):
        index_new += next
        #shift back the even indices for uniformity
        if (j%2==0):
            index_new = index_new - 1
        boundary = data[item].sort_values().values[index_new]
        bins.append(boundary)
    all_bins.append(bins)
print 'bin boundaries set for quantization'

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


#################################
#################################
###FIND OPTIMUM BIN BOUNDARIES###
#################################
#################################
print all_bins #bins before starting perturbations
#randomly pick a boundary and optimize till convergence
max_exp_gain = -float('Inf')
max_temp = -float('Inf')
patience = 100
countdown = patience
random.seed(1234)
while(countdown > 0):
    dim = random.randrange(len(all_bins))
    index_boundary = random.randrange(len(all_bins[dim]))
    #find previous boundary
    if index_boundary == 0:
        prev_boundary = 0
    else:
        prev_boundary = all_bins[dim][index_boundary-1]
    #find next boundary
    if index_boundary == (len(all_bins[dim])-1):
        next_boundary = 1
    else:
        next_boundary = all_bins[dim][index_boundary+1]
    flag = False
    already_boundary = all_bins[dim][index_boundary]
    mid_picked = 100 #lol, just assigning a number
    while (mid_picked == 100 or mid_picked == prev_boundary or mid_picked == next_boundary):
        mid_picked = random.uniform(prev_boundary, next_boundary)
    all_bins[dim][index_boundary] = mid_picked
    # for i_delta in range(11):
    #     all_bins[dim][index_boundary] = start_boundary +(i_delta*delta)
    ##########
    #TRAINING#
    ##########
    #quantize training data
    for i,item in enumerate(names[:-1]):
        dq = pd.np.digitize(data[item], bins=all_bins[i],right=True)
        data['{}_quantized'.format(item)] = dq
    # print 'training data quantized'
    #calculate frequency of d,c
    freq = np.array([[0, 0]]*possible)
    counter0 = Counter(tuple(a[1:]) for a in data.values[:,5:11] if a[0]==0)
    counter1 = Counter(tuple(a[1:]) for a in data.values[:,5:11] if a[0]==1)
    for key in counter0.keys():
        freq[int(get_lin_add(key, X))][0] = counter0[key]
    for key in counter1.keys():
        freq[int(get_lin_add(key, X))][1] = counter1[key]
    #calculate P(d,c)
    p_d_c = freq/count
    #calculate P(d|c)
    p_d_given_c = freq/count #just initialize
    for row in range(possible):
        p_d_given_c[row][0] = p_d_given_c[row][0]/p_c0
        p_d_given_c[row][1] = p_d_given_c[row][1]/p_c1
    #calculate P(c|d) (uses p_c0_given and p_c1_given)
    p_c_given_d = freq/count #just initialize
    for row in range(possible):
        denominator = (p_d_given_c[row][0]*p_c0_given) + (p_d_given_c[row][1]*p_c1_given)
        if (denominator!=0):
            p_c_given_d[row][0] = (p_d_given_c[row][0]*p_c0_given)/denominator
            p_c_given_d[row][1] = (p_d_given_c[row][1]*p_c1_given)/denominator
    #calculate P(c,d): to be used (**not same as P(d,c) calculated before)
    #(uses p_c0_given and p_c1_given)
    p_c_d = freq/count #just initialize
    for row in range(possible):
        p_c_d[row][0] = p_d_given_c[row][0]*p_c0_given
        p_c_d[row][1] = p_d_given_c[row][1]*p_c1_given
    #calculate fd(c)
    fd = np.array([[0,0]]*possible)
    for row in range(possible):
        sum0 = p_c_d[row][0]*e[0,0]+p_c_d[row][1]*e[1,0]
        sum1 = p_c_d[row][0]*e[0,1]+p_c_d[row][1]*e[1,1]
        if sum0 > sum1:
            fd[row][0] = 1
        else:
            fd[row][1] = 1
    ############
    #VALIDATION#
    ############
    #quantize validation data
    for i,item in enumerate(names[:-1]):
        dq = pd.np.digitize(data_val[item], bins=all_bins[i])
        data_val['{}_quantized'.format(item)] = dq
    # print 'validation data quantized'
    #calculate frequency of d,c
    freq_val = np.array([[0, 0]]*possible)
    counter0_val = Counter(tuple(a[1:]) for a in data_val.values[:,5:11] if a[0]==0)
    counter1_val = Counter(tuple(a[1:]) for a in data_val.values[:,5:11] if a[0]==1)
    for key in counter0_val.keys():
        freq_val[int(get_lin_add(key, X))][0] = counter0_val[key]
    for key in counter1_val.keys():
        freq_val[int(get_lin_add(key, X))][1] = counter1_val[key]
    #calculate P(d,c)
    p_d_c_val = freq_val/count_val
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
    if exp_gain > max_exp_gain:
        flag = True
        max_exp_gain = exp_gain
        # optimal_i = i_delta
    print 'sum =', conf_mat.sum(), 'Exp gain = ', exp_gain, 'max Exp gain =', max_exp_gain, countdown

    if (flag==False):
        all_bins[dim][index_boundary] = already_boundary

    if (max_exp_gain == max_temp):
        countdown = countdown - 1
    else:
        countdown = patience
    max_temp = max_exp_gain

    #sequential check
    if (countdown == 0):
        flag_sequential = True
        for dim_seq in range(len(all_bins)):
            if (not flag_sequential):
                break
            for index_boundary_seq in range(len(all_bins[dim_seq])):
                if (not flag_sequential):
                    break
                # find previous boundary
                if index_boundary_seq == 0:
                    prev_boundary_seq = 0
                else:
                    prev_boundary_seq = all_bins[dim_seq][index_boundary_seq - 1]
                # find next boundary
                if index_boundary_seq == (len(all_bins[dim_seq]) - 1):
                    next_boundary_seq = 1
                else:
                    next_boundary_seq = all_bins[dim_seq][index_boundary_seq + 1]
                # find delta
                delta_seq = float(
                    (0.5 * (next_boundary_seq - prev_boundary_seq)) / (10))  # 11 new boundaries to be covered for every boundary
                # max_exp_gain = -float('Inf')
                start_boundary_seq = ((all_bins[dim_seq][index_boundary_seq] + prev_boundary_seq) / float(2))
                # flag = False
                already_boundary_seq = all_bins[dim_seq][index_boundary_seq]
                for i_delta_seq in range(11):
                    all_bins[dim_seq][index_boundary_seq] = start_boundary_seq + (i_delta_seq * delta_seq)
                    ##########
                    #TRAINING#
                    ##########
                    # quantize training data
                    for i, item in enumerate(names[:-1]):
                        dq = pd.np.digitize(data[item], bins=all_bins[i], right=True)
                        data['{}_quantized'.format(item)] = dq
                    # print 'training data quantized'
                    # calculate frequency of d,c
                    freq = np.array([[0, 0]] * possible)
                    counter0 = Counter(tuple(a[1:]) for a in data.values[:, 5:11] if a[0] == 0)
                    counter1 = Counter(tuple(a[1:]) for a in data.values[:, 5:11] if a[0] == 1)
                    for key in counter0.keys():
                        freq[int(get_lin_add(key, X))][0] = counter0[key]
                    for key in counter1.keys():
                        freq[int(get_lin_add(key, X))][1] = counter1[key]
                    # calculate P(d,c)
                    p_d_c = freq / count
                    # calculate P(d|c)
                    p_d_given_c = freq / count  # just initialize
                    for row in range(possible):
                        p_d_given_c[row][0] = p_d_given_c[row][0] / p_c0
                        p_d_given_c[row][1] = p_d_given_c[row][1] / p_c1
                    # calculate P(c|d) (uses p_c0_given and p_c1_given)
                    p_c_given_d = freq / count  # just initialize
                    for row in range(possible):
                        denominator = (p_d_given_c[row][0] * p_c0_given) + (p_d_given_c[row][1] * p_c1_given)
                        if (denominator != 0):
                            p_c_given_d[row][0] = (p_d_given_c[row][0] * p_c0_given) / denominator
                            p_c_given_d[row][1] = (p_d_given_c[row][1] * p_c1_given) / denominator
                    # calculate P(c,d): to be used (**not same as P(d,c) calculated before)
                    # (uses p_c0_given and p_c1_given)
                    p_c_d = freq / count  # just initialize
                    for row in range(possible):
                        p_c_d[row][0] = p_d_given_c[row][0] * p_c0_given
                        p_c_d[row][1] = p_d_given_c[row][1] * p_c1_given
                    # calculate fd(c)
                    fd = np.array([[0, 0]] * possible)
                    for row in range(possible):
                        sum0 = p_c_d[row][0] * e[0, 0] + p_c_d[row][1] * e[1, 0]
                        sum1 = p_c_d[row][0] * e[0, 1] + p_c_d[row][1] * e[1, 1]
                        if sum0 > sum1:
                            fd[row][0] = 1
                        else:
                            fd[row][1] = 1
                    ############
                    #VALIDATION#
                    ############
                    # quantize validation data
                    for i, item in enumerate(names[:-1]):
                        dq = pd.np.digitize(data_val[item], bins=all_bins[i])
                        data_val['{}_quantized'.format(item)] = dq
                    # print 'validation data quantized'
                    # calculate frequency of d,c
                    freq_val = np.array([[0, 0]] * possible)
                    counter0_val = Counter(tuple(a[1:]) for a in data_val.values[:, 5:11] if a[0] == 0)
                    counter1_val = Counter(tuple(a[1:]) for a in data_val.values[:, 5:11] if a[0] == 1)
                    for key in counter0_val.keys():
                        freq_val[int(get_lin_add(key, X))][0] = counter0_val[key]
                    for key in counter1_val.keys():
                        freq_val[int(get_lin_add(key, X))][1] = counter1_val[key]
                    # calculate P(d,c)
                    p_d_c_val = freq_val / count_val
                    # calculate confusion matrix
                    conf_mat = np.array([[0, 0], [0, 0]], dtype=float)  # initialize
                    for tr in range(k):
                        for assigned in range(k):
                            sum = 0
                            for i in range(possible):
                                sum = sum + p_d_c_val[i, tr] * fd[i, assigned] #p_c_d_val changed to p_d_c_val
                            conf_mat[tr, assigned] = sum
                    # calculate economic gain
                    exp_gain = (e * conf_mat).sum()
                    if exp_gain > max_exp_gain:
                        # flag = True
                        max_exp_gain = exp_gain
                        max_temp = max_exp_gain
                        optimal_i_seq = i_delta_seq
                        all_bins[dim_seq][index_boundary_seq] = start_boundary_seq + (optimal_i_seq * delta_seq)
                        countdown = patience
                        print 'sum =', conf_mat.sum(), 'Exp gain = ', exp_gain, 'max Exp gain =', max_exp_gain, countdown
                        print all_bins
                        flag_sequential = False
                        break
                    else:
                        all_bins[dim_seq][index_boundary_seq] = already_boundary_seq
                        print 'checking if convergence is successful..'


##################################
##################################
###OPTIMUM BIN BOUNDARIES FOUND###
##################################
##################################


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
print max_exp_gain
print all_bins

print '\n'
print '-----------------'+str(time()-start_time)+'-----------------'