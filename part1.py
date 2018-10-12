import numpy as np
import math
from functools import reduce
from operator import mul

print('input the number of feature vectors')
n = int(input('numbers are:'))

print ('enter the value of Mn for each dimension')
M = np.zeros(n)
for i in range(n):
    M[i] = int(input(i))

print('Enter the number of classes')
k = int(input('enter :'))

print('enter the economic gain matrix row wise')
economic_gain = np.zeros((k, k))
for i in range(k):
    print('row')
    for j in range(k):
        economic_gain[i][j] = input(j)
dim_prob = 1
for ele in M:
    dim_prob = dim_prob * ele
length = int(dim_prob)
prob_DgvnC = np.zeros((length, 2))

tuple = np.zeros(n)
p = np.zeros(k)

# compute linear address
#linear address
def get_mul_list(M):
    x = []
    for i in range(len(M) - 1):
        x.append(reduce(mul, M[i + 1:], 1))
    x.append(1)
    return x
X = get_mul_list(M)	
def get_lin_add(d_m): #d={d_1,d_2,...,d_N}
    sum = 0
    for i in range(n):
        sum += X[i]*d_m[i]
    print("linear address is:")
    print(sum)
    return sum


total_d_known = int(input("enter the number of measurement vector d"))
d = np.zeros((total_d_known, n))
print("Input d and P(d/c) the class conditional probabilities  class wise 0,1,2,... ")
for measurement in range(total_d_known):
  
    print('enter the value of d')
    for y in range(n):  # warning if probability is not in 0 and 1
        d[measurement][y] = int(input())

    for z in range(k):
        print('class ')
        p[z] = float(input(z))
        if 1 < p[z] or p[z] < 0:
            print('please give probability between 0 and 1')
            break
   # la = d[0]
    #for j in range(n):
     #   la = la * M[j] + (d[j])
    #a = int(la)
    a= int(get_lin_add(d[measurement]))
    for w in range(k):
        prob_DgvnC[a][w] = p[w]

print(' Enter the values of prior class probabilities p_c where 0<=p_c<=1 classwise in the order of class 0,1,2.. ')
p_c = np.zeros(k)
for i in range(k):
    p_c[i] = float(input('enter now'))
from math import *
# Compute P(c/d)
prob_CgvnD = np.zeros((length, 2))
prob_D = np.zeros(length)
for i in range(length):
    for j in range(k):
        prob_D[i] = prob_D[i] + prob_DgvnC[i][j] * p_c[j]
for i in range(length):
    for j in range(k):
        if (prob_D[i]!=0):
            prob_CgvnD[i][j] = (prob_DgvnC[i][j] * p_c[j]) / prob_D[i]
# compute p(c,d)
prob_CandD = np.zeros((length, k))

for i in range(length):
    for j in range(k):
        prob_CandD[i][j] = prob_CgvnD[i][j] * prob_D[i]
# Compute decision rule:
search = np.zeros(k)
F_DgvnC = np.zeros((length, k))

for i in range(length):
    for j in range(k):
        search[j] = np.dot(prob_CandD[i,:],economic_gain[j,:])                                                                                         
    z = np.argmax(search)
    F_DgvnC[i][z] = 1

# Calculate the confusion matrix
confusion_matrix = np.zeros((k, k))
for i in range(k):
    for j in range(k):
        confusion_matrix[i][j] = np.dot(prob_CandD[:, i],  F_DgvnC[:, j])

# Calculate the expected gain
expected_gain = 0
for i in range(k):
    for j in range(k):
        expected_gain = expected_gain + confusion_matrix[i][j] * economic_gain[i][j]
print(expected_gain)