import numpy as np
import pandas as pd
from math import *
import random
from collections import Counter as C 
import itertools

filename = "pr_data.txt"
df = pd.read_csv(filename ,delim_whitespace=True, names = ["d0" , "d1" , "d2" , "d3" , "d4" ,"c"])
total_length = len(df)

total_train=int(total_length*1/3)
count_validate=2*total_train
data=df.head(n=total_train)
validate= df[total_train:count_validate]
m= int(len(data))
M=10000
l=int(m/10)
length_1 = 1/l
x=0
z=0
bin = np.zeros(l)
for i in range(l):
	bin[i]= length_1 + x
	x=bin[i]
names_bins = ["bin_d0","bin_d1","bin_d2","bin_d3","bin_d4"]
for ele in names_bins:
	data[ele] = pd.np.digitize(data.values[:,z],bins = bin,right=True)	
	z=z+1
	
	
#define entropy
def entropy(d):
	d_unique, mk = np.unique(d,return_counts = True)   
	m1=mk.shape[0]
	n0 = l - m1
	p_k = np.zeros(m1)
	sum_p_k = 0
	for i in range(m1):
		p_k[i] = mk[i]/m
		sum_p_k += p_k[i]*(log(p_k[i],2))
	#entropy for d0
	C = (n0-1)/(2*(d.shape[0])*log(2))
	H_i = -1*sum_p_k + C
	return(H_i)

# K select algorithm	
def select(d, n):
	
   
    d = list(d)
   
    while True:
        pivot = random.choice(d)
        pcount = 0
        under, over = [], []
        uappend, oappend = under.append, over.append
        for elem in d:
            if elem < pivot:
                uappend(elem)
            elif elem > pivot:
                oappend(elem)
            else:
                pcount += 1
        if (n < len(under)):
            d = under
        elif n < len(under) + pcount:
            return pivot
        else:
            d = over
            n -= len(under) + pcount

# calculate Lj
def main():
	H=np.zeros(5)
	sum_H=0
	for i in range(6,11):
		d=data.iloc[:,i]
		arr_d = np.array(d)
		H[i-6] = entropy(arr_d)
		sum_H+=H[i-6]
	f=np.zeros(5)
	global L
	L=np.zeros(5)
	bin_size=np.zeros(5)
	for i in range(5):
		f[i]=H[i]/sum_H
		L[i]=floor(M**f[i])
		print(L[i])
main()