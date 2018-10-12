##########################################################
###DESCRIPTIONS OF THE SCRIPT FILES AND HOW TO USE THEM###
##########################################################

Programming language: Python version 2.7
Python packages required: Numpy, Pandas

1. entropy.py

Input REQUIRED before running:
(i) filename (this should be pr_data)
(ii) Mem (Mem = 10,000 is used in the code by default)


2. boundary.py

Input REQUIRED before running:
(i) M (this is a list of quantization levels for each dimension, i.e Lj found)
(ii) filename (this should be the training dataset file)
(iii) filename_val (this should be the validation dataset file)
(iv) k (number of classes)
(v) p_c0_given, p_c1_given (prior probabilities)
(vi) e (economic gain matrix)


3. smoothing.py

Input REQUIRED before running:
(i) k_smoothing (the smoothing parameter)
(ii) M (this is a list of quantization levels for each dimension, i.e Lj found)
(iii) filename (this should be the training dataset file)
(iv) filename_val (this should be the validation dataset file; set this to be the testset if you want to test it on the testset)
(v) k (number of classes)
(vi) p_c0_given, p_c1_given (prior probabilities)
(vii) e (economic gain matrix)
(viii) all_bins (this is a list of bin boundaries for all dimensions, obtained after bin boundaries optimization)

4. part1.py

The above script has functions for part 1 of the assignment.

Note: We split the given dataset pr_data to 3 files train.txt, val.txt and testset.txt each having same number of lines. We use these files for training, validation and testing respectively.