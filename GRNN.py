import matplotlib.pyplot as plt
import copy
from scipy.spatial import distance
from scipy.spatial import distance_matrix
import math
import random
import pandas as pd
import numpy as np

#create empty list and define sigma
Ij_all=[]
f_Ij_all=[]
sum_Ij_all=[]
output_all=[]
sigma=0.5

#read training dataset 
df=pd.read_excel('TrainData.xls')
#assign input-pattern weights with X 
weight_input=df.iloc[:, 0]
#assign pattern-summation weights with Y
weight_output=df['Y=erf(x)']
#assign B unit
B=np.ones(len(weight_input),)

#read prediction dataset
df_pred=pd.read_excel('PredictData.xls')
#assign inputs with X
inputs_pred=df_pred.iloc[:, 0]

#calculate the absolute value of weights minus input pattern
#1D input, so no sum
for k in range (0,len(inputs_pred)):
    for i in weight_input:
        Ij=abs(i-inputs_pred[k])
        Ij_all.append(Ij)

#split Ij_all into 9 lists, each containing 26 values
Ij_all_split=[Ij_all[i:i + len(weight_input)] for i in range(0, len(Ij_all), len(weight_input))]

#calculate exponential f value for each Ij
for i in Ij_all_split:
    for j in i:
        f_Ij=math.exp(-j/(2*(sigma**2)))
        f_Ij_all.append(f_Ij)
        
#split Ij_all into 9 lists, each containing 26 values
fIj_all_split=[f_Ij_all[i:i + len(weight_input)] for i in range(0, len(f_Ij_all), len(weight_input))]

#calculate final predicted output
for i in fIj_all_split:
    A_output=np.dot(i, weight_output)
    B_output=np.dot(i, B)
    output=A_output/B_output
    output_all.append(output)
    
#plot training data and prediction data
plt.plot(weight_input, weight_output, '-o', c='r')
plt.plot(inputs_pred, output_all, '-o', c='b')
plt.show()    

    
