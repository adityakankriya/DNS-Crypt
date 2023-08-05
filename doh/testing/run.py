import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import sys

def l1_result(X):
    Y_pred=model1.predict(X)
    if len(Y_pred.shape)==2:
        return np.array([1 if i[0]>0.5 else 0 for i in Y_pred])
    return Y_pred

def l2_result(X):
    Y_pred=model2.predict(X)
    if len(Y_pred.shape)==2:
        return np.array([1 if i[0]>0.5 else 0 for i in Y_pred])
    return Y_pred

def convert(a,b):
    if a==0:
        return 1
    return b

#Opening Model at l1
file=open('../systems/l1/model'+str(sys.argv[1])+'.pkl','rb')
model1=pickle.load(file)
file.close()

#Opening Model at l2
file=open('../systems/l2/model'+str(sys.argv[2])+'.pkl','rb')
model2=pickle.load(file)
file.close()

data=pd.read_csv('input/final.csv')
data.dropna( axis=0, how="any",inplace=True)
data=data.iloc[:,:-1]

X_test=data[['FlowBytesSent', 'FlowSentRate',
       'FlowBytesReceived', 'FlowReceivedRate', 'PacketLengthVariance',
       'PacketLengthStandardDeviation', 'PacketLengthMean',
       'PacketLengthMedian', 'PacketLengthMode', 'PacketLengthSkewFromMedian',
       'PacketLengthSkewFromMode', 'PacketLengthCoefficientofVariation',
       'PacketTimeVariance', 'PacketTimeStandardDeviation', 'PacketTimeMean',
       'PacketTimeMedian', 'PacketTimeMode', 'PacketTimeSkewFromMedian',
       'PacketTimeSkewFromMode', 'PacketTimeCoefficientofVariation',
       'ResponseTimeTimeVariance', 'ResponseTimeTimeStandardDeviation',
       'ResponseTimeTimeMean', 'ResponseTimeTimeMedian',
       'ResponseTimeTimeMode', 'ResponseTimeTimeSkewFromMedian',
       'ResponseTimeTimeSkewFromMode',
       'ResponseTimeTimeCoefficientofVariation']].values

l1=['Non-DoH','DoH']
l2=['Malicious','Benign']

data['DoH']=l1_result(X_test)
data['Benign']=l2_result(X_test)
data['Benign']=data.apply(lambda row:convert(row['DoH'],row['Benign']),axis=1)
data.to_csv('output/output.csv',index=False)
