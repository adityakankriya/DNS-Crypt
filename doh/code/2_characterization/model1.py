#Loading Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings("ignore")


#Data Loading
data_path='../../processed_data/'
data1=pd.read_csv(data_path+'l2-benign.csv')
data2=pd.read_csv(data_path+'l2-malicious.csv')
data=data1.append(data2,ignore_index=True)

#Data Preprocessing
dataset=data[['FlowBytesSent', 'FlowSentRate',
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
       'ResponseTimeTimeCoefficientofVariation','Label']]
dataset.dropna( axis=0, how="any",inplace=True)
X=dataset[['FlowBytesSent', 'FlowSentRate',
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
       'ResponseTimeTimeCoefficientofVariation']]
Y=dataset['Label']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,stratify=Y)
scaler = MinMaxScaler().fit(X_train)
X_train=scaler.transform(X_train)
Y_train=[1 if i == 'Benign' else 0 for i in Y_train]
X_test=scaler.transform(X_test)
Y_test=[1 if i == 'Benign' else 0 for i in Y_test]

#Running Model
rf=RandomForestClassifier(max_depth=10, random_state=0)
rf.fit(X_train,Y_train)

#Testing
Y_pred=rf.predict(X_test)
target=['Non-DoH','DoH']
file=open('../../output/2_characterization/model1.txt','w')
file.write(classification_report(Y_test,Y_pred,target_names=target))
file.close()

#Saving Model
file=open('../../systems/l2/model1.pkl','wb')
pickle.dump(rf,file)
file.close()
