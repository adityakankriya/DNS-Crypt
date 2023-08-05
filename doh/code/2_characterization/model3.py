#Loading Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from tensorflow.keras import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM, Conv1D, MaxPool1D
from tensorflow.keras.callbacks import EarlyStopping
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
X_train_unsqueeze=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test_unsqueeze=X_test.reshape(X_test.shape[0],X_test.shape[1],1)


#Running Model
def create_model(segment_size):
    model = Sequential()
    model.add(Dense(10, input_dim=segment_size, activation='relu'))
    model.add(Flatten())
    model.add(Dense(segment_size * 6, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(segment_size * 2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model

modelDNN=create_model(X_train.shape[1])
history=modelDNN.fit(X_train,np.array(Y_train),epochs=50,batch_size=32,validation_split=0.3,callbacks=EarlyStopping(monitor='val_loss'))

#Testing
Y_pred=modelDNN.predict(X_test)
Y_pred=[1 if i[0]>0.5 else 0 for i in Y_pred]
target=['Malicious','Benign']
file=open('../../output/2_characterization/model3.txt','w')
file.write(classification_report(Y_test,Y_pred,target_names=target))
file.close()

#Saving Model
file=open('../../systems/l2/model3.pkl','wb')
pickle.dump(modelDNN,file,protocol= pickle.HIGHEST_PROTOCOL)
file.close()

