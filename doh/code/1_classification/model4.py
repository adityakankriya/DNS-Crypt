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
data1=pd.read_csv(data_path+'l1-doh.csv')
data2=pd.read_csv(data_path+'l1-nondoh.csv')
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
Y_train=[1 if i == 'DoH' else 0 for i in Y_train]
X_test=scaler.transform(X_test)
Y_test=[1 if i == 'DoH' else 0 for i in Y_test]
X_train_unsqueeze=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test_unsqueeze=X_test.reshape(X_test.shape[0],X_test.shape[1],1)


#Running Model
def create_model_cnn(shape):
    model = Sequential()
    model.add(Conv1D(shape[0] * 2, kernel_size=3, input_shape=shape, activation='relu'))
    model.add(MaxPool1D())
    model.add(Flatten())
    model.add(Dense(shape[0] * 6, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(shape[0] * 2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model

modelCNN=create_model_cnn(X_train_unsqueeze.shape[1:])
history=modelCNN.fit(X_train_unsqueeze,np.array(Y_train),epochs=50,batch_size=32,validation_split=0.3,callbacks=EarlyStopping(monitor='val_loss'))

#Testing
Y_pred=modelCNN.predict(X_test)
Y_pred=[1 if i[0]>0.5 else 0 for i in Y_pred]
target=['Non-DoH','DoH']
file=open('../../output/1_classification/model4.txt','w')
file.write(classification_report(Y_test,Y_pred,target_names=target))
file.close()

#Saving Model
file=open('../../systems/l1/model4.pkl','wb')
pickle.dump(modelCNN,file,protocol= pickle.HIGHEST_PROTOCOL)
file.close()

