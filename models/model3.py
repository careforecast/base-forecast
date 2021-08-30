import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import LSTM, Dense, Conv2D, GlobalMaxPooling2D, Dropout,MaxPooling2D
import os
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint,EarlyStopping
from utility.preprocessing3 import clean_data, preprocessing, smote,normalise_data, preprocess_column, preprocess_column,split_data
from tensorflow.keras.regularizers import l1_l2

def custom_model(input_dim=10,input_length=10):
    
    model = Sequential()
    model.add(LSTM(4, input_dim = input_dim, input_length = input_length))
    model.add(Dropout(0.5))
    model.add(Dense(16,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model
    
    
def train_model(X_train,y_train,X_val,y_val, size=100):
    save_path="save_model/model_{}.h5".format(size)
    filepath=os.path.join('save_model','model_{}'.format(size))
    if not os.path.isdir(filepath):
        os.mkdir(filepath)
    input_length = X_train.shape[1]
    input_dim = X_train.shape[2]
    redlor=ReduceLROnPlateau(monitor='val_accuracy', factor=0.001, patience=5, verbose=1,mode='max', min_delta=0.0003,min_lr=1e-10)
    checkpoint=ModelCheckpoint(os.path.join(filepath,'model.h5'),monitor="val_accuracy",verbose=1,save_best_only=True,mode="max",save_freq="epoch")
    early_stop=EarlyStopping(monitor="val_accuracy",min_delta=0.0002,patience=5,verbose=1,mode="max",baseline=0.82,restore_best_weights=True,)
    model = Sequential()
    model.add(LSTM(6, input_dim = input_dim, input_length = input_length))
    model.add(Dense(16,activation='relu',kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train,
              batch_size=32, epochs=50,
              validation_data=(X_val,y_val),
              callbacks=[redlor,checkpoint, early_stop],
              verbose = 1)
    print('Model saved to ',os.getcwd()+'/'+filepath)
    return history
    
    

if __name__ == '__main__':
    DATA_SIZE=10000
    BASE_PATH=f'data/{DATA_SIZE}'
    train_df=pd.read_csv(f'{BASE_PATH}/train.csv')
    X_train,y_train,X_test,y_test,X_val,y_val=split_data(train_df)
    history = train_model(X_train,y_train,X_val,y_val,DATA_SIZE)    
