import pandas as pd
import os,shutil,math,scipy,cv2
import numpy as np
import matplotlib.pyplot as plt
import random as rn
import sys
import argparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,CSVLogger,ReduceLROnPlateau,LearningRateScheduler
from tensorflow.keras import backend as K

def weight_mapping(filename, label):
        return label[label['filename']==filename]['weight'].values[0]

def means(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return mean_squared_error(y_true,y_pred)


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    print('first = ',y_true)
    print('second = ',y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser_options = ['--path', '--option']

    for parser_option in parser_options:
        parser.add_argument(parser_option)

    args = parser.parse_args()

    imgsize = 200
    x = []
    y = []
    food = []
    label = pd.read_csv(os.path.join(args.path, 'csv', args.option+'.csv'))
    path_data = os.path.join(args.path, args.option)

    for file in tqdm(os.listdir(path_data)):
        weight = weight_mapping(file, label)
        img = cv2.imread(os.path.join(path_data, file), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (imgsize, imgsize))
        x.append(np.array(img))
        y.append(weight)
        food.append(label[label['filename']==file]['food'].values[0])

    x = np.array(x)
    y = np.array(y)
    
    print(args.option)
    if args.option=='train':
        base_model = InceptionResNetV2(include_top=False,
                    input_shape = (imgsize,imgsize,3),
                    weights = 'imagenet')
        model = Sequential()
        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dropout(0.25))
        model.add(Dense(1, activation='relu'))
        model.summary()

        model.compile(
            loss='mean_squared_error',
            optimizer='adam',
        )
        augs_gen = ImageDataGenerator(
            rotation_range=180,  
            horizontal_flip=True,  
            vertical_flip=True,
            dtype='uint8'
        )
        history = model.fit(
            augs_gen.flow(x, y, batch_size=10),
            steps_per_epoch  = 75,
            epochs = 100, 
            verbose = 1,
        )
        model.save("model_estimation.h5")
    elif args.option=='test':
        model = load_model('model_estimation.h5')
        y_pred = model.predict(x)

        result = pd.DataFrame({'real':y.flatten(), 'predict':y_pred.flatten()})
        result['food'] = food

        report = dict()
        report['id'] = list(result['food'].unique())
        report['mape'] = []
        report['mse'] = []
        print('Mean absolute percentage error')
        for i in result['food'].unique():
            food = result[result['food']==i]
            mape = mean_absolute_percentage_error(food['real'],food['predict'])
            report['mape'].append(mape)
            print(i+': '+str(mape))
            #print('ytrue'+)
            
        print('Mean squared error')
        for i in result['food'].unique():
            food = result[result['food']==i]
            mse = means(food['real'],food['predict'])
            report['mse'].append(mse)
            print('mse: '+i+': '+str(mse))