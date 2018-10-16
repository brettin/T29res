import pandas as pd
import numpy as np
import os
import sys
import gzip
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keras as ke
from keras.layers import Input, Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Sequential, Model, model_from_json, model_from_yaml
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, LearningRateScheduler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler


file_path = os.path.dirname(os.path.realpath(__file__))

# candle
sys.path.append('/raid/brettin/Benchmarks/common')
import candle_keras

# candle
def initialize_parameters():
    t29_common = candle_keras.Benchmark(file_path, 't29_default_model.txt','keras',
                            prog='t29res.py',desc='resnet')

    # Need a pointer to the docs showing what is provided
    # by default
    additional_definitions = [
        {'name':'connections',
         'default':1,
         'type':int,
         'help':'The number of residual connections.'},
        {'name':'distance',
         'default':1,
         'type':int,
         'help':'Residual connection distance between dense layers.'}
    ]
    t29_common.additional_definitions = additional_definitions
    gParameters = candle_keras.initialize_parameters(t29_common)
    return gParameters


def load_data(nb_classes, PL, gParameters):
    train_path=gParameters['train_path']
    test_path=gParameters['test_path']
    df_train = (pd.read_csv(train_path,header=None).values).astype('float32')
    df_test = (pd.read_csv(test_path,header=None).values).astype('float32')

    print('df_train shape:', df_train.shape)
    print('df_test shape:', df_test.shape)

    df_y_train = df_train[:,0].astype('int')
    df_y_test = df_test[:,0].astype('int')

    Y_train = np_utils.to_categorical(df_y_train,nb_classes)
    train_classes = np.argmax(Y_train, axis=1)
    np.savetxt("train_classes.csv", train_classes, delimiter=",", fmt="%d")
    
    Y_test = np_utils.to_categorical(df_y_test,nb_classes)
    test_classes = np.argmax(Y_test, axis=1)
    np.savetxt("test_classes.csv", test_classes, delimiter=",", fmt="%d")
              
    df_x_train = df_train[:, 1:PL].astype(np.float32)
    df_x_test = df_test[:, 1:PL].astype(np.float32)
            
    # not sure the extra variable is needed, and is this a copy or reference
    X_train = df_x_train
    X_test = df_x_test
            
    scaler = MaxAbsScaler()
    mat = np.concatenate((X_train, X_test), axis=0)
    mat = scaler.fit_transform(mat)
       
    X_train = mat[:X_train.shape[0], :]
    X_test = mat[X_train.shape[0]:, :]
        
    return X_train, Y_train, X_test, Y_test

# This is required for candle compliance.
# It essentially wraps what was in the implicit main funcion
def run(gParameters):
    print ('gParameters: ', gParameters)

    X_train, Y_train, X_test, Y_test = load_data(nb_classes, PL, gParameters)

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    print('Y_train shape:', Y_train.shape)
    print('Y_test shape:', Y_test.shape)

    # load json and create model
    json_file = open('t29res.model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model_json = model_from_json(loaded_model_json)

    # load yaml and create model
    yaml_file = open('t29res.model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model_yaml = model_from_yaml(loaded_model_yaml)

    # load weights into new model
    loaded_model_json.load_weights("t29res.model.h5")
    print("Loaded json model from disk")

    # evaluate json loaded model on test data
    loaded_model_json.compile(loss='binary_crossentropy', optimizer=gParameters['optimizer'], metrics=['accuracy'])
    score_json = loaded_model_json.evaluate(X_test, Y_test, verbose=0)

    print('json Validation loss:', score_json[0])
    print('json Validation accuracy:', score_json[1])
    print("json %s: %.2f%%" % (loaded_model_json.metrics_names[1], score_json[1]*100))

    # load weights into new model
    loaded_model_yaml.load_weights("t29res.model.h5")
    print("Loaded yaml model from disk")

    # evaluate loaded model on test data
    loaded_model_yaml.compile(loss='binary_crossentropy', optimizer=gParameters['optimizer'], metrics=['accuracy'])
    score_yaml = loaded_model_yaml.evaluate(X_test, Y_test, verbose=0)

    print('yaml Validation loss:', score_yaml[0])
    print('yaml Validation accuracy:', score_yaml[1])
    print("yaml %s: %.2f%%" % (loaded_model_yaml.metrics_names[1], score_yaml[1]*100))

    # predict using loaded yaml model on test and training data
    predict_yaml_train = loaded_model_yaml.predict(X_train)
    predict_yaml_test = loaded_model_yaml.predict(X_test)

    print('Yaml_train_shape:', predict_yaml_train.shape)
    print('Yaml_test_shape:', predict_yaml_test.shape)

    predict_yaml_train_classes = np.argmax(predict_yaml_train, axis=1)
    predict_yaml_test_classes = np.argmax(predict_yaml_test, axis=1)

    np.savetxt("predict_yaml_train.csv", predict_yaml_train, delimiter=",", fmt="%.3f")
    np.savetxt("predict_yaml_test.csv", predict_yaml_test, delimiter=",", fmt="%.3f")

    np.savetxt("predict_yaml_train_classes.csv", predict_yaml_train_classes, delimiter=",",fmt="%d")
    np.savetxt("predict_yaml_test_classes.csv", predict_yaml_test_classes, delimiter=",",fmt="%d")

    return history

# This is also added for candle compliance so that the program can
# still be executed independently from the command line.
def main():

    gParameters = initialize_parameters()
    run(gParameters)

if __name__ == '__main__':
    main()
    try:
        ke.clear_session()
    except AttributeError:      # theano does not have this function
        pass

