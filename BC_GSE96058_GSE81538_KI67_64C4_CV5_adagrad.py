# Script for 5-fold cross validation for the combined GSE96058 and GSE81538 dataset

import pandas as pd
import numpy as np
import os
from os import path
import tensorflow as tf
from keras import backend as K 
from keras import initializers
from keras.models import Model, model_from_json
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalAvgPool1D, GlobalMaxPool1D
from keras.layers import Input, Flatten, Dense, Dropout
from keras.layers import concatenate, add, maximum, subtract, average
from keras.layers import BatchNormalization, Activation
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from keras import optimizers
from keras import regularizers
from keras import initializers
from imblearn.over_sampling import RandomOverSampler, BorderlineSMOTE
import timeit
import datetime
import matplotlib.pyplot as plt
import math
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import roc_auc_score

# define home directory and parameters
HOME_DIR = "D:/Cancers/BreastCancer/"
os.chdir("D:/Cancers/BreastCancer/")

# read in data 1
df = pd.read_csv('GSE96058_sorted_ki67.128_lab_feat.csv', header=None)
df = np.array(df, dtype=np.float32)

# read in data 2
df2 = pd.read_csv('GSE81538_sorted_ki67.128_lab_feat.csv', header=None)
df2 = np.array(df2, dtype=np.float32)

# combine the two datasets together
ki67 = np.concatenate((df, df2), axis=0)

ki67_exp = ki67[:, 1:]
ki67_exp = np.array(ki67_exp, dtype=np.float32)
ki67_exp.shape

ki67_label = ki67[:, 0]
ki67_label = np.array(ki67_label, dtype=np.int32)
ki67_label.shape

# use oversampling to balance the training set

# use borderlineSMOTE
oversample = BorderlineSMOTE()

# fit and apply the transform
ki67_exp, ki67_label = oversample.fit_resample(ki67_exp, ki67_label)

ki67_exp = np.reshape(ki67_exp, (-1, 64, 64, 4))

# del array objects to save memory
del df
del df2
del ki67

# batch size and numEpochs
numEpoch = 200
batchSize = 25
LR = 0.001
DECAY = 0.000
EPSILON = 0.99
DROPOUT = 0.5
DROP = 0.9
EPOCHS_DROP = 20

# instantiate L2 regularizers
reg2 = regularizers.l2(0.185)

# optimizer parameters
adagrad = optimizers.Adagrad(lr=LR, epsilon=EPSILON, decay=DECAY)

# learning rate scheduler
# define step decay function
class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))
        print('lr:', step_decay(len(self.losses)))

def step_decay(epoch):
    initial_lrate = LR
    drop = DROP
    epochs_drop = EPOCHS_DROP
    lrate = initial_lrate * math.pow(drop, math.floor((epoch)/epochs_drop))
    return lrate

# use class weights
CLASS_WEIGHT = {0: 1.1064, 1: 0.9122}

# embedding parameters
inputDim = 128
outputDim = 32
inputLength = 16384

# input layers. Chanage this as necessary
exp_input = Input(shape=(64, 64, 4, ))
exp_input2 = Flatten()(exp_input)

# define k-fold cross validation test harness
cvFold = 5
kfold = KFold(n_splits=cvFold, shuffle=True, random_state=789)
train_acc_cvscores = []
test_acc_cvscores = []
train_auc_cvscores = []
test_auc_cvscores = []
i = 0

for train_idx, test_idx in kfold.split(ki67_exp, ki67_label):
    checkpointer = ModelCheckpoint(
        filepath='./best_weights_' + str(i) + '.hdf5',
        monitor="val_acc",
        save_best_only=True,
        save_weights_only=False,
        verbose=2)

    earlyStoper = EarlyStopping(
        monitor='val_loss', 
        mode='min', 
        restore_best_weights=True, 
        patience = 5,
        verbose=2)
    
    # learning schedule callback
    loss_history = LossHistory()
    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [loss_history, lrate, checkpointer]
    
   # build the model with expression data
    exp_conv = Conv2D(
        filters=64, 
        kernel_size=3, 
        kernel_regularizer=reg2, 
        kernel_initializer='glorot_normal',
        activation='relu', 
        padding ='same')(exp_input)
    exp_conv = Conv2D(
        filters=64, 
        kernel_size=3, 
        kernel_regularizer=reg2, 
        kernel_initializer='glorot_normal',
        activation='relu', 
        padding ='same')(exp_conv)
    exp_conv = BatchNormalization(axis=-1, center=True, scale=False)(exp_conv)
    exp_conv = AveragePooling2D(pool_size = 2, strides = 2)(exp_conv)
    exp_conv = Conv2D(
        filters=128, 
        kernel_size=3, 
        kernel_regularizer=reg2, 
        kernel_initializer='glorot_normal',
        activation='relu', 
        padding ='same')(exp_conv)
    exp_conv = Conv2D(
        filters=128, 
        kernel_size=3, 
        kernel_regularizer=reg2, 
        kernel_initializer='glorot_normal',
        activation='relu', 
        padding ='same')(exp_conv)
    exp_conv = BatchNormalization(axis=-1, center=True, scale=False)(exp_conv)
    exp_conv = AveragePooling2D(pool_size = 2, strides = 2)(exp_conv)
    exp_conv = Conv2D(
        filters=256, 
        kernel_size=3, 
        kernel_regularizer=reg2, 
        kernel_initializer='glorot_normal',
        activation='relu', 
        padding ='same')(exp_conv)
    exp_conv = Conv2D(
        filters=256, 
        kernel_size=3, 
        kernel_regularizer=reg2, 
        kernel_initializer='glorot_normal',
        activation='relu', 
        padding ='same')(exp_conv)
    exp_conv = BatchNormalization(axis=-1, center=True, scale=False)(exp_conv)
    exp_conv = AveragePooling2D(pool_size = 2, strides = 2)(exp_conv)
    exp_conv = Conv2D(
        filters=512, 
        kernel_size=7, 
        kernel_regularizer=reg2, 
        kernel_initializer='glorot_normal',
        activation='relu', 
        padding ='same')(exp_conv)
    exp_conv = BatchNormalization(axis=-1, center=True, scale=False)(exp_conv)
    exp_conv = AveragePooling2D(pool_size = 1, strides = 1)(exp_conv)
    exp_conv = Flatten()(exp_conv)
    exp_conv = Dropout(rate=DROPOUT)(exp_conv)
    exp_conv = Dense(units=512, activation='relu')(exp_conv)
    exp_conv = Dropout(rate=DROPOUT)(exp_conv)
    exp_conv = Dense(units=512, activation='relu')(exp_conv)
    exp_conv = Dropout(rate=DROPOUT)(exp_conv)
    exp_conv = Dense(units=512, activation='relu')(exp_conv)
    exp_conv = Dropout(rate=DROPOUT)(exp_conv)
    exp_conv = Dense(units=64, activation='relu')(exp_conv)

    combined_output = Dense(units=1, activation='sigmoid')(exp_conv)
    classifier = Model(inputs=exp_input, outputs=combined_output)
    # summarize layers
    print("Model summary: \n", classifier.summary())

    # compile the model
    classifier.compile(
        optimizer=adagrad, 
        loss='binary_crossentropy', 
        metrics=['acc', tf.keras.metrics.AUC()])
    
    # fit the model with training data
    training_start_time = timeit.default_timer()
    history = classifier.fit(
        x = ki67_exp[train_idx],
        y = ki67_label[train_idx],
        batch_size = batchSize,
        epochs = numEpoch,
        validation_data = (ki67_exp[test_idx], ki67_label[test_idx]),
        callbacks=callbacks_list,
        class_weight=CLASS_WEIGHT,
        shuffle=True,
        verbose=2)

    training_end_time = timeit.default_timer()
    print("Training time: {:10.2f} min. \n" .format((training_end_time - training_start_time) / 60))   

    # save training history for late use
    # first convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history) 
    # then save it as a csv file
    hist_csv_file = './train_history_' + str(i) + '.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f, float_format='%0.6f')
    f.close

    # evaluate the model with testing data
    train_scores = classifier.evaluate(ki67_exp[train_idx], ki67_label[train_idx], verbose = 0 )
    test_scores = classifier.evaluate(ki67_exp[test_idx], ki67_label[test_idx], verbose = 0 )
    print("%s: %.2f%%" % (classifier.metrics_names[1], train_scores[1]*100))
    print("%s: %.2f%%" % (classifier.metrics_names[1], test_scores[1]*100))    
    train_acc_cvscores.append(train_scores[1]*100)
    test_acc_cvscores.append(test_scores[1]*100)
	
    # prediction 
    test_pred_prob = classifier.predict(ki67_exp[test_idx])
    train_pred_prob = classifier.predict(ki67_exp[train_idx])
    print("Predicted outcomes:\n")
    np.set_printoptions(precision=3, suppress=True)
    print(np.c_[ki67_label[test_idx], test_pred_prob])

    #Confution Matrix and Classification Report
    test_y_pred = np.where(test_pred_prob > 0.5, 1, 0)
    print('Confusion Matrix')
    print(confusion_matrix(ki67_label[test_idx], test_y_pred))
    print('Classification Report')
    target_names = ['Ki67-', 'Ki67+']
    print(classification_report(ki67_label[test_idx], test_y_pred, target_names=target_names))

    # plot training and validation history
    fig1 = plt.figure(i)
    plt.plot(
        history.history['acc'], 
        label='train (acc={:.3f})'.format(train_scores[1]), 
        linestyle='-')
    plt.plot(
        history.history['val_acc'], 
        label='test (acc={:.3f})'.format(test_scores[1]), 
        linestyle='-.')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.savefig('GSE96058_GSE81538_ki67_64C4_cv' + str(cvFold) + '_v2.2.0_train' + str(i) +'.png')
    plt.close(fig1)
    
    # get ROC data for training and testing
    y_pred_train = classifier.predict(ki67_exp[train_idx]).ravel()
    y_pred_test = classifier.predict(ki67_exp[test_idx]).ravel()

    fpr_train, tpr_train, thresholds_train = roc_curve(ki67_label[train_idx], y_pred_train)
    fpr_test, tpr_test, thresholds_test = roc_curve(ki67_label[test_idx], y_pred_test)

    auc_train = roc_auc_score(ki67_label[train_idx], y_pred_train)
    auc_test = roc_auc_score(ki67_label[test_idx], y_pred_test)

    train_auc_cvscores.append(auc_train)
    test_auc_cvscores.append(auc_test)

    # plot ROC 
    fig2=plt.figure(cvFold + i)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(
        fpr_train, 
        tpr_train, 
        label='train auc: {:.3f}'.format(auc_train), 
        linestyle='-')
    plt.plot(
        fpr_test, 
        tpr_test, 
        label='test auc: {:.3f}'.format(auc_test), 
        linestyle='-.')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig('GSE96058_GSE81538_ki67_64C4_cv' + str(cvFold) + '_v2.2.0_ROC_figure' + str(cvFold + i) + '.png')
    plt.close(fig2)
    
    i = i + 1
#
# print CV results
print("\nCV results for the training acc: \n")
print("%.2f%% (+/- %.2f%%)" % (np.mean(train_acc_cvscores, axis=0), np.std(train_acc_cvscores, axis=0)))
print("\nCV results for the testing acc: \n")
print("%.2f%% (+/- %.2f%%)" % (np.mean(test_acc_cvscores, axis=0), np.std(test_acc_cvscores, axis=0)))
print("\nCV results for the training auc: \n")
print("%.3f (+/- %.3f)" % (np.mean(train_auc_cvscores, axis=0), np.std(train_auc_cvscores, axis=0)))
print("\nCV results for the testing auc: \n")
print("%.3f (+/- %.3f)" % (np.mean(test_auc_cvscores, axis=0), np.std(test_auc_cvscores, axis=0)))

train_acc_mean=round(np.mean(train_acc_cvscores, axis=0)/100, 3)
train_acc_sd=round(np.std(train_acc_cvscores, axis=0)/100, 3)
test_acc_mean=round(np.mean(test_acc_cvscores, axis=0)/100, 3)
test_acc_sd=round(np.std(test_acc_cvscores, axis=0)/100, 3)
train_AUC_mean=round(np.mean(train_auc_cvscores, axis=0), 3)
train_AUC_sd=round(np.std(train_auc_cvscores, axis=0), 3)
test_AUC_mean=round(np.mean(test_auc_cvscores, axis=0), 3)
test_AUC_sd=round(np.std(test_auc_cvscores, axis=0), 3)

# plot results for all runs
fig3 = plt.figure(cvFold*2 + i)
plt.plot(
    history.history['acc'],
    label='train acc={0}±{1}'.format(train_acc_mean, train_acc_sd),
    linestyle='-',
	linewidth=1.5)
plt.plot(
    history.history['val_acc'],
    label='test acc={0}±{1}'.format(test_acc_mean, test_acc_sd),
    linestyle='-.', 
	linewidth=1.5)
plt.title("Model Training", fontsize=20)
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Accuracy", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc="best", fontsize=14)
plt.savefig('GSE96058_GSE81538_ki67_64C4_cv' + str(cvFold) + '_v2.2.0_train.png')
plt.close(fig3)

# plot ROC 
fig4=plt.figure(cvFold*3 + i)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(
    fpr_train,
    tpr_train,
    label='train auc={0}±{1}'.format(train_AUC_mean, train_AUC_sd),
    linestyle='-',
    linewidth=1.5)
plt.plot(
    fpr_test,
    tpr_test,
    label='test auc={0}±{1}'.format(test_AUC_mean, test_AUC_sd),
    linestyle='-.',
    linewidth=1.5)
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('ROC Curve', fontsize=20)
plt.legend(loc="best", fontsize=14)
plt.savefig('GSE96058_GSE81538_ki67_64C4_cv' + str(cvFold) + '_v2.2.0_ROC_figure.png')
plt.close(fig4)

now = datetime.datetime.now()
print("The run is done by: \n", now.strftime("%Y-%m-%d %H:%M"))


