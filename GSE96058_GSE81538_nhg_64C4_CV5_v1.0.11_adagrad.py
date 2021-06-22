# Five fold cross validation mddel for NHG using the combined GSE81538 and GSE96058 
# datasets. The two datasets were read in separately, and then combined by concatenation. 
# For both datasets, the files were in .csv format, with no headers. The first column was 
# class label, and the rest columns were normalized gene expression levels. 

# import necessary libaries and prcedures

import pandas as pd
import numpy as np
import os
from os import path
import tensorflow as tf
from keras import backend as K 
from keras.models import Model, model_from_json
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalAvgPool3D, GlobalMaxPool3D
from keras.layers import Input, Flatten, Dense, Dropout
from keras.layers import concatenate, add, maximum, subtract, multiply
from keras.layers import BatchNormalization, Activation
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras import optimizers
from keras import regularizers
from keras.utils.vis_utils import plot_model
import pydot
import timeit
import datetime
import matplotlib.pyplot as plt
import math

# define home directory and parameters
HOME_DIR = "D:/Cancers/BreastCancer/"
os.chdir("D:/Cancers/BreastCancer/")

# read in data 1
nhg1 = pd.read_csv('GSE96058_sorted_nhg128_lab_feat.csv', header=None)
nhg1 = np.array(nhg1, dtype=np.float32)

# read in data 2
nhg2 = pd.read_csv('GSE81538_sorted_nhg128_lab_feat.csv', header=None)
nhg2 = np.array(nhg2, dtype=np.float32)

# combine the two datasets together
nhg = np.concatenate((nhg1, nhg2), axis=0)

nhg_exp = nhg[:, 1:]
nhg_exp = np.array(nhg_exp, dtype=np.float32)
nhg_exp = np.reshape(nhg_exp, (-1, 64, 64, 4))
nhg_exp.shape

# get diagnosis
nhg_label = nhg[:, 0]
nhg_label = np.array(nhg_label, dtype=np.int32)
nhg_label = to_categorical(nhg_label)

# fix random seed for reproducibility
seed = 999
np.random.seed(seed)

# batch size and numEpochs
numEpoch = 500
batchSize = 100
LR = 0.001
DECAY = 0.001
EPSILON = 0.99
DROPOUT = 0.5
DROP = 0.99
EPOCHS_DROP = 25
numClasses = 3

# instantiate L1, L2 regularizers
# reg1 = regularizers.l1(0.01)
reg2 = regularizers.l2(0.185)

# optimizer parameters
# sgd = optimizers.SGD(lr=LR, decay=DECAY, momentum=0.9, nesterov=True)
# rms = optimizers.RMSprop(lr=LR, rho=0.9, epsilon=EPSILON, decay=DECAY)
# adam = optimizers.Adam(lr=LR, beta_1=0.95, beta_2=0.999, epsilon=EPSILON)
adagrad = optimizers.Adagrad(lr=LR, epsilon=EPSILON, decay=DECAY)
# nadam = optimizers.Nadam(lr=LR, beta_1=0.95, beta_2=0.999, epsilon=EPSILON, schedule_decay=DECAY)
# adadelta = optimizers.Adadelta(lr=LR, rho=0.95, epsilon=EPSILON, decay=DECAY)

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
CLASS_WEIGHT = {0: 2.5775, 1: 0.6841, 2: 0.8692}

# focal loss function
def focal_loss(y_true, y_pred):
    gamma = 2.5
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

# embedding parameters
inputDim = 256
outputDim = 64
inputLength = 16384

# input layers. Chanage this as necessary
exp_input = Input(shape=(64, 64, 4, ))
exp_input2 = Flatten()(exp_input)

# define k-fold cross validation test harness
cvFold = 5
# kfold = KFold(n_splits=cvFold, shuffle=True, random_state=999)
train_acc_cvscores = []
test_acc_cvscores = []
train_auc_cvscores = []
test_auc_cvscores = []
i = 0

# split samples into training and testing sets;
for i in range(cvFold):
    (trainX, testX, trainY, testY) = train_test_split(
        nhg_exp, nhg_label, 
        test_size=0.2, 
        shuffle=True, 
        random_state=999)

    # checkpointer
    checkpointer = ModelCheckpoint(
        filepath='./best_weights_' + str(i) + '.hdf5',
        monitor="val_categorical_accuracy",
        save_best_only=True,
        save_weights_only=False,
        verbose=2)

    # early stoper
    earlyStoper = EarlyStopping(
        monitor='val_loss', 
        mode='min', 
        restore_best_weights=True, 
        # min_delta =0.75,
        patience = 5,
        verbose=2)

    # learning schedule callback
    loss_history = LossHistory()
    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [loss_history, lrate, checkpointer]

    # model blocks start here
	# convolutional layers
    exp_conv = Conv2D(
        filters=64, 
        kernel_size=15, 
        kernel_regularizer=reg2, 
        activation='relu', 
        dilation_rate=(2,2),
        padding ='same')(exp_input)
    exp_conv = BatchNormalization(axis=-1, center=True, scale=False)(exp_conv)
    exp_conv = AveragePooling2D(pool_size = 2, strides = 2)(exp_conv)
    exp_conv = Conv2D(
        filters=128, 
        kernel_size=15, 
        kernel_regularizer=reg2, 
        activation='relu', 
        dilation_rate=(2,2),
        padding ='same')(exp_conv)
    exp_conv = BatchNormalization(axis=-1, center=True, scale=False)(exp_conv)
    exp_conv = AveragePooling2D(pool_size = 2, strides = 2)(exp_conv)
    exp_conv = Conv2D(
        filters=256, 
        kernel_size=15, 
        kernel_regularizer=reg2, 
        activation='relu', 
        padding ='same')(exp_conv)
    exp_conv = BatchNormalization(axis=-1, center=True, scale=False)(exp_conv)
    exp_conv = AveragePooling2D(pool_size = 2, strides = 2)(exp_conv)
    # exp_conv = Dropout(rate=DROPOUT)(exp_conv)
    exp_conv = Conv2D(
        filters=256, 
        kernel_size=1, 
        kernel_regularizer=reg2, 
        activation='relu', 
        dilation_rate=(2,2),
        padding ='same')(exp_conv)
    exp_conv = BatchNormalization(axis=-1, center=True, scale=False)(exp_conv)
    exp_conv = AveragePooling2D(pool_size = 1, strides = 1)(exp_conv)
    exp_conv = Flatten()(exp_conv)
    exp_conv = Dense(units=256, activation='relu')(exp_conv)
    exp_conv = Dropout(rate=DROPOUT)(exp_conv)
    exp_conv = Dense(units=64, activation='relu')(exp_conv)

    # embedding layer, which uses the same expresion data as input
    exp_emb = Embedding(
        input_dim=inputDim, 
        output_dim=outputDim, 
        embeddings_regularizer=reg2, 
        name='EXP_input2')(exp_input)
    exp_gap = GlobalAvgPool3D()(exp_emb)
    exp_gap = Dropout(rate=DROPOUT)(exp_gap)
    exp_gap = Dense(units=64, activation='relu')(exp_gap)

    # combine the convolution and embedding layers
    combined = add([exp_conv, exp_gap])

    # fully connected layers
    combined = Dense(units=64, activation='relu')(combined)
    combined = Dropout(rate=DROPOUT)(combined)
    combined_output = Dense(units=numClasses, activation='softmax')(combined)
    classifier = Model(inputs=exp_input, outputs=combined_output)
    # summarize layers
    print("Model summary: \n", classifier.summary())

    # compile the model
    classifier.compile(
        optimizer=adagrad, 
        # loss='categorical_crossentropy', 
        loss=[focal_loss],
        metrics=['categorical_accuracy', tf.keras.metrics.AUC()])
    
    # fit the model with training data
    training_start_time = timeit.default_timer()
    history = classifier.fit(
        x = trainX,
        y = trainY,
        batch_size = batchSize,
        epochs = numEpoch,
        validation_data = (testX, testY),
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
    train_scores = classifier.evaluate(trainX, trainY, verbose = 0 )
    test_scores = classifier.evaluate(testX, testY, verbose = 0 )
    print("%s: %.3f%%" % (classifier.metrics_names[1], train_scores[1]*100))
    print("%s: %.3f%%" % (classifier.metrics_names[1], test_scores[1]*100))    
    train_acc_cvscores.append(train_scores[1]*100)
    test_acc_cvscores.append(test_scores[1]*100)
	
    # make predictions on testing samples 
    test_pred_prob = classifier.predict(testX)
    train_pred_prob = classifier.predict(trainX)
    print("Predicted outcomes:\n")
    np.set_printoptions(precision=3, suppress=True)
    print(np.c_[testY, test_pred_prob])

    # write test sample predictions to drive
    model_pred = np.c_[testLabels, test_Y_pred]
    with open("model_pred.csv", "w") as f:
        np.savetxt(f, model_pred, delimiter=",", fmt=['%d', '%d', '%d', '%0.4f', '%0.4f', '%0.4f'])
    f.close

    # Confution Matrix and Classification Report
    test_y_pred = np.argmax(test_pred_prob, axis=1)
    test_y_true = np.argmax(testY, axis=1)

    print('Confusion Matrix')
    print(confusion_matrix(test_y_true, test_y_pred))
    print('Classification Report')
    target_names = ['Grade I', 'Grade II', 'Grade III']
    print(classification_report(test_y_true, test_y_pred, target_names=target_names))

    # plot training and validation history
    # define dash linestyles
    dashList = [(4,4,4,4), (4,1,4,1), (4,1,1,4), (1,1,1,1), (2,1,2,1), (3,1,3,1)] 
    colors = ['green', 'deeppink', 'navy', 'aqua', 'darkorange', 'cornflowerblue']
    lw=2
    
    fig1 = plt.figure(i)
    plt.plot(
        history.history['categorical_accuracy'], 
        label='train (acc={:.3f})'.format(train_scores[1]), 
        linestyle='-', 
        lw=lw)
    plt.plot(
        history.history['val_categorical_accuracy'], 
        label='test (acc={:.3f})'.format(test_scores[1]), 
        linestyle='-.', 
        lw=lw)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.savefig('GSE96058_GSE81538_nhg_cv' + str(cvFold) + '_v1.0.11_train' + str(i) +'.png')
    plt.close(fig1)
    
    # get ROC data for plotting
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
    # Multiclass ROC curve and ROC area for each class
    test_fpr = dict()
    test_tpr = dict()
    train_fpr = dict()
    train_tpr = dict()
    test_roc_auc = dict()
    train_roc_auc = dict()
    for j in range(numClasses):
        test_fpr[j], test_tpr[j], _ = roc_curve(testY[:, j], test_pred_prob[:, j])
        train_fpr[j], train_tpr[j], _ = roc_curve(trainY[:, j], train_pred_prob[:, j])
        test_roc_auc[j] = auc(test_fpr[j], test_tpr[j])
        train_roc_auc[j] = auc(train_fpr[j], train_tpr[j])

    train_auc_cvscores.append(train_roc_auc)
    test_auc_cvscores.append(test_roc_auc)

    # plot ROC 
    fig2 = plt.figure(cvFold + i)
    plt.plot([0, 1], [0, 1], 'k--')
    for j, color, dashes in zip(range(numClasses), colors, dashList):
        plt.plot(
            train_fpr[j], 
            train_tpr[j], 
            color=colors[j], 
            linestyle='-', 
            dashes=dashList[j], 
            lw=lw,
            label='train {0} (area = {1:0.2f})'.format(j, train_roc_auc[j]))
        plt.plot(
            test_fpr[j], 
            test_tpr[j], 
            color=colors[numClasses + j], 
            linestyle='-', 
            dashes=dashList[numClasses + j], 
            lw=lw,
            label='test {0} (area = {1:0.2f})'.format(j, test_roc_auc[j]))
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig('GSE96058_GSE81538_nhg_cv' + str(cvFold) + '_v1.0.11_ROC_figure' + str(cvFold + i) + '.png')
    plt.close(fig2)
    i = i + 1

# convert list of dict to np array before print
train_auc_cvscores = np.array(pd.DataFrame(train_auc_cvscores).values, dtype=np.float32)
test_auc_cvscores = np.array(pd.DataFrame(test_auc_cvscores).values, dtype=np.float32)

# print CV results
print("\nCV results for the training acc: ")
print("%.3f%% (+/- %.3f%%)" % (np.mean(train_acc_cvscores, axis=0), np.std(train_acc_cvscores, axis=0)))
print("\nCV results for the testing acc: ")
print("%.3f%% (+/- %.3f%%)" % (np.mean(test_acc_cvscores, axis=0), np.std(test_acc_cvscores, axis=0)))
np.set_printoptions(precision=3, suppress=True)
print("\ntraining AUCs:", train_auc_cvscores)
print("\ntest AUCs:", test_auc_cvscores)
print("\ntraining AUC means:", train_auc_cvscores.mean(axis=0))
print("training AUC std:", train_auc_cvscores.std(axis=0))
print("test AUC means:", test_auc_cvscores.mean(axis=0))
print("test AUC std:", test_auc_cvscores.std(axis=0))

# plot the model model structure to a file
plot_model(classifier, to_file='./model_plot.png', show_shapes=True, show_layer_names=True)

now = datetime.datetime.now()
print("The run is done by: \n", now.strftime("%Y-%m-%d %H:%M"))


