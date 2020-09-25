# script to classify Set I AIO with diagnosis phenotype
# the script uses MGS and SWD datasets as training data,
# and use CATIE as testing data.
# input files are comma separated value files with first column
# as label, i.e. diagnosis


import pandas as pd
import numpy as np
import os
import tensorflow as tf
from keras import backend as K 
from keras.models import Model, model_from_json
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Input, Flatten, Dense, Dropout, Reshape
from keras.layers import BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from keras import optimizers
from keras import regularizers
import timeit
import datetime
import math


# define home directory and parameters
os.chdir("D:/ResData/SCZ/MSC64_embedding/")
HOME_DIR = "D:/ResData/SCZ/MSC64_embedding/"

# read in training data (MGS + SWD)
ms = pd.read_csv('mgs_swd4096_diag_L2_additive.csv', header=None)
ms = np.array(ms, dtype=np.float32)
ms_snp = ms[:, 1:]
ms_snp = np.reshape(ms_snp, (12065, 64, 64, 1))
ms_snp.shape

# get diagnosis phenotype
ms_diagnosis = ms[:, 0]
ms_diagnosis = np.array(ms_diagnosis, dtype=np.int32)
ms_diagnosis.shape

# read in testing data (CATIE)
catie = pd.read_csv('catie4096_diag_L2_additive.csv', header=None)
catie = np.array(catie, dtype=np.float32)
catie_snp = catie[:, 1:]
catie_snp = np.reshape(catie_snp, (1492, 64, 64, 1))
ms_snp.shape

# get Y, i.e. diagnosis
catie_diagnosis = catie[:, 0]
catie_diagnosis = np.array(catie_diagnosis, dtype=np.int32)
catie_diagnosis.shape

# instantiate L1, L2 regularizers
reg1 = regularizers.l1(0.015)
reg2 = regularizers.l2(0.150)

# training parameters
batchSize = 100
numEpoch = 200
DROPOUT = 0.50
LR = 0.001
DROPOUT = 0.5
EPSILON = 0.95
DECAY = 0.00
DROP = 0.90
EPOCHS_DROP = 20

# optimizer parameters
adam = optimizers.Adam(lr=LR, beta_1=0.95, beta_2=0.999, epsilon=EPSILON)

# learning rate scheduler
# define step wise decay function
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

# set check pointer and same best model weights 
checkpointer = ModelCheckpoint(
    filepath='./best_weights.hdf5',
    monitor="val_acc",
    save_best_only=True,
    save_weights_only=False,
    verbose=2)

# learning rate schedule callback
loss_history = LossHistory()
lrate = LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate, checkpointer]

# embedding parameters
inputDim = 256
outputDim = 64
inputLength = 4096

# inputs
snpInput = Input(shape=(64, 64, 1, ))
snpInput2 = Flatten()(snpInput)

# build 2D convolutional layers
convModel = Conv2D(
    filters=64,
    kernel_size=15,
    activation='relu',
    kernel_regularizer=reg2,
    kernel_initializer='he_normal',
    padding = 'same',
    dilation_rate=(3,3),
    use_bias=False)(snpInput)
convModel = BatchNormalization(axis=-1, center=True, scale=False)(convModel)
convModel = AveragePooling2D(pool_size = 2, strides = 2)(convModel)
convModel = Conv2D(
    filters=128,
    kernel_size=15,
    activation='relu',
    kernel_regularizer=reg2,
    kernel_initializer='he_normal',
    padding = 'same',
    dilation_rate=(3,3),
    use_bias=False)(convModel)
convModel = BatchNormalization(axis=-1, center=True, scale=False)(convModel)
convModel = AveragePooling2D(pool_size = 2, strides = 2)(convModel)
convModel = Conv2D(
    filters=256,
    kernel_size=15,
    activation='relu',
    kernel_regularizer=reg2,
    kernel_initializer='he_normal',
    padding = 'same',
    dilation_rate=(3,3),
    use_bias=False)(convModel)
convModel = BatchNormalization(axis=-1, center=True, scale=False)(convModel)
convModel = AveragePooling2D(pool_size = 2, strides = 2)(convModel)
convModel = Flatten()(convModel)

conv_snp = Dense(units=512, activation='relu', kernel_regularizer=reg2)(convModel)
conv_snp = Dense(units=512, activation='relu', kernel_regularizer=reg2)(conv_snp)
conv_snp = Dropout(rate=DROPOUT)(conv_snp)
conv_snp = Dense(units=64, activation='relu')(conv_snp)
conv_snp = Dropout(rate=DROPOUT)(conv_snp)
combined_output = Dense(units=1, activation='sigmoid')(conv_snp)
classifier = Model(inputs=snpInput, outputs=combined_output)
# summarize layers
print("Model summary: \n", classifier.summary())

# compile the model
classifier.compile(
    optimizer=adam,
    loss='binary_crossentropy', 
    metrics=['acc'])
    
# fit the model with training data
training_start_time = timeit.default_timer()
history = classifier.fit(
    x = ms_snp,
    y = ms_diagnosis,
    batch_size = batchSize,
    epochs = numEpoch,
    validation_data = (catie_snp, catie_diagnosis),
    callbacks=callbacks_list,
    shuffle=True,
    verbose=2)
training_end_time = timeit.default_timer()
print("Model 1 training time: {:10.2f} min. \n" .format((training_end_time - training_start_time) / 60))    

# evaluate the model with testing data
train_scores = classifier.evaluate(ms_snp, ms_diagnosis, verbose = 0)
test_scores = classifier.evaluate(catie_snp, catie_diagnosis, verbose = 0)
print("Model training accuracy: {:6.2f}%".format(train_scores[1]*100))
print("Model testing accuracy: {:6.2f}%".format(test_scores[1]*100))

# prediction 
pred_prob = classifier.predict(catie_snp)
print("Model prediction:\n")
np.set_printoptions(precision=3, suppress=True)
print(np.c_[catie_diagnosis, pred_prob])

# Confution Matrix and Classification Report
test_Y_pred = classifier.predict(catie_snp)
train_Y_pred = classifier.predict(ms_snp)
test_y_pred = np.where(test_Y_pred > 0.5, 1, 0)
train_y_pred = np.where(train_Y_pred > 0.5, 1, 0)
print('Confusion Matrix')
print(confusion_matrix(catie_diagnosis, test_y_pred))
print('Classification Report')
target_names = ['CTRL', 'SCZ']
print(classification_report(catie_diagnosis, test_y_pred, target_names=target_names))

# write predictions to a output file
model_pred = np.c_[catie_diagnosis, pred_prob]
f = open("model_pred.csv", "w")
np.savetxt(f, model_pred, delimiter=",")
f.close

# save model and model weights
# serialize model to JSON
model_json = classifier.to_json()
with open("classifier.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("./model_weights.hdf5")
print("Saved model to disk")
 
# load best model weights and calculate best model
# performance
 
# load json and create model
json_file = open('./classifier.json', 'r')
best_model_json = json_file.read()
json_file.close()
best_model = model_from_json(best_model_json)
# load best weights into new model
best_model.load_weights("./best_weights.hdf5")
print("Loaded best weights from disk")

# complie the best model
best_model.compile(
    optimizer=adam, 
    loss='binary_crossentropy', 
    # loss=[focal_loss], 
    metrics=['acc'])
 
# evaluate best model on test data
best_train_scores = best_model.evaluate(ms_snp, ms_diagnosis, verbose = 0)
best_test_scores = best_model.evaluate(catie_snp, catie_diagnosis, verbose = 0)
print("Best model training accuracy: {:6.2f}".format(best_train_scores[1]*100))
print("Best model testing accuracy: {:6.2f}".format(best_test_scores[1]*100))

# best model prediction 
pred_prob = best_model.predict(catie_snp)
print("Best model prediction:\n")
print(np.c_[catie_diagnosis, pred_prob])

# write best model predictions to the drive
best_pred = np.c_[catie_diagnosis, pred_prob]
f2 = open("best_pred.csv", "w")
np.savetxt(f2, best_pred, delimiter=",")
f2.close

# plot training and validation history
from matplotlib import pyplot
pyplot.figure(1)
pyplot.plot(history.history['acc'], label='SNV acc = {:.3f}'.format(train_scores[1]))
pyplot.plot(history.history['val_acc'], label='SNV val acc = {:.3f}'.format(test_scores[1]))
pyplot.title('model accuracy')
pyplot.ylabel('accuracy')
pyplot.xlabel('epoch')
pyplot.legend(loc='upper left')
pyplot.savefig('ms64_catie64_v1.0_adam_train.png')

# get ROC data for training and testing data
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
y_pred_train = classifier.predict(ms_snp).ravel()
y_pred_test = classifier.predict(catie_snp).ravel()

fpr_train, tpr_train, thresholds_train = roc_curve(ms_diagnosis, y_pred_train)
fpr_test, tpr_test, thresholds_test = roc_curve(catie_diagnosis, y_pred_test)

from sklearn.metrics import auc
auc_train = auc(fpr_train, tpr_train)
auc_test = auc(fpr_test, tpr_test)

# plot ROC 
pyplot.figure(2)
pyplot.plot([0, 1], [0, 1], 'k--')
pyplot.plot(fpr_train, tpr_train, label='SNV train (area = {:.3f})'.format(auc_train))
pyplot.plot(fpr_test, tpr_test, label='SNV test (area = {:.3f})'.format(auc_test))
pyplot.xlabel('False positive rate')
pyplot.ylabel('True positive rate')
pyplot.title('ROC curve')
pyplot.legend(loc='lower right')
pyplot.savefig('ms64_catie64_v1.0_adam_ROC.png')

now = datetime.datetime.now()
print("The run is done by: \n", now.strftime("%Y-%m-%d %H:%M"))

