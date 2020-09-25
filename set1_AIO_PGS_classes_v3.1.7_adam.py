# Set I SNP AIO classification for polygenic risk score stratified
# classes. Inputs are csv files with labels on the first column.
# The reports are categorical accuracy, and class specific AUCs.

import pandas as pd
import numpy as np
import os
import tensorflow as tf
from keras import backend as K 
from keras.models import Model, model_from_json
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Input, Flatten, Dense, Dropout, Reshape
from keras.layers import GlobalMaxPool3D, GlobalAvgPool3D
from keras.layers import GlobalMaxPool2D, GlobalAvgPool2D
from keras.layers import BatchNormalization, Activation
from keras.layers.embeddings import Embedding
from keras.layers import concatenate, add, maximum, multiply, average, subtract
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import to_categorical
from keras import optimizers
from keras import regularizers
import matplotlib.pyplot as plt
import timeit
import datetime
import math

# define home directory and parameters
os.chdir("D:/ResData/SCZ/MSC64_embedding/")
HOME_DIR = "D:/ResData/SCZ/MSC64_embedding/"

# read in  train data in csv file
ms_scz = pd.read_csv('mgs_swd4096_diag_0.25sd_L2_additive.csv', header=None)
ms_scz = np.array(ms_scz, dtype=np.float32)
ms_scz_snp = ms_scz[:, 1:]
ms_scz_snp = np.reshape(ms_scz_snp, (12065, 64, 64, 1))
ms_scz_snp.shape

# get Y, i.e. diagnosis
ms_scz_diagnosis= ms_scz[:, 0]
ms_scz_diagnosisms_1hot_diagnosis = np.array(ms_scz_diagnosis, dtype=np.int32)
ms_1hot_diagnosis = to_categorical(ms_scz_diagnosis)		# one hot coding
ms_scz_diagnosis.shape
ms_1hot_diagnosis.shape

# read in  test data
catie_scz = pd.read_csv('catie4096_diag_0.25sd_L2_additive.csv', header=None)
catie_scz = np.array(catie_scz, dtype=np.float32)
catie_scz_snp = catie_scz[:, 1:]
catie_scz_snp = np.reshape(catie_scz_snp, (1492, 64, 64, 1))
catie_scz_snp.shape

# get Y, i.e. diagnosis
catie_scz_diagnosis = catie_scz[:, 0]
catie_scz_diagnosis = np.array(catie_scz_diagnosis, dtype=np.int32)
catie_1hot_diagnosis = to_categorical(catie_scz_diagnosis)		# one hot coding
catie_scz_diagnosis.shape
catie_1hot_diagnosis.shape

# instantiate L1, L2 regularizers
reg1 = regularizers.l1(0.01)
reg2 = regularizers.l2(0.125)

# training parameters
batchSize = 20
numEpoch = 100
LR = 0.0010
numClasses = 3
DROPOUT = 0.5
EPSILON = 1.05
DECAY = 0.00
DROP = 0.80
EPOCHS_DROP = 20

# optimizer parameters
adam = optimizers.Adam(lr=LR, beta_1=0.95, beta_2=0.999, epsilon=EPSILON)

# use class weights
CLASS_WEIGHT = {0: 0.35, 1: 0.30, 2: 0.35}

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

checkpointer = ModelCheckpoint(
    filepath='./best_weights.hdf5',
    monitor="val_categorical_accuracy",
    save_best_only=True,
    save_weights_only=False,
    verbose=2)

earlyStoper = EarlyStopping(
    monitor='val_loss', 
    mode='min', 
    restore_best_weights=True, 
    min_delta =0.75,
    # patience = 5,
    verbose=2)

# learning schedule callback
loss_history = LossHistory()
lrate = LearningRateScheduler(step_decay)
callbacks_list = [loss_history, lrate, checkpointer]

def focal_loss(y_true, y_pred):
    gamma = 4
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

# embedding parameters
inputDim = 256
outputDim = 32
inputLength = 4096

# inputs
sczInput = Input(shape=(64, 64, 1, ))
sczInput2 = Flatten()(sczInput)

# model 1, SCZ SNPs
# build the models: first convolutional 2D model
convModel = Conv2D(
    filters=64, 
    kernel_size=11, 
    activation='relu', 
    kernel_regularizer=reg2,
    kernel_initializer='he_normal',
    dilation_rate=(2, 2),	
    padding = 'same', 
    use_bias=False)(sczInput)
convModel = BatchNormalization(axis=-1, center=True, scale=False)(convModel)
convModel = AveragePooling2D(pool_size = 2, strides = 2)(convModel)
convModel = Conv2D(
    filters=128, 
    kernel_size=11, 	
    activation='relu',  
    kernel_regularizer=reg2,
    kernel_initializer='he_normal',
    dilation_rate=(2, 2),	
    padding = 'same', 
    use_bias=False)(convModel)
convModel = BatchNormalization(axis=-1, center=True, scale=False)(convModel)
convModel = AveragePooling2D(pool_size = 2, strides = 2)(convModel)
convModel = Conv2D(
    filters=256, 
    kernel_size=11, 	
    activation='relu',  
    kernel_regularizer=reg2,
    kernel_initializer='he_normal',
    dilation_rate=(2, 2),	
    padding = 'same', 
    use_bias=False)(convModel)
convModel = BatchNormalization(axis=-1, center=True, scale=False)(convModel)
convModel = AveragePooling2D(pool_size = 2, strides = 2)(convModel)
# convModel = GlobalAvgPool2D()(convModel)
convModel = Flatten()(convModel)
scz_sm = Dense(units=512, activation='relu', kernel_regularizer=reg1)(convModel)
scz_sm = Dense(units=512, activation='relu', kernel_regularizer=reg1)(scz_sm)
scz_sm = Dropout(rate=DROPOUT)(scz_sm)
scz_sm = Dense(units=64, activation='relu')(scz_sm)

# Second, global pooling that uses the same snp emb data as input
snp_emb = Embedding(
    input_dim=inputDim, 
	output_dim=outputDim, 
	embeddings_regularizer=reg2, 
	name='SNP_input')(sczInput)
snp_gap = GlobalAvgPool3D()(snp_emb)
snp_gap = Dense(units=64, activation='relu')(snp_gap)

# combine snp conv 2D with snp embedding
concat = concatenate([scz_sm, snp_gap])

dropout = Dropout(rate=DROPOUT)(concat)
combined_output = Dense(units=numClasses, activation='softmax')(dropout)
classifier = Model(inputs=sczInput, outputs=combined_output)
# summarize layers
print("Model summary: \n", classifier.summary())

# compile the model
classifier.compile(
    optimizer=adam, 
    # loss=[focal_loss], 
    loss='categorical_crossentropy', 
    metrics=['categorical_accuracy'])
 		
# fit the model3 with training data
training_start_time = timeit.default_timer()
history = classifier.fit(
    x = ms_scz_snp,
    y = ms_1hot_diagnosis,
    batch_size = batchSize,
    epochs = numEpoch,
    validation_data = (catie_scz_snp, catie_1hot_diagnosis),
    # class_weight=CLASS_WEIGHT,
    callbacks=callbacks_list,
    shuffle=True,
    verbose=2)		

training_end_time = timeit.default_timer()
print("Model training time: {:10.2f} min. \n" .format((training_end_time - training_start_time) / 60))    

# evaluate the model with testing data
train_scores = classifier.evaluate(ms_scz_snp, ms_1hot_diagnosis, verbose = 0 )
test_scores = classifier.evaluate(catie_scz_snp, catie_1hot_diagnosis, verbose = 0 )

print("Model training accuracy: {:.2f}%".format(train_scores[1]*100))
print("Model testing accuracy: {:.2f}%".format(test_scores[1]*100))

pred_prob = classifier.predict(catie_scz_snp)
np.set_printoptions(precision=3, suppress=True)
print("SCZ model Predicted outcomes:\n")
print(np.c_[catie_1hot_diagnosis, pred_prob])

# write predictions to the drive
model_pred = np.c_[catie_1hot_diagnosis, pred_prob]
f = open("model_pred.csv", "w")
np.savetxt(f, model_pred, delimiter=",", fmt='%0.4f')
f.close

#Confution Matrix and Classification Report
test_Y_pred = classifier.predict(catie_scz_snp)
train_Y_pred = classifier.predict(ms_scz_snp)
test_y_pred = np.argmax(test_Y_pred, axis=1)
train_y_pred = np.argmax(train_Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(catie_scz_diagnosis, test_y_pred))
print('Classification Report')
target_names = ['Normal', 'Unknown', 'SCZ']
print(classification_report(catie_scz_diagnosis, test_y_pred, target_names=target_names))

# save model to JSON file
model_json = classifier.to_json()
with open("./classifier.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("./classifier_weights.hdf5")
print("Saved model weights to disk")
 
# load best model to evaluate its performance
 
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
    optimizer=adagrad, 
    # loss=[focal_loss], 
    loss='categorical_crossentropy', 
    metrics=['categorical_accuracy'])
 
# evaluate loaded model on test data
best_train_scores = best_model.evaluate(ms_scz_snp, ms_1hot_diagnosis, verbose = 0)
best_test_scores = best_model.evaluate(catie_scz_snp, catie_1hot_diagnosis, verbose = 0)
print("Best model training accuracy: {:6.2f}".format(best_train_scores[1]*100))
print("Best model testing accuracy: {:6.2f}".format(best_test_scores[1]*100))

# prediction 
pred_prob = best_model.predict(catie_scz_snp)
np.set_printoptions(precision=3, suppress=True)
print("Best model predicted outcomes:\n")
print(np.c_[catie_1hot_diagnosis, pred_prob])

# write predictions to the drive
best_pred = np.c_[catie_1hot_diagnosis, pred_prob]
f2 = open("best_pred.csv", "w")
np.savetxt(f2, best_pred, delimiter=",", fmt='%0.4f')
f2.close

# plot training and validation history
# make 5 dash linestyles
dashList = [(5,0), (5,2), (1,1), (3, 3), (4, 1)] 
colors = ['darkorange', 'limegreen', 'deeppink', 'navy', 'aqua']

from matplotlib import pyplot

fig1=pyplot.figure(1)
pyplot.plot(
    history.history['categorical_accuracy'], 
	label='train acc = {:.3f}'.format(train_scores[1]), 
	linestyle='-', 
	dashes=dashList[0], 
	linewidth=2)
pyplot.plot(
    history.history['val_categorical_accuracy'], 
	label='val acc = {:.3f}'.format(test_scores[1]), 
	linestyle='-', 
	dashes=dashList[1], 
	linewidth=2)
pyplot.title('model accuracy', fontsize=20)
pyplot.ylabel('accuracy', fontsize=16)
pyplot.xlabel('epoch', fontsize=16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
pyplot.legend(loc='best')
# pyplot.show()
pyplot.savefig('set1_AIO_PGS_classes_v3.1.7_adam_train.png')
pyplot.close(fig1)

# Multiclass ROC curve and ROC area for each class
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(numClasses):
    fpr[i], tpr[i], _ = roc_curve(catie_1hot_diagnosis[:, i], pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(catie_1hot_diagnosis.ravel(), pred_prob.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(numClasses)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(numClasses):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= numClasses

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw = 2
fig2=plt.figure(2)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='red', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='green', linestyle='-.', linewidth=4)


for i, color, dashes in zip(range(numClasses), colors, dashList):
    plt.plot(fpr[i], tpr[i], color=color, linestyle='-', dashes=dashList[i], lw=lw,
             label='class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title('Multi-Class ROC', fontsize=20)
plt.legend(loc="lower right", fontsize=14)
# pyplot.show() 	# enable this if want to see the figure instead of saving to file
pyplot.savefig('set1_AIO_PGS_classes_v3.1.7_adam_ROC.png')
pyplot.close(fig2)

now = datetime.datetime.now()
print("The run is done by: \n", now.strftime("%Y-%m-%d %H:%M"))

