# Script for classification of AIOs from Set II SNVs
# with diagnosis phenotype. 
# training data is the combination of MGS and SWD datasets
# testing data is the CATIE dataset. Inputs are .csv files
# with the first column as lables.

import pandas as pd
import numpy as np
import os
from os import path
import tensorflow as tf
from keras import backend as K 
from keras.models import Model, model_from_json
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.convolutional import Conv1D, MaxPooling1D, AveragePooling1D
from keras.layers import GlobalAvgPool3D, GlobalMaxPool3D
from keras.layers import GlobalAvgPool2D, GlobalMaxPool2D
from keras.layers import Input, Flatten, Dense, Dropout
from keras.layers import concatenate, add, maximum, subtract, multiply
from keras.layers import BatchNormalization, Activation
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from keras import optimizers
from keras import regularizers
import timeit
import datetime
import matplotlib.pyplot as plt
import math

# define home directory and parameters
HOME_DIR = "D:/SCZ/MSC_5e5_210/"
os.chdir("D:/SCZ/MSC_5e5_210/")

# initialize the paths to our training and testing CSV files
TRAIN_CSV = "MGS_SWD_5e5_maf0.05_210_diag_feat.csv"
TEST_CSV = "CATIE_5e5_maf0.05_210_diag_feat.csv"

# batch size and numEpochs
NUM_EPOCHS = 100
BS = 10
LR = 0.001
DECAY = 0.00
EPSILON = 0.99
DROPOUT = 0.5
DROP = 0.90
EPOCHS_DROP = 25
NUM_TRAIN_IMAGES = 0
NUM_TEST_IMAGES = 0

# instantiate L1, L2 regularizers
reg1 = regularizers.l1(0.02)
reg2 = regularizers.l2(0.125)

# optimizer parameters
adagrad = optimizers.Adagrad(lr=LR, epsilon=EPSILON, decay=DECAY)

# define data input generator:
def csv_image_generator(inputPath, bs, lb, mode="train"):
    # open the CSV and PGS files for reading
    f = open(inputPath, "r")
    # loop indefinitely
    while True:
        # initialize our batches of images and labels
        images = []
        labels = []
        # keep looping until we reach our batch size
        while len(images) < bs:  # images and PGSs have the same length
            # attempt to read the next line of the CSV file
            line = f.readline()
            # check to see if the line is empty, indicating we have
            # reached the end of the file
            if line == "":
                # reset the file pointer to the beginning of the file
                # and re-read the line
                f.seek(0)
                line = f.readline()
                # if we are evaluating we should now break from our
                # loop to ensure we don't continue to fill up the
                # batch from samples at the beginning of the file
                if mode == "eval":
                    break
            # extract the label and construct the image
            line = line.strip().split(",")
            label = line[0]
            image = np.array(line[1:], dtype=np.float64)
            image = image.reshape((210, 210, 1))
            # update our corresponding batches lists
            images.append(image)
            labels.append(label)
        # one-hot encode the labels
        labels = lb.transform(np.array(labels))
        # yield the batch to the calling function
        yield (np.array(images), labels)

# open the training CSV file, then initialize the unique set of class
# labels in the dataset along with the testing labels
f = open(TRAIN_CSV, "r")
labels = set()
labels2 = []
testLabels = []

# loop over all rows of the CSV file
for line in f:
    # extract the class label, update the labels list, and increment
    # the total number of training images
    label = line.strip().split(",")[0]
    labels.add(label)
    labels2.append(label)
    NUM_TRAIN_IMAGES += 1

# close the training CSV file and open the testing CSV file
f.close()
f = open(TEST_CSV, "r")

# loop over the lines in the testing file
for line in f:
    # extract the class label, update the test labels list, and
    # increment the total number of testing images
    label = line.strip().split(",")[0]
    testLabels.append(label)
    NUM_TEST_IMAGES += 1

# close the testing CSV file
f.close()

# create the label binarizer for one-hot encoding labels, then encode
# the testing labels
lb = LabelBinarizer()
lb.fit(list(labels))
testLabels = lb.transform(testLabels)

# initialize both the training and testing image generators
trainGen = csv_image_generator(TRAIN_CSV, BS, lb, mode="train")
testGen = csv_image_generator(TEST_CSV, BS, lb, mode="train")

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
    monitor="val_acc",
    save_best_only=True,
    save_weights_only=False,
    verbose=2)

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

# embedding parameters
inputDim = 256
outputDim = 32
inputLength = 44100

# input layers. Chanage this as necessary
snp_input = Input(shape=(210, 210, 1, ))
snp_input2 = Flatten()(snp_input)

# SNP convolutional layers
# build the snp models: first convolutional 2D model
snp_conv = Conv2D(
    filters=64, 
    kernel_size=11, 
    kernel_regularizer=reg2, 
    activation='relu', 
    padding ='same')(snp_input)
snp_conv = BatchNormalization(axis=-1, center=True, scale=False)(snp_conv)
snp_conv = AveragePooling2D(pool_size = 2, strides = 2)(snp_conv)
snp_conv = Conv2D(
    filters=128, 
    kernel_size=11, 
    kernel_regularizer=reg2, 
    activation='relu', 
    padding ='same')(snp_conv)
snp_conv = BatchNormalization(axis=-1, center=True, scale=False)(snp_conv)
snp_conv = AveragePooling2D(pool_size = 2, strides = 2)(snp_conv)
snp_conv = Conv2D(
    filters=256, 
    kernel_size=11, 
    kernel_regularizer=reg2, 
    activation='relu', 
    padding ='same')(snp_conv)
snp_conv = BatchNormalization(axis=-1, center=True, scale=False)(snp_conv)
snp_conv = AveragePooling2D(pool_size = 2, strides = 2)(snp_conv)
snp_conv = Flatten()(snp_conv)
snp_conv = Dense(units=64, activation='relu')(snp_conv)

# SNP embedding layers which uses the same snp data as input
snp_emb = Embedding(
    input_dim=inputDim, 
    output_dim=outputDim, 
    embeddings_regularizer=reg2, 
    name='SNP_input2')(snp_input)
snp_gap = GlobalAvgPool3D()(snp_emb)
snp_gap = Dense(units=64, activation='relu')(snp_gap)

# combine the SNP layers
combined = concatenate([snp_conv, snp_gap])

combined = Dense(units=256, activation='relu')(combined)
combined = Dropout(rate=DROPOUT)(combined)
combined = Dense(units=64, activation='relu')(combined)
combined = Dropout(rate=DROPOUT)(combined)
combined_output = Dense(units=1, activation='sigmoid')(combined)
classifier = Model(inputs=snp_input, outputs=combined_output)
# summarize layers
print("Model summary: \n", classifier.summary())

# compile the model
classifier.compile(
    optimizer=adagrad, 
    loss='binary_crossentropy', 
    metrics=['acc'])
    
# fit the model with training data
training_start_time = timeit.default_timer()
H = classifier.fit_generator(
    generator=trainGen,
    steps_per_epoch=NUM_TRAIN_IMAGES // BS,
    validation_data=testGen,
    validation_steps=NUM_TEST_IMAGES // BS,
    epochs=NUM_EPOCHS,
    callbacks=callbacks_list,
    verbose=2)

training_end_time = timeit.default_timer()
print("Training time: {:10.2f} min. \n" .format((training_end_time - training_start_time) / 60))    
# Evaluation
# re-initialize our testing data generator, this time for evaluating
testGen = csv_image_generator(TEST_CSV, BS, lb, mode="eval")

# make predictions on the testing images, finding the index of the
predIdxs = classifier.predict_generator(testGen, steps=(NUM_TEST_IMAGES // BS) + 1)
predIdxs = np.argmax(predIdxs, axis=1)

#Confution Matrix and Classification Report
trainGen = csv_image_generator(TRAIN_CSV, BS, lb, mode="eval")
testGen = csv_image_generator(TEST_CSV, BS, lb, mode="eval")

test_Y_pred = classifier.predict_generator(testGen, steps=(NUM_TEST_IMAGES // BS) + 1)
train_Y_pred = classifier.predict_generator(trainGen, steps=(NUM_TRAIN_IMAGES // BS) + 1)
test_y_pred = np.where(test_Y_pred > 0.5, 1, 0)
train_y_pred = np.where(train_Y_pred > 0.5, 1, 0)
test_y_true = np.argmax(testLabels)
train_y_true = np.argmax(labels)

print('Confusion Matrix')
print(confusion_matrix(testLabels, test_y_pred))
print('Classification Report')
target_names = ['CTRL', 'SCZ']
print(classification_report(testLabels, test_y_pred, target_names=target_names))

# evaluate the model with testing data
trainGen = csv_image_generator(TRAIN_CSV, BS, lb, mode="eval")
testGen = csv_image_generator(TEST_CSV, BS, lb, mode="eval")

train_scores = classifier.evaluate_generator(trainGen, steps=(NUM_TRAIN_IMAGES // BS) + 1, verbose = 0 )
test_scores = classifier.evaluate_generator(testGen, steps=(NUM_TEST_IMAGES // BS) + 1, verbose = 0 )
print("Training accuracy {:0.2f}%".format(train_scores[1]*100))
print("Testing accuracy {:0.2f}%".format(test_scores[1]*100))

# write prediction to drive
model_pred = np.c_[testLabels, test_Y_pred]
f = open("model_pred.csv", "w")
np.savetxt(f, model_pred, delimiter=",", fmt='%0.4f')
f.close

# serialize model to JSON
model_json = classifier.to_json()
with open("classifier.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("model_weights.hdf5")
print("Saved model to disk")
 
# Evaluate the best model performance
 
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
    loss='binary_crossentropy', 
    metrics=['acc'])
 
# evaluate loaded model on test data
trainGen = csv_image_generator(TRAIN_CSV, BS, lb, mode="eval")
testGen = csv_image_generator(TEST_CSV, BS, lb, mode="eval")

best_train_scores = best_model.evaluate_generator(trainGen, steps=(NUM_TRAIN_IMAGES // BS) + 1)
best_test_scores = best_model.evaluate_generator(testGen, steps=(NUM_TEST_IMAGES // BS) + 1)
print("Best model training accuracy: {:0.2f}%".format(best_train_scores[1]*100))
print("Best model testing accuracy: {:0.2f}%".format(best_test_scores[1]*100))

# prediction
trainGen = csv_image_generator(TRAIN_CSV, BS, lb, mode="eval")
testGen = csv_image_generator(TEST_CSV, BS, lb, mode="eval")
 
pred_prob = best_model.predict_generator(testGen, steps=(NUM_TEST_IMAGES // BS) + 1)
print("Best model predicted outcomes:\n")
print(np.c_[testLabels, pred_prob])

# write best model prediction to drive
best_pred = np.c_[testLabels, pred_prob]
f2 = open("best_pred.csv", "w")
np.savetxt(f2, best_pred, delimiter=",", fmt='%0.4f')
f2.close

# plot the training loss and accuracy
N = NUM_EPOCHS
plt.figure(1)
plt.plot(np.arange(0, N), H.history["acc"], label="train acc={:0.3f}".format(train_scores[1]), linestyle='solid')
plt.plot(np.arange(0, N), H.history["val_acc"], label="val acc={:0.3f}".format(test_scores[1]), linestyle='dashed')
# plt.plot(np.arange(0, N), H.history["auc"], label="train auc={:0.3f}".format(train_scores[2]))
# plt.plot(np.arange(0, N), H.history["val_auc"], label="val auc={:0.3f}".format(train_scores[2]))
plt.title("Training Accuracy", fontsize=20)
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Accuracy", fontsize=16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.legend(loc="best", fontsize=14)
plt.savefig("Set2_AIO_diagnosis_v1.0_adagrad_training.png")

# get ROC data for plotting
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

labels2 = np.array(labels2, dtype=np.int32)
testLabels = np.array(testLabels, dtype=np.int32)
fpr_train, tpr_train, thresholds_train = roc_curve(labels2, train_Y_pred)
fpr_test, tpr_test, thresholds_test = roc_curve(testLabels, test_Y_pred)

auc_train = auc(fpr_train, tpr_train)
auc_test = auc(fpr_test, tpr_test)

# plot ROC 
plt.figure(2)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_train, tpr_train, label='Train (area = {:0.3f})'.format(auc_train), linestyle='solid')
plt.plot(fpr_test, tpr_test, label='Test (area = {:0.3f})'.format(auc_test), linestyle='dashed')
plt.title('ROC curve', fontsize=20)
plt.xlabel("False Positive Rate", fontsize=16)
plt.ylabel("True Positive Rate", fontsize=16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.legend(loc="best", fontsize=14)
plt.savefig('Set2_AIO_diagnosis_v1.0_adagrad_ROC.png')

now = datetime.datetime.now()
print("The run is done by: \n", now.strftime("%Y-%m-%d %H:%M"))


