# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 13:20:01 2020
@author: crist
"""
import numpy as np
import tensorflow as tf
import cv2,os
from imutils import paths
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import ConfusionMatrixDisplay


#Initialize :
#the init learning rate
#number of epochs to train for
#batch size
init_LR =  0.001
Epochs = 20
BS = 10

# get the list of images in the dataset directory
# initialize the list of data (i.e., images) and class label of images
datasetPath='/*/dataset' #change with your path file
imgPath = list(paths.list_images(datasetPath))

Imdata = []
Imlabels = []

#loop function for:
    #extracting the class label
    #image loading, gray conversion, resizing to 224x224
    #update the data and labels list
for img in imgPath:
	clabel = img.split(os.path.sep)[1]
	image = cv2.imread(img)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))

	Imdata.append(image)
	Imlabels.append(clabel)

# Conversion to np.array of Data + pixel range ([0,255]) and Labels 
Imdata = np.array(Imdata) / 255.0
Imlabels = np.array(Imlabels)

# one-hot encoding on the labels using LabelBinarizer()
lb = LabelBinarizer()
Imlabels = lb.fit_transform(Imlabels)
Imlabels = to_categorical(Imlabels) 
 
#split the Data into training (60%), validation (20%) and testing (20%)
#split to train, test
(train_X,test_X, train_Y,test_Y) = train_test_split(Imdata, Imlabels,
	test_size=0.2, random_state=1)
#split again train into validation and train
(train_X,val_X, train_Y,val_Y) = train_test_split(train_X,train_Y,
	test_size=0.25, random_state=1)

# initialization of data augmentation object
traindataAugm = ImageDataGenerator(
	rotation_range=15,
	fill_mode="nearest")

# load the VGG16 network and ensure the head FC layers are left off and with weights pre-trained on ImageNet
bmodel = tf.keras.applications.VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# build the head of the model that will be placed on top of the base model (bmodel)
model_h = bmodel.output
model_h = AveragePooling2D(pool_size=(2, 2))(model_h)
model_h = Flatten(name="flatten")(model_h)
model_h = Dense(84, activation="relu")(model_h)
model_h = Dropout(0.5)(model_h)
model_h = Dense(2, activation="softmax")(model_h)

# the head FC model should be placed on top of the base model and this will be the model we will train
model = Model(inputs=bmodel.input, outputs=model_h)

#  freeze the CONV weights of VGG16 and FC layer head will be trained
for layer in bmodel.layers:
    layer.trainable = False
    
    print(layer.name, layer.trainable)

# compile the model
print("----------Compiling model----------")
optim = Adam(lr=init_LR, decay=init_LR / Epochs)
model.compile(loss="binary_crossentropy", optimizer=optim,
	metrics=["accuracy"])

# train the head of the network
print("----------Training head----------")
trainHead = model.fit_generator(
	traindataAugm.flow(train_X, train_Y, batch_size=BS),
	steps_per_epoch=len(train_X) // BS,
	validation_data=(val_X, val_Y),
	validation_steps=len(val_X) // BS,
	epochs=Epochs)

# make predictions on the testing set
print("----------Evaluating test set----------")
testpred = model.predict(test_X, batch_size=BS)

# set label with corresponding largest predicted probability

testpred_a = (testpred[:,1] > 0.5)*1

#Classification report #########################
print(classification_report(test_Y.argmax(axis=1),
    testpred_a,target_names=lb.classes_))

# Compute ROC curve and ROC area for each class
FPr = dict()
TPr = dict()
roc_auc = dict()
for i in range(test_Y.shape[-1]):
    FPr[i], TPr[i], _ = roc_curve(test_Y[:, i], testpred[:, i])
    roc_auc[i] = auc(FPr[i], TPr[i])

# Compute micro-average ROC curve and ROC area
FPr["micro"], TPr["micro"], _ = roc_curve(test_Y.ravel(), testpred.ravel())
roc_auc["micro"] = auc(FPr["micro"], TPr["micro"])

# plt.figure()
plt.style.use("dark_background")
plt.figure()
plt.plot(FPr["micro"], TPr["micro"], lw=2,label='ROC curve (AUC =%.2f)' % roc_auc["micro"],color='r') 
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--',label='Random guess')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(False)
plt.legend(loc="lower right")
plt.savefig('ROCcurve.png')


#compute the confusion matrix and use it to derive the raw
#sensitivity,specificity and accuracy
cf_m = confusion_matrix(test_Y.argmax(axis=1), testpred_a)
sens = cf_m[0, 0] / (cf_m[0, 0] + cf_m[0, 1])
spec = cf_m[1, 1] / (cf_m[1, 1] + cf_m[1, 0])
cf_m_sum = np.sum(sum(cf_m))
acc = (cf_m[0, 0] + cf_m[1, 1]) / cf_m_sum

# show the confusion matrix, accuracy, sensitivity, and specificity
print(cf_m)

print('acc=%.2f' % (acc))
print('sens=%.2f' % (sens))
print('spec=%.2f'% (spec))

# Plot Confustion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cf_m,
                              display_labels=lb.classes_)


# NOTE: Fill all variables here with default values of the plot confusion matrix
disp = disp.plot(include_values=True,cmap='Purples',
                 values_format=None,ax=None)

plt.show()

# plot the training loss and accuracy
nE = Epochs
plt.style.use("dark_background")
plt.figure()
plt.plot(np.arange(0, nE), trainHead.history["loss"], label="train_loss")
plt.plot(np.arange(0, nE), trainHead.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, nE), trainHead.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, nE), trainHead.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Pneumonia Dataset")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="middle right")
plt.savefig('acc_loss.png')

