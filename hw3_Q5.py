import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Activation, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from sklearn.metrics import confusion_matrix, roc_auc_score

# set image generators
train_dir='./chest_xray/train/'
test_dir='./chest_xray/test/'
validation_dir='./chest_xray/val/'

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_dir, batch_size=20, class_mode='binary')

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
        validation_dir, batch_size=20, class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        test_dir, batch_size=20, class_mode='binary')



# loading the model
from tensorflow.keras.models import load_model
model = load_model('chest_x-ray_4_extra_after_4.h5')
model.summary()

# Model Evaluation
y_test = test_generator.classes
y_pred = model.predict_generator(test_generator)
matrix = confusion_matrix(y_test, y_pred>0.5)
auc = roc_auc_score(y_test, y_pred)


TP = matrix[0][0]
FN = matrix[0][1]
FP = matrix[1][0]
TN = matrix[1][1]

Accuracy = (TP+TN)/624
Recall = TP/390
Precision = TP/(TP+FP)
Specificity = TN/(TN+FP)
F1 = (2*Precision*Recall)/(1*Precision+Recall)

print('---------------------')
print('\n')
print('Confusion Matrix')
print(matrix, '\n')
print('\nAccuracy: ', Accuracy)
print('\nPrecision: ', Precision)
print('\nRecall: ', Recall)
print('\nSpecificity: ', Specificity)
print('\nF1: ', F1)
print('\nAUC: ', auc)
