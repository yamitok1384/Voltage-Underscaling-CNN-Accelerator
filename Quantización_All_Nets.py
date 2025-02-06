import tensorflow as tf
import os
import numpy as np
from Training import GetDatasets
from Nets_original  import GetNeuralNetworkModel
from Stats import WeightQuantization, ActivationStats, CheckAccuracyAndLoss, QuantizationEffect, GetReadAndWrites
from Simulation import buffer_simulation, save_obj, load_obj

tf.random.set_seed(1234)
np.random.seed(1234)



#Se llama a la función QuantizationEffect que realiza la cuantización de los pesos para obtener los bits
#tanto de parte entera como fraccionaria para las activaciones y los pesos con la presiciones para cada
#variante.
#El objetivo es definir los bits tanto en la parte fracc como en la parte entera con los cuales
#ontenemos una precisión óptima para los pesos entrenados.

train_batch_size = test_batch_size = 1



# Alexnet
train_batch_size = test_batch_size = 1
train_set,valid_set,test_set = GetDatasets('colorectal_histology',(80,5,15),(227,227), 8, train_batch_size, test_batch_size)

#Obtenemos el modelo

aging_active = [False]*11
AlexNet   = GetNeuralNetworkModel('AlexNet',(227,227,3),8, quantization = False, aging_active = aging_active)
loss      = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
AlexNet.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

AlexNet.evaluate(test_set)



# Se cargan los pesos
cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'AlexNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')
AlexNet.load_weights(wgt_dir)



df = QuantizationEffect('AlexNet',test_set,wgt_dir,(227,227,3),8,test_batch_size)
#save_obj(df,'Data/Quantization/AlexNet/Colorectal Dataset/Quantization')




#Densenet



trainSet,validSet,testSet = GetDatasets('colorectal_histology',(80,5,15),(224,224), 8, train_batch_size, test_batch_size)


aging_active= [False]*188
DenseNet   = GetNeuralNetworkModel('DenseNet',(224,224,3),8, quantization = False, aging_active=aging_active)
loss      = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
DenseNet.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])


cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'DenseNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir, 'Weights')
DenseNet.load_weights(wgt_dir)

(OrigLoss,OrigAcc) = DenseNet.evaluate(testSet)


df = QuantizationEffect('DenseNet',test_set,wgt_dir,(224,224,3),8,test_batch_size)
#save_obj(df,'Data/Quantization/AlexNet/Colorectal Dataset/Quantization')


#MobilNet
train_batch_size = test_batch_size = 1
trainSet,validSet,testSet = GetDatasets('colorectal_histology',(80,5,15),(224,224), 8, train_batch_size, test_batch_size)
aging_active = [False]*29

MobileNet = GetNeuralNetworkModel('MobileNet',(224,224,3),8, quantization = False, aging_active=aging_active)
loss      = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
MobileNet.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

(OrigLoss,OrigAcc) = MobileNet.evaluate(testSet)


cwd = os.getcwd()
wgtDir = os.path.join(cwd, 'Data')
wgtDir = os.path.join(wgtDir, 'Trained Weights')
wgtDir = os.path.join(wgtDir, 'MobileNet')
wgtDir = os.path.join(wgtDir, 'Colorectal Dataset')
wgtDir = os.path.join(wgtDir,'Weights')
MobileNet.load_weights(wgtDir)


df = QuantizationEffect('MobileNet',testSet,wgtDir,(224,224,3),8,test_batch_size)
save_obj(df,'Data/Quantization/MobileNet/Quantization')


#SqueezNet

train_set,valid_set,test_set = GetDatasets('colorectal_histology',(80,5,15),(224,224), 8, train_batch_size, test_batch_size)

SqueezeNet     = GetNeuralNetworkModel('SqueezeNet',(224,224,3),8, quantization = False, aging_active=False)
loss      = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
SqueezeNet.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

(OrigLoss,OrigAcc) = SqueezeNet.evaluate(test_set)


cwd = os.getcwd()
wgtDir = os.path.join(cwd, 'Data')
wgtDir = os.path.join(wgtDir, 'Trained Weights')
wgtDir = os.path.join(wgtDir, 'SqueezeNet')
wgtDir = os.path.join(wgtDir, 'Colorectal Dataset')
wgtDir = os.path.join(wgtDir,'Weights')
SqueezeNet.load_weights(wgtDir)


df = QuantizationEffect('SqueezeNet',testSet,wgtDir,(224,224,3),8,test_batch_size)
save_obj(df,'Data/Quantization/SqueezeNet/Colorectal Dataset/Quantization')


#VGG16



train_set,valid_set,test_set = GetDatasets('colorectal_histology',(80,5,15),(227,227), 8, train_batch_size, test_batch_size)


activation_aging = [True] * 21
VGG16   = GetNeuralNetworkModel('VGG16',(227,227,3),8, quantization = False, aging_active=activation_aging)
loss      = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
VGG16.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])


cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'VGG16')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir, 'Weights')
VGG16.load_weights(wgt_dir)

(OrigLoss,OrigAcc) = VGG16.evaluate(test_set)
print(test_set)

df = QuantizationEffect('VGG16',testSet,wgtDir,(224,224,3),8,test_batch_size)
save_obj(df,'Data/Quantization/SqueezeNet/Colorectal Dataset/Quantization')


#ZFNet


train_set,valid_set,test_set = GetDatasets('colorectal_histology',(80,5,15),(224,224), 8, train_batch_size, test_batch_size)

ZFNet   = GetNeuralNetworkModel('ZFNet',(224,224,3),8, quantization = False, aging_active=False)
loss      = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
ZFNet.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])


cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'ZFNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir, 'Weights')
ZFNet.load_weights(wgt_dir)

(OrigLoss,OrigAcc) = ZFNet.evaluate(test_set)
print(test_set)


df = QuantizationEffect('ZFNet',testSet,wgtDir,(224,224,3),8,test_batch_size)
save_obj(df,'Data/Quantization/SqueezeNet/Colorectal Dataset/Quantization')