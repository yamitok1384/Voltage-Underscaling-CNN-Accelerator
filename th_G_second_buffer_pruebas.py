
#%load_ext autoreload
#%autoreload 2

import tensorflow as tf
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import pathlib
import matplotlib.pyplot as plt
from Training import GetDatasets
from Nets  import GetNeuralNetworkModel
from Stats import WeightQuantization, ActivationStats, CheckAccuracyAndLoss, QuantizationEffect, GetReadAndWrites
from Simulation import buffer_simulation, save_obj, load_obj, get_all_outputs
import math


tf.random.set_seed(1234)
np.random.seed(1234)

import tensorflow as tf
import numpy as np


train_batch_size = test_batch_size = 32

train_set,valid_set,test_set = GetDatasets('colorectal_histology',(80,5,15),(227,227), 8, train_batch_size, test_batch_size)

import tensorflow_datasets as tfds
tfds.load('colorectal_histology')

# AlexNet   = GetNeuralNetworkModel('AlexNet',(227,227,3),8, quantization = False, aging_active=False)
# loss      = tf.keras.losses.CategoricalCrossentropy()
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
# AlexNet.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# cwd = os.getcwd()
# wgt_dir = os.path.join(cwd, 'Data')
# wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
# wgt_dir = os.path.join(wgt_dir, 'AlexNet')
# wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
# wgt_dir = os.path.join(wgt_dir, 'Weights')
# AlexNet.load_weights(wgt_dir)
#
# (OrigLoss,OrigAcc) = AlexNet.evaluate(test_set)
# print(test_set)
#
# for index,layer in enumerate(AlexNet.layers):
#     print(index,layer.name)
# print('Las capas 0,3,9,11,17,19,25,31,37,40,45 y 50  contienen la informacion para su procesamiento en MMU')
# print('Las capas 2,8,10,16,18,24,30,36,38,44,49 y 53 contienen las activaciones que son escritas en memoria')
#
# num_address  = 290400
# Indices      = [0,3,9,11,17,19,25,31,37,40,45,50]
# samples      = 2
# # Sin Power Gating:
# Data         = GetReadAndWrites(AlexNet,Indices,num_address,samples,CNN_gating=False)
# stats        = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
# Baseline_Acceses   = pd.DataFrame(stats).reset_index(drop=False)
# # Con Power Gating
# Data     = GetReadAndWrites(AlexNet,Indices,num_address,samples,CNN_gating=True)
# stats    = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
# CNN_gating_Acceses = pd.DataFrame(stats).reset_index(drop=False)
#save_obj(Data,'Data/Acceses/AlexNet/Colorectal_Dataset/Baseline')
#save_obj(stats,'Data/Acceses/AlexNet/Colorectal_Dataset/CNN_gating_Adj')

# CheckAccuracyAndLoss('AlexNet', test_set, wgt_dir, act_frac_size = 11, act_int_size = 4, wgt_frac_size = 11, wgt_int_size = 4,
#                     input_shape = (227,227,3), output_shape = 8, batch_size = test_batch_size);
#
# ActivationStats(AlexNet,test_set,11,4,24)

# from copy import deepcopy
from Stats import CheckAccuracyAndLoss
from Simulation import save_obj, load_obj
from datetime import datetime
import itertools
from Training import GetDatasets
import numpy as np
import os

cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'AlexNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')
    

trainBatchSize = testBatchSize = 16
_,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(227,227), 8, trainBatchSize, testBatchSize)


Accs_x= []
Loss_x=[]



network_size   = 290400*16   # Tamaño del buffer (en bits)
num_of_samples = 1       # Numero de muestras de distintas configuraciones de fallos a testear por cada valor de Accs/Loss

n_bits_total = np.ceil(network_size).astype(int)    #numero de bits totales

buffer = np.array(['x']*(network_size))
print(buffer)
for index in range(0,num_of_samples):

    loss,acc_x   = CheckAccuracyAndLoss('AlexNet', test_dataset, wgt_dir, output_shape=8, input_shape = (227,227,3),
                                            act_frac_size = 11, act_int_size = 4, wgt_frac_size = 11, wgt_int_size = 4,
                                            batch_size=testBatchSize, verbose = 0, aging_active = False, weights_faults = True)


    print(acc_x)
    print(loss)



    Accs_x.append(acc_x)
    Loss_x.append(loss)
    print('accurancy sin inyectar errores', Accs_x)
    print('perdida sin inyectar errores', Loss_x)
    print('activaciones pesos en 1 aging_active = False, weights_faults = True', Accs_x)
    print('perdida pesos en 1 aging_active = False, weights_faults = True', loss)
    print('nan en perdidas', np.isnan(loss))
    print('nan en accurancy', np.isnan(Accs_x))

print(str(n_bits_total)+' completada: ', datetime.now().strftime("%H:%M:%S"))
# save_obj(Accs_x,'Data/Errors/AlexNet/Colorectal Dataset/Accs_x')
# #save_obj(Loss,'Data/Errors/AlexNet/Colorectal Dataset/Loss')

## from copy import deepcopy
from Stats import CheckAccuracyAndLoss
from Simulation import save_obj, load_obj
#from FileAnalizeWgtPrueba import analize_file_w, analize_file_uno_w,analize_file_uno_ceros_w, save_file, load_file
from FileAnalizeWgt import analize_file_w, analize_file_uno_w,analize_file_uno_ceros_w, save_file, load_file
#from FileAnalizeWgtRemodelado import analize_file_w, analize_file_uno_w,analize_file_uno_ceros_w, save_file, load_file
from funciones import buffer_vectores
import collections
from datetime import datetime
import itertools
from Training import GetDatasets
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pandas as pd
import os, sys
#
#
# cwd = os.getcwd()
# wgt_dir = os.path.join(cwd, 'Data')
# wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
# wgt_dir = os.path.join(wgt_dir, 'AlexNet')
# wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
# wgt_dir = os.path.join(wgt_dir,'Weights')
#
#
# trainBatchSize = testBatchSize = 16
# _,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(227,227), 8, trainBatchSize, testBatchSize)
#

Accs = []
Accs_w = []
Accs_a_w = []
Loss_w=[]



buffer_size= 16777216 *2
#
#
ruta_bin = 'Data/Fault Characterization/VC707/RawData/VC707-0.54.bin'
# #ruta_bin = 'Data/Fault Characterization/KC705_B/RowData/'
directorio = pathlib.Path(ruta_bin)
# num_of_samples = 1
#
#
# ficheros = [fichero.name for fichero in directorio.iterdir()]
# ficheros.sort()
# for index in range(0,num_of_samples):
#     for i, j in enumerate(ficheros):
#         directorio= os.path.join(ruta_bin, j)
buffer = (analize_file_w(directorio, buffer_size))
error_mask, locs = (buffer_vectores(buffer))

# loss,acc   = CheckAccuracyAndLoss('AlexNet', test_dataset, wgt_dir, output_shape=8, input_shape = (227,227,3),
#                                     act_frac_size = 11, act_int_size = 4, wgt_frac_size = 11, wgt_int_size = 4,
#                                     batch_size=testBatchSize, verbose = 0, aging_active = True, weights_faults = False,
#                                     faulty_addresses = locs, masked_faults = error_mask)
#
loss_w, acc_w = CheckAccuracyAndLoss('AlexNet', test_dataset, wgt_dir, output_shape=8, input_shape=(227, 227, 3),
                                   act_frac_size=11, act_int_size=4, wgt_frac_size=11, wgt_int_size=4,
                                   batch_size=testBatchSize, verbose=0, aging_active=False, weights_faults=True,
                                   faulty_addresses=locs, masked_faults=error_mask)
#
# loss,acc_a_w   = CheckAccuracyAndLoss('AlexNet', test_dataset, wgt_dir, output_shape=8, input_shape = (227,227,3),
#                                     act_frac_size = 11, act_int_size = 4, wgt_frac_size = 11, wgt_int_size = 4,
#                                     batch_size=testBatchSize, verbose = 0, aging_active = True, weights_faults = True,
#                                     faulty_addresses = locs, masked_faults = error_mask)


#Accs.append(acc)
Accs_w.append(acc_w)
Loss_w.append(loss_w)
#Accs_a_w.append(acc_a_w)
print('activaciones pesos en 1 aging_active = False, weights_faults = True', Accs_w)
print('perdida pesos en 1 aging_active = False, weights_faults = True',Loss_w)
print('nan en perdidas aging_active = False, weights_faults = True',np.isnan(Loss_w))
print('nan en accurancy aging_active = False, weights_faults = True',np.isnan(Accs_w))
# #
#
#
# Acc=pd.DataFrame(Accs)
# Acc_w =pd.DataFrame(Accs_w)
# Acc_a_w =pd.DataFrame(Accs_a_w)
#buf_cero = pd.concat([Acc,Acc_w, Acc_a_w], axis=1, join='outer')
#buf_cero.columns =['Acc_cero', 'A_w_cero', 'Acc_a_w_cero']
#buf_cero.to_excel('resultado.xlsx', sheet_name='ceros_707', index=False)





# print(str()+' operación completada: ', datetime.now().strftime("%H:%M:%S"))
# #lsave_file(Accs,'Data/Fault Characterization/Accs')
#
# #save_obj(Loss,'Data/Errors/AlexNet/Colorectal Dataset/Loss')
#


trainBatchSize = testBatchSize = 16
_,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(227,227), 8, trainBatchSize, testBatchSize)






cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'AlexNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')

#
Accs_1 = []
Accs_w_1 = []
Accs_a_w_1 = []
Loss_w_1=[]

# #
# #
# #
buffer_size= 16777216 *2
# #
# #
# #
# #
ruta_bin = 'Data/Fault Characterization/VC707/RawData/VC707-0.54.bin'
#ruta_bin = 'Data/Fault Characterization/KC705_B/RowData/'
directorio = pathlib.Path(ruta_bin)
#
#
# ficheros = [fichero.name for fichero in directorio.iterdir()]
# ficheros.sort()
# print(directorio)

# for i, j in enumerate(ficheros):
#     directorio = os.path.join(ruta_bin, j)
buffer = (analize_file_uno_w(directorio, buffer_size))
error_mask, locs = (buffer_vectores(buffer))
print(directorio)

#         loss,acc_1   = CheckAccuracyAndLoss('AlexNet', test_dataset, wgt_dir, output_shape=8, input_shape = (227,227,3),
#                                                 act_frac_size = 11, act_int_size = 4, wgt_frac_size = 11, wgt_int_size = 4,
#                                                 batch_size=testBatchSize, verbose = 0, aging_active = True, weights_faults = False,
#                                                 faulty_addresses = locs, masked_faults = error_mask)
#
#         acc_w_1   = CheckAccuracyAndLoss('AlexNet', test_dataset, wgt_dir, output_shape=8, input_shape = (227,227,3),
#                                                 act_frac_size = 11, act_int_size = 4, wgt_frac_size = 11, wgt_int_size = 4,
#                                                 batch_size=testBatchSize, verbose = 0, aging_active = True, weights_faults = True,
#                                                 faulty_addresses = locs, masked_faults = error_mask)
#         loss_w_1     = CheckAccuracyAndLoss('AlexNet', test_dataset, wgt_dir, output_shape=8, input_shape = (227,227,3),
#                                                 act_frac_size = 11, act_int_size = 4, wgt_frac_size = 11, wgt_int_size = 4,
#                                                 batch_size=testBatchSize, verbose = 0, aging_active = True, weights_faults = True,
#                                                 faulty_addresses = locs, masked_faults = error_mask)
#
loss_uno, acc_uno = CheckAccuracyAndLoss('AlexNet', test_dataset, wgt_dir, output_shape=8, input_shape=(227, 227, 3),
                                               act_frac_size=11, act_int_size=4, wgt_frac_size=11, wgt_int_size=4,
                                               batch_size=testBatchSize, verbose=0, aging_active=False, weights_faults=True,
                                               faulty_addresses=locs, masked_faults=error_mask)

Accs_w_1.append(acc_uno)
Loss_w_1.append(loss_uno)
print('accurancy inyectando solo en pesos 1',Accs_w_1)
print('perdida inyectando solo en pesos 1',Loss_w_1)
print('activaciones pesos en 1 aging_active = False, weights_faults = True', Accs_w_1)
print('perdida pesos en 1 aging_active = False, weights_faults = True',Loss_w_1)
print('nan en perdidas aging_active = False, weights_faults = True',np.isnan(Loss_w_1))
print('nan en accurancy aging_active = False, weights_faults = True',np.isnan(Accs_w_1))
# #
#
# Acc_1=pd.DataFrame(Accs_1)
# Acc_w_1 =pd.DataFrame(Accs_w_1)
# Acc_a_w_1 =pd.DataFrame(Accs_a_w_1)
# #buf_cero = pd.concat([Acc_1,Acc_w_1, Acc_a_w_1], axis=1, join='outer')
# #buf_cero.columns =['Acc_uno', 'A_w_uno', 'Acc_a_w_uno']
# #buf_cero.to_excel('resultado.xlsx', sheet_name='unos_707', index=False)
#
# #buf_cero = pd.concat([Acc,Acc_w, Acc_a_w], axis=1, join='outer')
# #buf_cero.columns =['Acc_cero', 'A_w_cero', 'Acc_a_w_cero']
# #buf_cero.to_excel('resultado.xlsx', sheet_name='ceros_707', index=False)
#
#
#
# Acc.head()




# print(str()+' operación completada: ', datetime.now().strftime("%H:%M:%S"))
# save_file(Accs_1,'Data/Fault Characterization/Accs')

#save_obj(Loss,'Data/Errors/AlexNet/Colorectal Dataset/Loss')


Accs_1_0 = []
Accs_w_1_0 = []
Loss_w_1_0=[]




buffer_size= 16777216 * 2


ruta_bin = 'Data/Fault Characterization/VC707/RawData/VC707-0.54.bin'
#ruta_bin = 'Data/Fault Characterization/KC705_B/RowData/'
directorio = pathlib.Path(ruta_bin)


# ficheros = [fichero.name for fichero in directorio.iterdir()]
# ficheros.sort()
#
# for index in range(0,num_of_samples):
#
#     for i, j in enumerate(ficheros):

        #directorio= os.path.join(ruta_bin, j)
buffer= (analize_file_uno_ceros_w(directorio, buffer_size))
error_mask, locs = (buffer_vectores(buffer))
print(directorio)
        
        # loss,acc_1_0   = CheckAccuracyAndLoss('AlexNet', test_dataset, wgt_dir, output_shape=8, input_shape = (227,227,3),
        #                                         act_frac_size = 11, act_int_size = 4, wgt_frac_size = 11, wgt_int_size = 4,
        #                                         batch_size=testBatchSize, verbose = 0, aging_active = True, weights_faults = False,
        #                                         faulty_addresses = locs, masked_faults = error_mask)
        
        # loss,acc_w_1_0   = CheckAccuracyAndLoss('AlexNet', test_dataset, wgt_dir, output_shape=8, input_shape = (227,227,3),
        #                                         act_frac_size = 11, act_int_size = 4, wgt_frac_size = 11, wgt_int_size = 4,
        #                                         batch_size=testBatchSize, verbose = 0, aging_active = False, weights_faults = True,
        #                                         faulty_addresses = locs, masked_faults = error_mask)
loss_w_1_0, acc_w_1_0 = CheckAccuracyAndLoss('AlexNet', test_dataset, wgt_dir, output_shape=8, input_shape=(227, 227, 3),
                                               act_frac_size=11, act_int_size=4, wgt_frac_size=11, wgt_int_size=4,
                                               batch_size=testBatchSize, verbose=0, aging_active=False, weights_faults=True,
                                               faulty_addresses=locs, masked_faults=error_mask)
#

        #
        #Accs_1_0.append(acc_1_0)
        #Accs_w_1_0.append(acc_w_1_0)
Accs_w_1_0.append(acc_w_1_0)
Loss_w_1_0.append(loss_w_1_0)


print(Accs_w_1_0)
print(Loss_w_1_0)
print('Es este')
print('accurancy pesos en 1 aging_active = False, weights_faults = True', Accs_w_1_0)
print('perdida pesos en 1 aging_active = Flase, weights_faults = True',Loss_w_1_0)
print('nan en perdidas aging_active = False, weights_faults = True',np.isnan(Loss_w_1_0))
print('nan en accu aging_active = False, weights_faults = True',np.isnan(Accs_w_1_0))
plt.plot(buffer)
plt.show


# print(Accs_w_1)
# print(loss_w_1)
# print('accurancy pesos en 1 aging_active = True, weights_faults = True', Accs_w_1)
# print('perdida pesos en 1 aging_active = True, weights_faults = True',Loss_w_1)
# print('np.isnan aging_active = True, weights_faults = True',np.isnan(Loss_w_1))
# #


# Arquit=pd.DataFrame(ficheros)
# Acc_1_0=pd.DataFrame(Accs_1_0)
# Acc_w_1_0 =pd.DataFrame(Accs_w_1_0)
# Acc_a_w_1_0 =pd.DataFrame(Accs_a_w_1_0)
# buf_cero = pd.concat([Arquit,Acc,Acc_w, Acc_a_w,Acc_1,Acc_w_1, Acc_a_w_1,Acc_1_0,Acc_w_1_0, Acc_a_w_1_0], axis=1, join='outer')
# buf_cero.columns =['Arquit','Acc_cero', 'A_w_cero', 'Acc_a_w_cero', 'Acc_uno', 'A_w_uno', 'Acc_a_w_uno' ,'Acc_uno_cero', 'A_w_uno_cero', 'Acc_a_w_uno_cero']
# buf_cero.to_excel('second_buf_resultado_Alexnet_comprob.xlsx', sheet_name='fichero_707', index=False)
#
# #buf_cero = pd.concat([Acc_1,Acc_w_1, Acc_a_w_1], axis=1, join='outer')
# #buf_cero.columns =['Acc_uno', 'A_w_uno', 'Acc_a_w_uno']
# #buf_cero.to_excel('resultado.xlsx', sheet_name='unos_707', index=False)
#
# #buf_cero = pd.concat([Acc,Acc_w, Acc_a_w], axis=1, join='outer')
# #buf_cero.columns =['Acc_cero', 'A_w_cero', 'Acc_a_w_cero']
# #buf_cero.to_excel('resultado.xlsx', sheet_name='ceros_707', index=False)
# Acc.head()


    
   



    
        
    

print(str()+' operación completada: ', datetime.now().strftime("%H:%M:%S"))
#save_file(Accs_1,'Data/Fault Characterization/Accs')

#save_obj(Loss,'Data/Errors/AlexNet/Colorectal Dataset/Loss')


