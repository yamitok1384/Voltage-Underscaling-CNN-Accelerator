#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

import tensorflow as tf
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from Training import GetDatasets
from Nets  import GetNeuralNetworkModel
from Stats import WeightQuantization, ActivationStats, CheckAccuracyAndLoss, QuantizationEffect, GetReadAndWrites
from Simulation import buffer_simulation, save_obj, load_obj


tf.random.set_seed(1234)
np.random.seed(1234)


# In[ ]:





# a) Get Dataset

# In[2]:


train_batch_size = test_batch_size = 32

train_set,valid_set,test_set = GetDatasets('colorectal_histology',(80,5,15),(224,224), 8, train_batch_size, test_batch_size)


# a) Get Model

# In[3]:


import tensorflow_datasets as tfds
tfds.load('colorectal_histology')


# Se crea la red, sin activar la cuantización ni el efecto de envejecimiento
# Las dimensiones de entrada de la imagen (224,224), el número de clases (8) y el tamaño de los batches

# In[4]:


ZFNet   = GetNeuralNetworkModel('ZFNet',(224,224,3),8, quantization = False, aging_active=False)
loss      = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
ZFNet.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])


#  b) Load/Save Weigths

# In[5]:


cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'ZFNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir, 'Weights')
ZFNet.load_weights(wgt_dir)


# Evaluación del accuracy y loss de la red

# In[6]:


(OrigLoss,OrigAcc) = ZFNet.evaluate(test_set)
print(test_set)


# 2) Stats

#  Write/Read Stats¶

# Se identifican (manualmente) las capas procesadadas(Convoluciones, Full conectadas y Pooling) junto a las capas que contienen los resultados que se escribiran en el buffer (capas luego de la funcion de activacion y/o Normalizacion)

# In[197]:


for index,layer in enumerate(ZFNet.layers):
    print(index,layer.name)
print('Las capas 0,3,9,11,17,19,25,31,37,40,45 y 50  contienen la informacion para su procesamiento en MMU')
print('Las capas 2,8,10,16,18,24,30,36,38,44,49 y 53 contienen las activaciones que son escritas en memoria')


# Con el siguiente bloque obtenemos el número de lecturas y escrituras por posición de memoria tanto usando la estrategia de CNN Gating o sin usarla

# In[13]:


Indices = [0,3,7,11,15,19,23,27,31,34,37,40]
Data    = GetReadAndWrites(ZFNet,Indices,279936,10,CNN_gating=False)
stats   = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
Baseline_Acceses   = pd.DataFrame(stats).reset_index(drop=False)
Data    = GetReadAndWrites(ZFNet,Indices,1024*1024,10,CNN_gating=True)
stats   = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
CNN_gating_Acceses = pd.DataFrame(stats).reset_index(drop=False)
save_obj(Baseline_Acceses,'Data/Acceses/ZFNet/Baseline_v2')
save_obj(CNN_gating_Acceses,'Data/Acceses/ZFNet/CNN_gating_Fix_v2')

# Analizar la posibilidad de usar menos bits

# In[11]:


CheckAccuracyAndLoss('ZFNet', test_set, wgt_dir, act_frac_size = 11, act_int_size = 4, wgt_frac_size = 15, wgt_int_size = 0,
                    input_shape = (224,224,3), output_shape = 8, batch_size = test_batch_size);


#  c) Activation Stats

# Para la configuración anterior, se observa el valor medio,maximo,minimo y el ratio de saturación tanto de las activaciones procesadas dentro de la unidad matricial de multiplicacion como de las almacenadas en el buffer. Nota: el ultimo parametro indica el numero de iteraciones que se deben realizar hasta agotar el dataset, se obtiene como numero de imagenes del dataset dividido en el batch size.

# In[12]:


ActivationStats(ZFNet,test_set,11,4,24)


# 3) Buffer Simulation

#  a) Baseline

# Ahora para el Baseline simularemos el comportamiento de 1 buffer durante la inferencia de 3 imagenes (solo 3 como ejemplo), la red se crea ahora activando la cuantizacion pero no el envejecimiento. LI y AI son los definidos en el item 2) Stats

# ......estos ficheros ya están , solo cargar e interpretarlos

# Sin inyectar errores original

# In[51]:


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
wgt_dir = os.path.join(wgt_dir, 'ZFNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')


trainBatchSize = testBatchSize = 16
_,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(224,224), 8, trainBatchSize, testBatchSize)


Accs_x= []



network_size   = 290400*16   # Tamaño del buffer (en bits)
num_of_samples = 1       # Numero de muestras de distintas configuraciones de fallos a testear por cada valor de Accs/Loss

n_bits_total = np.ceil(network_size).astype(int)    #numero de bits totales

buffer = np.array(['x']*(network_size))
print(buffer)
for index in range(0,num_of_samples):

    loss,acc_x   = CheckAccuracyAndLoss('ZFNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
                                            act_frac_size = 11, act_int_size = 4, wgt_frac_size = 15, wgt_int_size = 0,
                                            batch_size=testBatchSize, verbose = 0, aging_active = False, weights_faults = False)


    print(acc_x)



    Accs_x.append(acc_x)

print(str(n_bits_total)+' completada: ', datetime.now().strftime("%H:%M:%S"))
save_obj(Accs_x,'Data/Errors/ZFNet/Colorectal Dataset/Accs_x_05')
#save_obj(Loss,'Data/Errors/ZFNet/Colorectal Dataset/Loss')


# # Analizando el fichero en su estado original

# In[38]:


## from copy import deepcopy
from Stats import CheckAccuracyAndLoss
from Simulation import save_obj, load_obj
from FileAnalizeWgt import analize_file_w, analize_file_uno_w,analize_file_uno_ceros_w, save_file, load_file
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
from openpyxl import Workbook
from openpyxl import load_workbook


cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'ZFNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')


trainBatchSize = testBatchSize = 16
_,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(224,224), 8, trainBatchSize, testBatchSize)


Accs = []
Accs_w = []
Accs_a_w = []



buffer_size= 16777216 * 2


ruta_bin = 'Data/Fault Characterization/VC707/RawData'
#ruta_bin = 'Data/Fault Characterization/KC705_B/RowData/'
directorio = pathlib.Path(ruta_bin)


ficheros = [fichero.name for fichero in directorio.iterdir()]


for i, j in enumerate(ficheros):
    directorio= os.path.join(ruta_bin, j)
    buffer= (analize_file_w(directorio, buffer_size))
    error_mask, locs = (buffer_vectores(buffer))
    print(directorio)

    loss,acc   = CheckAccuracyAndLoss('ZFNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
                                            act_frac_size = 11, act_int_size = 4, wgt_frac_size = 15, wgt_int_size = 0,
                                            batch_size=testBatchSize, verbose = 0, aging_active = True, weights_faults = False,
                                            faulty_addresses = locs, masked_faults = error_mask)

    loss,acc_w   = CheckAccuracyAndLoss('ZFNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
                                            act_frac_size = 11, act_int_size = 4, wgt_frac_size = 15, wgt_int_size = 0,
                                            batch_size=testBatchSize, verbose = 0, aging_active = False, weights_faults = True,
                                            faulty_addresses = locs, masked_faults = error_mask)

    loss,acc_a_w   = CheckAccuracyAndLoss('ZFNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
                                            act_frac_size = 11, act_int_size = 4, wgt_frac_size = 15, wgt_int_size = 0,
                                            batch_size=testBatchSize, verbose = 0, aging_active = True, weights_faults = True,
                                            faulty_addresses = locs, masked_faults = error_mask)


    Accs.append(acc)
    Accs_w.append(acc_w)
    Accs_a_w.append(acc_a_w)


Acc=pd.DataFrame(Accs)
Acc_w =pd.DataFrame(Accs_w)
Acc_a_w =pd.DataFrame(Accs_a_w)
#buf_cero = pd.concat([Acc,Acc_w, Acc_a_w], axis=1, join='outer')
#buf_cero.columns =['Acc_cero', 'A_w_cero', 'Acc_a_w_cero']
#buf_cero.to_excel('resultado.xlsx', sheet_name='ceros_707', index=False)


Acc.head()



print(str()+' operación completada: ', datetime.now().strftime("%H:%M:%S"))
#lsave_file(Accs,'Data/Fault Characterization/Accs')

#save_obj(Loss,'Data/Errors/ZFNet/Colorectal Dataset/Loss')


# # Inyectando errores en 1

# In[42]:




Accs_1 = []
Accs_w_1 = []
Accs_a_w_1 = []




buffer_size= 16777216 *2


ruta_bin = 'Data/Fault Characterization/VC707/RawData'
#ruta_bin = 'Data/Fault Characterization/KC705_B/RowData/'
directorio = pathlib.Path(ruta_bin)


ficheros = [fichero.name for fichero in directorio.iterdir()]
print(directorio)

for i, j in enumerate(ficheros):
    directorio= os.path.join(ruta_bin, j)
    buffer= (analize_file_uno_w(directorio, buffer_size))
    error_mask, locs = (buffer_vectores(buffer))
    print(directorio)

    loss,acc_1   = CheckAccuracyAndLoss('ZFNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
                                            act_frac_size = 11, act_int_size = 4, wgt_frac_size = 15, wgt_int_size = 0,
                                            batch_size=testBatchSize, verbose = 0, aging_active = True, weights_faults = False,
                                            faulty_addresses = locs, masked_faults = error_mask)

    loss,acc_w_1   = CheckAccuracyAndLoss('ZFNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
                                            act_frac_size = 11, act_int_size = 4, wgt_frac_size = 15, wgt_int_size = 0,
                                            batch_size=testBatchSize, verbose = 0, aging_active = False, weights_faults = True,
                                            faulty_addresses = locs, masked_faults = error_mask)

    loss,acc_a_w_1   = CheckAccuracyAndLoss('ZFNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
                                            act_frac_size = 11, act_int_size = 4, wgt_frac_size = 15, wgt_int_size = 0,
                                            batch_size=testBatchSize, verbose = 0, aging_active = True, weights_faults = True,
                                            faulty_addresses = locs, masked_faults = error_mask)


    Accs_1.append(acc_1)
    Accs_w_1.append(acc_w_1)
    Accs_a_w_1.append(acc_a_w_1)


Acc_1=pd.DataFrame(Accs_1)
Acc_w_1 =pd.DataFrame(Accs_w_1)
Acc_a_w_1 =pd.DataFrame(Accs_a_w_1)
#buf_cero = pd.concat([Acc_1,Acc_w_1, Acc_a_w_1], axis=1, join='outer')
#buf_cero.columns =['Acc_uno', 'A_w_uno', 'Acc_a_w_uno']
#buf_cero.to_excel('resultado.xlsx', sheet_name='unos_707', index=False)

#buf_cero = pd.concat([Acc,Acc_w, Acc_a_w], axis=1, join='outer')
#buf_cero.columns =['Acc_cero', 'A_w_cero', 'Acc_a_w_cero']
#buf_cero.to_excel('resultado.xlsx', sheet_name='ceros_707', index=False)



Acc.head()




print(str()+' operación completada: ', datetime.now().strftime("%H:%M:%S"))
save_file(Accs_1,'Data/Fault Characterization/Accs')

#save_obj(Loss,'Data/Errors/ZFNet/Colorectal Dataset/Loss')


# # Inyectando errores aleatorios ceros y unos

# In[44]:



Accs_1_0 = []
Accs_w_1_0 = []
Accs_a_w_1_0 = []




buffer_size= 16777216 * 2


ruta_bin = 'Data/Fault Characterization/VC707/RawData'
#ruta_bin = 'Data/Fault Characterization/KC705_B/RowData/'
directorio = pathlib.Path(ruta_bin)


ficheros = [fichero.name for fichero in directorio.iterdir()]

for i, j in enumerate(ficheros):
    directorio= os.path.join(ruta_bin, j)
    buffer= (analize_file_uno_ceros_w(directorio, buffer_size))
    error_mask, locs = (buffer_vectores(buffer))
    print(directorio)

    loss,acc_1_0   = CheckAccuracyAndLoss('ZFNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
                                            act_frac_size = 11, act_int_size = 4, wgt_frac_size = 15, wgt_int_size = 0,
                                            batch_size=testBatchSize, verbose = 0, aging_active = True, weights_faults = False,
                                            faulty_addresses = locs, masked_faults = error_mask)

    loss,acc_w_1_0   = CheckAccuracyAndLoss('ZFNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
                                            act_frac_size = 11, act_int_size = 4, wgt_frac_size = 15, wgt_int_size = 0,
                                            batch_size=testBatchSize, verbose = 0, aging_active = False, weights_faults = True,
                                            faulty_addresses = locs, masked_faults = error_mask)

    loss,acc_a_w_1_0   = CheckAccuracyAndLoss('ZFNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
                                            act_frac_size = 11, act_int_size = 4, wgt_frac_size = 15, wgt_int_size = 0,
                                            batch_size=testBatchSize, verbose = 0, aging_active = True, weights_faults = True,
                                            faulty_addresses = locs, masked_faults = error_mask)


    Accs_1_0.append(acc_1_0)
    Accs_w_1_0.append(acc_w_1_0)
    Accs_a_w_1_0.append(acc_a_w_1_0)


Acc_1_0=pd.DataFrame(Accs_1_0)
Acc_w_1_0 =pd.DataFrame(Accs_w_1_0)
Acc_a_w_1_0 =pd.DataFrame(Accs_a_w_1_0)
buf_cero = pd.concat([Acc,Acc_w, Acc_a_w,Acc_1,Acc_w_1, Acc_a_w_1,Acc_1_0,Acc_w_1_0, Acc_a_w_1_0], axis=1, join='outer')
buf_cero.columns =['Acc_cero', 'A_w_cero', 'Acc_a_w_cero', 'Acc_uno', 'A_w_uno', 'Acc_a_w_uno' ,'Acc_uno_cero', 'A_w_uno_cero', 'Acc_a_w_uno_cero']
buf_cero.to_excel('ZFNet_buffer_2.xlsx', sheet_name='fichero_707_buffer2', index=False)

#buf_cero = pd.concat([Acc_1,Acc_w_1, Acc_a_w_1], axis=1, join='outer')
#buf_cero.columns =['Acc_uno', 'A_w_uno', 'Acc_a_w_uno']
#buf_cero.to_excel('resultado.xlsx', sheet_name='unos_707', index=False)

#buf_cero = pd.concat([Acc,Acc_w, Acc_a_w], axis=1, join='outer')
#buf_cero.columns =['Acc_cero', 'A_w_cero', 'Acc_a_w_cero']
#buf_cero.to_excel('resultado.xlsx', sheet_name='ceros_707', index=False)
Acc.head()









print(str()+' operación completada: ', datetime.now().strftime("%H:%M:%S"))
#save_file(Accs_1,'Data/Fault Characterization/Accs')

#save_obj(Loss,'Data/Errors/ZFNet/Colorectal Dataset/Loss')


# In[ ]:




