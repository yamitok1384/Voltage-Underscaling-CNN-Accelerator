#!/usr/bin/env python
# coding: utf-8

# ## SqueezeNet

# In[1]:

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

import os
import pickle as pickle
import tensorflow as tf
import numpy as np
#from Stats_original import WeightQuantization, IntroduceFaultsInWeights, GenerateFaultsList
#from Nets_test_shift import GetNeuralNetworkModel
from Stats import WeightQuantization,IntroduceFaultsInWeights,GenerateFaultsList,CheckAccuracyAndLoss
from funciones import compilNet, same_elements,buffer_vectores,Base,IsoAECC,ECC, Flip,ScratchPad,FlipPatchBetter,ShiftMask,WordType,TestBins,TestBinsAllActvs
from Nets_original import GetNeuralNetworkModel
from Training import GetDatasets
from Simulation import get_all_outputs
from Simulation import buffer_simulation, save_obj, load_obj
from datetime import datetime
import pandas as pd
import math


capa=[]


total_locs=[]
values_diff_zero = []
mask = [16384]
mask=np.array(16384)
print(mask.dtype)

vol=0.53
inc= 0

test= pd.DataFrame(['INTERVALO', 'CONTADOR'])
print(test)
test.to_excel('MoRS/Modelo3_col_8_'+ str(vol)+'/Analisis_Resultados/bins/test'+ str(vol)+'_'+ str(inc) + '.xlsx', sheet_name='fichero_707', index=False)

# error_mask = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/error_mask_' + str(inc))
# locs = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/locs_' + str(inc))
error_mask =load_obj('MoRS/Modelo3_col_8_'+ str(vol)+'/mask/error_mask_' + str(inc))
locs = load_obj('MoRS/Modelo3_col_8_'+ str(vol)+'/mask/locs_' + str(inc))
error_mask_new, locs,locs_patch,word_change = ScratchPad(error_mask, locs)
locs_LO, locs_HO, locs_H_L_O = WordType(error_mask_new, locs)

# WordType(error_mask_new)
# print(len(error_mask))
# print(len(locs))



#
trainBatchSize = testBatchSize = 1
# _, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (227, 227), 8, trainBatchSize, testBatchSize)
# # #
# # # # # In[3]:
# # # # # # Numero de bits para activaciones (a) y pesos (w)
# word_size  = 16
# afrac_size = 11
# aint_size  = 4
# wfrac_size = 11
# wint_size  = 4
# #
# # # Directorio de los pesos
# #
# cwd = os.getcwd()
# wgt_dir = os.path.join(cwd, 'Data')
# wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
# wgt_dir = os.path.join(wgt_dir, 'AlexNet')
# wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
# wgt_dir = os.path.join(wgt_dir,'Weights')
#
#
# acc_list = []
# iterator = iter(test_dataset)
#
# # ## Creo la red sin fallos
#
# diff_zero_total=[]
# total_locs=[]
# activation_aging = [False] * 11
#
# frac_size = 11
#
#
# Net = GetNeuralNetworkModel('AlexNet', (227,227,3), 8, aging_active=activation_aging,faulty_addresses=locs, masked_faults=error_mask,
#                             word_size=word_size, frac_size=afrac_size, batch_size=testBatchSize)
# Net.load_weights(wgt_dir).expect_partial()
# WeightQuantization(model=Net, frac_bits=wfrac_size, int_bits=wint_size)
# loss = tf.keras.losses.CategoricalCrossentropy()
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# Net.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
# loss, acc = Net.evaluate(test_dataset)
#
#
# # #### Binss Solo para las activaciones con LO y HO afectadas
# #
# #
# # ### Analizar valores distintos de 0 en el bit 14 de cada valor
# #
# write_layer = [2,8,10,16,18,24,30,36,38,44,49,53]
# #write_layer = [2,8]
#
# #intervalo,np_count= TestBins(write_layer,test_dataset,Net,locs_LO)
# intervalo,np_count= TestBinsAllActvs(write_layer,test_dataset,Net)
#
#
#
# df_intervalo= pd.DataFrame(intervalo)
# df_contador= pd.DataFrame(np_count)
# #print(df_contador)
# df_concat= pd.concat([df_intervalo,df_contador],axis=1, join='outer')
# df_concat.columns =['INTERVALO', 'CONTADOR']
# print(df_concat)
# df_concat.to_excel('MoRS/Modelo3_col_8_'+ str(vol)+'/Analisis_Resultados/bins/AlexNet_new_bins_allActivs_Sinfallos'+ str(vol)+'_'+ str(inc) + '.xlsx', sheet_name='fichero_707', index=False)
#
# del df_concat
# del Net
#
# # del df_concat
#
# trainBatchSize = testBatchSize = 1
# _, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)
#
# # # In[3]:
# # # Numero de bits para activaciones (a) y pesos (w)
# word_size  = 16
# afrac_size = 11
# aint_size  = 4
# wfrac_size = 14
# wint_size  = 1
#
# # Tamaño del buffer de activaciones == al tamaño de la capa con mayor numero de activaciones (290400 pesos de 16 bits cada uno)
#
# # Directorio de los pesos
# cwd = os.getcwd()
# wgt_dir = os.path.join(cwd, 'Data')
# wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
# wgt_dir = os.path.join(wgt_dir, 'MobileNet')
# wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
# wgt_dir = os.path.join(wgt_dir, 'Weights')
#
#
#
# acc_list = []
# iterator = iter(test_dataset)
# diff_zero_total=[]
# total_locs=[]
# # ## Creo la red sin fallos
# #
# frac_size = 11
#
# activation_aging = [False] * 29
#
#
# Net = GetNeuralNetworkModel('MobileNet', (224, 224, 3), 8, aging_active=activation_aging,faulty_addresses=locs, masked_faults=error_mask,
#                              word_size=word_size, frac_size=afrac_size, batch_size=testBatchSize)
# Net.load_weights(wgt_dir).expect_partial()
# WeightQuantization(model=Net, frac_bits=wfrac_size, int_bits=wint_size)
# loss = tf.keras.losses.CategoricalCrossentropy()
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# Net.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
# loss_sf, acc_sf = Net.evaluate(test_dataset)
#
#
#
# write_layer=[2,9,15,21,28,34,40,46,53,59,65,71,78,84,90,96,102,108,114,120,126,132,138,144,151,157,163,169,173,179]
#
# #intervalo,np_count= TestBins(write_layer,test_dataset,Net,locs_LO)
# intervalo,np_count= TestBinsAllActvs(write_layer,test_dataset,Net)
#
# df_intervalo= pd.DataFrame(intervalo)
# df_contador= pd.DataFrame(np_count)
# #print(df_contador)
# df_concat= pd.concat([df_intervalo,df_contador],axis=1, join='outer')
# df_concat.columns =['INTERVALO', 'CONTADOR']
# print(df_concat)
# df_concat.to_excel('MoRS/Modelo3_col_8_'+ str(vol)+'/Analisis_Resultados/bins/MobileNet_new_bins_allActivs_Sinfallos'+ str(vol)+'_'+ str(inc) + '.xlsx', sheet_name='fichero_707', index=False)
#
# del df_concat
# del Net
#
#
#
#
# trainBatchSize = testBatchSize = 1
# _, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)
# # #
# # # # # In[3]:
# # # # # Numero de bits para activaciones (a) y pesos (w)
# word_size  = 16
# afrac_size = 11
# aint_size  = 4
# wfrac_size = 14
# wint_size  = 1
# #
# # # Directorio de los pesos
# #
# cwd = os.getcwd()
# wgt_dir = os.path.join(cwd, 'Data')
# wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
# wgt_dir = os.path.join(wgt_dir, 'ZFNet')
# wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
# wgt_dir = os.path.join(wgt_dir,'Weights')
#
#
# acc_list = []
# iterator = iter(test_dataset)
#
# # ## Creo la red sin fallos
#
#
# diff_zero_total=[]
# total_locs=[]
# activation_aging = [False] * 11
# frac_size = 11
#
# Net = GetNeuralNetworkModel('ZFNet', (224, 224, 3), 8, aging_active=activation_aging,faulty_addresses=locs, masked_faults=error_mask,
#                             word_size=word_size, frac_size=afrac_size, batch_size=testBatchSize)
# Net.load_weights(wgt_dir).expect_partial()
# WeightQuantization(model=Net, frac_bits=wfrac_size, int_bits=wint_size)
# loss = tf.keras.losses.CategoricalCrossentropy()
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# Net.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
# loss, acc = Net.evaluate(test_dataset)
#
#
#
# ### Analizar valores distintos de 0 en el bit 14 de cada valor
# write_layer = [2,6,10,14,18,22,26,30,32,36,39,43]
# #intervalo,np_count= TestBins(write_layer,test_dataset,Net,locs_LO)
# intervalo,np_count= TestBinsAllActvs(write_layer,test_dataset,Net)
#
#
# df_intervalo= pd.DataFrame(intervalo)
# df_contador= pd.DataFrame(np_count)
# #print(df_contador)
# df_concat= pd.concat([df_intervalo,df_contador],axis=1, join='outer')
# df_concat.columns =['INTERVALO', 'CONTADOR']
# print(df_concat)
# df_concat.to_excel('MoRS/Modelo3_col_8_'+ str(vol)+'/Analisis_Resultados/bins/ZFNet_new_bins_allActivs_Sinfallos'+ str(vol)+'_'+ str(inc) + '.xlsx', sheet_name='fichero_707', index=False)
# del df_concat
# del Net
#
#
# trainBatchSize = testBatchSize = 1
# _, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)
#
# # # In[3]:
# # # Numero de bits para activaciones (a) y pesos (w)
# word_size = 16
# afrac_size = 12
# aint_size = 3
# wfrac_size = 15
# wint_size = 0
#
#
# # Tamaño del buffer de activaciones == al tamaño de la capa con mayor numero de activaciones (290400 pesos de 16 bits cada uno)
#
# # Directorio de los pesos
# cwd = os.getcwd()
# wgt_dir = os.path.join(cwd, 'Data')
# wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
# wgt_dir = os.path.join(wgt_dir, 'VGG16')
# wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
# wgt_dir = os.path.join(wgt_dir, 'Weights')
#
# activation_aging = [False] * 21
#
# Net = GetNeuralNetworkModel('VGG16', (224, 224, 3), 8, aging_active=activation_aging,faulty_addresses=locs, masked_faults=error_mask,
#                              word_size=word_size, frac_size=afrac_size, batch_size=testBatchSize)
# Net.load_weights(wgt_dir).expect_partial()
# WeightQuantization(model=Net, frac_bits=wfrac_size, int_bits=wint_size)
# loss = tf.keras.losses.CategoricalCrossentropy()
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# Net.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
# loss_sf, acc_sf = Net.evaluate(test_dataset)
#
#
#
# write_layer = [2, 6, 10, 12, 16, 20, 22, 26, 30, 34, 36, 40, 44, 48, 50, 54, 58, 62, 64, 69, 73, 77]
# #intervalo,np_count= TestBins(write_layer,test_dataset,Net,locs_LO)
# intervalo,np_count= TestBinsAllActvs(write_layer,test_dataset,Net)
#
#
# df_intervalo= pd.DataFrame(intervalo)
# df_contador= pd.DataFrame(np_count)
# #print(df_contador)
# df_concat= pd.concat([df_intervalo,df_contador],axis=1, join='outer')
# df_concat.columns =['INTERVALO', 'CONTADOR']
# print(df_concat)
# df_concat.to_excel('MoRS/Modelo3_col_8_'+ str(vol)+'/Analisis_Resultados/bins/VGG16_new_bins_allActivs_Sinfallos'+ str(vol)+'_'+ str(inc) + '.xlsx', sheet_name='fichero_707', index=False)
#
# del df_concat
# del Net
#
# #
#
#
#
# trainBatchSize = testBatchSize = 1
# _, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)
#
# # # In[3]:
# # # Numero de bits para activaciones (a) y pesos (w)
# word_size  = 16
# afrac_size = 9
# aint_size  = 6
# wfrac_size = 15
# wint_size  = 0
# #
# # # Directorio de los pesos
# #
# cwd = os.getcwd()
# wgt_dir = os.path.join(cwd, 'Data')
# wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
# wgt_dir = os.path.join(wgt_dir, 'SqueezeNet')
# wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
# wgt_dir = os.path.join(wgt_dir,'Weights')
#
#
# acc_list = []
#
#
# # ## Creo la red sin fallos
#
# # buffer_size= 16777216
#
#
#
# activation_aging = [False] * 22
# Net = GetNeuralNetworkModel('SqueezeNet', (224, 224, 3), 8, aging_active=activation_aging,faulty_addresses=locs, masked_faults=error_mask,
#                             word_size=word_size, frac_size=afrac_size, batch_size=testBatchSize)
# Net.load_weights(wgt_dir).expect_partial()
# WeightQuantization(model=Net, frac_bits=wfrac_size, int_bits=wint_size)
# loss = tf.keras.losses.CategoricalCrossentropy()
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# Net.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
# loss, acc = Net.evaluate(test_dataset)
#
#
# write_layer = [2,6,8,12, 19,23, 30,34, 41,43,47, 54,58, 65,69, 76,80 , 87,89,93, 100,103,107]
# #intervalo,np_count= TestBins(write_layer,test_dataset,Net,locs_LO)
# intervalo,np_count= TestBinsAllActvs(write_layer,test_dataset,Net)
#
# df_intervalo= pd.DataFrame(intervalo)
# df_contador= pd.DataFrame(np_count)
# #print(df_contador)
# df_concat= pd.concat([df_intervalo,df_contador],axis=1, join='outer')
# df_concat.columns =['INTERVALO', 'CONTADOR']
# print(df_concat)
# df_concat.to_excel('MoRS/Modelo3_col_8_'+ str(vol)+'/Analisis_Resultados/bins/SqueezeNet_new_bins_allActivs_Sinfallos'+ str(vol)+'_'+ str(inc) + '.xlsx', sheet_name='fichero_707', index=False)
#
# del df_concat
# del Net


trainBatchSize = testBatchSize = 1
_, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)

# # In[3]:
# # Numero de bits para activaciones (a) y pesos (w)
word_size  = 16
afrac_size = 12
aint_size  = 3
wfrac_size = 13
wint_size  = 2

# Tamaño del buffer de activaciones == al tamaño de la capa con mayor numero de activaciones (290400 pesos de 16 bits cada uno)

# Directorio de los pesos
cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'DenseNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir, 'Weights')



acc_list = []
iterator = iter(test_dataset)
differnce_list=[]
diff_sum_Shift =[]
diff_sum_FP =[]
## Creo la red sin fallos

activation_aging = [False] * 188
Net = GetNeuralNetworkModel('DenseNet', (224, 224, 3), 8, aging_active=activation_aging,faulty_addresses=locs, masked_faults=error_mask,
                             word_size=word_size, frac_size=afrac_size, batch_size=testBatchSize)
Net.load_weights(wgt_dir).expect_partial()
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
WeightQuantization(model=Net, frac_bits=wfrac_size, int_bits=wint_size)
Net.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
loss_sf, acc_sf = Net.evaluate(test_dataset)

write_layer = [2,9,11,15,21,23,11,28,34,36,24,41,47,49,37,54,60,62,50,67,73,75,63,80,86,88,76,93,96,98,102,108,110,98,
     115,121,123,111,128,134,136,124,141,147,49,137,154,160,162,150,167,173,175,163,180,186,188,176,193,199,201,189,206,
     212,214,202,219,225,27,215,232,238,240,228,245,251,253,241,258,261,263,267,273,275,263, 280,286,288,276,293,299,301,
     289,306,312,314,302,319,325,327,315,332,338,340,328,345,351,353,341, 358,364,366,353,371,377,379,367,384,390,392,380,
     397,403,405,393,410,416,418,406,423,429,431,419, 436,442,444,432,449,455,457,445,462,468,470,458,475,481,483,471,488,
     494,496,484,501,507,509,497, 514,520,522,510,527,533,535,523,540,546,548,536,553,559,561,549,566,572,574,562,579,582,
     584,588,594,596,584, 601,607,609,597,614,620,622,610,627,633,635,623,640,646,648,636,653,659,661,649,666,672,674,662,
     679,685,687,675,692,698,700,688,705,711,713,701,718,724,726,714,731,737,739,727,744,750,752,740, 757,763,765,753,770,
     776,778,766,783,789,791,779,796,799,803]

intervalo,np_count = TestBins(write_layer,test_dataset,Net,locs_LO)
df_intervalo= pd.DataFrame(intervalo)
df_contador= pd.DataFrame(np_count)
#print(df_contador)
df_concat= pd.concat([df_intervalo,df_contador],axis=1, join='outer')
df_concat.columns =['INTERVALO', 'CONTADOR']
print(df_concat)
df_concat.to_excel('MoRS/Modelo3_col_8_'+ str(vol)+'/Analisis_Resultados/bins/DenseNet_new_bins_test_LO_mask_Sinfallos'+ str(vol)+'_'+ str(inc) + '.xlsx', sheet_name='fichero_707', index=False)

## Analizar valores distintos de 0 en el bit 14 de cada valor



# for i, j in enumerate(write_layer):
#     print('.........................................',j)
#     index = 0
#     stop = False
#
#     iterator = iter(test_dataset)
#     #while index < len(test_dataset) and stop==False:
#     while index < 750 :
#         print('index............................',index)
#         image = next(iterator)[0]
#         outputs = get_all_outputs(Net, image)
#         #print('outputs[j].size',outputs[j].size)
#         output=outputs[j]
#         output.flatten(order='F')
#         datos = output
#         actvs = output.size
#         #print('tamaño de las actvs', actvs)
#         locs_lo = np.array(locs_LO)
#         locs_affected = locs_lo[(locs_lo < output.size)]
#         #print('tamalo de lo afectado', len(locs_affected))
#         locs_size = len(locs_affected)
#         #print('tamaño de locs_ffected', locs_size)
#         affectedValues = np.take(output, locs_affected)
#         #counts, bin_edges = np.histogram(datos, bins)
#         counts, bin_edges = np.histogram(affectedValues, bins)
#         intervalo = []
#         contar = []
#         for low, hight, count in zip(bin_edges, np.roll(bin_edges, -1), counts):
#             print(f"{count}")
#             intervalo.append(f'{low}-{hight}')
#             # rint('size',count.size)
#             contar.append(count)
#
#         np_contar = np.array(contar)
#         np_count = np_count + np_contar
#         print('np_count', np_count)
#         index = index + 1
#         # if index == 2:
#         #     stop = True
# df_intervalo= pd.DataFrame(intervalo)
# df_contador= pd.DataFrame(np_count)
# #print(df_contador)
# df_concat= pd.concat([df_intervalo,df_contador],axis=1, join='outer')
# df_concat.columns =['INTERVALO', 'CONTADOR']
# print(df_concat)
#df_concat.to_excel('MoRS/Modelo3_mas_fallos_col_8_experimentos/estatics/bins/MobilNet_new_bins_test_LO_mask_test_funcion.xlsx', sheet_name='fichero_707', index=False)

#

##Bins
###Para todas las activaciones
# mask = [16384]
# mask=np.array(16384)
# total_values=[]
# locs_by_layer=[]
# ### Analizar valores distintos de 0 en el bit 14 de cada valor
# write_layer = [2,8,10,16,18,24,30,36,38,44,49,53]
#
#
# np_count = np.full(16, 0)
# #bins = [-64,-32,-16,-8,-4,-2,-1,0, 1, 2, 4, 8, 16, 32, 64]
# bins = [-256,-128,-64,-32,-16,-8,-4,-2,-1,0,1 ,2, 4, 8,16,32,64,128,256]
# save=[]
# for i, j in enumerate(write_layer):
#     print('.........................................',j)
#     index = 0
#     stop = False
#
#     iterator = iter(test_dataset)
#     #while index < len(test_dataset) and stop==False:
#     while index < 750 :
#         print('index............................',index)
#         image = next(iterator)[0]
#         outputs = get_all_outputs(Net, image)
#         print('outputs[j].size',outputs[j].size)
#         output=outputs[j]
#         output.flatten(order='F')
#         datos = output
#
#         counts, bin_edges = np.histogram(datos, bins)
#         intervalo = []
#         contar = []
#         for low, hight, count in zip(bin_edges, np.roll(bin_edges, -1), counts):
#             print(f"{count}")
#             intervalo.append(f'{low}-{hight}')
#             # rint('size',count.size)
#             contar.append(count)
#
#             np_contar = np.array(contar)
#         np_count = np_count + np_contar
#         print('np_count', np_count)
#         index = index + 1
#         # if index == 2:
#         #     stop = True
# df_intervalo= pd.DataFrame(intervalo)
# df_contador= pd.DataFrame(np_count)
# #print(df_contador)
# df_concat= pd.concat([df_intervalo,df_contador],axis=1, join='outer')
# df_concat.columns =['INTERVALO', 'CONTADOR']
# print(df_concat)
# df_concat.to_excel('MoRS/Modelo3_mas_fallos_col_8_experimentos/estatics/bins/AlexNet_new_bins_test_LO_mask.xlsx', sheet_name='fichero_707', index=False)
#
# #


# ### Solo para las activaciones afectadas por las direcciones con fallos
# mask = [16384]
# mask=np.array(16384)
# total_values=[]
# locs_by_layer=[]
# ### Analizar valores distintos de 0 en el bit 14 de cada valor
# write_layer = [2,8,10,16,18,24,30,36,38,44,49,53]
#
# for i, j in enumerate(write_layer):
#     print('.........................................',j)
#     save_values=[]
#     index = 0
#     stop = False
#
#     iterator = iter(test_dataset)
#     #while index < len(test_dataset) and stop==False:
#     while index < 750 and stop == False:
#         print('index............................',index)
#         image = next(iterator)[0]
#         outputs = get_all_outputs(Net, image)
#         print('outputs[j].size',outputs[j].size)
#         output=outputs[j]
#         output.flatten(order='F')
#         locs = np.array(locs)
#         print('maximo valor de todo el locs',np.amax(locs))
#         print('tamaño del locs',len(locs))
#         locs_affected = locs[(locs < output.size)]
#         print('maximo valor de todo el locs afectados', np.amax(locs_affected))
#         #print('locs_ffected', locs_affected)
#         locs_size = len(locs_affected)
#         print('tamaño de locs_ffected', locs_size)
#         total_locs.append(locs_size)
#         print('total_locs,total_locs')
#         #print('output',output)
#         affectedValues = np.take(output, locs_affected)
#         Ogdtype = affectedValues.dtype
#         # print(Ogdtype)
#         shift = 2 ** (15)
#         factor = 2 ** frac_size
#         output = affectedValues * factor
#         output = output.astype(np.int32)
#         output = np.where(np.less(output, 0), -output + shift, output)
#         original = output
#         #print('original', original)
#         original = np.bitwise_and(original, mask)
#         #print('original',original)
#         valores_afectados = np.not_equal(original, 0)
#         diff_zero_values = np.count_nonzero(valores_afectados)
#         print('diff_zero_values',diff_zero_values)
#         if diff_zero_values > 0:
#             print('dentro del if')
#             save_values.append(diff_zero_values)
#             print('save_values',save_values)
#             index = index + 1
#         else:
#             index = index+1
#
#         if index == 750:
#             stop = True
#             sum_FP = np.sum(save_values)
#             print('sum_FP',sum_FP)
#             total_values.append(sum_FP)
#             print(total_values)
#             locs_by_layer.append(locs_size)
#             capa.append(Net.layers[j].__class__.__name__)
#         else:
#             stop = False
#
# print('total_values',total_values)
# print('total_values',len(total_values))
# print('locs_by_layer',locs_by_layer)
# print('locs_by_layer',len(locs_by_layer))
# sum_total_values=np.sum(total_values)
# sum_len_layer=np.sum(locs_by_layer)
# ratio=np.sum(total_values)/np.sum(locs_by_layer)
# print('suma de los valores totales',sum_total_values)
# print('np.sum(locs_by_layer)',sum_len_layer)
# print('ratio', ratio)
# capa.append('ratio')
# total_values.append(sum_total_values)
# locs_by_layer.append(ratio)
# df_total_values = pd.DataFrame(total_values)
# df_locs_by_layer = pd.DataFrame(locs_by_layer)
# df_capa = pd.DataFrame(capa)
# buf_test_shift = pd.concat([df_capa,df_total_values,df_locs_by_layer], axis=1, join='outer')
# buf_test_shift.columns = ['Capa','values_af','locs_affected']
# print('buf_test_shift', buf_test_shift)
# buf_test_shift.to_excel('MoRS/Modelo3_mas_fallos_col_8_experimentos/estatics/AlexNet_test_shift_ratio_affect_sinfallos.xlsx', sheet_name='fichero_707', index=False)
#
#
#