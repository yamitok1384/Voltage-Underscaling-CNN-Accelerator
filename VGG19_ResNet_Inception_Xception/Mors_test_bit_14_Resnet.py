#!/usr/bin/env python
# coding: utf-8



# In[1]:

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

import os
import pickle as pickle
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.optimizers import Adam
import numpy as np
#from Nets_test_shift import GetNeuralNetworkModel
from funciones import compilNet, same_elements,buffer_vectores,Base,IsoAECC,ECC, Flip,FlipPatch,FlipPatchBetter,ShiftMask
from Stats import WeightQuantization, IntroduceFaultsInWeights, GenerateFaultsList
from Nets import GetNeuralNetworkModel
from Training import GetDatasets
from Simulation_29_08 import get_all_outputs
from Simulation_29_08 import buffer_simulation, save_obj, load_obj
from datetime import datetime
import pandas as pd


(train_ds, validation_ds, test_ds), info = tfds.load(
    "colorectal_histology",
    split=["train[:85%]", "train[85%:95%]", "train[95%:]"],
    with_info=True,
    as_supervised=True,
    shuffle_files= True,
)

num_classes = info.features['label'].num_classes

size = (150, 150)
batch_size = 1

train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
validation_ds = validation_ds.map(lambda x, y: (tf.image.resize(x, size), y))
test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, size), y))

train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=1)
validation_ds = validation_ds.cache().batch(batch_size).prefetch(buffer_size=1)
test_ds = test_ds.cache().batch(batch_size).prefetch(buffer_size=1)


capa=[]


total_locs=[]
values_diff_zero = []
mask = [16384]
mask=np.array(16384)
print(mask.dtype)
#### solo par alas capas qu eescriben por cada red
## acumular los locs y luego hacer la razon entre el total diferente de cero y los locs afectados
# def DifferenceZero(output,locs,frac_size):
#     print('ciclo')
#     #print('tamaño el outpust',len(outputs))
#     #write_layer = [2,6,8,12, 19,23, 30,34, 41,43,47, 54,58, 65,69, 76,80 , 87,89,93, 100,103,107]
#
#     output.flatten(order='F')
#     locs=locs[(locs < len(j))]
#     locs_size = len(locs)
#     total_locs.append(locs_size)
#     print('locs_size', locs_size)
#     affectedValues = np.take(j, locs)
#     capa.append(Net.layers[i].__class__.__name__)
#     #locs_by_layer.append(locs_size)
#     Ogdtype = affectedValues.dtype
#     # print(Ogdtype)
#     shift = 2 ** (15)
#     factor = 2 ** frac_size
#     output = affectedValues * factor
#     output = output.astype(np.int32)
#     output = np.where(np.less(output, 0), -output + shift, output)
#     original = output
#     original = np.bitwise_and(original, mask)
#     valores_afectados = np.not_equal(original, 0)
#     diff_zero_values = np.count_nonzero(valores_afectados)
#     print('diff_zero_values', diff_zero_values)
#     # if diff_zero_values is None:
    #return sum_values_diff_zero, sum_total_locs


#####SqueezeNet#########################################################################

#

# #
# # # # In[3]:
# # # # Numero de bits para activaciones (a) y pesos (w)

#
# # Directorio de los pesos
#
wgt_dir= ('../weights/ResNet50/weights.data')



acc_list = []
iterator = iter(test_ds)

# ## Creo la red sin fallos

diff_zero_total=[]
total_locs=[]
activation_aging = [False] * 22

#write_layer = [2,6,8,12, 19,23, 30,34, 41,43,47, 54,58, 65,69, 76,80 , 87,89,93, 100,103,107]
word_size  = 16
afrac_size = 11
aint_size  = 4
wfrac_size = 14
wint_size  = 1
error_mask_x=load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/error_mask_all_x')
locs  = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/locs_0')

Net = GetNeuralNetworkModel('ResNet', (150,150,3), 8, aging_active=activation_aging,faulty_addresses=locs, masked_faults=error_mask_x,
                            word_size=word_size, frac_size=afrac_size, batch_size=test_ds)
Net.load_weights(wgt_dir).expect_partial()
WeightQuantization(model=Net, frac_bits=wfrac_size, int_bits=wint_size)
loss='sparse_categorical_crossentropy'
optimizer = Adam()
Net.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
loss, acc = Net.evaluate(test_ds)
print('original', acc)


save=[]
for index,layer in enumerate(Net.layers):
    if Net.layers[index].__class__.__name__ == 'Lambda':
        print(Net.layers[index].__class__.__name__)
        save.append(index)
df_layers_lambdas= pd.DataFrame(save)

df_layers_lambdas.to_excel('MoRS/Modelo3_mas_fallos_col_8_experimentos/estatics/ResNet_df_layers_lambdas.xlsx', sheet_name='fichero_707', index=False)


# frac_size = 7
# mask = [16384]
# mask=np.array(16384)
# total_values=[]
# locs_by_layer=[]
# ### Analizar valores distintos de 0 en el bit 14 de cada valor
# # # write_layer =
# write_layer = [3, 9, 15, 20, 28, 32, 36, 40, 43, 47, 51, 55, 58, 63, 68, 76, 80, 84, 88, 91, 95, 99, 103, 106, 110,
#  114, 121, 126, 131, 139, 143, 147, 151, 154, 158, 162, 166, 169, 173, 177, 181, 184, 196, 199, 211, 219,
#  224, 232, 236, 240, 244, 247, 255, 259, 262,266, 268]
#
# for index,layer in enumerate(Net.layers):
#     save_values = []
#     img = 0
#     stop = False
#     print(Net.layers[index].__class__.__name__)
#     if Net.layers[index].__class__.__name__ == 'Lambda':
#         print(Net.layers[index].__class__.__name__)
#
#
#         iterator = iter(test_ds)
#         #while index < len(test_dataset) and stop==False:
#         while img < len(test_dataset) and stop == False:
#             print('img............................',img)
#             image = next(iterator)[0]
#             outputs = get_all_outputs(Net, image)
#             print('outputs[index].size', outputs[index].size)
#             output = outputs[index]
#             output.flatten(order='F')
#             locs = np.array(locs)
#             print('maximo valor de todo el locs',np.amax(locs))
#             print('tamaño del locs',len(locs))
#             locs_affected = locs[(locs < output.size)]
#             print('maximo valor de todo el locs afectados', np.amax(locs_affected))
#             #print('locs_ffected', locs_affected)
#             locs_size = len(locs_affected)
#             print('tamaño de locs_ffected', locs_size)
#             #total_locs.append(locs_size)
#             #print('total_locs,total_locs')
#             #print('output',output)
#             affectedValues = np.take(output, locs_affected)
#             Ogdtype = affectedValues.dtype
#             # print(Ogdtype)
#             shift = 2 ** (15)
#             factor = 2 ** frac_size
#             output = affectedValues * factor
#             output = output.astype(np.int32)
#             output = np.where(np.less(output, 0), -output + shift, output)
#             original = output
#             #print('original', original)
#             original = np.bitwise_and(original, mask)
#             #print('original',original)
#             valores_afectados = np.not_equal(original, 0)
#             diff_zero_values = np.count_nonzero(valores_afectados)
#             print('diff_zero_values',diff_zero_values)
#             if diff_zero_values > 0:
#                 print('dentro del if')
#                 save_values.append(diff_zero_values)
#                 print('save_values',save_values)
#                 img = img + 1
#             else:
#                 img = img + 1
#
#             if img == len(test_dataset):
#                 stop = True
#                 sum_FP = np.sum(save_values)
#                 print('sum_FP',sum_FP)
#                 total_values.append(sum_FP)
#                 print(total_values)
#                 locs_by_layer.append(locs_size)
#                 capa.append(Net.layers[index].__class__.__name__)
#             else:
#                 stop = False
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
# buf_test_shift.to_excel('MoRS/Modelo3_mas_fallos_col_8_experimentos/estatics/ResNet_test_shift_ratio_affect.xlsx', sheet_name='fichero_707', index=False)
#
#
#








