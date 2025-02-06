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
from Nets_test_shift import GetNeuralNetworkModel
from Stats import WeightQuantization,IntroduceFaultsInWeights,GenerateFaultsList,CheckAccuracyAndLoss
from funciones import compilNet, same_elements,buffer_vectores,Base,IsoAECC,ECC, Flip,FlipPatch,FlipPatchBetter,ShiftMask
#from Nets_original import GetNeuralNetworkModel
from Training import GetDatasets
from Simulation import get_all_outputs
from Simulation import buffer_simulation, save_obj, load_obj
from datetime import datetime
import pandas as pd
import math


capa=[]

diff_nets_int_abs=[]
diff_nets_int=[]
def DifferenceOuts(outputs, outputs1,outputs2):
    diff_F_P = []
    diff_shift = []
    #write_layer = [2, 6, 10, 12, 16, 20, 22, 26, 30, 34, 36, 40, 44, 48, 50, 54, 58, 62, 64, 69, 73, 77]

    for index in range(0, len(outputs)):
        #if index == write_layer[ciclo]:
        #print('Capa', index, Net.layers[index].__class__.__name__)
        # a = outputs[index] == outputs1[index]
        # size_output = a.size
        # output_true = np.sum(a)
        #numero.append(index)
        capa.append(Net.layers[index].__class__.__name__)
        # print('capa', capa)
        # list_output_true.append(output_true)
        # list_size_output.append(size_output)
        # amount_dif = size_output - output_true
        # list_amount_dif.append(amount_dif)
        diff_nets_shif = np.sum(tf.math.abs(tf.math.subtract(outputs[index], outputs1[index])))
        diff_nets_FP = np.sum(tf.math.abs(tf.math.subtract(outputs[index], outputs2[index])))
        diff_F_P.append(diff_nets_FP)
        diff_shift.append(diff_nets_shif)
        sum_FP = np.sum(diff_F_P)
        sum_shift = np.sum(diff_shift)
    # si se hace pra una imagen se retorna esto
        #print('diff_nets', diff_nets)
    #     diff_F_P.append(diff_nets_FP)
    #     diff_shift.append(diff_nets_shif)
    # sum_FP = np.sum(diff_F_P)
    # sum_shift = np.sum(diff_shift)
    # diff_F_P.append(sum_FP)
    # diff_shift.append(sum_shift)
    # # print('sum_FP',sum_FP)
    # # print('sum_shift', sum_shift)
    # df_capa = pd.DataFrame(capa)
    # df_diff_F_P = pd.DataFrame(diff_F_P)
    # df_diff_shift = pd.DataFrame(diff_shift)
    # print('df_diff_F_P', df_diff_F_P)
    # print('df_diff_shift', df_diff_shift)
    # test_Mors = pd.concat([df_capa,df_diff_F_P,df_diff_shift], axis=1, join='outer')
    # test_Mors.columns = ['Capa','df_diff_F_P','I-df_diff_shift']
    # test_Mors.to_excel('MoRS/Modelo3_mas_fallos_col_8_experimentos/estatics/Squez_Test_shift.xlsx', sheet_name='fichero_707', index=False)
    # si se hace pra una imagen se retorna esto
    return sum_FP,sum_shift

total_locs=[]

mask = [16384]
mask=np.array(16384)
print(mask.dtype)
#### solo par alas capas qu eescriben por cada red
## acumular los locs y luego hacer la razon entre el total diferente de cero y los locs afectados
def DifferenceZero(outputs,locs,frac_size):
    #print('tamaño el outpust',len(outputs))
    write_layer = [2,6,8,12, 19,23, 30,34, 41,43,47, 54,58, 65,69, 76,80 , 87,89,93, 100,103,107]
    #write_layer = [2, 6]

    total_locs = []
    values_diff_zero = []


    for i, j in enumerate(outputs):
        capa = []
        locs_by_layer = []
        print('i',i)
        if i in write_layer:
            #print('i dentro del if', i)
            print('capa', Net.layers[i].__class__.__name__)
            print('tamaño de j', len(j))
            j= j.flatten(order='F')
            print('tamaño de j',len(j))
            locs=locs[(locs < j.size)]
            #locs = locs[(locs < len(j))]
            locs_size=len(locs)
            print('locs_afectado tamaño', locs_size)
            total_locs.append(locs_size)

            affectedValues = np.take(j, locs)
            capa.append(Net.layers[i].__class__.__name__)
            locs_by_layer.append(locs_size)
            Ogdtype = affectedValues.dtype
            #print(Ogdtype)
            shift = 2 ** (15)
            factor = 2 ** frac_size
            output = affectedValues * factor
            #print(output)
            output = output.astype(np.int32)
            #print(output.dtype)
            #print(output)
            output = np.where(np.less(output, 0), -output + shift, output)
            original = output
            print('original', original)
            original = np.bitwise_and(original, mask)
            print('original',original)
            valores_afectados = np.not_equal(original, 0)
            diff_zero_values = np.count_nonzero(valores_afectados)
            print ('diff_zero_values', diff_zero_values)
            #if diff_zero_values is None:

            if diff_zero_values> 0  :
                    print('diff_zero_values', diff_zero_values)

                    print('capa',Net.layers[i].__class__.__name__)
                    #print(type(diff_zero_values))
                     #print('diff_zero_values',diff_zero_values)
                    #print('diff_zero_values', diff_zero_values)
                    values_diff_zero.append(diff_zero_values)
                    #print('values_diff_zero',values_diff_zero)
                    #output = tf.where(tf.greater_equal(output, shift), shift - output, output)
                    #output = tf.cast(output / factor, dtype=Ogdtype)
    # print('values_diff_zero',values_diff_zero)
    # print( 'total_locs', total_locs)
    sum_values_diff_zero = np.sum(values_diff_zero)
    sum_total_locs = np.sum(total_locs)
    # print('sum_values_diff_zero',sum_values_diff_zero)
    # print('sum_total_locs', sum_total_locs)
    return sum_values_diff_zero,sum_total_locs







error_mask_x=load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/error_mask_all_x')
#print(error_mask_x)
locs  = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/locs_0')
print(len(error_mask_x))
print(len(locs))


#####SqueezeNet#########################################################################

# #
trainBatchSize = testBatchSize = 1
_, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)


iterator = iter(test_dataset)
# # #
# # # # # In[3]:
# # # # # Numero de bits para activaciones (a) y pesos (w)
word_size  = 16
afrac_size = 9
aint_size  = 6
wfrac_size = 15
wint_size  = 0
#
# # Directorio de los pesos
#
cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'SqueezeNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')


acc_list = []


# ## Creo la red sin fallos





#write_layer = [2,6,8,12, 19,23, 30,34, 41,43,47, 54,58, 65,69, 76,80 , 87,89,93, 100,103,107]
activation_aging = [True] * 22
Net = GetNeuralNetworkModel('SqueezeNet', (224, 224, 3), 8, aging_active=activation_aging,faulty_addresses=locs, masked_faults=error_mask_x,
                            word_size=word_size, frac_size=afrac_size, batch_size=testBatchSize)
Net.load_weights(wgt_dir).expect_partial()
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
WeightQuantization(model=Net, frac_bits=wfrac_size, int_bits=wint_size)
Net.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
loss, acc = Net.evaluate(test_dataset)
#




####ZFNet#########################################################################
# word_size  = 16
# afrac_size = 11
# aint_size  = 4
# wfrac_size = 14
# wint_size  = 1
#
# # Tamaño del buffer de activaciones == al tamaño de la capa con mayor numero de activaciones (290400 pesos de 16 bits cada uno)
# #
# # Directorio de los pesos
# cwd = os.getcwd()
# wgt_dir = os.path.join(cwd, 'Data')
# wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
# wgt_dir = os.path.join(wgt_dir, 'ZFNet')
# wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
# wgt_dir = os.path.join(wgt_dir, 'Weights')
#
#
#
# acc_list = []
# iterator = iter(test_dataset)
#
#
# #write_layer = [2, 6, 10, 14, 18, 22, 26, 30, 32, 36, 39, 43]
# activation_aging = [True] * 11
# Net = GetNeuralNetworkModel('ZFNet', (224, 224, 3), 8, aging_active=activation_aging,faulty_addresses=locs, masked_faults=error_mask_x,
#                              word_size=word_size, frac_size=afrac_size, batch_size=testBatchSize)
# Net.load_weights(wgt_dir).expect_partial()
# loss = tf.keras.losses.CategoricalCrossentropy()
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# WeightQuantization(model=Net, frac_bits=wfrac_size, int_bits=wint_size)
# Net.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
# loss, acc = Net.evaluate(test_dataset)
# print('acc ZFNet',acc )

####VGG16#########################################################################

# word_size  = 16
# afrac_size = 12
# aint_size  = 3
# wfrac_size = 15
# wint_size  = 0
#
# cwd = os.getcwd()
# wgt_dir = os.path.join(cwd, 'Data')
# wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
# wgt_dir = os.path.join(wgt_dir, 'VGG16')
# wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
# wgt_dir = os.path.join(wgt_dir,'Weights')
#
# activation_aging = [True] * 21
#
# #write_layer = [2, 6, 10, 12, 16, 20, 22, 26, 30, 34, 36, 40, 44, 48, 50, 54, 58, 62, 64, 69, 73, 77]
# Net = GetNeuralNetworkModel('VGG16', (224, 224, 3), 8, aging_active=activation_aging,faulty_addresses=locs, masked_faults=error_mask_x,
#                              word_size=word_size, frac_size=afrac_size, batch_size=testBatchSize)
# Net.load_weights(wgt_dir).expect_partial()
# loss = tf.keras.losses.CategoricalCrossentropy()
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# WeightQuantization(model=Net, frac_bits=wfrac_size, int_bits=wint_size)
# Net.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
# loss_sf, acc_sf = Net.evaluate(test_dataset)


# ####MobileNet#########################################################################

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
#
# # ## Creo la red sin fallos
# write_layer=[2,9,15,21,28,34,40,46,53,59,65,71,78,84,90,96,102,108,114,120,126,132,138,144,151,157,163,169,173,179]
# activation_aging = [True] * 29
# Net = GetNeuralNetworkModel('MobileNet', (224, 224, 3), 8, aging_active=activation_aging, faulty_addresses=locs, masked_faults=error_mask_x,
#                              word_size=word_size, frac_size=afrac_size, batch_size=testBatchSize)
# Net.load_weights(wgt_dir).expect_partial()
# loss = tf.keras.losses.CategoricalCrossentropy()
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# WeightQuantization(model=Net, frac_bits=wfrac_size, int_bits=wint_size)
# Net.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
# loss_sf, acc_sf = Net.evaluate(test_dataset)




# # ####DenseNet#########################################################################
#
# word_size  = 16
# afrac_size = 12
# aint_size  = 3
# wfrac_size = 13
# wint_size  = 2
#
#
#
# # Directorio de los pesos
# cwd = os.getcwd()
# wgt_dir = os.path.join(cwd, 'Data')
# wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
# wgt_dir = os.path.join(wgt_dir, 'DenseNet')
# wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
# wgt_dir = os.path.join(wgt_dir, 'Weights')
#
#
#
# acc_list = []
# iterator = iter(test_dataset)
#
# # ## Creo la red sin fallos
# #
# activation_aging = [True] * 188
# Net = GetNeuralNetworkModel('DenseNet', (224, 224, 3), 8, aging_active=activation_aging, faulty_addresses=locs, masked_faults=error_mask_x,
#                              word_size=word_size, frac_size=afrac_size, batch_size=testBatchSize)
# Net.load_weights(wgt_dir).expect_partial()
# loss = tf.keras.losses.CategoricalCrossentropy()
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# WeightQuantization(model=Net, frac_bits=wfrac_size, int_bits=wint_size)
# Net.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
# loss_sf, acc_sf = Net.evaluate(test_dataset)


###Alexnet#########################################################################

# word_size  = 16
# afrac_size = 11
# aint_size  = 4
# wfrac_size = 11
# wint_size  = 4
#
# trainBatchSize = testBatchSize = 1
# _, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (227, 227), 8, trainBatchSize, testBatchSize)
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
#
#
# acc_list = []
# iterator = iter(test_dataset)
#
# # ## Creo la red sin fallos
# write_layer=[2,8,10,16,18,24,30,36,38,44,49,53]
# activation_aging = [True] * 11
# Net = GetNeuralNetworkModel('AlexNet', (227,227,3), 8, aging_active=activation_aging,faulty_addresses=locs, masked_faults=error_mask_x,
#                              word_size=word_size, frac_size=afrac_size, batch_size=testBatchSize)
# Net.load_weights(wgt_dir).expect_partial()
# loss = tf.keras.losses.CategoricalCrossentropy()
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# WeightQuantization(model=Net, frac_bits=wfrac_size, int_bits=wint_size)
# Net.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
# loss_sf, acc_sf = Net.evaluate(test_dataset)


### Analizar valores distintos de 0 en el bit 14 de cada valor
index = 1
diff_zero_total=[]
total_locs=[]

while index < len(test_dataset) :
    locs_numpy=np.array(locs)
    image = next(iterator)[0]
    print('index+++++++++++++++++++++++++++', index)
    # print('imagen',index, image)
    outputs = get_all_outputs(Net, image)
    #print('outputs',outputs)
    diff_zero,len_locs= DifferenceZero(outputs, locs_numpy,afrac_size)
    #diff_zero = DifferenceZero(outputs, locs_numpy, afrac_size)
    #total_locs.append(len_locs)
    print('diff_zero',diff_zero)
    #if diff_zero is not None and diff_zero!= 0 :
        #print('diff_zero', diff_zero)
    diff_zero_total.append(diff_zero)
    total_locs.append(len_locs)
    index += 1
print(len(diff_zero_total))
print('diff_zero_total',diff_zero_total)
print('total_locs',total_locs)
n_diff_zero_total=np.array(diff_zero_total)
np_total_locs=np.array(total_locs)
sin_nan=n_diff_zero_total[n_diff_zero_total != np.array(None)]
print('cantidad vaoles',len(sin_nan))
total_non=len(n_diff_zero_total)-len(sin_nan)
print('total de None', total_non)
#sin_nan= n_diff_zero_total[np.isnan(n_diff_zero_total)] = 0
sum_FP = np.sum(sin_nan)
sum_locs_afected=np.sum(total_locs)
ratio=sum_FP/total_locs[0]
print('sum_FP',sum_FP)
print('sum_locs_afected',sum_locs_afected)
print('ratio',ratio)







