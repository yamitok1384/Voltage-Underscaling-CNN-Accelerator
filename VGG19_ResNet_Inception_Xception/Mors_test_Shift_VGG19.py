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
from funciones import compilNet, same_elements,buffer_vectores,Base,IsoAECC,ECC, Flip,FlipPatch,FlipPatchBetter,ShiftMask,WordType
from Stats import WeightQuantization, IntroduceFaultsInWeights, GenerateFaultsList
from Nets import GetNeuralNetworkModel
from Training import GetDatasets
from Simulation_29_08 import get_all_outputs
from Simulation_29_08 import buffer_simulation, save_obj, load_obj
from datetime import datetime
import pandas as pd
#from funciones import compilNet, same_elements

# In[2]:


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

word_size  = 16
afrac_size = 7
aint_size  = 8
wfrac_size = 15
wint_size  = 0

wgt_dir= ('../weights/VGG19/weights.data')




# In[4]:




# In[5]:

# a matriz que devuelve en true los elementos iguales entre el modelo con fallo y sin fallo
# size_output cantidad de elementos iguales entre el modelo con fallo y sin fallo
# amount_dif : cantidad de activaciones diferentes entre modelo con fallo y sin fallo
# capa=[]
# diff_F_P=[]
# diff_shift=[]
# diff_nets_int_abs=[]
# diff_nets_int=[]

### para 1 imagen del dataset
# def DifferenceOuts(outputs, outputs1,outputs2):
#     #write_layer = [2, 6, 10, 12, 16, 20, 22, 26, 30, 34, 36, 40, 44, 48, 50, 54, 58, 62, 64, 69, 73, 77]
#
#     for index in range(0, len(outputs)):
#         #if index == write_layer[ciclo]:
#         print('Capa', index, Net.layers[index].__class__.__name__)
#         # a = outputs[index] == outputs1[index]
#         # size_output = a.size
#         # output_true = np.sum(a)
#         #numero.append(index)
#         capa.append(Net.layers[index].__class__.__name__)
#         # print('capa', capa)
#         # list_output_true.append(output_true)
#         # list_size_output.append(size_output)
#         # amount_dif = size_output - output_true
#         # list_amount_dif.append(amount_dif)
#         diff_nets_shif = np.sum(tf.math.abs(tf.math.subtract(outputs[index], outputs1[index])))
#         diff_nets_FP = np.sum(tf.math.abs(tf.math.subtract(outputs[index], outputs2[index])))
#
#         #print('diff_nets', diff_nets)
#         diff_F_P.append(diff_nets_FP)
#         diff_shift.append(diff_nets_shif)
#     sum_FP = np.sum(diff_F_P)
#     sum_shift = np.sum(diff_shift)
#     diff_F_P.append(sum_FP)
#     diff_shift.append(sum_shift)
#     print('sum_FP',sum_FP)
#     print('sum_shift', sum_shift)
#     df_capa = pd.DataFrame(capa)
#     df_diff_F_P = pd.DataFrame(diff_F_P)
#     df_diff_shift = pd.DataFrame(diff_shift)
#     print('df_diff_F_P', df_diff_F_P)
#     print('df_diff_shift', df_diff_shift)
#     test_Mors = pd.concat([df_capa,df_diff_F_P,df_diff_shift], axis=1, join='outer')
#     test_Mors.columns = ['Capa','df_diff_F_P','I-df_diff_shift']
#     test_Mors.to_excel('MoRS/Modelo3_mas_fallos_col_8_experimentos/estatics/Incept_Test_shift.xlsx', sheet_name='fichero_707', index=False)
#
#     return sum_shift,sum_FP


capa=[]

diff_nets_int_abs=[]
diff_nets_int=[]
def DifferenceOuts(outputs, outputs1,outputs2):
    diff_F_P = []
    diff_shift = []

    write_layer = [2, 4, 7, 9, 11, 14, 16, 18, 20, 22, 25, 27, 29, 31, 33, 36, 38, 40, 42, 44, 47, 52, 56, 57]

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


iterator = iter(test_ds)
differnce_list=[]
diff_sum_Shift =[]
diff_sum_FP =[]

activation_aging = [False] * 28

Net = GetNeuralNetworkModel('VGG19', (150,150,3), 8, aging_active=activation_aging,
                             word_size=word_size, frac_size=afrac_size, batch_size=batch_size)
Net.load_weights(wgt_dir).expect_partial()
loss='sparse_categorical_crossentropy'
optimizer = Adam()
WeightQuantization(model=Net, frac_bits=wfrac_size, int_bits=wint_size)
Net.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
loss, acc = Net.evaluate(test_ds)
print('original', acc)


error_mask = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/error_mask_0')
locs = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/locs_0')

error_mask_new, locs,word_change = FlipPatchBetter(error_mask, locs)

activation_aging = [True] * 28



F_patch = GetNeuralNetworkModel('VGG19', (150,150,3), 8, faulty_addresses=locs, masked_faults=error_mask_new,
                             aging_active=activation_aging, word_size=word_size, frac_size=afrac_size,
                             batch_size=batch_size)
F_patch.load_weights(wgt_dir).expect_partial()
WeightQuantization(model=F_patch, frac_bits=wfrac_size, int_bits=wint_size)
loss='sparse_categorical_crossentropy'
optimizer = Adam()
F_patch.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
loss, acc = F_patch.evaluate(test_ds)
print('acc, f_p', acc)


del error_mask_new

error_mask = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/error_mask_0')
locs = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/locs_0')
error_mask_new, locs,locs_patch,word_change = FlipPatch(error_mask, locs)


activation_aging = [True] * 28



S_Patch = GetNeuralNetworkModel('VGG19', (150,150,3), 8, faulty_addresses=locs, masked_faults=error_mask_new,
                             aging_active=activation_aging, word_size=word_size, frac_size=afrac_size,
                             batch_size=batch_size)
S_Patch.load_weights(wgt_dir).expect_partial()
WeightQuantization(model=S_Patch, frac_bits=wfrac_size, int_bits=wint_size)
loss='sparse_categorical_crossentropy'
optimizer = Adam()
S_Patch.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
loss, acc = S_Patch.evaluate(test_ds)
print('acc, f_p', acc)


del error_mask_new



wgt_dir= ('../weights/VGG19/weights.data')

error_mask = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/error_mask_0')
locs = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/locs_0')
error_mask_new, locs,locs_patch,word_change = FlipPatch(error_mask, locs)
locs_LO, locs_HO, locs_H_L_O = WordType(error_mask_new, locs)

activation_aging = [True] * 28

from Nets_test_shift import GetNeuralNetworkModel

Net_Shift = GetNeuralNetworkModel('VGG19', (150,150,3), 8, faulty_addresses=locs, masked_faults=error_mask_new,
                             aging_active=activation_aging, word_size=word_size, frac_size=afrac_size,
                             batch_size=batch_size)
Net_Shift.load_weights(wgt_dir).expect_partial()
loss='sparse_categorical_crossentropy'
optimizer = Adam()
WeightQuantization(model=Net_Shift, frac_bits=wfrac_size, int_bits=wint_size)
Net_Shift.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
loss, acc = Net_Shift.evaluate(test_ds)
print('shift',acc)


#
#
#
#


write_layer =  [2,4,7,9,11,14,16,18,20,22,25,27,29,31,33,36,38,40,42,44,47,52,56,57]
#write_layer =  [2,4]

iffernce_list=[]
diff_sum_Shift =[]
diff_sum_FP =[]
diff_sum_SP = []
actvs_afect_Shif=[]
actvs_afect_FP=[]
actvs_afect_SP =[]
total_locs_lo = []
actvs_by_capa = []
# actvs_lo_capa = []
# actvs_lo_SP_capa=[]


for i, j in enumerate(write_layer):
    print('capas', j)
    index = 0
    stop = False
    diff_F_P = []
    diff_shift = []
    diff_S_P =[]
    list_actvs_afect_Shif = []
    list_actvs_afect_Flip = []
    list_actvs_afect_S_Patch = []
    iterator = iter(test_ds)

    #while index <= len(test_dataset):
    while index <= 250 and stop == False :
        image = next(iterator)[0]
        print('index+++++++++++++++++++++++++++', index)
        # print('imagen',index, image)
        outputs = get_all_outputs(Net, image)
        outputs1 = get_all_outputs(S_Patch, image)
        outputs2 = get_all_outputs(F_patch, image)
        outputs3 = get_all_outputs(Net_Shift, image)
        output = outputs[j]
        output1 = outputs1[j]
        output2 = outputs2[j]
        output3 = outputs3[j]
        output.flatten(order='F')
        #print('output', output)
        output1.flatten(order='F')
        #print('output', output1)
        output2.flatten(order='F')
        #print('output', output2)
        output3.flatten(order='F')
        locs_lo = np.array(locs_LO)
        print('locs_lo',locs_lo)
        # locs_patch = np.array(locs_patch)
        actvs = output.size
        print('tamaño de las actvs',actvs)
        locs_affected = locs_lo[(locs_lo < output.size)]
        locs_size = len(locs_affected)
        print('tamaño de locs_ffected', locs_size)
        # actvs_LO = locs_patch[(locs_patch < actvs)]
        # amount_actvs_lo = len(actvs_LO)

        # actvs_LO_SP = locs_patch_SP[(locs_patch_SP < actvs)]
        # amount_actvs_lo_SP = len(actvs_LO_SP)
        affectedValues = np.take(output, locs_affected)
        #affectedValues= tf.gather_nd(output, locs_affected)
        #print('affectedValues',affectedValues)
        affectedValues_SPatch = np.take(output1, locs_affected)
        affectedValues_Fpatch = np.take(output2, locs_affected)
        affectedValues_Net_Shift = np.take(output3, locs_affected)
        # Razón entre la diff de cada técnica entre el total de actvs que affectab la diffreneci
        # para ello calculo la difrenecia entre las variables implicada y cuento las actovaciones distinas de 0
        #diff_nets_shif = np.sum(tf.math.abs(tf.math.subtract(output, output1)))
##operaciones para todas las activaciones
#         diff_nets_S_patch = np.sum(np.abs(np.subtract(output, output1)))
#         diff_nets_F_P = np.sum(np.abs(np.subtract(output, output2)))
#         diff_nets_Shift = np.sum(np.abs(np.subtract(output, output3)))
##Opreaciones solo para las LO afectadas
        diff_nets_S_patch = np.sum(np.abs(np.subtract(affectedValues, affectedValues_SPatch)))
        diff_nets_F_P = np.sum(np.abs(np.subtract(affectedValues, affectedValues_Fpatch)))
        diff_nets_Shift = np.sum(np.abs(np.subtract(affectedValues, affectedValues_Net_Shift)))

        # diferencia (queria escribir un valor y se escribió otro)
        actvs_size_nets_S_patch = np.subtract(affectedValues, affectedValues_SPatch)
        actvs_afect_diff_S_patch = np.count_nonzero(actvs_size_nets_S_patch)
        actvs_size_nets_FP = np.subtract(affectedValues, affectedValues_Fpatch)
        actvs_afect_diff_FP = np.count_nonzero(actvs_size_nets_FP)
        actvs_size_nets_Shift = np.subtract(affectedValues, affectedValues_Net_Shift)
        actvs_afect_diff_Shif = np.count_nonzero(actvs_size_nets_Shift)
##operaciones para todas las activaciones
        # actvs_size_nets_S_patch = np.subtract (output, output1)
        # actvs_afect_diff_S_patch = np.count_nonzero(actvs_size_nets_S_patch)
        # actvs_size_nets_FP = np.subtract(output, output2)
        # actvs_afect_diff_FP = np.count_nonzero(actvs_size_nets_FP)
        # actvs_size_nets_Shift = np.subtract(output, output3)
        # actvs_afect_diff_Shif = np.count_nonzero(actvs_size_nets_Shift)

        diff_S_P.append(diff_nets_S_patch)
        #print('lita', diff_F_P)
        diff_F_P.append(diff_nets_F_P)
        #print('lita', diff_shift)
        diff_shift.append(diff_nets_Shift)
        list_actvs_afect_S_Patch.append(actvs_afect_diff_S_patch)
        list_actvs_afect_Flip.append(actvs_afect_diff_FP)
        #print('list_actvs_afect_Flip', list_actvs_afect_Flip)
        list_actvs_afect_Shif.append(actvs_afect_diff_Shif)
        # print('list_actvs_afect_Shif', list_actvs_afect_Shif)
        index = index + 1
        if index == 250:
            stop = True
            sum_SP = np.sum(diff_S_P)
            sum_FP = np.sum(diff_F_P)
            sum_shift = np.sum(diff_shift)
            sum_actvs_afect_S_patch = np.sum(list_actvs_afect_S_Patch)
            # print('sum_actvs_afect_Flip', sum_actvs_afect_Flip)
            sum_actvs_afect_Flip = np.sum(list_actvs_afect_Flip)
            # print('sum_actvs_afect_Flip', sum_actvs_afect_Flip)
            sum_actvs_afect_Shif = np.sum(list_actvs_afect_Shif)
            #print('sum_actvs_afect_Shif',sum_actvs_afect_Shif)
            diff_sum_SP.append(sum_SP)
            diff_sum_FP.append(sum_FP)
            diff_sum_Shift.append(sum_shift)
            actvs_afect_SP.append(sum_actvs_afect_S_patch)
            actvs_afect_FP.append(sum_actvs_afect_Flip)
            actvs_afect_Shif.append(sum_actvs_afect_Shif)
            capa.append(Net.layers[j].__class__.__name__)
            actvs_by_capa.append(actvs)
            total_locs_lo.append(locs_size)
            # actvs_lo_capa.append(amount_actvs_lo)
            # actvs_lo_SP_capa.append(amount_actvs_lo_SP)
        else:
            stop = False



## Sumo todo par acalcular el valor total de la suma de todas las diferencias  par aluego agregarlo a la lista al final
total_S_p = np.sum(diff_sum_SP)
total_F_p = np.sum(diff_sum_FP)
total_shift = np.sum(diff_sum_Shift)
## Sumo el toral de activaciones por fallos para la técnica correspondiente par aluego calcular el ratio
#sum_actvs_afect_Shif = np.sum(actvs_afect_Shif)
diff_sum_SP.append(total_S_p)
diff_sum_Shift.append(total_shift)
diff_sum_FP.append(total_F_p)
capa.append('Total')

actvs_by_capa.append(np.sum(actvs_by_capa))
total_locs_lo.append(np.sum(total_locs_lo))
#actvs_lo_SP_capa.append(np.sum(actvs_lo_capa))
actvs_afect_SP.append(np.sum(actvs_afect_SP))
actvs_afect_FP.append(np.sum(actvs_afect_FP))
actvs_afect_Shif.append(np.sum(actvs_afect_Shif))


df_diff_S_P = pd.DataFrame(diff_sum_SP)
df_diff_F_P = pd.DataFrame(diff_sum_FP)
df_diff_shift = pd.DataFrame(diff_sum_Shift)


df_actvs_SP = pd.DataFrame(actvs_afect_SP)
df_actvs_FP = pd.DataFrame(actvs_afect_FP)
df_actvs_Shit = pd.DataFrame(actvs_afect_Shif)

df_capa = pd.DataFrame(capa)
df_capa_actvs = pd.DataFrame(actvs_by_capa)
df_actvs_lo_capa = pd.DataFrame(total_locs_lo)
#df_actvs_lo_capa = pd.DataFrame(actvs_lo_capa)
test_Mors = pd.concat([df_capa,df_capa_actvs, df_actvs_lo_capa,df_diff_F_P,df_actvs_FP, df_diff_S_P,   df_actvs_SP, df_diff_shift,df_actvs_Shit], axis=1, join='outer')
test_Mors.columns = ['capa',    'Total_ctvs','df_actvs_lo','diff_F_P','actvs_FP','df_diff_S_P', 'Actvs_S_P', 'diff_shift','actvs_Shit']
print(test_Mors)
test_Mors.to_excel('MoRS/Modelo3_mas_fallos_col_8_experimentos/estatics/VGG19_ratio_by_layers_by_technique_test_solo_LO.xlsx', sheet_name='fichero_707', index=False)




#
# differnce_list=[]
# diff_sum_Shift =[]
# diff_sum_FP =[]
# actvs_afect_Shif=[]
# actvs_afect_FP=[]
# actvs_by_capa = []
# actvs_lo_capa = []
# for i, j in enumerate(write_layer):
#     print('capas', j)
#     index = 0
#     stop = False
#     diff_F_P = []
#     diff_shift = []
#     list_actvs_afect_Shif = []
#     list_actvs_afect_Flip = []
#     iterator = iter(test_ds)
#
#     #while index <= 2 and stop == False :
#     while index <= len(test_ds) and stop == False :
#         image = next(iterator)[0]
#         print('index+++++++++++++++++++++++++++', index)
#         # print('imagen',index, image)
#         outputs = get_all_outputs(Net, image)
#         outputs1 = get_all_outputs(Net_Shift, image)
#         outputs2 = get_all_outputs(Net_Filp_Patch, image)
#         output = outputs[j]
#         output1 = outputs1[j]
#         output2 = outputs2[j]
#         output.flatten(order='F')
#         #print('output', output)
#         output1.flatten(order='F')
#         #print('output', output1)
#         output2.flatten(order='F')
#         #print('output', output2)
#         locs_patch = np.array(locs_patch)
#         actvs = output.size
#         print('tamaño de las actvs',actvs)
#         actvs_LO = locs_patch[(locs_patch < actvs)]
#         amount_actvs_lo = len(actvs_LO)
#         #print('actuv lo afectadas en la capa',amount_actvs_lo )
#         # print('Capa', index, Net.layers[index].__class__.__name__)
#         # Razón entre la diff de cada técnica entre el total de actvs que affectab la diffreneci
#         # para ello calculo la difrenecia entre las variables implicada y cuento las actovaciones distinas de 0
#         #diff_nets_shif = np.sum(tf.math.abs(tf.math.subtract(output, output1)))
#         diff_nets_shif = np.sum(np.abs(np.subtract(output, output1)))
#         #print('diff_nets_shif',diff_nets_shif)
#         diff_nets_FP = np.sum(np.abs(np.subtract(output, output2)))
#         #print('diff_nets_FP', diff_nets_FP)
#         # diferencia (queria escribir un valor y se escribió otro)
#         actvs_size_nets_shif = np.subtract (output, output1)
#         actvs_afect_diff_Shif = np.count_nonzero(actvs_size_nets_shif)
#         #print('actvs_afect_shift', actvs_afect_diff_Shif)
#         actvs_size_nets_FP = np.subtract(output, output2)
#         actvs_afect_diff_FP = np.count_nonzero(actvs_size_nets_FP)
#         #print('actvs_afect_FP', actvs_afect_diff_FP)
#         diff_F_P.append(diff_nets_FP)
#         #print('lita', diff_F_P)
#         diff_shift.append(diff_nets_shif)
#         #print('lita', diff_shift)
#         list_actvs_afect_Shif.append(actvs_afect_diff_Shif)
#         #print('list_actvs_afect_Shif', list_actvs_afect_Shif)
#         list_actvs_afect_Flip.append(actvs_afect_diff_FP)
#         #print('list_actvs_afect_Flip', list_actvs_afect_Flip)
#         index = index + 1
#         #if index == 2:
#         if index == len(test_ds):
#             stop = True
#             sum_FP = np.sum(diff_F_P)
#             sum_shift = np.sum(diff_shift)
#             sum_actvs_afect_Shif = np.sum(list_actvs_afect_Shif)
#             #print('sum_actvs_afect_Shif',sum_actvs_afect_Shif)
#             sum_actvs_afect_Flip = np.sum(list_actvs_afect_Flip)
#             #print('sum_actvs_afect_Flip', sum_actvs_afect_Flip)
#             diff_sum_Shift.append(sum_shift)
#             diff_sum_FP.append(sum_FP)
#             actvs_afect_Shif.append(sum_actvs_afect_Shif)
#             actvs_afect_FP.append(sum_actvs_afect_Flip)
#             capa.append(Net.layers[j].__class__.__name__)
#             actvs_by_capa.append(actvs)
#             actvs_lo_capa.append(amount_actvs_lo)
#         else:
#             stop = False
#
#
# ## Sumo todo par acalcular el valor total de la suma de todas las diferencias  par aluego agregarlo a la lista al final
# total_shift = np.sum(diff_sum_Shift)
# ## Sumo el toral de activaciones por fallos par ala técnica correspondiente par aluego calcular el ratio
# sum_actvs_afect_Shif = np.sum(actvs_afect_Shif)
# diff_sum_Shift.append(total_shift)
# ## Sumo todo par acalcular el valor total de la suma de todas las diferencias  par aluego agregarlo a la lista al final
# total_F_p = np.sum(diff_sum_FP)
# ## Sumo el toral de activaciones por fallos par ala técnica correspondiente par aluego calcular el ratio
# diff_sum_FP.append(total_F_p)
# capa.append('Total')
# actvs_by_capa.append(np.sum(actvs_by_capa))
# actvs_lo_capa.append(np.sum(actvs_lo_capa))
# actvs_afect_FP.append(np.sum(actvs_afect_FP))
# actvs_afect_Shif.append(np.sum(actvs_afect_Shif))
#
#
# df_diff_shift = pd.DataFrame(diff_sum_Shift)
# df_diff_F_P = pd.DataFrame(diff_sum_FP)
# df_actvs_Shit = pd.DataFrame(actvs_afect_Shif)
# df_actvs_FP = pd.DataFrame(actvs_afect_FP)
# df_capa = pd.DataFrame(capa)
# df_capa_actvs = pd.DataFrame(actvs_by_capa)
# df_actvs_lo_capa = pd.DataFrame(actvs_lo_capa)
# test_Mors = pd.concat([df_capa,df_capa_actvs,df_actvs_lo_capa,df_diff_F_P,df_actvs_FP,df_diff_shift,df_actvs_Shit], axis=1, join='outer')
# test_Mors.columns = ['capa','Total_ctvs','Actvs_lo_capa','diff_F_P','actvs_FP','diff_shift','actvs_Shit']
# print(test_Mors)
# test_Mors.to_excel('MoRS/Modelo3_mas_fallos_col_8_experimentos/estatics/VGG19_ratio_by_layers_Test_shift_all_dataset.xlsx', sheet_name='fichero_707', index=False)
#

# index = 1
#
# while index <= len(test_ds):
#     image = next(iterator)[0]
#     #print('index+++++++++++++++++++++++++++', index)
#     # print('imagen',index, image)
#     outputs = get_all_outputs(Net, image)
#     outputs1 = get_all_outputs(Net_Shift, image)
#     outputs2 = get_all_outputs(Net_Filp_Patch, image)
#     # salidas del modelo sin fallas para la primer imagen del dataset de prueba
#     #outputs2 = get_all_outputs(Net_Shift, image)
#     # salidas del modelo con fallas para la primer imagen del dataset de prueba
#     net_FP,net_shift =DifferenceOuts(outputs, outputs1,outputs2)
#     diff_sum_Shift.append(net_shift)
#     diff_sum_FP.append(net_FP)
#     index = index + 1
# total_shift = np.sum(diff_sum_Shift)
# diff_sum_Shift.append(total_shift)
# total_F_p = np.sum(diff_sum_FP)
# diff_sum_FP.append(total_F_p)
# df_diff_shift = pd.DataFrame(diff_sum_Shift)
# df_diff_F_P = pd.DataFrame(diff_sum_FP)
# test_Mors = pd.concat([df_diff_F_P,df_diff_shift], axis=1, join='outer')
# test_Mors.columns = ['df_diff_F_P','df_diff_shift']
# test_Mors.to_excel('MoRS/Modelo3_mas_fallos_col_8_experimentos/estatics/VGG19_Test_shift_all_dataset.xlsx', sheet_name='fichero_707', index=False)


# X = [x for x, y in test_ds]
# print('aolo 1 vez')
# outputs = get_all_outputs(Net, X[0])
# outputs1 = get_all_outputs(Net_Shift, X[0])
# outputs2 = get_all_outputs(Net_Filp_Patch, X[0])
# # salidas del modelo sin fallas para la primer imagen del dataset de prueba
#  #outputs2 = get_all_outputs(Net_Shift, image)
# # salidas del modelo con fallas para la primer imagen del dataset de prueba
# sum_Shift, sum_FP=DifferenceOuts(outputs, outputs1,outputs2)




# #
# # Net2 = GetNeuralNetworkModel('VGG19', (150,150,3), 8, faulty_addresses=locs, masked_faults=error_mask,
# #                              aging_active=True, word_size=word_size, frac_size=afrac_size,
# #                              batch_size=batch_size)
# # Net2.load_weights(wgt_dir).expect_partial()
# # loss='sparse_categorical_crossentropy'
# # optimizer = Adam()
# # WeightQuantization(model=Net2, frac_bits=wfrac_size, int_bits=wint_size)
# # Net2.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
# # loss, acc = Net2.evaluate(test_ds)
# # acc_list.append(acc)
# # # list_ciclo.append(i)
# # index = 1
# # while index <= len(test_ds):
# #     image = next(iterator)[0]
# #     print('index+++++++++++++++++++++++++++', index)
# #     # print('imagen',index, image)
# #     outputs1 = get_all_outputs(Net1, image)
# #     # salidas del modelo sin fallas para la primer imagen del dataset de prueba
# #     outputs2 = get_all_outputs(Net2, image)
# #     # salidas del modelo con fallas para la primer imagen del dataset de prueba
# #     diff_nets_sum = SofmaxDiffElements(outputs1, outputs2, acc_list)
# #     diff_layer_softmax.append(diff_nets_sum)
# #     index = index + 1
# # df_diff_softmax_ECC = pd.DataFrame(diff_layer_softmax)
# # mean_diff_layer_softmaxnp.append(np.mean(diff_layer_softmax))
# #
# # del diff_layer_softmax
# #
# #
# #
# #
# #
# # locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/mask_volteada_x/mask_volteada/vc_707/locs_054')
# # error_mask = load_obj(    'Data/Fault Characterization/variante_mask_vc_707/mask_volteada_x/mask_volteada/vc_707/error_mask_054')
# # # acc_list = []
# # iterator = iter(test_ds)
# # diff_layer_softmax = []
# #
# # Net2 = GetNeuralNetworkModel('VGG19', (150,150,3), 8, faulty_addresses=locs, masked_faults=error_mask,
# #                              aging_active=True, word_size=word_size, frac_size=afrac_size,
# #                              batch_size=batch_size)
# # Net2.load_weights(wgt_dir).expect_partial()
# # loss='sparse_categorical_crossentropy'
# # optimizer = Adam()
# # WeightQuantization(model=Net2, frac_bits=wfrac_size, int_bits=wint_size)
# # Net2.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
# # loss, acc = Net2.evaluate(test_ds)
# # acc_list.append(acc)
# # # list_ciclo.append(i)
# # index = 1
# # while index <= len(test_ds):
# #     image = next(iterator)[0]
# #     print('index+++++++++++++++++++++++++++', index)
# #     # print('imagen',index, image)
# #     outputs1 = get_all_outputs(Net1, image)
# #     # salidas del modelo sin fallas para la primer imagen del dataset de prueba
# #     outputs2 = get_all_outputs(Net2, image)
# #     # salidas del modelo con fallas para la primer imagen del dataset de prueba
# #     diff_nets_sum = SofmaxDiffElements(outputs1, outputs2, acc_list)
# #     diff_layer_softmax.append(diff_nets_sum)
# #     index = index + 1
# # df_diff_softmax_Flip = pd.DataFrame(diff_layer_softmax)
# # mean_diff_layer_softmaxnp.append(np.mean(diff_layer_softmax))
# # print('media df_diff_softmax_Flip' ,mean_diff_layer_softmaxnp)
# # del diff_layer_softmax
#
#
#
# error_mask = load_obj('Data/Fault Characterization/variante_mask_vc_707/mask_VBW_en_x/mask_VBW_base/vc_707/error_mask_054')
# locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/mask_VBW_en_x/mask_VBW_base/vc_707/locs_054')
# # acc_list = []
# iterator = iter(test_ds)
# diff_layer_softmax = []
# # mean_diff_layer_softmaxnp=[]
#
#
# Net2 = GetNeuralNetworkModel('VGG19', (150,150,3), 8, faulty_addresses=locs, masked_faults=error_mask,
#                              aging_active=activation_aging, word_size=word_size, frac_size=afrac_size,
#                              batch_size=batch_size)
# Net2.load_weights(wgt_dir).expect_partial()
# loss='sparse_categorical_crossentropy'
# optimizer = Adam()
# WeightQuantization(model=Net2, frac_bits=wfrac_size, int_bits=wint_size)
# Net2.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
# loss, acc = Net2.evaluate(test_ds)
# acc_list.append(acc)
# # list_ciclo.append(i)
# index = 1
# while index <= len(test_ds):
#     image = next(iterator)[0]
#     print('index+++++++++++++++++++++++++++', index)
#     # print('imagen',index, image)
#     outputs1 = get_all_outputs(Net1, image)
#     # salidas del modelo sin fallas para la primer imagen del dataset de prueba
#     outputs2 = get_all_outputs(Net2, image)
#     # salidas del modelo con fallas para la primer imagen del dataset de prueba
#     diff_nets_sum = SofmaxDiffElements(outputs1, outputs2, acc_list)
#     diff_layer_softmax.append(diff_nets_sum)
#     index = index + 1
# df_diff_softmax_Flip_Patch = pd.DataFrame(diff_layer_softmax)
# mean_diff_layer_softmaxnp.append(np.mean(diff_layer_softmax))
# del diff_layer_softmax
#
#
#
#
# print('direrencias todas las redes',mean_diff_layer_softmaxnp)
# df_diff_layer_softmax = pd.DataFrame(mean_diff_layer_softmaxnp)
# df_acc_list = pd.DataFrame(acc_list)
#
# # buf_same_elemen = pd.concat([df_diff_softmax_base,df_diff_softmax_IA_ECC,df_diff_softmax_ECC,df_diff_softmax_Flip,df_diff_softmax_Flip_Patch], axis=1, join='outer')
# # buf_same_elemen.columns = ['Base','I-A ECC','ECC','Flip', 'F + P']
# # acc_media_comprobacion = pd.concat([df_diff_layer_softmax,df_acc_list], axis=1, join='outer')
# # acc_media_comprobacion.columns = ['media_diff','ACC']
# # print(buf_same_elemen)
# # with pd.ExcelWriter('VGG19/métricas/VGG19_diff_softmax.xlsx') as writer:
# #     buf_same_elemen.to_excel(writer, sheet_name='softmax_exp', index=False)
# #     acc_media_comprobacion.to_excel(writer, sheet_name='acc_media_comprobacion', index=False)
#
#
# buf_same_elemen = pd.concat([df_diff_softmax_Flip_Patch], axis=1, join='outer')
# buf_same_elemen.columns = [ 'F + P']
# acc_media_comprobacion = pd.concat([df_diff_layer_softmax,df_acc_list], axis=1, join='outer')
# acc_media_comprobacion.columns = ['media_diff','ACC']
# print(buf_same_elemen)
# with pd.ExcelWriter('VGG19/métricas/VGG19_diff_F_P_16xxx.xlsx') as writer:
#     buf_same_elemen.to_excel(writer, sheet_name='softmax_exp', index=False)
#     acc_media_comprobacion.to_excel(writer, sheet_name='acc_media_comprobacion', index=False)


#Calcular la diferencia entre la part eentera del modelo con fallos y sin fallos
# Cargar_errores = True
#
# if Cargar_errores:
#     locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask/vc_707/locs_054')
#     error_mask = load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask/vc_707/error_mask_054')
#
#
# activation_aging = np.array([False] * 11)
# acc_list = []
# list_ciclo = []
#
# with pd.ExcelWriter('AlexNet/AlexNet_ratio_part_int_base.xlsx') as writer:
#     for i, valor in enumerate(activation_aging):
#         ciclo = i
#         activation_aging[i] = True
#         activation_aging[i - 1] = False
#         print(activation_aging)
#         # activation_aging = [False,False,False,False,True,False,False,False,False,False,False]
#         # activation_aging= False
#         Net2 = GetNeuralNetworkModel('AlexNet', (227, 227, 3), 8, faulty_addresses=locs, masked_faults=error_mask,
#                                      aging_active=activation_aging, word_size=word_size, frac_size=afrac_size,
#                                      batch_size=batch_size)
#         Net2.load_weights(wgt_dir).expect_partial()
#         loss = tf.keras.losses.CategoricalCrossentropy()
#         optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
#         WeightQuantization(model=Net2, frac_bits=wfrac_size, int_bits=wint_size)
#         Net2.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
#         loss, acc = Net2.evaluate(test_ds)
#         acc_list.append(acc)
#         # list_ciclo.append(i)
#
#         X = [x for x, y in test_ds]
#         # salidas del modelo sin fallas para la primer imagen del dataset de prueba
#         outputs1 = get_all_outputs(Net1, X[0])
#         # salidas del modelo con fallas para la primer imagen del dataset de prueba
#         outputs2 = get_all_outputs(Net2, X[0])
#         # print(outputs1)
#         # print(outputs2)
#         # print('list_ciclo',list_ciclo)
#         # same_elements(outputs1,outputs2,list_ciclo,acc_list)
#         diferenc_intpart(outputs1, outputs2, ciclo, acc_list)
#     if ciclo == len(activation_aging) - 1:
#         last_layer = ciclo + 1
#         diferenc_intpart(outputs1, outputs2, last_layer, 0)
#
#     df_numero = pd.DataFrame(numero)
#     df_capa = pd.DataFrame(capa)
#     df_acc = pd.DataFrame(acc_list)
#     df_list_size_output = pd.DataFrame(list_size_output)
#     df_list_output_diff = pd.DataFrame(list_amount_dif)
#     df_diff_nets = pd.DataFrame(diff_layer)
#     df_diff_netsint_abso=pd.DataFrame(diff_nets_int_abs)
#     df_diff_netsint_signo=pd.DataFrame(diff_nets_int)
#
#     buf_same_elemen = pd.concat([df_numero, df_capa, df_list_size_output, df_acc, df_list_output_diff, df_diff_nets, df_diff_netsint_abso,
#          df_diff_netsint_signo], axis=1, join='outer')
#     buf_same_elemen.columns = ['Num', 'Capa', 'T_actv', 'Acc', '#_Act_diff', 'diff_nets', 'diff_netsint_abso', 'diff_netsint_signo']
#     buf_same_elemen.to_excel(writer, sheet_name='datos1', startcol=2, index=False)
#     writer.save()

print('Ejecución  completada: ', datetime.now().strftime("%H:%M:%S"))






