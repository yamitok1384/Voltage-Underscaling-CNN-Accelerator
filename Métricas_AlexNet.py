#!/usr/bin/env python
# coding: utf-8

# # Accesos al buffer: Lecturas y Escrituras

# In[1]:



import os
import pickle as pickle
import tensorflow as tf
import numpy as np
from Nets  import GetNeuralNetworkModel
from StatsProvisional import WeightQuantization, ActivationStats, CheckAccuracyAndLoss, QuantizationEffect, GetReadAndWrites
from Training import GetDatasets, GetPilotNetDataset
import pandas as pd
from datetime import datetime
from Simulation import buffer_simulation, save_obj, load_obj
from funciones import VeryBadWords,FlipPatchBetter,VBWGoToScratch,DeleteTercioRamdom, WordType,Flip




# Con el siguiente bloque obtiene  el número de lecturas y escrituras por posición de memoria tanto usando los experimentos sin usarlos

# In[2]:

vol=0.54
inc = 0
#
error_mask = load_obj('MoRS/Modelo3_col_8_' + str(vol) + '/mask/error_mask_' + str(inc))
locs = load_obj('MoRS/Modelo3_col_8_' + str(vol) + '/mask/locs_' + str(inc))
# df_error_mask = pd.DataFrame(error_mask)
# df_locs = pd.DataFrame(locs)
# mask_locs = pd.concat( [df_error_mask, df_locs],            axis=1, join='outer')
# mask_locs.columns = ['df_error_mask', 'df_locs']
# mask_locs.to_excel('MoRS/Analisis_Resultados/Energía_VBW/layers_analysis_read_writing_sizes/mask_locs_'+ str(vol)+'_'+ str(inc) + '.xlsx', sheet_name='fichero_707', index=False)





error_mask_H_L_O,locs_H_L_O,index_locs_VBW = VeryBadWords(error_mask,locs)
# df_VBW = pd.DataFrame(error_mask_H_L_O)
# df_index = pd.DataFrame(index_locs_VBW)
df_locs = pd.DataFrame(locs_H_L_O)
# index_locs = pd.concat( [df_index, df_VBW, df_locs],            axis=1, join='outer')
# index_locs.columns = ['df_index', 'df_VBW','df_locs']
# index_locs.to_excel('MoRS/Analisis_Resultados/Energía_VBW/layers_analysis_read_writing_sizes/inex_losc'+ str(vol)+'_'+ str(inc) + '.xlsx', sheet_name='fichero_707', index=False)
df_locs.to_excel('MoRS/Analisis_Resultados/Energía_VBW/layers_analysis_read_writing_sizes/losc_originales_Mors_test'+ str(vol)+'_'+ str(inc) + '.xlsx', sheet_name='fichero_707', index=False)


#Estas líneas s eutilizan para 0.52 y 0.51 porque para estos voltajes hemos traspolado los fallos y hay demasiadas
#VBW
#error_mask_new, locs_new= DeleteTercioRamdom(error_mask,locs,index_locs_VBW)

#error_mask_H_L_O,locs_H_L_O,index_locs_VBW = VeryBadWords(error_mask_new, locs_new)

df_VBW=pd.DataFrame(locs_H_L_O)
with pd.ExcelWriter('MoRS/Analisis_Resultados/Energía_VBW/layers_analysis_read_writing_sizes/AlexNet_VBW_enviar' + str(vol) + '.xlsx') as writer:
         df_VBW.to_excel(writer, sheet_name='base', index=False)
print('tipos  de paabra despues ')
WordType(error_mask, locs)

# print('tamaño d ela mascara depues', len(error_mask_new))
# print('tamaño d ela locs depues', len(locs_new))
# print('index_locs_VBW', len(index_locs_VBW))



#error_mask = load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/error_mask_054')
#locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/locs_054')
#error_mask=error_mask[0:10]
print(error_mask[0:10])
print(locs[0:10])
#error_mask=error_mask[9000:9010]
#locs=locs[0:10]
print(len(locs))
print(len(error_mask))


# # AlexNet

# In[3]:


#
word_size  = 16
afrac_size = 11
aint_size  = 4
wfrac_size = 11
wint_size  = 4
trainBatchSize = testBatchSize = 1
_, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (227, 227), 8, trainBatchSize, testBatchSize)


cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'AlexNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')


# # In[5]:
#
#
activation_aging = [True]*11
AlexNet = GetNeuralNetworkModel('AlexNet', (227,227,3), 8, faulty_addresses=locs, masked_faults=error_mask,
                                aging_active=activation_aging, word_size=word_size, frac_size=afrac_size,
                               batch_size = testBatchSize)
AlexNet.load_weights(wgt_dir).expect_partial()
WeightQuantization(model=AlexNet, frac_bits=wfrac_size, int_bits=wint_size)
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
AlexNet.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
loss,acc =AlexNet.evaluate(test_dataset)

#
# # In[7]:
#


num_address  =1048576

Indices      = [0,3,9,11,17,19,25,31,37,40,45,50] #Capas con la informacion de procesamiento
#locs_VBW = [101102, 124520, 302065, 331634, 340492, 340493, 360087, 360088, 377722, 377723, 419841, 458633, 458634, 465007, 465008, 465034, 465197, 465389, 544769, 544770, 545758, 590341, 590402, 590403, 590404, 590405, 590406, 590407, 590408, 590409, 590424, 590610, 590706, 590707, 590790, 590804, 611329, 807694, 596228, 371814, 153245, 431863, 870432, 431606, 854134, 464132, 75599, 984684, 957428, 144540, 155294, 449309, 343438, 224550, 515399, 278216, 103082, 153193, 910945, 685437, 504077, 450176, 49594, 699634, 734341, 635221, 639518, 276509, 385631, 40833, 887620, 259787, 105166, 487082, 421367, 829519, 1002831, 827753, 750402, 342656, 641683, 132117, 846177, 670210, 507666, 183528, 108520, 335181, 645235, 306439, 986119, 910692, 870824, 260319, 435744, 768399, 94637, 525999, 1026694, 26510, 816438, 225078, 190841, 422703, 463124, 467197, 1030806, 379426, 871962, 746460, 360883, 971049, 559437, 989409, 145877, 845559, 1018805, 283649, 79627, 912268, 1042255, 676817, 309244, 682316, 493406, 151515, 58733, 403778, 402881, 793085, 416518, 4606, 305748, 143466, 16917, 28154, 504505, 91708, 1013618, 350501, 367555, 993020, 563837, 128, 77845, 697509, 448560, 25033]

samples      = 1 #Numero de imagenes
# Sin Power Gating:
Data         = GetReadAndWrites(AlexNet,Indices,num_address,samples,CNN_gating=False,network_name='AlexNet')
stats        = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
Baseline_Acceses   = pd.DataFrame(stats).reset_index(drop=False)
df_writes_Read =  pd.concat([Baseline_Acceses], axis=1, join='outer')
df_writes_Read .columns = ['index','Lecturas','Escrituras']
#print(df_writes_Read)
with pd.ExcelWriter('AlexNet/métricas/AlexNet_reads_and_write_num_adress_Mors_test_29_08' + str(vol) + '_StatsProvisional_less_VBW.xlsx') as writer:
         df_writes_Read.to_excel(writer, sheet_name='base', index=False)
#
# save_obj(Baseline_Acceses,'Data/Acceses/AlexNet/Baseline_4')

#
#
# #del Dataframe anterior obtengo uno nuevo para los índices especificados
# VBW: Indice de las palabras que son consideradas very bad word porque aunque s ele aplique las tecnicas siguen los errores porque su estructura es xx11xx00....xx11
#VBW = [101102, 124520, 302065, 331634, 340492, 340493, 360087, 360088, 377722, 377723, 419841, 458633, 458634, 465007, 465008, 465034, 465197, 465389, 544769, 544770, 545758, 590341, 590402, 590403, 590404, 590405, 590406, 590407, 590408, 590409, 590424, 590610, 590706, 590707, 590790, 590804, 611329]
# # #VBW = [101102, 124520, 302065, 331634, 340492, 340493, 360087, 360088, 377722, 377723, 419841, 458633, 458634, 465007, 465008, 465034, 465197, 465389, 544769, 544770, 545758, 590341, 590402, 590403, 590404, 590405, 590406, 590407, 590408, 590409, 590424, 590610, 590706, 590707, 590790, 590804, 611329, 807694, 596228, 371814, 153245, 431863, 870432, 431606, 854134, 464132, 75599, 984684, 957428, 144540, 155294, 449309, 343438, 224550, 515399, 278216, 103082, 153193, 910945, 685437, 504077, 450176, 49594, 699634, 734341, 635221, 639518, 276509, 385631, 40833, 887620, 259787, 105166, 487082]
# # #VBW =[101102, 124520, 302065, 331634, 340492, 340493, 360087, 360088, 377722, 377723, 419841, 458633, 458634, 465007, 465008, 465034, 465197, 465389, 544769, 544770, 545758, 590341, 590402, 590403, 590404, 590405, 590406, 590407, 590408, 590409, 590424, 590610, 590706, 590707, 590790, 590804, 611329, 807694, 596228, 371814, 153245, 431863, 870432, 431606, 854134, 464132, 75599, 984684, 957428, 144540, 155294, 449309, 343438, 224550, 515399, 278216, 103082, 153193, 910945, 685437, 504077, 450176, 49594, 699634, 734341, 635221, 639518, 276509, 385631, 40833, 887620, 259787, 105166, 487082, 421367, 829519, 1002831, 827753, 750402, 342656, 641683, 132117, 846177, 670210, 507666, 183528, 108520, 335181, 645235, 306439, 986119, 910692, 870824, 260319, 435744, 768399, 94637, 525999, 1026694, 26510, 816438, 225078, 190841, 422703, 463124, 467197, 1030806, 379426, 871962, 746460, 360883]
# # VBW = [101102, 124520, 302065, 331634, 340492, 340493, 360087, 360088, 377722, 377723, 419841, 458633, 458634, 465007, 465008, 465034, 465197, 465389, 544769, 544770, 545758, 590341, 590402, 590403, 590404, 590405, 590406, 590407, 590408, 590409, 590424, 590610, 590706, 590707, 590790, 590804, 611329, 807694, 596228, 371814, 153245, 431863, 870432, 431606, 854134, 464132, 75599, 984684, 957428, 144540, 155294, 449309, 343438, 224550, 515399, 278216, 103082, 153193, 910945, 685437, 504077, 450176, 49594, 699634, 734341, 635221, 639518, 276509, 385631, 40833, 887620, 259787, 105166, 487082, 421367, 829519, 1002831, 827753, 750402, 342656, 641683, 132117, 846177, 670210, 507666, 183528, 108520, 335181, 645235, 306439, 986119, 910692, 870824, 260319, 435744, 768399, 94637, 525999, 1026694, 26510, 816438, 225078, 190841, 422703, 463124, 467197, 1030806, 379426, 871962, 746460, 360883, 971049, 559437, 989409, 145877, 845559, 1018805, 283649, 79627, 912268, 1042255, 676817, 309244, 682316, 493406, 151515, 58733, 403778, 402881, 793085, 416518, 4606, 305748, 143466, 16917, 28154, 504505, 91708, 1013618, 350501, 367555, 993020, 563837, 128, 77845, 697509, 448560, 25033]

#Esta son las VBW para el modelo_0 obtenido con MORS

writes_Read_VBW=df_writes_Read.iloc[index_locs_VBW]
#
with pd.ExcelWriter('AlexNet/métricas/AlexNet_index_locs_VBW_' + str(vol) + '_29_08.xlsx') as writer:
   writes_Read_VBW.to_excel(writer, sheet_name='base', index=False)
#
#
#
writes_Read_VBW=df_writes_Read.iloc[index_locs_VBW]
print('writes_Read_VBW', writes_Read_VBW)
suma_por_columnas = writes_Read_VBW.sum()
print(suma_por_columnas)
#
#
reads_list=np.asarray(Data['Reads'])
k=0
m=16
list_values_max =[]
for i in range(len(reads_list) // 16):
    # print(i)
    values_max = np.amax(reads_list[k:m])
    if values_max!= 0:
        list_values_max.append(values_max)


    k = m
    m = k + 16
sum_values_max = np.sum(list_values_max)
print('sum_values_max', sum_values_max)
list_values_max.append(sum_values_max)
print(len(list_values_max))
#print(list_values_max)
df_read_layers = pd.DataFrame(list_values_max)
df_read_layers .columns = ['Lecturas x 16']
with pd.ExcelWriter('AlexNet/métricas/new_for_max_read_x_cada_16_direcciones_PE_16_Mors_29_08' + str(vol) + '_StatsProvisional_entre16.xlsx') as writer:
         df_read_layers.to_excel(writer, sheet_name='base', index=False)
# #
#




samples = 1
LI = [0,3,9,11,17,19,25,31,37,40,45,50]
AI = [2,8,10,16,18,24,30,36,38,44,49,53]
Buffer,ciclos =  buffer_simulation(AlexNet, test_dataset, integer_bits = 4, fractional_bits = 11, samples = samples, start_from = 0,
                                  bit_invertion = False, bit_shifting = False, CNN_gating = False,
                                  buffer_size = 1048576, write_mode ='default', save_results = True, network_type = 'AlexNet',
                                  results_dir = 'Data/Stats/AlexNet/test_29_08',
                                  layer_indexes = LI , activation_indixes = AI)
print('ciclos',ciclos)
print('Buffer',Buffer)
print(str()+' operación ciclos completada: ', datetime.now().strftime("%H:%M:%S"))


# voltaj = [59, 58, 57, 56, 55, 54]
# # print(voltaj)
# Voltajes = pd.DataFrame(voltaj)
# activation_aging = np.array([False] * 11)
# activation_aging[0] = True
# vol = 59
# print(len(activation_aging))
# acc_list = []
# vol_list = []
# list_ciclo = []
# capa = []
# iterator = iter(test_dataset)
# conf = False
# j = 0
#
# # with pd.ExcelWriter('AlexNet/acc_vol.xlsx') as writer:
# for j, valor in enumerate(activation_aging):
#     print('capa', j)
#
#     for i, v in enumerate(voltaj):
#         # if vol==53:
#         #    break
#         # ciclo=i
#         print('i**********', i)
#         print('v**********', v)
#
#         error_mask = load_obj(
#             'Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/error_mask_0' + str(v))
#         locs = load_obj(
#             'Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/locs_0' + str(v))
#         # print('vol',vol)
#         # vol = vol - 1
#         print(activation_aging)
#
#         Net2 = GetNeuralNetworkModel('AlexNet', (227, 227, 3), 8, faulty_addresses=locs, masked_faults=error_mask,
#                                      aging_active=activation_aging, word_size=word_size, frac_size=afrac_size,
#                                      batch_size=testBatchSize)
#         Net2.load_weights(wgt_dir).expect_partial()
#         loss = tf.keras.losses.CategoricalCrossentropy()
#         optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
#         Net2.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
#         WeightQuantization(model=Net2, frac_bits=11, int_bits=4)
#         loss, acc = Net2.evaluate(test_dataset)
#
#         if acc < 0.890666663646697:
#             print('acc menor o igual', acc)
#             acc_list.append(acc)
#             vol_list.append(voltaj[i])
#             capa.append(j)
#             activation_aging = np.array([False] * 11)
#             j = j + 1
#             activation_aging[j] = True
#             i = 0
#             conf = True
#             print(capa)
#             print(vol_list)
#             print(acc_list)
#             break
#         else:
#             if conf == True:
#                 i = 0
#                 activation_aging = np.array([False] * 11)
#                 activation_aging[j] = True
#             else:
#                 print('dentro del else')
#                 j = 0
#                 activation_aging[j] = True
#             if v == 54:
#                 print('v dnetro del else', v)
#                 acc_list.append(acc)
#                 vol_list.append(v)
#                 capa.append(j)
#                 print(capa)
#                 print(vol_list)
#                 print(acc_list)

# df_capa = pd.DataFrame(capa)
# df_acc = pd.DataFrame(acc_list)
# df_vol = pd.DataFrame(vol_list)
# buf_acc_capa = pd.concat([df_capa, df_acc, df_vol], axis=1, join='outer')
# buf_acc_capa.columns = ['capa', 'Accu', 'vol']
# print(buf_acc_capa)
# buf_acc_capa.to_excel('buf_acc_capa.xlsx', sheet_name='AlexNet', index=False)