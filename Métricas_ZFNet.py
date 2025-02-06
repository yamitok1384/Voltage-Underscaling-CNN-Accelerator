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
from funciones import VeryBadWords,FlipPatchBetter,VBWGoToScratch,Layers



# Con el siguiente bloque obtiene  el número de lecturas y escrituras por posición de memoria tanto usando los experimentos sin usarlos

# In[2]:


vol=0.53
# error_mask =load_obj('Data/Fault Characterization/variante_mask_vc_707/High_order_37VBW/vc_707/error_mask_0' + str(vol))
# locs=load_obj('Data/Fault Characterization/variante_mask_vc_707/High_order_37VBW/vc_707/locs_0' + str(vol))

error_mask = load_obj('MoRS/Modelo3_col_8_' + str(vol) + '/mask/error_mask_' + str(inc))
locs = load_obj('MoRS/Modelo3_col_8_' + str(vol) + '/mask/locs_' + str(inc))
#error_mask_H_L_O,locs,VBW, = VBWGoToScratch(error_mask,locs)
error_mask_H_L_O,locs,VBW, = VeryBadWords(error_mask,locs)
#print( 'VBW', VBW)
print(len(VBW))#error_mask=error_mask[0:10]
print(error_mask[0:10])
print(locs[0:10])
#error_mask=error_mask[9000:9010]
#locs=locs[0:10]
print(len(locs))
print(len(error_mask))


trainBatchSize = testBatchSize = 1
_, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)



word_size  = 16
afrac_size = 11
aint_size  = 4
wfrac_size = 14
wint_size  = 1


cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'ZFNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')


# In[4]:


activation_aging = [False]*11

ZFNet = GetNeuralNetworkModel('ZFNet', (224,224,3), 8, faulty_addresses=locs, masked_faults=error_mask,
                                 aging_active=activation_aging, word_size=word_size, frac_size=afrac_size,
                                 batch_size = testBatchSize)
ZFNet.load_weights(wgt_dir).expect_partial()
WeightQuantization(model=ZFNet, frac_bits=wfrac_size, int_bits=wint_size)
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
ZFNet.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
loss,acc  = ZFNet.evaluate(test_dataset)

LI = [0,3,7 ,11,15,19,23,27,31,34,37,40]
AI = [2,6,10,14,18,22,26,30,32,36,39,43]

Layers(ZFNet,LI,AI,'ZFNet')

# num_address  =1048576
# samples      = 1
# Indices = [0,3,7,11,15,19,23,27,31,34,37,40]
# Data    = GetReadAndWrites(ZFNet,Indices,num_address,samples,CNN_gating=False,network_name='ZFNet')
# stats   = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
# Baseline_Acceses   = pd.DataFrame(stats).reset_index(drop=False)
# df_writes_Read =  pd.concat([Baseline_Acceses], axis=1, join='outer')
# df_writes_Read.columns = ['index','Lecturas','Escrituras']
# with pd.ExcelWriter('ZFNet/métricas/ZFNet_reads_and_write_num_adress_Mors_oput.xlsx') as writer:
#        df_writes_Read.to_excel(writer, sheet_name='base', index=False)
# #save_obj(Baseline_Acceses,'ZFNet/métricas/Baseline_4')
# print('df_writes_Read', df_writes_Read)
# #
# #
# #
# #
# #
# # #VBW = [101102, 124520, 302065, 331634, 340492, 340493, 360087, 360088, 377722, 377723, 419841, 458633, 458634, 465007, 465008, 465034, 465197, 465389, 544769, 544770, 545758, 590341, 590402, 590403, 590404, 590405, 590406, 590407, 590408, 590409, 590424, 590610, 590706, 590707, 590790, 590804, 611329]
# # #VBW = [101102, 124520, 302065, 331634, 340492, 340493, 360087, 360088, 377722, 377723, 419841, 458633, 458634, 465007, 465008, 465034, 465197, 465389, 544769, 544770, 545758, 590341, 590402, 590403, 590404, 590405, 590406, 590407, 590408, 590409, 590424, 590610, 590706, 590707, 590790, 590804, 611329, 807694, 596228, 371814, 153245, 431863, 870432, 431606, 854134, 464132, 75599, 984684, 957428, 144540, 155294, 449309, 343438, 224550, 515399, 278216, 103082, 153193, 910945, 685437, 504077, 450176, 49594, 699634, 734341, 635221, 639518, 276509, 385631, 40833, 887620, 259787, 105166, 487082]
# # #VBW =[101102, 124520, 302065, 331634, 340492, 340493, 360087, 360088, 377722, 377723, 419841, 458633, 458634, 465007, 465008, 465034, 465197, 465389, 544769, 544770, 545758, 590341, 590402, 590403, 590404, 590405, 590406, 590407, 590408, 590409, 590424, 590610, 590706, 590707, 590790, 590804, 611329, 807694, 596228, 371814, 153245, 431863, 870432, 431606, 854134, 464132, 75599, 984684, 957428, 144540, 155294, 449309, 343438, 224550, 515399, 278216, 103082, 153193, 910945, 685437, 504077, 450176, 49594, 699634, 734341, 635221, 639518, 276509, 385631, 40833, 887620, 259787, 105166, 487082, 421367, 829519, 1002831, 827753, 750402, 342656, 641683, 132117, 846177, 670210, 507666, 183528, 108520, 335181, 645235, 306439, 986119, 910692, 870824, 260319, 435744, 768399, 94637, 525999, 1026694, 26510, 816438, 225078, 190841, 422703, 463124, 467197, 1030806, 379426, 871962, 746460, 360883]
# # VBW = [101102, 124520, 302065, 331634, 340492, 340493, 360087, 360088, 377722, 377723, 419841, 458633, 458634, 465007, 465008, 465034, 465197, 465389, 544769, 544770, 545758, 590341, 590402, 590403, 590404, 590405, 590406, 590407, 590408, 590409, 590424, 590610, 590706, 590707, 590790, 590804, 611329, 807694, 596228, 371814, 153245, 431863, 870432, 431606, 854134, 464132, 75599, 984684, 957428, 144540, 155294, 449309, 343438, 224550, 515399, 278216, 103082, 153193, 910945, 685437, 504077, 450176, 49594, 699634, 734341, 635221, 639518, 276509, 385631, 40833, 887620, 259787, 105166, 487082, 421367, 829519, 1002831, 827753, 750402, 342656, 641683, 132117, 846177, 670210, 507666, 183528, 108520, 335181, 645235, 306439, 986119, 910692, 870824, 260319, 435744, 768399, 94637, 525999, 1026694, 26510, 816438, 225078, 190841, 422703, 463124, 467197, 1030806, 379426, 871962, 746460, 360883, 971049, 559437, 989409, 145877, 845559, 1018805, 283649, 79627, 912268, 1042255, 676817, 309244, 682316, 493406, 151515, 58733, 403778, 402881, 793085, 416518, 4606, 305748, 143466, 16917, 28154, 504505, 91708, 1013618, 350501, 367555, 993020, 563837, 128, 77845, 697509, 448560, 25033]
#
# #Esta son las VBW para el modelo_0 obtenido con MORS
#
# writes_Read_VBW=df_writes_Read.iloc[VBW]
# with pd.ExcelWriter('ZFNet/métricas/ZFNet_reads_and_write_VBW_Mors_oupt.xlsx') as writer:
#         writes_Read_VBW.to_excel(writer, sheet_name='base', index=False)
#
#
#
# writes_Read_VBW=df_writes_Read.iloc[VBW]
# print('writes_Read_VBW', writes_Read_VBW)
# suma_por_columnas = writes_Read_VBW.sum()
# print(suma_por_columnas)
#
# reads_list=np.asarray(Data['Reads'])
# k=0
# m=16
# list_values_max =[]
# for i in range(len(reads_list) // 16):
#     # print(i)
#     values_max = np.amax(reads_list[k:m])
#     if values_max!= 0:
#         list_values_max.append(values_max)
#
#
#     k = m
#     m = k + 16
# sum_values_max = np.sum(list_values_max)
# print('sum_values_max', sum_values_max)
# list_values_max.append(sum_values_max)
# print(len(list_values_max))
# #print(list_values_max)
# df_read_layers = pd.DataFrame(list_values_max)
# df_read_layers .columns = ['Lecturas x 16']
# with pd.ExcelWriter('ZFNet/métricas/ZFNet_max_lect_x16_Mors_ouput.xlsx') as writer:
#          df_read_layers.to_excel(writer, sheet_name='base', index=False)


#
# # ciclos
#
# # In[ ]:
#
#
# samples = 1
# LI = [0,3,7 ,11,15,19,23,27,31,34,37,40]
# AI = [2,6,10,14,18,22,26,30,32,36,39,43]
# Buffer,ciclos = =buffer_simulation(ZFNet,test_dataset, integer_bits = 4, fractional_bits = 11, samples = samples, start_from = 0,
#                  bit_invertion = False, bit_shifting = False, CNN_gating = False,  buffer_size = 1048576,write_mode ='default', network_type = 'ZFNet',
#                  results_dir = 'Data/Stats/ZFNet/mask_x3/',
#                  layer_indexes = LI , activation_indixes = AI)


print(str()+' operación completada: ', datetime.now().strftime("%H:%M:%S"))