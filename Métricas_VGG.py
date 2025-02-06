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
from funciones import VeryBadWords,FlipPatchBetter,VBWGoToScratch



# Con el siguiente bloque obtiene  el número de lecturas y escrituras por posición de memoria tanto usando los experimentos sin usarlos

# In[2]:


vol=0.53
inc = 0
#
error_mask = load_obj('MoRS/Modelo3_col_8_' + str(vol) + '/mask/error_mask_' + str(inc))
locs = load_obj('MoRS/Modelo3_col_8_' + str(vol) + '/mask/locs_' + str(inc))
# #error_mask_H_L_O,locs,VBW, = VBWGoToScratch(error_mask,locs)
error_mask_H_L_O,locs,VBW, = VeryBadWords(error_mask,locs)
#print( 'VBW', VBW)


# Con el siguiente bloque obtiene  el número de lecturas y escrituras por posición de memoria tanto usando los experimentos sin usarlos

# In[2]:



# print(error_mask[0:10])
# print(locs[0:10])
# #error_mask=error_mask[9000:9010]
# #locs=locs[0:10]
# print(len(locs))
# print(len(error_mask))


trainBatchSize = testBatchSize = 1
_, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)






word_size  = 16
afrac_size = 12
aint_size  = 3
wfrac_size = 15
wint_size  = 0

# Tamaño del buffer de activaciones == al tamaño de la capa con mayor numero de activaciones (290400 pesos de 16 bits cada uno)
#abuffer_size = 16777216
# Directorio de los pesos
cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'VGG16')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')


# In[4]:


activation_aging = [False]*21

VGG16 = GetNeuralNetworkModel('VGG16', (224,224,3), 8, faulty_addresses=locs, masked_faults=error_mask,aging_active=activation_aging, word_size=word_size, frac_size=afrac_size,
                                 batch_size = testBatchSize)

VGG16.load_weights(wgt_dir).expect_partial()
WeightQuantization(model=VGG16, frac_bits=wfrac_size, int_bits=wint_size)
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
VGG16.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
#loss,acc  = VGG16.evaluate(test_dataset)

#
# # In[ ]:
#

indice = []
read_write_layers = []
layer_size =[]
write = []
read = []
write_size = []
read_size = []
LI = [0,3,7,11,13,17,21,23,27,31,35,37,41,45,49,51,55,59,63,66,70,74]
AI = [2,6,10,12,16,20,22,26,30,34,36,40,44,48,50,54,58,62,64,69,73,77]

for i, j in enumerate(VGG16.layers):
    print(VGG16.layers[i].__class__.__name__)
    print(np.prod(VGG16.layers[i].input_shape[1:]).astype(np.int64))


    if i in AI:
        indice.append(i)
        read_write_layers.append(VGG16.layers[i].__class__.__name__+ str('_') + str('write'))
        layer_size.append(np.prod(VGG16.layers[i].input_shape[1:]).astype(np.int64))
        write.append(VGG16.layers[i].__class__.__name__)
        write_size.append(np.prod(VGG16.layers[i].input_shape[1:]).astype(np.int64))
    if i in LI:
        indice.append(i)
        read_write_layers.append(VGG16.layers[i].__class__.__name__+ str('_') + str('read'))
        read.append(VGG16.layers[i].__class__.__name__)
        layer_size.append(np.prod(VGG16.layers[i].input_shape[1:]).astype(np.int64))
        read_size.append(np.prod(VGG16.layers[i].input_shape[1:]).astype(np.int64))


# df_indice = pd.DataFrame(indice)
# df_write_read_layers = pd.DataFrame(read_write_layers)
# print(df_write_read_layers)
# def_size_layers= pd.DataFrame(layer_size)
# print(def_size_layers)
# df_write_read = pd.concat([df_indice,df_write_read_layers, def_size_layers], axis=1, join='outer')
# df_write_read.columns = ['Indice','Capa', 'tamaño']
# print(df_write_read)
# df_write_read.to_excel('MoRS/Analisis_Resultados/Energía_VBW/layers_analysis_read_writing_sizes/Capas_read_write_enviar_VGG16.xlsx', sheet_name='fichero_707', index=False)


df_write_layers= pd.DataFrame(write)
df_write_indice = pd.DataFrame(AI)
df_read_size = pd.DataFrame(write_size)
df_write_layers = pd.concat([df_write_indice,df_write_layers, df_read_size], axis=1, join='outer')
df_write_layers.columns = ['Indice','Capa', 'tamaño']
df_write_layers.to_excel('MoRS/Analisis_Resultados/Energía_VBW/layers_analysis_read_writing_sizes/Capas_write_enviar_VGG16.xlsx', sheet_name='fichero_707', index=False)

df_read_layers = pd.DataFrame(read)
df_read_indice = pd.DataFrame(LI)
df_read_size = pd.DataFrame(read_size)
df_read_layers = pd.concat([df_read_indice,df_read_layers, df_read_size], axis=1, join='outer')
df_read_layers.columns = ['Indice','Capa', 'tamaño']
df_read_layers.to_excel('MoRS/Analisis_Resultados/Energía_VBW/layers_analysis_read_writing_sizes/Capas_read_enviar_VGG16.xlsx', sheet_name='fichero_707', index=False)


#
# num_address  =1048576
# #num_address  =3211264
# samples      = 1
#
# Indices = [0,3,7,11,13,17,21,23,27,31,35,37,41,45,49,51,55,59,63,66,70,74]
# Data    = GetReadAndWrites(VGG16,Indices,num_address,samples,CNN_gating=False, network_name='VGG16')
# stats   = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
# Baseline_Acceses   = pd.DataFrame(stats).reset_index(drop=False)
# df_writes_Read =  pd.concat([Baseline_Acceses], axis=1, join='outer')
# df_writes_Read.columns = ['index','Lecturas','Escrituras']
# save_obj(df_writes_Read,'Data/Acceses/VGG16/Experiment')
# with pd.ExcelWriter('VGG16/métricas/VGG16_reads_and_write_num_adress_Mors.xlsx') as writer:
#         df_writes_Read.to_excel(writer, sheet_name='base', index=False)

#VBW = [101102, 124520, 302065, 331634, 340492, 340493, 360087, 360088, 377722, 377723, 419841, 458633, 458634, 465007, 465008, 465034, 465197, 465389, 544769, 544770, 545758, 590341, 590402, 590403, 590404, 590405, 590406, 590407, 590408, 590409, 590424, 590610, 590706, 590707, 590790, 590804, 611329]
#Esta son las VBW para el modelo_0 obtenido con MORS

# print(len(VBW))
# writes_Read_VBW=df_writes_Read.iloc[VBW]
#
#
# writes_Read_VBW=df_writes_Read.iloc[VBW]
# print('writes_Read_VBW', writes_Read_VBW)
# suma_por_columnas = writes_Read_VBW.sum()
# print(suma_por_columnas)
# with pd.ExcelWriter('VGG16/métricas/VGG16_reads_and_write_VBW_new.xlsx') as writer:
#     writes_Read_VBW.to_excel(writer, sheet_name='base', index=False)
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
# print ('suma d etodos los read entre 16',(np.sum(Data['Reads'])/16))
# #print(list_values_max)
# df_read_layers = pd.DataFrame(list_values_max)
# df_read_layers .columns = ['Lecturas x 16']
# with pd.ExcelWriter('VGG16/métricas/max_lecturas_x_cada_16_dir_Mors.xlsx') as writer:
#          df_read_layers.to_excel(writer, sheet_name='base', index=False)
# #

# In[ ]:
#
# #
# samples = 1
# LI = [0,3,7,11,13,17,21,23,27,31,35,37,41,45,49,51,55,59,63,66,70,74]
# AI = [2,6,10,12,16,20,22,26,30,34,36,40,44,48,50,54,58,62,64,69,73,77]
# Buffer,ciclos = buffer_simulation(VGG16,test_dataset, integer_bits = 3, fractional_bits = 12, samples = samples, start_from = 0,
#                  bit_invertion = False, bit_shifting = False, CNN_gating = False, write_mode ='default', save_results = False,
#                  network_type = 'VGG16',
#                  results_dir = 'Data/Stats/VGG16/mask/', buffer_size = num_address,
#                  layer_indexes = LI , activation_indixes = AI)
#
# print(str()+' operación completada: ', datetime.now().strftime("%H:%M:%S"))