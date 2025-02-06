import os
import pickle as pickle
import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
from Nets  import GetNeuralNetworkModel
from Stats_lect_index import WeightQuantization, ActivationStats, CheckAccuracyAndLoss, QuantizationEffect, GetReadAndWrites
from Training import GetDatasets, GetPilotNetDataset
from funciones import VeryBadWords,FlipPatchBetter,VBWGoToScratch,DeleteTercioRamdom, WordType,Flip
import pandas as pd
from keras.optimizers import Adam
from datetime import datetime
from Simulation_29_08 import buffer_simulation, save_obj, load_obj


vol=0.54
inc = 0

# Con el siguiente bloque obtiene  el número de lecturas y escrituras por posición de memoria tanto usando los experimentos sin usarlos

# In[2]:
error_mask = load_obj('MoRS/Modelo3_col_8_' + str(vol) + '/mask/error_mask_' + str(inc))
locs = load_obj('MoRS/Modelo3_col_8_' + str(vol) + '/mask/locs_' + str(inc))
print('error_mask', len(error_mask))
print('locs', len(locs))



error_mask_H_L_O,locs_H_L_O,index_locs_VBW = VeryBadWords(error_mask,locs)
# df_VBW = pd.DataFrame(error_mask_H_L_O)
# df_index = pd.DataFrame(index_locs_VBW)
df_locs = pd.DataFrame(locs_H_L_O)
# index_locs = pd.concat( [df_index, df_VBW, df_locs],            axis=1, join='outer')
# index_locs.columns = ['df_index', 'df_VBW','df_locs']
# index_locs.to_excel('MoRS/Analisis_Resultados/Energía_VBW/layers_analysis_read_writing_sizes/inex_losc'+ str(vol)+'_'+ str(inc) + '.xlsx', sheet_name='fichero_707', index=False)
df_locs.to_excel('MoRS/Analisis_Resultados/Energía_VBW/VGG19_VBW_Mors'+ str(vol)+'_'+ str(inc) + '.xlsx', sheet_name='fichero_707', index=False)


#Estas líneas de código hast a*****s eutilizan para 0.52 y 0.51 porque para estos voltajes hemos traspolado los fallos y hay demasiadas
#VBW
#error_mask_new, locs_new= DeleteTercioRamdom(error_mask,locs,index_locs_VBW)

#error_mask_H_L_O,locs_H_L_O,index_locs_VBW = VeryBadWords(error_mask_new, locs_new)

# df_VBW=pd.DataFrame(locs_H_L_O)
# with pd.ExcelWriter('MoRS/Analisis_Resultados/Energía_VBW/layers_analysis_read_writing_sizes/AlexNet_VBW_enviar' + str(vol) + '.xlsx') as writer:
#          df_VBW.to_excel(writer, sheet_name='base', index=False)
# print('tipos  epaabra despues ')
# WordType(error_mask, locs)
#
# print('tamaño d ela mascara depues', len(error_mask_new))
# print('tamaño d ela locs depues', len(locs_new))
# print('index_locs_VBW', len(index_locs_VBW))
#******************************+


word_size  = 16
afrac_size = 7
aint_size  = 8
wfrac_size = 15
wint_size  = 0

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


wgt_dir= ('../weights/VGG19/weights.data')



activation_aging = [True] * 28
#
# Acá la creamos, notese que como no se introduciran fallos en activaciones no es necesario pasar locs ni masks

VGG19 = GetNeuralNetworkModel('VGG19', (150,150,3), 8, aging_active=activation_aging,faulty_addresses=locs, masked_faults=error_mask,
                            word_size=word_size, frac_size=afrac_size, batch_size=batch_size)
VGG19.load_weights(wgt_dir).expect_partial()
WeightQuantization(model=VGG19, frac_bits=wfrac_size, int_bits=wint_size)
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
VGG19.compile(optimizer=Adam(),   loss='sparse_categorical_crossentropy',  metrics=['accuracy'])
loss, acc = VGG19.evaluate(test_ds)
#



num_address  =1048575
#num_address  = 2359808
#


# #Capas con la informacion de procesamiento
Indices      = [0,1,3,5,8,10,12,15,17,19,21,23,26,28,30,32,34,37,39,41,43,45,49,53,57]
 #= [1,3,5,8,10,12,15,17,19,21,23,26,28,30,32,34,37,39,41,43,45,49,53]
samples      = 1 #Numero de imagenes
# Sin Power Gating:
Data         = GetReadAndWrites(VGG19,Indices,num_address,samples,CNN_gating=False,network_name='VGG19')
stats        = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
Baseline_Acceses   = pd.DataFrame(stats).reset_index(drop=False)
df_writes_Read =  pd.concat([Baseline_Acceses], axis=1, join='outer')
df_writes_Read .columns = ['index','Lecturas','Escrituras']
print(df_writes_Read)
print(sum(df_writes_Read['Lecturas']))
print(sum(df_writes_Read['Escrituras']))
#save_obj(Baseline_Acceses,'VGG19/métricas/')
with pd.ExcelWriter('VGG19/métricas/VGG19_reads_and_write_num_adress_Mors' + str(vol) + '_StatsProvisional_VBW.xlsx') as writer:
          df_writes_Read.to_excel(writer, sheet_name='base', index=False)


error_mask_H_L_O,locs_H_L_O,index_locs_VBW = VeryBadWords(error_mask, locs)

df_VBW=pd.DataFrame(locs_H_L_O)
with pd.ExcelWriter('Data/Stats/VGG19/Energía_VBW/VGG19_locs_HLO' + str(vol) + '.xlsx') as writer:
         df_VBW.to_excel(writer, sheet_name='base', index=False)


#VBW = [101102, 124520, 302065, 331634, 340492, 340493, 360087, 360088, 377722, 377723, 419841, 458633, 458634, 465007, 465008, 465034, 465197, 465389, 544769, 544770, 545758, 590341, 590402, 590403, 590404, 590405, 590406, 590407, 590408, 590409, 590424, 590610, 590706, 590707, 590790, 590804, 611329]
#VBW = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/estatics/locs_HLO_0')


#Esta son las VBW para el modelo_0 obtenido con MORS

writes_Read_VBW=df_writes_Read.iloc[VBW]

with pd.ExcelWriter('VGG19/métricas/VGG19_reads_and_write_VBW_Mors_test_' + str(vol) + '_StatsProvisional_entre16.xlsx') as writer:
   writes_Read_VBW.to_excel(writer, sheet_name='base', index=False)

writes_Read_VBW=df_writes_Read.iloc[index_locs_VBW]



# print('writes_Read_VBW', writes_Read_VBW)
with pd.ExcelWriter('VGG19/métricas/VGG19_writes_Read_VBW_Mors.xlsx') as writer:
    writes_Read_VBW.to_excel(writer, sheet_name='base', index=False)

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
print('máximas letcturas',df_read_layers)
with pd.ExcelWriter('VGG19/métricas/max_lecturas_x_cada_16_direcciones_Mors.xlsx') as writer:
         df_read_layers.to_excel(writer, sheet_name='base', index=False)



samples = 1

LI = [0,1,3,5,8,10,12,15,17,19,21,23,26,28,30,32,34,37,39,41,43,45,49,53]

AI = [2,4,7,9,11,14,16,18,20,22,25,27,29,31,33,36,38,40,42,44,47,52,56,57]



Buffer,ciclos =  buffer_simulation(VGG19, test_ds, integer_bits = 8, fractional_bits = 7, samples = samples, start_from = 0,
                                  bit_invertion = False, bit_shifting = False, CNN_gating = False,
                                  buffer_size = 1048576, write_mode ='default', save_results = True, network_type = 'VGG19',
                                  results_dir = 'Data/Stats/VGG19/',
                                  layer_indexes = LI , activation_indixes = AI)

print('ciclos',ciclos)

print(str()+' operación ciclos completada: ', datetime.now().strftime("%H:%M:%S"))
