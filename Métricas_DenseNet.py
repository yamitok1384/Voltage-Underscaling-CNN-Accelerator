
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

vol=0.54
inc = 0
# error_mask =load_obj('Data/Fault Characterization/variante_mask_vc_707/High_order_37VBW/vc_707/error_mask_0' + str(vol))
# locs=load_obj('Data/Fault Characterization/variante_mask_vc_707/High_order_37VBW/vc_707/locs_0' + str(vol))
error_mask = load_obj('MoRS/Modelo3_col_8_' + str(vol) + '/mask/error_mask_' + str(inc))
locs = load_obj('MoRS/Modelo3_col_8_' + str(vol) + '/mask/locs_' + str(inc))
#error_mask_H_L_O,locs,VBW, = VBWGoToScratch(error_mask,locs)

error_mask_H_L_O,locs,VBW, = VeryBadWords(error_mask,locs)


# Con el siguiente bloque obtiene  el número de lecturas y escrituras por posición de memoria tanto usando los experimentos sin usarlos

# In[2]:





#error_mask = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/error_mask_0')
#locs = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/locs_0')
#error_mask=error_mask[0:10])

#error_mask=error_mask[9000:9010]
#locs=locs[0:10]
print(len(locs))
print(len(error_mask))
trainBatchSize = testBatchSize = 1
_, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)

word_size  = 16
afrac_size = 12
aint_size  = 3
wfrac_size = 13
wint_size  = 2

# Tamaño del buffer de activaciones == al tamaño de la capa con mayor numero de activaciones (290400 pesos de 16 bits cada uno)
#abuffer_size = 16777216
# Directorio de los pesos
cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'DenseNet')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir,'Weights')


# In[4]:


activation_aging = [False]*188

DenseNet = GetNeuralNetworkModel('DenseNet', (224,224,3), 8, faulty_addresses=locs, masked_faults=error_mask,
                                 aging_active=activation_aging, word_size=word_size, frac_size=afrac_size,
                                 batch_size = testBatchSize)
DenseNet.load_weights(wgt_dir).expect_partial()
WeightQuantization(model=DenseNet, frac_bits=wfrac_size, int_bits=wint_size)
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
DenseNet.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
#loss,acc =DenseNet.evaluate(test_dataset)



LI = [0,4,11,12,16,     22,25,29,     35,38,42,     48,51,55,     61,64,68,     74,77,81,     87,90,94,97,99,103,      109,
     112,116,      122,125,129,      135,138,142,      148,151,155,      161,164,168,      174,177,181,      187,
     190,194,      200,203,207,      213,216,220,      226,229,233,      239,242,246,      252,255,259,262,264,268,      274,
     277,281,      287,290,294,      300,303,307,      313,316,320,      326,329,333,      339,342,346,      352,
     355,359,      365,368,372,      378,381,385,      391,394,398,      404,407,411,      417,420,424,      430,
     433,437,      443,446,450,      456,459,463,      469,472,476,      482,485,489,      495,498,502,      508,
     511,515,      521,524,528,      534,537,541,      547,550,554,      560,563,567,      573,576,580,583,585,589,      595,
     598,602,      608,611,615,      621,624,628,      634,637,641,      647,650,654,      660,663,667,      673,
     676,680,      686,689,693,      699,702,706,      712,715,719,      725,728,732,      738,741,745,      751,
     754,758,      764,767,771,      777,780,784,      790,793,797,800]
AI = [2,9,11,15,21,(23,11),28,34,(36,24),41,47,(49,37),54,60,(62,50),67,73,(75,63),80,86,(88,76),93,96,98,102,108,(110,98),
     115,121,(123,111),128,134,(136,124),141,147,(149,137),154,160,(162,150),167,173,(175,163),180,186,(188,176),
     193,199,(201,189),206,212,(214,202),219,225,(227,215),232,238,(240,228),245,251,(253,241),258,261,263,267,273,(275,263),
     280,286,(288,276),293,299,(301,289),306,312,(314,302),319,325,(327,315),332,338,(340,328),345,351,(353,341),
     358,364,(366,353),371,377,(379,367),384,390,(392,380),397,403,(405,393),410,416,(418,406),423,429,(431,419),
     436,442,(444,432),449,455,(457,445),462,468,(470,458),475,481,(483,471),488,494,(496,484),501,507,(509,497),
     514,520,(522,510),527,533,(535,523),540,546,(548,536),553,559,(561,549),566,572,(574,562),579,582,584,588,594,(596,584),
     601,607,(609,597),614,620,(622,610),627,633,(635,623),640,646,(648,636),653,659,(661,649),666,672,(674,662),
     679,685,(687,675),692,698,(700,688),705,711,(713,701),718,724,(726,714),731,737,(739,727),744,750,(752,740),
     757,763,(765,753),770,776,(778,766),783,789,(791,779),796,799,803]

Layers(DenseNet,LI,AI,'DenseNet')
#
#
# # In[ ]:
#
#
#
num_address  =1048576
samples      = 1
Indices=[0,4,11,12,16,     22,25,29,     35,38,42,     48,51,55,     61,64,68,     74,77,81,     87,90,94,97,99,103,      109,
     112,116,      122,125,129,      135,138,142,      148,151,155,      161,164,168,      174,177,181,      187,
     190,194,      200,203,207,      213,216,220,      226,229,233,      239,242,246,      252,255,259,262,264,268,      274,
     277,281,      287,290,294,      300,303,307,      313,316,320,      326,329,333,      339,342,346,      352,
     355,359,      365,368,372,      378,381,385,      391,394,398,      404,407,411,      417,420,424,      430,
     433,437,      443,446,450,      456,459,463,      469,472,476,      482,485,489,      495,498,502,      508,
     511,515,      521,524,528,      534,537,541,      547,550,554,      560,563,567,      573,576,580,583,585,589,      595,
     598,602,      608,611,615,      621,624,628,      634,637,641,      647,650,654,      660,663,667,      673,
     676,680,      686,689,693,      699,702,706,      712,715,719,      725,728,732,      738,741,745,      751,
     754,758,      764,767,771,      777,780,784,      790,793,797,800]
len(Indices)
Data     = GetReadAndWrites(DenseNet,Indices,num_address,samples,CNN_gating=False,network_name='DenseNet')
stats    = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
Baseline_Acceses   = pd.DataFrame(stats).reset_index(drop=False)
df_writes_Read =  pd.concat([Baseline_Acceses], axis=1, join='outer')
df_writes_Read.columns = ['index','Lecturas','Escrituras']
# with pd.ExcelWriter('DenseNet/métricas/DenseNet_reads_and_write_num_' + str(vol) + '_adress_total.xlsx') as writer:
#          df_writes_Read.to_excel(writer, sheet_name='base', index=False)
# #ave_obj(Baseline_Acceses,'DenseNet/métricas/Baseline_4')



#
#VBW = [101102, 124520, 302065, 331634, 340492, 340493, 360087, 360088, 377722, 377723, 419841, 458633, 458634, 465007, 465008, 465034, 465197, 465389, 544769, 544770, 545758, 590341, 590402, 590403, 590404, 590405, 590406, 590407, 590408, 590409, 590424, 590610, 590706, 590707, 590790, 590804, 611329, 807694, 596228, 371814, 153245, 431863, 870432, 431606, 854134, 464132, 75599, 984684, 957428, 144540, 155294, 449309, 343438, 224550, 515399, 278216, 103082, 153193, 910945, 685437, 504077, 450176, 49594, 699634, 734341, 635221, 639518, 276509, 385631, 40833, 887620, 259787, 105166, 487082, 421367, 829519, 1002831, 827753, 750402, 342656, 641683, 132117, 846177, 670210, 507666, 183528, 108520, 335181, 645235, 306439, 986119, 910692, 870824, 260319, 435744, 768399, 94637, 525999, 1026694, 26510, 816438, 225078, 190841, 422703, 463124, 467197, 1030806, 379426, 871962, 746460, 360883, 971049, 559437, 989409, 145877, 845559, 1018805, 283649, 79627, 912268, 1042255, 676817, 309244, 682316, 493406, 151515, 58733, 403778, 402881, 793085, 416518, 4606, 305748, 143466, 16917, 28154, 504505, 91708, 1013618, 350501, 367555, 993020, 563837, 128, 77845, 697509, 448560, 25033]
# VBW = [101102, 124520, 302065, 331634, 340492, 340493, 360087, 360088, 377722, 377723, 419841, 458633, 458634, 465007, 465008, 465034, 465197, 465389, 544769, 544770, 545758, 590341, 590402, 590403, 590404, 590405, 590406, 590407, 590408, 590409, 590424, 590610, 590706, 590707, 590790, 590804, 611329]
# #
# #Esta son las VBW para el modelo_0 obtenido con MORS

writes_Read_VBW=df_writes_Read.iloc[VBW]
# with pd.ExcelWriter('DenseNet/métricas/DenseNet_reads_and_write_VBW' + str(vol) + '.xlsx') as writer:
#         writes_Read_VBW.to_excel(writer, sheet_name='base', index=False)
#


writes_Read_VBW=df_writes_Read.iloc[VBW]
print('writes_Read_VBW', writes_Read_VBW)
suma_por_columnas = writes_Read_VBW.sum()
print(suma_por_columnas)
#
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
# with pd.ExcelWriter('DenseNet/métricas/max_lecturas_x_cada_16' + str(vol) + '.xlsx') as writer:
#          df_read_layers.to_excel(writer, sheet_name='base', index=False)

# ciclos : analizar por qué start_fron esta en 137 correo con 0  aver que pasa

# In[ ]:

# samples= 1
# LI = [0,4,11,12,16,     22,25,29,     35,38,42,     48,51,55,     61,64,68,     74,77,81,     87,90,94,97,99,103,      109,
#      112,116,      122,125,129,      135,138,142,      148,151,155,      161,164,168,      174,177,181,      187,
#      190,194,      200,203,207,      213,216,220,      226,229,233,      239,242,246,      252,255,259,262,264,268,      274,
#      277,281,      287,290,294,      300,303,307,      313,316,320,      326,329,333,      339,342,346,      352,
#      355,359,      365,368,372,      378,381,385,      391,394,398,      404,407,411,      417,420,424,      430,
#      433,437,      443,446,450,      456,459,463,      469,472,476,      482,485,489,      495,498,502,      508,
#      511,515,      521,524,528,      534,537,541,      547,550,554,      560,563,567,      573,576,580,583,585,589,      595,
#      598,602,      608,611,615,      621,624,628,      634,637,641,      647,650,654,      660,663,667,      673,
#      676,680,      686,689,693,      699,702,706,      712,715,719,      725,728,732,      738,741,745,      751,
#      754,758,      764,767,771,      777,780,784,      790,793,797,800]
# AI = [2,9,11,15,21,(23,11),28,34,(36,24),41,47,(49,37),54,60,(62,50),67,73,(75,63),80,86,(88,76),93,96,98,102,108,(110,98),
#      115,121,(123,111),128,134,(136,124),141,147,(149,137),154,160,(162,150),167,173,(175,163),180,186,(188,176),
#      193,199,(201,189),206,212,(214,202),219,225,(227,215),232,238,(240,228),245,251,(253,241),258,261,263,267,273,(275,263),
#      280,286,(288,276),293,299,(301,289),306,312,(314,302),319,325,(327,315),332,338,(340,328),345,351,(353,341),
#      358,364,(366,353),371,377,(379,367),384,390,(392,380),397,403,(405,393),410,416,(418,406),423,429,(431,419),
#      436,442,(444,432),449,455,(457,445),462,468,(470,458),475,481,(483,471),488,494,(496,484),501,507,(509,497),
#      514,520,(522,510),527,533,(535,523),540,546,(548,536),553,559,(561,549),566,572,(574,562),579,582,584,588,594,(596,584),
#      601,607,(609,597),614,620,(622,610),627,633,(635,623),640,646,(648,636),653,659,(661,649),666,672,(674,662),
#      679,685,(687,675),692,698,(700,688),705,711,(713,701),718,724,(726,714),731,737,(739,727),744,750,(752,740),
#      757,763,(765,753),770,776,(778,766),783,789,(791,779),796,799,803]
# buffer_simulation(DenseNet,test_dataset, integer_bits = 3, fractional_bits = 12, samples= samples, start_from = 0,
#                  bit_invertion = False, bit_shifting = False, CNN_gating = False, write_mode ='default',save_results = True, network_type = 'DenseNet',
#                  results_dir = 'Data/Stats/DenseNet/mask_x3/', buffer_size = num_address,
#                  layer_indexes = LI , activation_indixes = AI)

print(str()+' operación completada: ', datetime.now().strftime("%H:%M:%S"))