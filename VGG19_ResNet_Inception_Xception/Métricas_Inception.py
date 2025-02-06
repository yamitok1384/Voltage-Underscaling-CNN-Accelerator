import os
import pickle as pickle
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from Nets  import GetNeuralNetworkModel
from Stats_lect_index import WeightQuantization, ActivationStats, CheckAccuracyAndLoss, QuantizationEffect, GetReadAndWrites
from Training import GetDatasets, GetPilotNetDataset
import pandas as pd
from keras.optimizers import Adam
from datetime import datetime
from Simulation import buffer_simulation, save_obj, load_obj


error_mask = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/error_mask_0')
locs = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/locs_0')
#error_mask=error_mask[0:10]
print(error_mask[0:10])
print(locs[0:10])
#error_mask=error_mask[9000:9010]
#locs=locs[0:10]
print(len(locs))
print(len(error_mask))




word_size  = 16
afrac_size = 7
aint_size  = 8
wfrac_size = 11
wint_size  = 12

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


wgt_dir= ('../weights/Inception/weights.data')



activation_aging = [True] * 170
#
# Acá la creamos, notese que como no se introduciran fallos en activaciones no es necesario pasar locs ni masks
Inception = GetNeuralNetworkModel('Inception', (150,150,3), 8, aging_active=activation_aging,faulty_addresses=locs, masked_faults=error_mask,
                            word_size=word_size, frac_size=afrac_size, batch_size=batch_size)
Inception.load_weights(wgt_dir).expect_partial()
WeightQuantization(model=Inception, frac_bits=wfrac_size, int_bits=wint_size)
loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
Inception.compile(optimizer=Adam(),   loss='sparse_categorical_crossentropy',  metrics=['accuracy'])
loss, acc = Inception.evaluate(test_ds)
#



#
# a = ActivationStats(Inception,test_ds,11,4,250)
# print(a)
#

num_address  =1048576






#

#
# #Capas con la informacion de procesamiento
# Indices      = [0,1,7,13,19,23,29,35,41,42,53,54,66,70,76,82,83,92,96,97,98,99,121,127,128,137,
#                 141,142,143,144,166,172,173,182,186,187,188,189,211,217,218,227,231,232,233,
#                 234,256,262,268,269,276,284,290,296,297,308,309,318,322,323,324,325,347,353,
#                 359,360,371,372,381,385,386,387,388,410,416,422,423,434,435,444,448,449,450,
#                 451,473,479,485,486,497,498,507,511,512,513,514,536,542,548,549,560,561,570,
#                 574,575,576,577,599,605,611,612,623,624,633,637,638,639,640,662,668,674,675,
#                 686,687,696,700,701,702,703,725,731,737,738,749,750,755,765,771,777,778,789,
#                 790,791,792,793,798,799,829,835,841,842,853,854,855,856,857,862,863,893,899,
#                 905,906,917,918,919,920,921,926,927,957,959,963]
#
#
#
#
# samples      = 1 #Numero de imagenes
# # Sin Power Gating:
# Data         = GetReadAndWrites(Inception,Indices,num_address,samples,CNN_gating=False,network_name='Inception')
# stats        = {'Lecturas': Data['Reads'],'Escrituras': Data['Writes']}
# Baseline_Acceses   = pd.DataFrame(stats).reset_index(drop=False)
# df_writes_Read =  pd.concat([Baseline_Acceses], axis=1, join='outer')
# df_writes_Read .columns = ['index','Lecturas','Escrituras']
# print(df_writes_Read)
# print(sum(df_writes_Read['Lecturas']))
# print(sum(df_writes_Read['Escrituras']))
# #save_obj(Baseline_Acceses,'Inception/métricas/Inception')
# # with pd.ExcelWriter('Inception/métricas/Inception_reads_and_write_num_adress_Mors'
# #                     '.xlsx') as writer:
# #           df_writes_Read.to_excel(writer, sheet_name='base', index=False)
# #
#
# # VBW = [101102, 124520, 302065, 331634, 340492, 340493, 360087, 360088, 377722, 377723, 419841, 458633, 458634, 465007, 465008, 465034, 465197, 465389, 544769, 544770, 545758, 590341, 590402, 590403, 590404, 590405, 590406, 590407, 590408, 590409, 590424, 590610, 590706, 590707, 590790, 590804, 611329]
# VBW = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/estatics/locs_HLO_0')
# #
# #
# writes_Read_VBW=df_writes_Read.iloc[VBW]
#
#
# writes_Read_VBW=df_writes_Read.iloc[VBW]
# # print('writes_Read_VBW', writes_Read_VBW)
# with pd.ExcelWriter('Inception/métricas/Inception_writes_Read_VBW_Mors.xlsx') as writer:
#     writes_Read_VBW.to_excel(writer, sheet_name='base', index=False)
#
# #
# #
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
# print('máximas letcturas',df_read_layers)
# with pd.ExcelWriter('Inception/métricas/max_lecturas_x_cada_16_direcciones_Mors.xlsx') as writer:
#          df_read_layers.to_excel(writer, sheet_name='base', index=False)
#
#

samples = 1

LI = [0,1,7,13,19,23,29,35,41,42,53,54,66,70,76,82,83,92,96,97,98,99,121,127,128,137,
                141,142,143,144,166,172,173,182,186,187,188,189,211,217,218,227,231,232,233,
                234,256,262,268,269,276,284,290,296,297,308,309,318,322,323,324,325,347,353,
                359,360,371,372,381,385,386,387,388,410,416,422,423,434,435,444,448,449,450,
                451,473,479,485,486,497,498,507,511,512,513,514,536,542,548,549,560,561,570,
                574,575,576,577,599,605,611,612,623,624,633,637,638,639,640,662,668,674,675,
                686,687,696,700,701,702,703,725,731,737,738,749,750,755,765,771,777,778,789,
                790,791,792,793,798,799,829,835,841,842,853,854,855,856,857,862,863,893,899,
                905,906,917,918,919,920,921,926,927,957,959,963]

AI = [6, 11, 12, 18, 26, 27, 34, 40, 51, 52, 63, 64, 67, 73, 74, 80, 81, 90, 91, 94, 95,
116, 117, 118, 119, 126, 135, 136, 139, 140, 161, 162, 163, 164, 171, 180, 181, 183, 184, 185,
206, 207, 208, 209, 216, 225, 226, 229, 230, 251, 252, 253, 254, 261, 267, 270, 271, 280, 281,
282, 289, 295, 298, 299, 306, 307, 317, 320, 321, 327, 328, 329, 343, 344, 345, 352, 358, 369,
370, 383, 384, 406, 407, 408, 415, 421, 432, 433, 446, 447, 469, 470, 471, 478, 484, 495, 496,
509, 510, 532, 533, 534, 541, 547, 558, 559, 572, 573, 595, 596, 597, 604, 610, 621, 622, 635,
636, 658, 659, 660, 667, 673, 684, 685, 698, 699, 721, 722, 723, 730, 736, 747, 748, 761, 762,
763, 770, 776, 786, 787, 788, 797, 820, 821, 822, 823, 834, 840, 851, 852, 858, 859, 860, 861,
884, 885, 886, 887, 891, 898, 904, 915, 916, 925, 950, 951, 954, 955, 958, 962]



Buffer,ciclos =  buffer_simulation(Inception, test_ds, integer_bits = 8, fractional_bits = 7, samples = samples, start_from = 0,
                                  bit_invertion = False, bit_shifting = False, CNN_gating = False,
                                  buffer_size = 1048576, write_mode ='default', save_results = True, network_type = 'Inception',
                                  results_dir = 'Data/Stats/Inception/',
                                  layer_indexes = LI , activation_indixes = AI)


df_cycles = pd.DataFrame(ciclos)
df_cycles.to_excel('cycles_Inception.xlsx')
print(str()+' operación ciclos completada: ', datetime.now().strftime("%H:%M:%S"))
