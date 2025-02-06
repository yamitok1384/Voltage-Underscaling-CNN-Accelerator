import os
import pickle as pickle
import tensorflow as tf
import numpy as np
from Nets  import GetNeuralNetworkModel
from StatsReadWrite import WeightQuantization, ActivationStats, CheckAccuracyAndLoss, QuantizationEffect, GetReadAndWrites
from Training import GetDatasets, GetPilotNetDataset
import pandas as pd
from datetime import datetime
from Simulation import buffer_simulation, save_obj, load_obj


#redes=[ 'AlexNet','ZFNet', 'MobileNet', 'VGG16','SqueezeNet', 'DenseNet']
redes=['DenseNet']
cycles_Alex = []
cycles_Dense = []
cycles_Mobile = []
cycles_VGG = []
cycles_Squez = []
cycles_ZF = []
cycles =[]
Accs_A = []
#vol=[0.51,0.52,0.53,0.54]

vol=0.51
inc = 0
samples = 1


error_mask = load_obj('MoRS/Modelo3_col_8_' + str(vol) + '/mask/error_mask_' + str(inc))
locs = load_obj('MoRS/Modelo3_col_8_' + str(vol) + '/mask/locs_' + str(inc))
print('error_mask', len(error_mask))
print('locs', len(locs))

# activation_aging = [True]*11
for i in range(1):




#     word_size = 16
#     afrac_size = 11
#     aint_size = 4
#     wfrac_size = 11
#     wint_size = 4
    trainBatchSize = testBatchSize = 1
    # _, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (227, 227), 8, trainBatchSize, testBatchSize)
#
#     cwd = os.getcwd()
#     wgt_dir = os.path.join(cwd, 'Data')
#     wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
#     wgt_dir = os.path.join(wgt_dir, 'AlexNet')
#     wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
#     wgt_dir = os.path.join(wgt_dir, 'Weights')
# #
# #
# #
#     AlexNet = GetNeuralNetworkModel('AlexNet', (227,227,3), 8, faulty_addresses=locs, masked_faults=error_mask,
#                                     aging_active=activation_aging, word_size=word_size, frac_size=afrac_size,
#                                    batch_size = testBatchSize)
#     AlexNet.load_weights(wgt_dir).expect_partial()
#     WeightQuantization(model=AlexNet, frac_bits=wfrac_size, int_bits=wint_size)
#     loss = tf.keras.losses.CategoricalCrossentropy()
#     optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
#     AlexNet.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
#     loss,acc =AlexNet.evaluate(test_dataset)


    # LI = [0, 3, 9, 11, 17, 19, 25, 31, 37, 40, 45, 50]
    # AI = [2, 8, 10, 16, 18, 24, 30, 36, 38, 44, 49, 53]
    # Buffer, ciclos = buffer_simulation(AlexNet, test_dataset, integer_bits=4, fractional_bits=11, samples=samples,
    #                                    start_from=0,
    #                                    bit_invertion=False, bit_shifting=False, CNN_gating=False,
    #                                    buffer_size=1048576, write_mode='default', save_results=True,
    #                                    network_type='AlexNet',
    #                                    results_dir='Data/Stats/AlexNet/',
    #                                    layer_indexes=LI, activation_indixes=AI)
    # cycles.append(ciclos)
    # Accs_A.append(acc)
    # df_Accs_A=pd.DataFrame(Accs_A)
    # df_cycles = pd.DataFrame(cycles)
    #
    #
    # print(str() + ' operación ciclos completada: ', datetime.now().strftime("%H:%M:%S"))



#     _, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)
#
#     word_size = 16
#     afrac_size = 11
#     aint_size = 4
#     wfrac_size = 14
#     wint_size = 1
#
#     cwd = os.getcwd()
#     wgt_dir = os.path.join(cwd, 'Data')
#     wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
#     wgt_dir = os.path.join(wgt_dir, 'ZFNet')
#     wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
#     wgt_dir = os.path.join(wgt_dir, 'Weights')
#
#
#
#     activation_aging = [True] * 11
#
# # for i in range(len(vol)):
# #     error_mask = load_obj('MoRS/Modelo3_col_8_' + str(vol[i]) + '/mask/error_mask_' + str(inc))
# #     locs = load_obj('MoRS/Modelo3_col_8_' + str(vol[i]) + '/mask/locs_' + str(inc))
#
#
#
#     ZFNet = GetNeuralNetworkModel('ZFNet', (224, 224, 3), 8, faulty_addresses=locs, masked_faults=error_mask,
#                                   aging_active=activation_aging, word_size=word_size, frac_size=afrac_size,
#                                   batch_size=testBatchSize)
#     ZFNet.load_weights(wgt_dir).expect_partial()
#     WeightQuantization(model=ZFNet, frac_bits=wfrac_size, int_bits=wint_size)
#     loss = tf.keras.losses.CategoricalCrossentropy()
#     optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
#     ZFNet.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
#     loss, acc = ZFNet.evaluate(test_dataset)
#
#     LI = [0, 3, 7, 11, 15, 19, 23, 27, 31, 34, 37, 40]
#     AI = [2,6,10,14,18,22,26,30,32,36,39,43]
#
#     Buffer,ciclos =buffer_simulation(ZFNet,test_dataset, integer_bits = 4, fractional_bits = 11, samples = samples, start_from = 0,
#                      bit_invertion = False, bit_shifting = False, CNN_gating = False,  buffer_size = 1048576,write_mode ='default', network_type = 'ZFNet',
#                      results_dir = 'Data/Stats/ZFNet/mask_x3/',
#                      layer_indexes = LI , activation_indixes = AI)
#     print('ciclos',ciclos)
#     cycles.append(ciclos)
#     Accs_A.append(acc)
#     df_redes = pd.DataFrame(redes)
#     df_Accs_A = pd.DataFrame(Accs_A)
#     df_cycles = pd.DataFrame(cycles)
#
#     cycles_by_nets = pd.concat([df_redes, df_Accs_A, df_cycles], axis=1, join='outer')
#     cycles_by_nets.columns = ['Red', 'ACC', 'Ciclos', ]
#     print(cycles_by_nets)
#     cycles_by_nets.to_excel('cycles_ZFNet.xlsx')



    # _, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)


    # In[3]:


    #Numero de bits para activaciones (a) y pesos (w)
    # word_size  = 16
    # afrac_size = 11
    # aint_size  = 4
    # wfrac_size = 14
    # wint_size  = 1
    #
    # # Tamaño del buffer de activaciones == al tamaño de la capa con mayor numero de activaciones (290400 pesos de 16 bits cada uno)
    # #abuffer_size = 16777216
    # # Directorio de los pesos
    # cwd = os.getcwd()
    # wgt_dir = os.path.join(cwd, 'Data')
    # wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
    # wgt_dir = os.path.join(wgt_dir, 'MobileNet')
    # wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
    # wgt_dir = os.path.join(wgt_dir,'Weights')
    #
    #
    #
    #
    # activation_aging = [True]*29
    # MobileNet = GetNeuralNetworkModel('MobileNet', (224,224,3), 8, faulty_addresses=locs, masked_faults=error_mask,
    #                                  aging_active=activation_aging, word_size=word_size, frac_size=afrac_size,  batch_size = testBatchSize)
    # MobileNet.load_weights(wgt_dir).expect_partial()
    # WeightQuantization(model=MobileNet, frac_bits=wfrac_size, int_bits=wint_size)
    # loss = tf.keras.losses.CategoricalCrossentropy()
    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    # MobileNet.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
    # loss,acc  = MobileNet.evaluate(test_dataset)
    #
    #
    #
    # LI = [0,4,10,16,23,29,35,41,48,54,60,66,73,79,85,91,97,103,109,115,121,127,133,139,146,152,158,164,170,175]
    # AI = [2,9,15,21,28,34,40,46,53,59,65,71,78,84,90,96,102,108,114,120,126,132,138,144,151,157,163,169,173,179]
    # Buffer,ciclos =buffer_simulation(MobileNet,test_dataset, integer_bits = 4, fractional_bits = 11, samples = samples, start_from = 0,
    #                  bit_invertion = False, bit_shifting = False, CNN_gating = False, write_mode ='default', save_results = True, network_type = 'MobileNet',
    #                  results_dir = 'Data/Stats/MobileNet/mask_x/' ,
    #                  buffer_size = 1048576, layer_indexes = LI , activation_indixes = AI)
    # print('ciclos', ciclos)
    # cycles.append(ciclos)
    # Accs_A.append(acc)
    # df_redes=pd.DataFrame(redes)
    # df_Accs_A=pd.DataFrame(Accs_A)
    # df_cycles = pd.DataFrame(cycles)

    # cycles_by_nets= pd.concat([df_redes,df_Accs_A,df_cycles], axis=1,join='outer')
    # cycles_by_nets.columns = ['Red','ACC', 'Ciclos',]
    # print(cycles_by_nets)
    # cycles_by_nets.to_excel('cycles.xlsx')



    # trainBatchSize = testBatchSize = 1
    # _, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)
    #
    # word_size = 16
    # afrac_size = 12
    # aint_size = 3
    # wfrac_size = 15
    # wint_size = 0
    #
    # # Tamaño del buffer de activaciones == al tamaño de la capa con mayor numero de activaciones (290400 pesos de 16 bits cada uno)
    # # abuffer_size = 16777216
    # # Directorio de los pesos
    # cwd = os.getcwd()
    # wgt_dir = os.path.join(cwd, 'Data')
    # wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
    # wgt_dir = os.path.join(wgt_dir, 'VGG16')
    # wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
    # wgt_dir = os.path.join(wgt_dir, 'Weights')
    #
    # # In[4]:
    #
    # activation_aging = [True] * 21
    #
    # VGG16 = GetNeuralNetworkModel('VGG16', (224, 224, 3), 8, faulty_addresses=locs, masked_faults=error_mask,
    #                               aging_active=activation_aging, word_size=word_size, frac_size=afrac_size,
    #                               batch_size=testBatchSize)
    #
    # VGG16.load_weights(wgt_dir).expect_partial()
    # WeightQuantization(model=VGG16, frac_bits=wfrac_size, int_bits=wint_size)
    # loss = tf.keras.losses.CategoricalCrossentropy()
    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    # VGG16.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
    # loss, acc = VGG16.evaluate(test_dataset)
    #
    # LI = [0, 3, 7, 11, 13, 17, 21, 23, 27, 31, 35, 37, 41, 45, 49, 51, 55, 59, 63, 66, 70, 74]
    # AI = [2,6,10,12,16,20,22,26,30,34,36,40,44,48,50,54,58,62,64,69,73,77]
    # Buffer,ciclos = buffer_simulation(VGG16,test_dataset, integer_bits = 3, fractional_bits = 12, samples = samples, start_from = 0,
    #                  bit_invertion = False, bit_shifting = False, CNN_gating = False, write_mode ='default', save_results = False,
    #                  network_type = 'VGG16',
    #                  results_dir = 'Data/Stats/VGG16/mask/', buffer_size =1048576,
    #                  layer_indexes = LI , activation_indixes = AI)
    # print('ciclos', ciclos)
    # cycles.append(ciclos)
    # Accs_A.append(acc)
    # df_redes = pd.DataFrame(redes)
    # df_Accs_A = pd.DataFrame(Accs_A)
    # df_cycles = pd.DataFrame(cycles)
    #
    # cycles_by_nets = pd.concat([df_redes, df_Accs_A, df_cycles], axis=1, join='outer')
    # cycles_by_nets.columns = ['Red', 'ACC', 'Ciclos', ]
    # print(cycles_by_nets)
    # cycles_by_nets.to_excel('MoRS\Analisis_Resultados\Energía_VBW\cycles\SVGG16.xlsx')
    #
    # print(str() + ' operación completada: ', datetime.now().strftime("%H:%M:%S"))
    #
    #
    #
    # word_size = 16
    # afrac_size = 9
    # aint_size = 6
    # wfrac_size = 15
    # wint_size = 0
    #
    # trainBatchSize = testBatchSize = 1
    # _, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)
    #
    # cwd = os.getcwd()
    # wgt_dir = os.path.join(cwd, 'Data')
    # wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
    # wgt_dir = os.path.join(wgt_dir, 'SqueezeNet')
    # wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
    # wgt_dir = os.path.join(wgt_dir, 'Weights')
    #
    # # In[4]:
    #
    # activation_aging = [True] * 22
    #
    # # Acá la creamos, notese que como no se introduciran fallos en activaciones no es necesario pasar locs ni masks
    # SqueezeNet = GetNeuralNetworkModel('SqueezeNet', (224, 224, 3), 8, faulty_addresses=locs, masked_faults=error_mask,
    #                                    aging_active=activation_aging, word_size=word_size, frac_size=afrac_size,
    #                                    batch_size=testBatchSize)
    #
    # loss = tf.keras.losses.CategoricalCrossentropy()
    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    # SqueezeNet.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
    # SqueezeNet.load_weights(wgt_dir).expect_partial()
    # WeightQuantization(model=SqueezeNet, frac_bits=wfrac_size, int_bits=wint_size)
    #
    # samples = 1
    # LI = [0,3,7, 9,(13,14),20,(24,25),31,(35,36),42,44,(48,49),55,(59,60),66,(70,71),77,(81,82),88,90,(94,95),101,104]
    # AI = [2,6,8,12,     19,23,     30,34,     41,43,47,     54,58,     65,69,     76,80 ,    87,89,93,    100,103,107]
    # Buffer,ciclos =buffer_simulation(SqueezeNet,test_dataset, integer_bits = 6, fractional_bits = 9, samples = samples, start_from = 0,
    #                  bit_invertion = False, bit_shifting = False, CNN_gating = False, write_mode ='default', save_results = False,
    #                  network_type = 'SqueezeNet',
    #                  results_dir = 'Data/Stats/SqueezeNet/mask/', buffer_size =1048576,
    #                  layer_indexes = LI , activation_indixes = AI)
    # print('ciclos', ciclos)
    # cycles.append(ciclos)
    # Accs_A.append(acc)
    # df_redes = pd.DataFrame(redes)
    # df_Accs_A = pd.DataFrame(Accs_A)
    # df_cycles = pd.DataFrame(cycles)
    #
    # cycles_by_nets = pd.concat([df_redes, df_Accs_A, df_cycles], axis=1, join='outer')
    # cycles_by_nets.columns = ['Red', 'ACC', 'Ciclos', ]
    # print(cycles_by_nets)
    # cycles_by_nets.to_excel('Cycles_SqueezeNet.xlsx')




    _, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)

    word_size = 16
    afrac_size = 12
    aint_size = 3
    wfrac_size = 13
    wint_size = 2

    # Tamaño del buffer de activaciones == al tamaño de la capa con mayor numero de activaciones (290400 pesos de 16 bits cada uno)
    # abuffer_size = 16777216
    # Directorio de los pesos
    cwd = os.getcwd()
    wgt_dir = os.path.join(cwd, 'Data')
    wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
    wgt_dir = os.path.join(wgt_dir, 'DenseNet')
    wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
    wgt_dir = os.path.join(wgt_dir, 'Weights')

    # In[4]:

    activation_aging = [True] * 188

    DenseNet = GetNeuralNetworkModel('DenseNet', (224, 224, 3), 8, faulty_addresses=locs, masked_faults=error_mask,
                                     aging_active=activation_aging, word_size=word_size, frac_size=afrac_size,
                                     batch_size=testBatchSize)
    DenseNet.load_weights(wgt_dir).expect_partial()
    WeightQuantization(model=DenseNet, frac_bits=wfrac_size, int_bits=wint_size)
    loss = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    DenseNet.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
    loss, acc = DenseNet.evaluate(test_dataset)

    samples = 1
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
    Buffer,ciclos =buffer_simulation(DenseNet,test_dataset, integer_bits = 3, fractional_bits = 12, samples= samples, start_from = 0,
                     bit_invertion = False, bit_shifting = False, CNN_gating = False, write_mode ='default',save_results = True, network_type = 'DenseNet',
                     results_dir = 'Data/Stats/DenseNet/mask_x3/', buffer_size = 1048576,
                     layer_indexes = LI , activation_indixes = AI)
    print('ciclos', ciclos)
    cycles.append(ciclos)
    Accs_A.append(acc)
    df_redes = pd.DataFrame(redes)
    df_Accs_A = pd.DataFrame(Accs_A)
    df_cycles = pd.DataFrame(cycles)

    cycles_by_nets = pd.concat([df_redes, df_Accs_A, df_cycles], axis=1, join='outer')
    cycles_by_nets.columns = ['Red', 'ACC', 'Ciclos', ]
    print(cycles_by_nets)
    cycles_by_nets.to_excel('cycles_DenseNet.xlsx')


    #

