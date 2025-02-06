#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
import pickle as pickle
import tensorflow as tf
import numpy as np
from Stats_Shift import WeightQuantization,IntroduceFaultsInWeights,GenerateFaultsList,CheckAccuracyAndLoss
from funciones import compilNet, same_elements,buffer_vectores,Base,IsoAECC,ECC, Flip,FlipPatchBetter,ScratchPad,ShiftMask,WordType
from Training import GetPilotNetDataset
from Training import GetDatasets
from Simulation import get_all_outputs
from Simulation import buffer_simulation, save_obj, load_obj
from datetime import datetime
import time
import collections
import pandas as pd
import os, sys
import pathlib


# In[14]:
#Acc_net: Desde una ruta definida con los ficgeros de error_mask y locs ya guardados con anterioridad para los voltajes analizados 
#los recorre y calcula el acc por cada voltaje para cada red de las estudiadas, finalmente los huarda en un documento excel para un
# futuro análisis más detallado, si se desea hacer para otras máscaras de error basta con cambiar las direcciones donde se encuentren



inc=0

for j in range(1):
    print('inc', inc)
    print('Ciclo externo',j)

    cwd = os.getcwd()
    wgt_dir = os.path.join(cwd, 'Data')
    wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
    wgt_dir = os.path.join(wgt_dir, 'AlexNet')
    wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
    wgt_dir = os.path.join(wgt_dir, 'Weights')

    run_time_A = []
    change_words = []
    Original_Acc = [0.890666663646697, 0.913333356380462, 0.881333351135253, 0.93066668510437, 0.805333316326141,  0.833333313465118]
    Original_Acc = pd.DataFrame(Original_Acc)

    Accs_A = []
    trainBatchSize = testBatchSize = 1
    _, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (227, 227), 8, trainBatchSize, testBatchSize)

# error_mask, locs = Flip(error_mask, locs)

#
    Funcion = [Base, IsoAECC, ECC, Flip, FlipPatchBetter, ScratchPad]
#     activation_aging = [True] * 11
#     #for i in range(1):
#
#     for i in range(1):
#         print('Ciclo interno', i)
#         inicio = time.time()
#         error_mask = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/error_mask_' + str(inc))
#         WordType(error_mask)
#         locs = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/locs_' + str(inc))
#
#         #error_mask_new, locs, word_change = Funcion[i](error_mask, locs)
#         error_mask_new, locs,word_change = FlipPatch(error_mask, locs)
#         WordType(error_mask_new)
#         #print(error_mask_new)
# #         error_mask = save_obj(error_mask_new,'MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/error_mask_f_p_better')
# #         error_mask_better = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/error_mask_f_p_better')
# #         #print(error_mask_better)
# #
# # WordType(error_mask_better)
#
#
#     # print('voltaje',vol)
#         print('tamaño de locs', len(locs))
    #print('tamaño de error_mask_new', len(error_mask))
#
#         loss, acc = CheckAccuracyAndLoss('AlexNet', test_dataset, wgt_dir, output_shape=8, input_shape=(227, 227, 3),
#                                      act_frac_size=11, act_int_size=4, wgt_frac_size=11, wgt_int_size=4, batch_size=testBatchSize,
#                                      verbose=0, aging_active=activation_aging, weights_faults=False,  faulty_addresses=locs, masked_faults=error_mask_new)
#
#         Accs_A.append(acc)
#         #change_words.append(word_change)
#         fin = time.time()
#         time_run = fin - inicio
#         run_time_A.append(time_run)
#     Acc_AlexNet = pd.DataFrame(Accs_A)
#    # Df_change_words = pd.DataFrame(change_words)
#     DF_run_time_a = pd.DataFrame(run_time_A)
#     print('Acc_AlexNet', Acc_AlexNet)
#     print(str() + ' operación completada AlexNet: ', datetime.now().strftime("%H:%M:%S"))
#
#
# Directorio de los pesos
    cwd = os.getcwd()
    wgt_dir = os.path.join(cwd, 'Data')
    wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
    wgt_dir = os.path.join(wgt_dir, 'SqueezeNet')
    wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
    wgt_dir = os.path.join(wgt_dir, 'Weights')
#
    Accs_S = []
    run_time_S = []
    list_words_fallos_S = []
#
    trainBatchSize = testBatchSize = 1
    _, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)

    activation_aging = [True] * 22
# # posiciones= [1,2,3]
#     #for i in range(1):
#     # for i in range(len(posiciones)):
    for i in range(1):
        inicio = time.time()
        error_mask = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/error_mask_' + str(inc))
        locs = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/locs_' + str(inc))
        print('tamaño de locs', len(locs))
        print('tamaño de error_mask', len(error_mask))

        #error_mask_new, locs, word_change = Funcion[i](error_mask, locs)
        error_mask_new, locs,locs_modif,word_change = FlipPatchBetter(error_mask, locs)
        # error_mask_shift,words_fallos= ShiftMask(error_mask_new,posiciones[i])
        print('tamaño de locs', len(locs))
        #print('tamaño de error_mask', len(error_mask_new))

        loss, acc = CheckAccuracyAndLoss('SqueezeNet', test_dataset, wgt_dir, output_shape=8, input_shape=(224, 224, 3),
                                     act_frac_size=9, act_int_size=6, wgt_frac_size=15, wgt_int_size=0,
                                     batch_size=testBatchSize, verbose=0, aging_active=activation_aging,
                                     weights_faults=False,  faulty_addresses=locs, masked_faults=error_mask_new)

        Accs_S.append(acc)
        print(acc)
        fin = time.time()
        time_run = fin - inicio
        run_time_S.append(time_run)
        # list_words_fallos_S.append(words_fallos)
    DF_run_time_s = pd.DataFrame(run_time_S)
    Acc_SqueezeNet = pd.DataFrame(Accs_S)
    #DF_words_fallos_S = pd.DataFrame(list_words_fallos_S)
    print('Acc_SqueezeNet', Acc_SqueezeNet)

    print(str() + ' operación completada SqueezeNet: ', datetime.now().strftime("%H:%M:%S"))

#     # Directorio de los pesos
#
#     cwd = os.getcwd()
#     wgt_dir = os.path.join(cwd, 'Data')
#     wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
#     wgt_dir = os.path.join(wgt_dir, 'DenseNet')
#     wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
#     wgt_dir = os.path.join(wgt_dir, 'Weights')
#
#     Accs_D = []
#     run_time_D = []
#
#     trainBatchSize = testBatchSize = 1
#     _, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)
#
#     activation_aging = [True] * 188
#     #for i in range(1):
#     for i in range(1):
#         inicio = time.time()
#         error_mask = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/error_mask_' + str(inc))
#         locs = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/locs_' + str(inc))
#
#         #error_mask_new, locs, word_change = Funcion[i](error_mask, locs)
#         error_mask_new, locs,word_change = FlipPatch(error_mask, locs)
#         print('tamaño de locs', len(locs))
#         #print('tamaño de error_mask', len(error_mask_new))
#
#         loss, acc = CheckAccuracyAndLoss('DenseNet', test_dataset, wgt_dir, output_shape=8, input_shape=(224, 224, 3),
#                                      act_frac_size=12, act_int_size=3, wgt_frac_size=13, wgt_int_size=2,
#                                      batch_size=testBatchSize, verbose=0, aging_active=activation_aging,
#                                      weights_faults=False,
#                                      faulty_addresses=locs, masked_faults=error_mask_new)
#
#         Accs_D.append(acc)
#         fin = time.time()
#         time_run = fin - inicio
#         run_time_D.append(time_run)
#     DF_run_time_d = pd.DataFrame(run_time_D)
#     Acc_DenseNet = pd.DataFrame(Accs_D)
#     print('Acc_DenseNet',Acc_DenseNet)
#
# # In[22]:
#
#     print(str() + ' operación completada DenseNet: ', datetime.now().strftime("%H:%M:%S"))
#
#     # Directorio de los pesos
#     cwd = os.getcwd()
#     wgt_dir = os.path.join(cwd, 'Data')
#     wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
#     wgt_dir = os.path.join(wgt_dir, 'MobileNet')
#     wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
#     wgt_dir = os.path.join(wgt_dir, 'Weights')
#     Accs_M = []
#     run_time_M = []
#
#     trainBatchSize = testBatchSize = 1
#     _, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)
#
#     activation_aging = [True] * 29
#     #for i in range(1):
#     for i in range(1):
#         inicio = time.time()
#         error_mask = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/error_mask_' + str(inc))
#         locs = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/locs_' + str(inc))
#
#         #error_mask_new, locs, word_change = Funcion[i](error_mask, locs)
#         error_mask_new, locs, word_change = FlipPatch(error_mask, locs)
#         print('tamaño de locs', len(locs))
#         #print('tamaño de error_mask', len(error_mask_new))
#
#         loss, acc = CheckAccuracyAndLoss('MobileNet', test_dataset, wgt_dir, output_shape=8, input_shape=(224, 224, 3),
#                                      act_frac_size=11, act_int_size=4, wgt_frac_size=14, wgt_int_size=1,
#                                      batch_size=testBatchSize, verbose=0, aging_active=activation_aging,
#                                      weights_faults=False,
#                                      faulty_addresses=locs, masked_faults=error_mask_new)
#         Accs_M.append(acc)
#         fin = time.time()
#         time_run = fin - inicio
#         run_time_M.append(time_run)
#     DF_run_time_m = pd.DataFrame(run_time_M)
#     Acc_MobileNet = pd.DataFrame(Accs_M)
#     print('Acc_MobileNet',Acc_MobileNet)
#
#     print(str() + ' operación completada MobileNet: ', datetime.now().strftime("%H:%M:%S"))
#
#         # In[26]:
#
#
#         # Directorio de los pesos
    cwd = os.getcwd()
    wgt_dir = os.path.join(cwd, 'Data')
    wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
    wgt_dir = os.path.join(wgt_dir, 'VGG16')
    wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
    wgt_dir = os.path.join(wgt_dir, 'Weights')

    Accs_V = []
    run_time_V = []
    list_words_fallos_V = []

    trainBatchSize = testBatchSize = 1
    _, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)

    activation_aging = [True] * 21
    #for i in range(1):
    # for i in range(len(posiciones)):
    for i in range(1):
        inicio = time.time()
        error_mask = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/error_mask_' + str(inc))
        locs = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/locs_' + str(inc))

        #error_mask_new, locs, word_change = Funcion[i](error_mask, locs)
        error_mask_new, locs,locs_modif,word_change = FlipPatchBetter(error_mask, locs)
        # error_mask_shift,words_fallos= ShiftMask(error_mask_new,posiciones[i])
        print('tamaño de locs', len(locs))
        #print('tamaño de error_mask', len(error_mask_new))
        from Stats_Shift import CheckAccuracyAndLoss
        loss, acc = CheckAccuracyAndLoss('VGG16', test_dataset, wgt_dir, output_shape=8, input_shape=(224, 224, 3),
                                             act_frac_size=12, act_int_size=3, wgt_frac_size=15, wgt_int_size=0,
                                             batch_size=testBatchSize, verbose=0, aging_active=activation_aging,
                                             weights_faults=False,
                                             faulty_addresses=locs, masked_faults=error_mask_new)

        Accs_V.append(acc)
        # list_words_fallos_V.append(words_fallos)
        fin = time.time()
        time_run = fin - inicio
        run_time_V.append(time_run)
    DF_run_time_v = pd.DataFrame(run_time_V)
    Acc_VGG16 = pd.DataFrame(Accs_V)
    #DF_words_fallos_V = pd.DataFrame(list_words_fallos_V)
    print('Acc_VGG16',Acc_VGG16)
#         #
#
#         #
#         # # In[30]:
#         #
#     print(str() + ' operación completada VGG16: ', datetime.now().strftime("%H:%M:%S"))
        #
        # #
        # # # Directorio de los pesos
    # cwd = os.getcwd()
    # wgt_dir = os.path.join(cwd, 'Data')
    # wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
    # wgt_dir = os.path.join(wgt_dir, 'ZFNet')
    # wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
    # wgt_dir = os.path.join(wgt_dir, 'Weights')
    #
    # Accs_Z = []
    # run_time_Z = []
    #
    # trainBatchSize = testBatchSize = 1
    # _, _, test_dataset = GetDatasets('colorectal_histology', (80, 5, 15), (224, 224), 8, trainBatchSize, testBatchSize)
    #
    # activation_aging = [True] * 11
    # #for i in range(1):
    # for i in range(1):
    #     inicio = time.time()
    #     error_mask_new = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/error_mask_' + str(inc))
    #     locs = load_obj('MoRS/Modelo3_mas_fallos_col_8_experimentos/mask/locs_' + str(inc))
    #
    #     #error_mask_new, locs,word_change = FlipPatchBetter(error_mask, locs)
    #     #error_mask_new, locs, locs_patch, word_change = FlipPatchBetter(error_mask, locs)
    #     #error_mask_new, locs, word_change = (error_mask, locs)
    #     print('tamaño de locs', len(locs))
    #     #print('tamaño de error_mask', len(error_mask_new))
    #     from Stats_Shift import WeightQuantization,IntroduceFaultsInWeights,GenerateFaultsList,CheckAccuracyAndLoss
    #
    #     loss, acc = CheckAccuracyAndLoss('ZFNet', test_dataset, wgt_dir, output_shape=8, input_shape=(224, 224, 3),
    #                                          act_frac_size=11, act_int_size=4, wgt_frac_size=15, wgt_int_size=0,
    #                                          batch_size=testBatchSize, verbose=0, aging_active=activation_aging,
    #                                          weights_faults=False,    faulty_addresses=locs, masked_faults=error_mask_new)
    #
    #     Accs_Z.append(acc)
    #     fin = time.time()
    #     time_run = fin - inicio
    #     run_time_Z.append(time_run)
    # DF_run_time_z = pd.DataFrame(run_time_Z)
    # Acc_ZFNet = pd.DataFrame(Accs_Z)
    # print('Acc_ZFNet',acc)
    # del error_mask_new
        # Funcion =[Base,IsoAECC, ECC, Flip,FlipPatch,FlipPatch]
    #DF_Funcion = pd.DataFrame(['Shift_all_actvs'])

    # Acc_all_exp = pd.concat( [ DF_Funcion, Acc_AlexNet, Acc_DenseNet, Acc_MobileNet, Acc_SqueezeNet, Acc_VGG16, Acc_ZFNet],
    #         axis=1, join='outer')
    # Acc_all_exp.columns = [ 'Técnica', 'AlexNet', 'DenseNet', 'MobileNet', 'SqueezeNet', 'VGG16', 'ZFNet']
    # print(Acc_all_exp)
    # Acc_all_exp.to_excel('MoRS/Modelo3_mas_fallos_col_8_experimentos/Analisis_Resultados/estatics/Shift/ACC_new_formula_flipat_and_sift_5_fault_modelo_0_53_'+ str(inc) + '.xlsx', sheet_name='fichero_707', index=False)

    # buf_time = pd.concat([DF_run_time_a, DF_run_time_d, DF_run_time_m, DF_run_time_s, DF_run_time_v, DF_run_time_z],
    #                      axis=1, join='outer')
    # buf_time.columns = ['AlexNet', 'DenseNet', 'MobileNet', 'SqueezeNet', 'VGG16', 'ZFNet']
    # print(buf_time)
    # buf_time.to_excel('MoRS/Modelo3_mas_fallos_col_8_experimentos/Analisis_Resultados/estatics/Shift/Shift_5/Time_Shift_5_error_mask_0_53_'+ str(inc) + '.xlsx', sheet_name='Time', index=False)

    #inc = inc + 1
#
