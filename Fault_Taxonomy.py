#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
import pickle as pickle
import tensorflow as tf
import numpy as np
from Stats import WeightQuantization,IntroduceFaultsInWeights,GenerateFaultsList,CheckAccuracyAndLoss
from funciones import compilNet, same_elements,buffer_vectores,LowOrder,HighOrder,VeryBadWords,VBWGoToScratch,Flip,DeleteTercioRamdom,MaskVeryBadWords,L0flippedHO
from Training import GetPilotNetDataset
from Training import GetDatasets
from Simulation import get_all_outputs
from Simulation import buffer_simulation, save_obj, load_obj
from datetime import datetime
import collections 
import pandas as pd
import os, sys
import pathlib
from openpyxl import Workbook



#Se calcula la presición para cada tipo de palabras (LO,L&HO,HO), además de la presición si enviamos todas las
#l&HO a las regiones de la memoria no ocupado por las capas de la red(VBWGoToScratch)
#LO(LowOrder): Palabras con fallos en el byte menos significativo
#HO(HighOrder):Palabras  con fallos en el byte más significativo
#L&HO(VeryBadWords):Palabras con bits de fallos tanto en ambos bytes
#L0flippedHO:
#VBWGoToScratch:

# In[14]:
#Acc_net: Desde una ruta definida con los ficgeros de error_mask y locs ya guardados con anterioridad para los voltajes analizados 
#los recorre y calcula el acc por cada voltaje para cada red de las estudiadas, finalmente los guarda en un documento excel para un
# futuro análisis más detallado, si se desea hacer para otras máscaras de error basta con cambiar las direcciones donde se encuentren

vol=0.51
inc = 0
Tecnicas=['LO','VBWGoToScratch','L&HO','HO']
# Uso VeryBadWords si quiero aplicar la tecnica de quiter 1/3 de palabra de L&HO(esto sería para Ultra-Low-Vdd)
#Funcion =[L0flippedHO,VBWGoToScratch]
#Funcion =[LowOrder,VeryBadWords,HighOrder]
#Funcion =[LowOrder,VBWGoToScratch,VeryBadWords,HighOrder]
Funcion =[LowOrder,VBWGoToScratch,MaskVeryBadWords,HighOrder]
#Funcion =[HighOrder]
#Tecnicas=['L0flippedHO','VBWGoToScratch']
Tecnicas=['LO','VBWGoToScratch','L&HO','HO']
#Tecnicas=['LO','L&HO','HO']
Df_Tecn=pd.DataFrame(Tecnicas*6)
print(Df_Tecn)
Df_Tecn.to_excel('MoRS/Modelo3_col_8_'+ str(vol)+'/Analisis_Resultados/Palabras_x_tipo/test_direccion_' + str(inc)+'.xlsx', sheet_name='fichero_707', index=False)

#Esta máscara e ssolo para rectificar estos códigos que los he optimizado
# error_mask = load_obj(  'Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/error_mask_0' + str(vol))
# locs = load_obj(  'Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/locs_0' + str(vol))


# print('tamaño mascara originalmente',len(error_mask))
# print('tamaño locs originalmente',len(locs))




for j in range(2):
# # PilotNet

    #voltaj=[54,55,56,57,58,59,60]
    # #Df_Vol=[54,56,58,60]
    # #Voltajes=pd.DataFrame(Df_Vol)
    # Df_Vol=[]

    error_mask = load_obj('MoRS/Modelo3_col_8_' + str(vol) + '/mask/error_mask_' + str(inc))
    locs = load_obj('MoRS/Modelo3_col_8_' + str(vol) + '/mask/locs_' + str(inc))

    print('tamaño de la mascara inicial',len(error_mask))
    print('tamaño de la locs inicial',len(locs))


    #Este codigo lo uso para voltaje sinferiores a 0.53
    #error_mask_H_L_O, locs_H_L_O, index_locs_VBW = VeryBadWords(error_mask, locs)
    #error_mask_less_VBW, locs_less_VBW = DeleteTercioRamdom(error_mask, locs, index_locs_VBW)
    #print('tamaño de la mascara final',len(error_mask_less_VBW))
    #print('tamaño de la locs final',len(locs_less_VBW))


    print('Modelo', inc)
    print('Modelo', vol)

    cwd = os.getcwd()
    wgt_dir = os.path.join(cwd, 'Data')
    wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
    wgt_dir = os.path.join(wgt_dir, 'AlexNet')
    wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
    wgt_dir = os.path.join(wgt_dir,'Weights')



    trainBatchSize = testBatchSize = 1
    _,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(227,227), 8, trainBatchSize, testBatchSize)



    Original_Acc = [0.890666663646697,0.913333356380462,0.881333351135253, 0.93066668510437, 0.805333316326141, 0.833333313465118]

    #vol=voltaj[0]
    Redes = []
    All_acc = []
    All_acc_normal = []
    Accs_A = []
    activation_aging = [True]*11
    #for i in range(1):
    for i in range(len(Funcion)):

        print(i)
        print('tipo', type(error_mask))
        print(type(locs))

        #error_mask_new, locs,word_change = Funcion[i](error_mask_less_VBW, locs_less_VBW)
        error_mask_new, locs,word_change = Funcion[i](error_mask, locs)


        #error_mask_new, locs = VBWGoToScratch(error_mask, locs)
        print('error_mask_new', len(error_mask_new))
        print('locs', len(locs))



        loss,acc   = CheckAccuracyAndLoss('AlexNet', test_dataset, wgt_dir, output_shape=8, input_shape = (227,227,3),
                                          act_frac_size = 11, act_int_size = 4, wgt_frac_size = 11, wgt_int_size = 4,
                                          batch_size=testBatchSize, verbose = 0, aging_active = activation_aging, weights_faults = False,
                                          faulty_addresses = locs, masked_faults = error_mask_new)

        Accs_A.append(acc)
        All_acc.append(acc)
        acc_normal= acc/0.89066666364669
        All_acc_normal.append(acc_normal)
        del error_mask_new
        Redes.append('AlexNet')
        print('tecnica y presicion',Funcion[i],acc)
    Acc_AlexNet=pd.DataFrame(Accs_A)
    print('Acc_AlexNet',Acc_AlexNet)
    print('All_acc',All_acc)
    print(str()+' operación completada AlexNet: ', datetime.now().strftime("%H:%M:%S"))


    # #Directorio de los pesos
    # cwd = os.getcwd()
    # wgt_dir = os.path.join(cwd, 'Data')
    # wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
    # wgt_dir = os.path.join(wgt_dir, 'DenseNet')
    # wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
    # wgt_dir = os.path.join(wgt_dir,'Weights')
    #
    # Accs_D=[]
    #
    # trainBatchSize = testBatchSize = 1
    # _,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(224,224), 8, trainBatchSize, testBatchSize)
    #
    #
    # #vol=voltaj[0]
    # activation_aging = [True] * 188
    # for i in range(len(Funcion)):
    #     #for i in range(1):
    #     #error_mask_new, locs,word_change = Funcion[i](error_mask_less_VBW, locs_less_VBW)
    #     error_mask_new, locs,word_change = Funcion[i](error_mask, locs)
    #     #error_mask_new, locs = L0flippedHO(error_mask, locs)
    #
    #
    #
    #     loss,acc   = CheckAccuracyAndLoss('DenseNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
    #                                       act_frac_size = 12, act_int_size = 3, wgt_frac_size = 13, wgt_int_size = 2,
    #                                       batch_size=testBatchSize, verbose = 0, aging_active = activation_aging, weights_faults = False,
    #                                       faulty_addresses = locs, masked_faults = error_mask_new)
    #
    #     Accs_D.append(acc)
    #     All_acc.append(acc)
    #     acc_normal = acc / 0.913333356380462
    #     All_acc_normal.append(acc_normal)
    #     del error_mask_new
    #     Redes.append('DenseNet')
    # Acc_DenseNet=pd.DataFrame(Accs_D)
    # print('Acc_DenseNet',Acc_DenseNet)
    # print('All_acc',All_acc)
    #
    #
    # # In[22]:
    #
    # print(str()+' operación completada DenseNet: ', datetime.now().strftime("%H:%M:%S"))
    #
    #
    #
    #
    # # Directorio de los pesos
    # cwd = os.getcwd()
    # wgt_dir = os.path.join(cwd, 'Data')
    # wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
    # wgt_dir = os.path.join(wgt_dir, 'MobileNet')
    # wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
    # wgt_dir = os.path.join(wgt_dir,'Weights')
    # Accs_M=[]
    #
    #
    # trainBatchSize = testBatchSize = 1
    # _,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(224,224), 8, trainBatchSize, testBatchSize)
    #
    #
    #
    # #vol=voltaj[0]
    # activation_aging = [True]*29
    # for i in range(len(Funcion)):
    #     #for i in range(1):
    #     #error_mask_new, locs,word_change = Funcion[i](error_mask_less_VBW, locs_less_VBW)
    #     error_mask_new, locs,word_change = Funcion[i](error_mask, locs)
    #     #error_mask_new, locs = L0flippedHO(error_mask, locs)
    #
    #     loss,acc   = CheckAccuracyAndLoss('MobileNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
    #                                       act_frac_size = 11, act_int_size = 4, wgt_frac_size = 14, wgt_int_size = 1,
    #                                       batch_size=testBatchSize, verbose = 0, aging_active = activation_aging, weights_faults = False,
    #                                       faulty_addresses = locs, masked_faults = error_mask_new)
    #     Accs_M.append(acc)
    #     All_acc.append(acc)
    #     acc_normal = acc / 0.881333351135253
    #     All_acc_normal.append(acc_normal)
    #     del error_mask_new
    #     Redes.append('MobileNet')
    # Acc_MobileNet=pd.DataFrame(Accs_M)
    # print('Acc_MobileNet',Acc_MobileNet)
    # print('All_acc',All_acc)
    #
    # print(str()+' operación completada MobileNet: ', datetime.now().strftime("%H:%M:%S"))
    #
    # # # In[26]:
    # #
    # #
    #
    #
    #
    # # Directorio de los pesos
    # cwd = os.getcwd()
    # wgt_dir = os.path.join(cwd, 'Data')
    # wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
    # wgt_dir = os.path.join(wgt_dir, 'SqueezeNet')
    # wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
    # wgt_dir = os.path.join(wgt_dir,'Weights')
    # Accs_S=[]
    #
    #
    # trainBatchSize = testBatchSize = 1
    # _,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(224,224), 8, trainBatchSize, testBatchSize)
    #
    #
    # #vol=voltaj[0]
    # activation_aging = [True] * 22
    # for i in range(len(Funcion)):
    #     #for i in range(1):
    #     #error_mask_new, locs,word_change = Funcion[i](error_mask_less_VBW, locs_less_VBW)
    #     error_mask_new, locs,word_change = Funcion[i](error_mask, locs)
    #     #error_mask_new, locs = L0flippedHO(error_mask, locs)
    #
    #     loss,acc   = CheckAccuracyAndLoss('SqueezeNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
    #                                       act_frac_size = 9, act_int_size = 6, wgt_frac_size = 15, wgt_int_size = 0,
    #                                       batch_size=testBatchSize, verbose = 0, aging_active = activation_aging, weights_faults = False,
    #                                       faulty_addresses = locs, masked_faults = error_mask_new)
    #
    #     Accs_S.append(acc)
    #     All_acc.append(acc)
    #     acc_normal = acc / 0.93066668510437
    #     All_acc_normal.append(acc_normal)
    #     del error_mask_new
    #     Redes.append('SqueezeNet')
    # Acc_SqueezeNet=pd.DataFrame(Accs_S)
    # print('Acc_SqueezeNet',Acc_SqueezeNet)
    # print('All_acc',All_acc)
    #
    # print(str()+' operación completada SqueezeNet: ', datetime.now().strftime("%H:%M:%S"))
    #
    #
    #
    # #Directorio de los pesos
    # cwd = os.getcwd()
    # wgt_dir = os.path.join(cwd, 'Data')
    # wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
    # wgt_dir = os.path.join(wgt_dir, 'VGG16')
    # wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
    # wgt_dir = os.path.join(wgt_dir,'Weights')
    # Accs_V=[]
    #
    #
    # trainBatchSize = testBatchSize = 1
    # _,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(224,224), 8, trainBatchSize, testBatchSize)
    #
    #
    #
    # #vol=voltaj[0]
    # activation_aging = [True]*21
    # for i in range(len(Funcion)):
    #     #for i in range(1):
    #     #error_mask_new, locs,word_change = Funcion[i](error_mask_less_VBW, locs_less_VBW)
    #     error_mask_new, locs,word_change = Funcion[i](error_mask, locs)
    #     #error_mask_new, locs = L0flippedHO(error_mask, locs)
    #
    #
    #     loss,acc   = CheckAccuracyAndLoss('VGG16', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
    #                                       act_frac_size = 12, act_int_size = 3, wgt_frac_size = 15, wgt_int_size = 0,
    #                                       batch_size=testBatchSize, verbose = 0, aging_active = activation_aging, weights_faults = False,
    #                                       faulty_addresses = locs, masked_faults = error_mask_new)
    #     Accs_V.append(acc)
    #     All_acc.append(acc)
    #     acc_normal = acc / 0.805333316326141
    #     All_acc_normal.append(acc_normal)
    #     del error_mask_new
    #     Redes.append('VGG16')
    # Acc_VGG16=pd.DataFrame(Accs_V)
    # print('Acc_VGG16',Acc_VGG16)
    # print('All_acc',All_acc)
    #
    # print(str()+' operación completada VGG16: ', datetime.now().strftime("%H:%M:%S"))
    #
    #
    # #
    # #
    # # # Directorio de los pesos
    # cwd = os.getcwd()
    # wgt_dir = os.path.join(cwd, 'Data')
    # wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
    # wgt_dir = os.path.join(wgt_dir, 'ZFNet')
    # wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
    # wgt_dir = os.path.join(wgt_dir,'Weights')
    #
    # Accs_Z=[]
    #
    # trainBatchSize = testBatchSize = 1
    # _,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(224,224), 8, trainBatchSize, testBatchSize)
    #
    #
    # #vol=voltaj[0]
    # activation_aging = [True] * 11
    #
    # for i in range(len(Funcion)):
    #     #for i in range(1):
    #     #error_mask_new, locs,word_change = Funcion[i](error_mask_less_VBW, locs_less_VBW)
    #     error_mask_new, locs,word_change = Funcion[i](error_mask, locs)
    #     #error_mask_new, locs = L0flippedHO(error_mask, locs)
    #     print('error_mask_new', len(error_mask_new))
    #     print('locs', len(locs))
    #
    #
    #     loss,acc   = CheckAccuracyAndLoss('ZFNet', test_dataset, wgt_dir, output_shape=8, input_shape = (224,224,3),
    #                                       act_frac_size = 11, act_int_size = 4, wgt_frac_size = 15, wgt_int_size = 0,
    #                                       batch_size=testBatchSize, verbose = 0, aging_active = activation_aging, weights_faults = False,
    #                                       faulty_addresses = locs, masked_faults = error_mask_new)
    #
    #     Accs_Z.append(acc)
    #     All_acc.append(acc)
    #     acc_normal = acc / 0.833333313465118
    #     All_acc_normal.append(acc_normal)
    #     del error_mask_new
    #     Redes.append('ZFNet')
    # Acc_ZFNet=pd.DataFrame(Accs_Z)
    # print('Acc_ZFNet',Acc_ZFNet)
    # print('All_acc',All_acc)


    Df_redes=pd.DataFrame(Redes)
    Df_acc=pd.DataFrame(All_acc)
    Df_acc_normal=pd.DataFrame(All_acc_normal)

    analize_by_part_Mors = pd.concat([Df_redes,Df_Tecn,Df_acc_normal,Df_acc], axis=1, join='outer')
    analize_by_part_Mors.columns =['Redes','Tecnic','acc_normal', 'acc']
    print(analize_by_part_Mors)
    #analize_by_part_Mors.to_excel('MoRS/Modelo3_col_8_'+ str(vol)+'/Analisis_Resultados/Palabras_x_tipo/Mask_L_F(HO)/ACC_Palabras_x_tipo_mask_' + str(inc)+'.xlsx', sheet_name='fichero_707', index=False)
    analize_by_part_Mors.to_excel('MoRS/Modelo3_col_8_' + str(vol) + '/Analisis_Resultados/Palabras_x_tipo/ACC_Palabras_x_tipo_mask_test_tesis' + str(inc) + '.xlsx',
                              sheet_name='fichero_707', index=False)


    inc = inc + 1

    # Voltajes=pd.DataFrame(Df_Vol)
    # buf_cero = pd.concat([Voltajes,Acc_AlexNet,Acc_DenseNet,Acc_MobileNet,Acc_SqueezeNet,Acc_VGG16,Acc_ZFNet,Acc_PilotNet], axis=1, join='outer')
    # #buf_cero = pd.concat([Voltajes,Acc_AlexNet,Acc_MobileNet,Acc_SqueezeNet,Acc_VGG16,Acc_ZFNet], axis=1, join='outer')
    # print(buf_cero)
    # buf_cero.columns =['Voltajes','AlexNet', 'DenseNet', 'MobileNet', 'SqueezeNet', 'VGG16', 'ZFNet' ,'PilotNet']
    # #buf_cero.columns =['Voltajes','AlexNet', 'MobileNet', 'SqueezeNet', 'VGG16', 'ZFNet']
    # buf_cero.to_excel('only_VBW_with_error_acc_error_mask_x.xlsx', sheet_name='fichero_707', index=False)

    # buf_cero = pd.concat([Voltajes,Acc_SqueezeNet,Acc_DenseNet,], axis=1, join='outer')
    # buf_cero.columns =['Voltajes', 'Acc_Sque', 'Acc_Dense']
    # buf_cero.to_excel('acc_baseline_sq_dens_060.xlsx', sheet_name='fichero_707', index=False)

    # print('buf_cero',buf_cero)
    print(str()+' operación completada: ', datetime.now().strftime("%H:%M:%S"))




# In[ ]:

