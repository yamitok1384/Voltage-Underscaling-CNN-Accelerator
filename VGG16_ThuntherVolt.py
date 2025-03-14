#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pickle as pickle
import tensorflow as tf
import numpy as np
from Stats import WeightQuantization, IntroduceFaultsInWeights, GenerateFaultsList
from Nets_original import GetNeuralNetworkModel
from Training import GetDatasets
from Simulation import get_all_outputs
from Simulation import buffer_simulation, save_obj, load_obj
from datetime import datetime
import pandas as pd
from funciones import compilNet, same_elements

word_size  = 16
afrac_size = 12
aint_size  = 3
wfrac_size = 15
wint_size  = 0

cwd = os.getcwd()
wgt_dir = os.path.join(cwd, 'Data')
wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
wgt_dir = os.path.join(wgt_dir, 'VGG16')
wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
wgt_dir = os.path.join(wgt_dir, 'Weights') 


trainBatchSize = testBatchSize = 1
_,_,test_dataset = GetDatasets('colorectal_histology',(80,5,15),(224,224), 8, trainBatchSize, testBatchSize)



#voltaj=('0.54','0.55','0.56','0.57','0.58','0.59')
voltaj=[59, 58, 57,56, 55,54]
#print(voltaj)
Voltajes=pd.DataFrame(voltaj) 
activation_aging = np.array([False]*21)
activation_aging[0] = True

vol=59
print(len(activation_aging))
acc_list=[]
vol_list=[]
list_ciclo=[]
capa=[]
iterator = iter(test_dataset)
conf = False
j = 0

 #with pd.ExcelWriter('AlexNet/acc_vol.xlsx') as writer:
for j, valor in enumerate(activation_aging):
    print('capa', j)
   
    for i, v in enumerate(voltaj):
        #if vol==53:
        #    break
        #ciclo=i
        print('i**********',i)
        print('v**********',v)
        print('j dentro del segundo ciclo', j)
            
        
        error_mask = load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/error_mask_0' + str(v))
        locs = load_obj('Data/Fault Characterization/variante_mask_vc_707/error_mask_x/error_mask_x/vc_707/locs_0' + str(v))
        #print('vol',vol)
        #vol = vol - 1
        print (activation_aging)
    
        VGG16 = GetNeuralNetworkModel('VGG16', (224,224,3), 8, faulty_addresses=locs, masked_faults=error_mask,
                                 aging_active=activation_aging, word_size=word_size, frac_size=afrac_size, 
                                 batch_size = testBatchSize)
        VGG16.load_weights(wgt_dir).expect_partial()
        loss = tf.keras.losses.CategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        VGG16.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
        WeightQuantization(model = VGG16, frac_bits = 12, int_bits = 3)
        loss,acc  = VGG16.evaluate(test_dataset)
        
        if acc < 0.805333316326141:
            print('acc menor o igual',acc)
            acc_list.append(acc)
            vol_list.append(voltaj[i] + 1)
            #vol_list.append(voltaj[i])
            capa.append(j)
            activation_aging = np.array([False]*21)
            j = j + 1
            activation_aging[j]=True 
            i=0
            conf = True
            print(capa)
            print(vol_list)
            print(acc_list)
            break
        else:
            if conf == True:
                i=0
                activation_aging = np.array([False]*21)
                activation_aging[j]= True
            else:
                print('dentro del else')
                j = 0
                activation_aging[j]= True
                
            if v==54:
                    print('v dnetro del else',v)
                    acc_list.append(acc)
                    vol_list.append(v)
                    capa.append(j)
                    print(capa)
                    print(vol_list)
                    print(acc_list)
                
df_capa=pd.DataFrame(capa)             
df_acc=pd.DataFrame(acc_list)
df_vol=pd.DataFrame(vol_list)
buf_acc_capa = pd.concat([df_capa,df_acc,df_vol], axis=1, join='outer')
buf_acc_capa.columns = ['capa','Accu', 'vol']
print(buf_acc_capa)
buf_acc_capa.to_excel('VGG_16_buf_acc_capa.xlsx', sheet_name='VGG16', index=False)  

