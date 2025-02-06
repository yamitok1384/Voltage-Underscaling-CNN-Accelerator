#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle as pickle
import tensorflow as tf
import numpy as np
import pandas as pd
import random
from Simulation import save_obj, load_obj
from funciones import buffer_vectores
import collections
from datetime import datetime



#Creamos  las máscaras de fallos teniendo en cuenta las direcciones del buffer donde hay fallos
# generados con MoRs

# buffer_size: Tamaño del buffer en bits(tamaño de la memoria de activación 2 MiB)
# vol: Voltaje para el cual queremos crear la máscara de fallos
# fich: variable para ir iterando sobre los ficheros generados pro Mors guardados
# index_buffer: Direcciones con fallos
# error_mask: lista de vectores de 16 bits con las máscaras de fallos
# locs:Lista de las direcciones del buffer conde hay fallos
##Ciclo para recorrer todo los ficheros e ir guardando las mácaras y las firechiones con nombres diferentes
##luego ejecutar el static_index



buffer_size = 16777216

## para ficheros pkl
vol='0.60'
fich = 0
for i in range(10):

    with open('MoRS/Modelo3_col_8_'+ str(vol)+'/index/index_'+ str(fich)+'.pkl', 'rb') as f:
        # Carga el contenido del archivo en una variable como lista de Python
        index_lista = pickle.load(f)


# Convierte la lista de Python en un arreglo de NumPy
    index_buffer = np.array(index_lista)
    print(index_buffer)

#Creo una lista numpy del tanalo del buffer todo en x(x significa que no hay fallos)

    mbuffer = np.array(['x']*(buffer_size))
    #print(len(mbuffer))

    count_fallos=0
# Ciclo para realizar la inyección de los fallos teniendo en cuenta las direcciones de los bits de fallos(generados por Mors)
#Para ellos en los índices con fallos inyectamos de forma random un 0 o un 1
    for i,x in enumerate(mbuffer):

        if i in index_buffer:
        #print(i)
            mbuffer[i]=random.randint(0, 1)
            count_fallos += 1
            #print(count_fallos)
#dist: Cantidad de fallos a 0 y a 1
    dist = collections.Counter(mbuffer)
            #print(mbuffer[i])
    print(len(mbuffer))
    print((mbuffer))
    print(dist)
    #print('count_fallos',count_fallos)


    mod=np.array(['x']*8)
    union=np.concatenate([mod, mbuffer])


#LLamamos a la función buffer_vectores la cual nos devuelve una lista de vectores(máscara de fallos)
#y una lista con los las direcciones en el buffer donde se encuentran esos fallos.
    error_mask, locs = (buffer_vectores(union[0:buffer_size]))
    #error_mask, locs = (buffer_vectores(mbuffer))

    print('tamaño de error mask',len(error_mask))
    print('tamaño de error locs',len(locs))

# Los guardamos en ficheros pkl para luego procesarlos
    save_obj(error_mask,'MoRS/Modelo3_col_8_'+ str(vol)+'/mask/error_mask_'+ str(fich))
    save_obj(locs,'MoRS/Modelo3_col_8_'+ str(vol)+'/mask/locs_'+str(fich))


    DF_mask = pd.DataFrame(error_mask)
    DF_locs = pd.DataFrame(locs)

    # Mask_locs = pd.concat([DF_mask, DF_locs], axis=1, join='outer')
    # Mask_locs.columns = ['error_mask','locs']
    #Mask_locs.to_excel('MoRS/Modelo3_col_8_0.51/Mask_locs/Mask_locs_' +str(fich) + '.xlsx', index=False)
    fich= fich+1



# Otra forma de hacerlo: esta forma la cree cuando guardaba las direcciones que me devuelve Mors en
#ficheros excels, pero para el ultra-volting donde hay muchos bits ocn fallos no me funciona por el límite
#de filas de excel y por eso lo hago con el código de arriba donde trabajo con ficheros .pkl

#
#ruta_bin = 'C:/Users/usuario/Desktop/MoRS/MoRS-master/Modelo3_col_8_0.57/index'
#ruta_bin = 'MoRS/Modelo3_col_8_0.51/index'
#ruta_bin = 'MoRS/Modelo3_mas_fallos_col_8_experimentos/index'
# directorio = pathlib.Path(ruta_bin)
#
# ficheros = [fichero.name for fichero in directorio.iterdir()]
# ficheros.sort()
# #
# fich=1
# for no_fichero, j in enumerate(ficheros):
#     #print('j',j)
#     print(str(j))
#
#     directorio = os.path.join(ruta_bin, j)
#     print(directorio)
#
#     mbuffer = np.array(['x']*(buffer_size))
#     print(len(mbuffer))
#     index = pd.read_excel(directorio)
#     #index=pd.read_excel('MoRS\index\index.xlsx')
#     np_array = index.values
#    # print(np_array)
#     count_fallos=0
#
#
#     for i,x in enumerate(mbuffer):
#         #print(x)
#         if i in np_array:
#             #print(i)
#             mbuffer[i]=random.randint(0, 1)
#             count_fallos += 1
#
#     dist = collections.Counter(mbuffer)
#         #print(mbuffer[i])
#     print(len(mbuffer))
#     print((mbuffer))
#     print(dist)
#     print('count_fallos',count_fallos)
#
#
#     mod=np.array(['x']*8)
#     union=np.concatenate([mod, mbuffer])
#     #union = union[:-8]
#     print('tanaño',len(union))
#
#     error_mask, locs = (buffer_vectores(union[0:buffer_size]))
#     #error_mask, locs = (buffer_vectores(mbuffer))
#
#     print('tamaño de error mask',len(error_mask))
#     print('tamaño de error locs',len(locs))
#
#     save_obj(error_mask,'MoRS/Modelo3_col_8_0.51/mask/error_mask_'+ str(fich))
#     save_obj(locs,'MoRS/Modelo3_col_8_0.51/mask/locs_'+str(fich))
#
#
#     DF_mask = pd.DataFrame(error_mask)
#     DF_locs = pd.DataFrame(locs)
#
#     Mask_locs = pd.concat([DF_mask, DF_locs], axis=1, join='outer')
#     Mask_locs.columns = ['error_mask','locs']
#     Mask_locs.to_excel('MoRS/Modelo3_col_8_0.51/Mask_locs/Mask_locs_' +str(fich) + '.xlsx', index=False)
#     fich= fich+1