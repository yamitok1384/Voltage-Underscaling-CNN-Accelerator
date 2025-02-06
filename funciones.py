#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

import tensorflow as tf
tf.config.run_functions_eagerly(True)

import numpy as np
import collections
import pandas as pd
from openpyxl import Workbook
from tipoError import ErrorAuno,ErrorAcero




def LayerSize(net,name):
    nput_shape_sizes = []
    layer_name = []
    activation_sizes = []

    for index, layer in enumerate(net.layers):
        layer_name.append(net.layers[index].__class__.__name__)
        activation_shape = layer.output_shape[1:]
        print('layer.output_shape[1:]', layer.output_shape[1:])
        total_activations = tf.reduce_prod(activation_shape).numpy()
        print('total_activations', total_activations)
        activation_sizes.append(total_activations)
        # print(index,layer.name)
        # print(index,layer.input_shape[1:])

    df_layer_name = pd.DataFrame(layer_name)
    #print('df_layer_name', df_layer_name)
    df_activation_sizes = pd.DataFrame(activation_sizes)
    #print('df_activation_sizes', df_activation_sizes)
    result = pd.concat([df_layer_name, df_activation_sizes], axis=1, join='outer')
    result.columns = ['layer_name', 'layer_size_ouput']
    result.to_excel(str(name) + '_layers.xlsx', index=False)


def DecimalToBinario(decimal):
    binario = ''
    dec_bin = []
    while decimal // 2 != 0:
        binario = str(decimal % 2) + binario
        decimal = decimal // 2

    return str(decimal) + binario

def buffer_vectores(buffer):
    print('soy el buffer dentro de funciones',len(buffer))
    print(' dentro de funciones Cantidad de elementos por tipo :', collections.Counter(buffer))
    #buffer_size=len(buffer)
    address_with_errors = np.reshape(buffer, (-1, 16))
    address_with_errors = ["".join(i) for i in address_with_errors]
    error_mask = [y for x, y in enumerate(address_with_errors) if y.count('x') < 16]
    locs = [x for x, y in enumerate(address_with_errors) if y.count('x') < 16]


    del address_with_errors
    return [error_mask, locs]

def buffer_vectores_mask(buffer):
    print('estoy dentro')
    buffer_size=len(buffer)
    address_with_errors = np.reshape(buffer, (-1, 16))
    address_with_errors = ["".join(i) for i in address_with_errors]
    error_mask = [y for x,y in enumerate(address_with_errors) if y.count('x') < 16]
    locs       = [x for x,y in enumerate(address_with_errors) if y.count('x') < 16]
    del address_with_errors
    #address_with_errors = ["".join(i) for i in address_with_errors[0:2000]]
    #error_mask = [y for x, y in enumerate(address_with_errors) if y.count('x') < 16]
    #locs = [x for x, y in enumerate(address_with_errors) if y.count('x') < 16]

    #del address_with_errors
    return error_mask


def error_word(error_mask):
    repeticiones = [0] * 16
    for vector in error_mask:
        for i,j in enumerate(vector):
            if j=='1':
                repeticiones[i]+=1
                
    #print(repeticiones)
    print('Cantidad de elementos',sum(repeticiones))
    return repeticiones





### Esta función devuelve los elementos iguales por capas.
def same_elements(outputs1,outputs2,ciclo,Net2):
    
    list_size_output=[]
    list_output_true=[]
    list_ratio=[]
    
    
    
    for index in range(0,len(outputs2)):
        
        print('Capa',index,Net2.layers[index].__class__.__name__)
        a=outputs1[index]== outputs2[index]
        size_output=a.size
        #print('Cantidad de elementos de la capa:',size_output)
        output_true=np.sum(a)
        list_output_true.append(output_true)
        list_size_output.append(size_output)
        
        #list_ciclo.append(ciclo)
        #print('Cantidad de elementos iguales entre modelo con fallos y sin fallos:', output_true)
        amount_dif=size_output-output_true
        #print('Cantidad de elementos diferentes entre modelo con fallos y sin fallos', amount_dif)
        ratio=(output_true*100)/size_output
        list_ratio.append(ratio)
        #print('Ratio:', ratio,'%')
        df_list_size_output=pd.DataFrame(list_size_output)
        df_list_output_true=pd.DataFrame(list_output_true)
        df_list_ratio=pd.DataFrame(list_ratio)
        df_list_capas=pd.DataFrame(list_ratio) 
        buf_same_elemen = pd.concat([df_list_size_output,df_list_output_true, df_list_ratio], axis=1, join='outer')
        buf_same_elemen.columns = ['Total_elements_layer', 'Same_elements', 'Ratio']
        #buf_same_elemen.to_excel(writer, sheet_name='ratio_'+ str(ciclo), index=False)
        return buf_same_elemen



###Esta función me permite saber dada un máscar ade fallos cuantas hay en cada clasificación de HO, LO
## y para L&H order

def WordType(error_mask,locs):
    print('Estoy dentro de WordType')
    count_bit_les = 0
    coun_bit_more = 0
    count_bit_conf = 0
    count_perfect_word = 0
    perfect_word = 0
    error_mask_ana = []
    error_mask_ana_inv = []
    error_mask_ana_antes = []
    locs_LO = []
    locs_HO = []
    locs_LHO = []
    for i, j in enumerate(error_mask):
        # print(i)
        bit_more = j[0:8].count('0') + j[0:8].count('1')
        bit_les = j[8:16].count('0') + j[8:16].count('1')

        if bit_les > 0 and bit_more == 0:
            count_bit_les += 1
            error_mask_ana.append(j)
            locs_LO.append(locs[i])



        elif bit_les == 0 and bit_more > 0:
            coun_bit_more += 1
            error_mask_ana.append(j)
            locs_HO.append(locs[i])

        elif bit_les == 0 and bit_more == 0:

            error_mask_ana.append(j)
            count_perfect_word += 1


        else:

            count_bit_conf += 1
            error_mask_ana.append(j)
            locs_LHO.append(locs[i])
            #locs_bin.append(format((locs[i]), "b"))
    print('count_bit_les', count_bit_les)
    print('coun_bit_more', coun_bit_more)
    print('count_bit_conf', count_bit_conf)
    print('count_perfect_word', count_perfect_word)
    print('LO', len(locs_LO))
    print('HO', len(locs_HO))
    print('LHO', len(locs_LHO))
    print('suma de perf wored y cpunt_bit_less debe ser igual qu ela máscara',count_bit_les+ count_perfect_word )
    print('Tamaño mascara final', len(error_mask_ana))
    return  locs_LO, locs_HO, locs_LHO


#Deja los fallos en el byte menos  significativo de la palabra , corrige las VBW(las palabras)
#que tienen fallos tanto en el bit más significativo como en el menos significativo
#el objetivo es calcular el acc con esta máscra para saber cuanto afectan las palabras clasificadas como LO
#(low order)

def LowOrder(error_mask,locs):
    print('estoy dentro LowOrder')


    count_bit_les = 0
    coun_bit_more = 0
    count_bit_conf = 0
    error_mask_ana = []
    locs_ana = []
    locs_LO = []
    locs_HO = []
    locs_H_L_O = []
    for i, j in enumerate(error_mask):
        bit_more = j[0:8].count('0') + j[0:8].count('1')
        bit_les = j[8:16].count('0') + j[8:16].count('1')
        if bit_les > 0 and bit_more == 0:
            count_bit_les += 1
            j = str("".join(j[0:8].replace('0', 'x'))) + str(j[8:16])
            j = str("".join(j[0:8].replace('1', 'x'))) + str(j[8:16])
            # j = str("".join(j.replace('0' ,'x')))
            # j = str("".join(j.replace('1','x')))
            error_mask_ana.append(j)
            #print('saber porque da fuera de rango',locs[i])
            locs_LO.append(locs[i])



        elif bit_les == 0 and bit_more > 0:
            coun_bit_more += 1
            j = str("".join(j[0:8].replace('0', 'x'))) + str(j[8:16])
            j = str("".join(j[0:8].replace('1', 'x'))) + str(j[8:16])
            error_mask_ana.append(j)
            locs_HO.append(locs[i])

        else:
            # dejarlas en xxxxxx
            count_bit_conf += 1
            k = 0
            stop = False
            j = str("".join(j[0:16].replace('0', 'x')))
            j = str("".join(j[0:16].replace('1', 'x')))
            error_mask_ana.append(j)
            locs_H_L_O.append(locs[i])

            k += 1
    return error_mask_ana,locs,count_bit_les

#Deja los fallos en el byte más significativo de la palabra , corrige las VBW(las palabras)
#que tienen fallos tanto en el bit más significativo como en el menos significativo
#el objetivo es calcular el acc con esta máscra para saber cuanto afectan las palabras clasificadas como HO
#(high order)
def HighOrder(error_mask,locs):
    print('estoy dentro HighOrder')
    count_bit_les = 0
    coun_bit_more = 0
    count_bit_conf = 0
    error_mask_ana = []
    locs_ana = []
    locs_LO = []
    locs_HO = []
    locs_H_L_O = []
    for i, j in enumerate(error_mask):
        # print(i)
        bit_more = j[0:8].count('0') + j[0:8].count('1')
        bit_les = j[8:16].count('0') + j[8:16].count('1')
        if bit_les > 0 and bit_more == 0:
            count_bit_les += 1
            j = str(j[0:8]) + str("".join(j[8:16].replace('0', 'x')))
            j = str(j[0:8]) + str("".join(j[8:16].replace('1', 'x')))
            error_mask_ana.append(j)

        elif bit_les == 0 and bit_more > 0:
            coun_bit_more += 1
            j = str(j[0:8]) + str("".join(j[8:16].replace('0', 'x')))
            j = str(j[0:8]) + str("".join(j[8:16].replace('1', 'x')))
            error_mask_ana.append(j)
        else:
            # dejarlas en xxxxxx
            count_bit_conf += 1
            k = 0
            stop = False
            j = str("".join(j[0:16].replace('0', 'x')))
            j = str("".join(j[0:16].replace('1', 'x')))
            error_mask_ana.append(j)
            #locs_ana.append(locs[i])

            k += 1
    return error_mask_ana, locs,coun_bit_more
## Deja solo los errores en aquellas palabras que tienen fallos tanto en el byte menos significativo
##como en el más significativo, las demas las coloca todas en 'xxxx'
def VeryBadWords(error_mask,locs):
    print('estoy dentro VeryBadWords')
    count_bit_les = 0
    coun_bit_more = 0
    count_bit_conf = 0
    count_perfect_word = 0
    error_mask_ana = []
    error_mask_H_L_O = []
    index_locs_VBW =[]
    locs_ana = []
    locs_LO = []
    locs_HO = []
    locs_H_L_O = []
    for i, j in enumerate(error_mask):
        # print(i)
        bit_more = j[0:8].count('0') + j[0:8].count('1')
        bit_les = j[8:16].count('0') + j[8:16].count('1')
        perfect_word = j[0:16].count('0') + j[0:16].count('1')



        if bit_les > 0 and bit_more == 0:
            count_bit_les += 1
            j = str("".join(j.replace('0', 'x')))
            j = str("".join(j.replace('1', 'x')))
            error_mask_ana.append(j)
        elif bit_les == 0 and bit_more > 0:
            coun_bit_more += 1
            j = str("".join(j.replace('0', 'x')))
            j = str("".join(j.replace('1', 'x')))
            error_mask_ana.append(j)
        elif perfect_word == 0:
            count_perfect_word += 1
            error_mask_ana.append(j)
            locs_ana.append(locs[i])

        else:

            count_bit_conf += 1
            k = 0
            stop = False
            error_mask_ana.append(j)
            error_mask_H_L_O.append(j)
            locs_ana.append(locs[i])
            locs_H_L_O.append(locs[i])
            index_locs_VBW.append(i)
            k += 1
    print('perfect_word',perfect_word)
    print('count_bit_les', count_bit_les)
    print('coun_bit_more', coun_bit_more)
    print('count_bit_conf', count_bit_conf)
    print('index_locs_VBW', len(index_locs_VBW))

    return (error_mask_H_L_O,locs_H_L_O,index_locs_VBW)

# Esta función me devuelve la máscara con las LO y las HO en xxxx y solo las palabra clasificadas como L&HO con los fallos , debo
# pensar en fucionarla con la de arriba que es lo mismo solo qu eme retorna mas parametros
def MaskVeryBadWords(error_mask,locs):
    print('estoy dentro VeryBadWords')
    count_bit_les = 0
    coun_bit_more = 0
    count_bit_conf = 0
    count_perfect_word = 0
    error_mask_ana = []
    error_mask_H_L_O = []
    index_locs_VBW =[]
    locs_ana = []
    locs_LO = []
    locs_HO = []
    locs_H_L_O = []
    for i, j in enumerate(error_mask):
        # print(i)
        bit_more = j[0:8].count('0') + j[0:8].count('1')
        bit_les = j[8:16].count('0') + j[8:16].count('1')
        perfect_word = j[0:16].count('0') + j[0:16].count('1')



        if bit_les > 0 and bit_more == 0:
            count_bit_les += 1
            j = str("".join(j.replace('0', 'x')))
            j = str("".join(j.replace('1', 'x')))
            error_mask_ana.append(j)
        elif bit_les == 0 and bit_more > 0:
            coun_bit_more += 1
            j = str("".join(j.replace('0', 'x')))
            j = str("".join(j.replace('1', 'x')))
            error_mask_ana.append(j)
        elif perfect_word == 0:
            count_perfect_word += 1
            error_mask_ana.append(j)
            locs_ana.append(locs[i])

        else:

            count_bit_conf += 1
            k = 0
            stop = False
            error_mask_ana.append(j)
            error_mask_H_L_O.append(j)
            locs_ana.append(locs[i])
            locs_H_L_O.append(locs[i])
            index_locs_VBW.append(i)
            k += 1
    print('perfect_word',perfect_word)
    print('count_bit_les', count_bit_les)
    print('coun_bit_more', coun_bit_more)
    print('count_bit_conf', count_bit_conf)
    print('index_locs_VBW', len(index_locs_VBW))
    WordType(error_mask_ana, locs)
    word_chanced=count_bit_les+coun_bit_more+count_perfect_word
    print('palabras todo en xx',word_chanced)

    return (error_mask_ana,locs,word_chanced)
### Esta función mantiene las palabra LO y a las Ho les hago flip y a las LHO las corrijo

def L0flippedHO(mask,locs):
    print('Dentro L0flippedHO')
    #np_mask = np.array(mask)
    #np_locs = np.array(locs)


    'Función  L0flippedHO'
    #mask_flip, locs,a,b = Flip(mask, locs)

    count_bit_les = 0
    coun_bit_more = 0
    count_bit_conf = 0
    count_perfect_word = 0
    error_mask_L0flippedHO = []
    errormask_HO=[]
    locs_ana = []
    locs_LO = []
    locs_HO = []
    #locs_HO = np.empty((0,))
    locs_H_L_O = []
    for i, j in enumerate(mask):
        bit_more = j[0:8].count('0') + j[0:8].count('1')
        bit_les = j[8:16].count('0') + j[8:16].count('1')
        perfect_word = j[0:16].count('0') + j[0:16].count('1')
        if bit_les > 0 and bit_more == 0:
            count_bit_les += 1
            # j = str("".join(j[0:8].replace('0', 'x'))) + str(j[8:16])
            # j = str("".join(j[0:8].replace('1', 'x'))) + str(j[8:16])
            # # j = str("".join(j.replace('0' ,'x')))
            # # j = str("".join(j.replace('1','x')))
            error_mask_L0flippedHO.append(j)
            #locs_LO.append(locs[i])



        elif bit_les == 0 and bit_more > 0:
            coun_bit_more += 1
            # j = str("".join(j[0:8].replace('0', 'x'))) + str(j[8:16])
            # j = str("".join(j[0:8].replace('1', 'x'))) + str(j[8:16])
            error_volteado = (j[::-1])
            error_mask_L0flippedHO.append(error_volteado)
            #errormask_HO.append(j)
            #print('hay paralabra con bit_more')
            #locs_HO = np.append(locs_HO, locs[i])
            locs_HO.append(i)
        elif perfect_word == 0:
            count_perfect_word += 1
            error_mask_L0flippedHO.append(j)



        else:
            # dejarlas en xxxxxx
            count_bit_conf += 1
            k = 0
            stop = False
            j = str("".join(j[0:16].replace('0', 'x')))
            j = str("".join(j[0:16].replace('1', 'x')))
            error_mask_L0flippedHO.append(j)
            #locs_H_L_O.append(locs[i])

            k += 1
    print('coun_bit_more',coun_bit_more)
    print('count_bit_les', count_bit_les)
    print('count_bit_conf', count_bit_conf)
    print('count_perfect_word', count_perfect_word)
    print('tamaño de la mascara optenida',len(error_mask_L0flippedHO))
    print('tamaño de los elementos de locs_HO', len(locs_HO))

    # HO_flip,a,b,c=Flip(errormask_HO,locs)
    # print('tamaño de la HO_flip debe ser igual que count_bit_conf', len(HO_flip))
    # for indice, new_value in zip(locs_HO, HO_flip):
    #     #print('indice',indice)
    #     error_mask_L0flippedHO[indice] = new_value


    #error_mask_L0flippedHO[locs_HO] = HO_flip
    #print('tamaño de la mascara optenida', len(error_mask_L0flippedHO))
    word_change=len(locs_HO)
    WordType(error_mask_L0flippedHO,locs)
    return error_mask_L0flippedHO,locs,word_change


#Con esta función se calcula el acc de las redes pasandole la máscara base
#Se le psa una máscara y la develve sin hacerle modificación , la he creadopara poder
#comlocar todos los experimentos en un ciclo
def Base(error_mask, locs):
    words=len(error_mask)/len(locs)
    if words==1:

        return error_mask,locs,0,0

#Iso-area ECC :Tomar las direcciones de 4 en 4 y si existen en estas máscaras 1 error lo corrijo, sino lo dejo
#igual, las direcciones van desde 0 hasta (1048576-1)(1024x1024x1)

def IsoAECC(error_mask, locs):
    error_mask_IA_ecc=error_mask
    ecc = 0
    t_adress = 1048576
    locs_modi = []

    for i in range(0, t_adress, 4):
        # print ('i',i)
        error = 0
        z = 0

        for j in range(i, i + 4):
            # print('i+4',i+4 )
            # print('j', j)
            if j in locs:
                z = locs.index(j)
                # print('z',z)

                error = error + (error_mask_IA_ecc[z].count('0') + error_mask_IA_ecc[z].count('1'))
            # print('error', error)
        if error == 1:
            # print('z',z)

            # print('mask_error[z]', error_mask[z])
            error_mask_IA_ecc[z] = 'xxxxxxxxxxxxxxxx'
            # print('mask_error[z]', error_mask[z])
            locs_modi.append(z)
            # print('locs_modi',locs_modi)
            ecc += 1

            # print('ecc', ecc)

    #print(ecc)
    #print(len(error_mask))

    print('cantidad de elementos a los que se le aplicó la técnica', ecc)


    return error_mask_IA_ecc, locs,locs_modi,ecc


##ECC: Las palabras del modelo de fallo  con un error colocarla
## todo en 'xxxxxxxxxxxxxxxx', es simular el Codigo de Corrección de errore sconvencional
def ECC(error_mask, locs):
    count_one_err = 0
    coun_more_err = 0
    error_mask_ecc = []
    loc_ecc=[]


    for i, j in enumerate(error_mask):
        # print(i)
        mask_one_error = j[0:16].count('x')
        if mask_one_error == 15:
            count_one_err += 1
            j = str("".join(j.replace('0', 'x')))
            j = str("".join(j.replace('1', 'x')))
            error_mask_ecc.append(j)
            loc_ecc.append(locs[i])


        else:
            coun_more_err += 1
            error_mask_ecc.append(j)
    print('Cantidad de palabras corregidas con ECC', count_one_err )
    return error_mask_ecc,locs,loc_ecc,count_one_err

##Esta fución aplica la técnica de volteo a las máscaras con esta técnica lo que hacemos es convertir voltear
#aquellas palabras que tengan la parte alta afectada para reordenada de forma tal que los errores no afecten en demasia
#la precición de la Red

def Flip(error_mask, locs):
    error_mask_flip = []
    locs_flip = []
    count_flip = 0
    locs_sin_flip = []
    error_mask_flip_ = []
    count_vbw=0
    # k = 0
    # marca ='*'
    for i, j in enumerate(error_mask):
        k = 0
        stop = False
        while k < 8 and stop == False:
            if ('0' in j[k] or '1' in j[k]) and 'x' in j[15 - k]:
                error_volteado = (j[::-1])
                # print('error_volteado',error_volteado)
                error_mask_flip.append(error_volteado)
                count_flip = count_flip + 1
                flip_locs = locs[i]
                locs_flip.append(flip_locs)
                stop = True

            else:
                if '0' in j[15 - k] or '1' in j[15 - k]:
                    error_mask_flip.append(j)
                    locs_sin_flip.append(locs[i])

                    stop = True
                k += 1

        if stop == False:
            error_mask_flip.append(j)
            locs_sin_flip.append(locs[i])
            #error_mask_flip_.append(j)


    print('cantidad  volteada', count_flip)
    print('tamaño de Máscara volteada ', len(locs_flip))
    print('Tamaño máscara', len(error_mask_flip))
    print('Tamaño locs', len(locs))
    return error_mask_flip, locs,locs_flip, count_flip


# Eta función optiene las direcciones y la máscara de aquellas palabras clasificadas
#como L&HO, es decir aquellas que tienen errores tanto en la parte alta como el la parte baja de
#la palabra , también nos da el vias (módulo de la dirección entre 256) para saber en que bram estará e
#el fallo
def StaticIndexVbw(error_mask,locs):
    print('StaticIndexVbw')
    print('error_mask',len(error_mask))
    print('locs',len(locs))
    count_bit_les = 0
    coun_bit_more = 0
    count_bit_conf =0
    error_mask_H_L_O =[]
    locs_H_L_O =[]
    vias =[]
    cant_elem=[]
    for i, j in enumerate(error_mask):
        bit_more = j[0:8].count('0') + j[0:8].count('1')
        bit_les = j[8:16].count('0') + j[8:16].count('1')
        if bit_les > 0 and bit_more == 0:
            count_bit_les += 1

        elif bit_les == 0 and bit_more > 0:
            coun_bit_more += 1

        else:
            #print('vbw dendro de index', j)
            count_bit_conf += 1
            error_mask_H_L_O.append(j)
            locs_H_L_O.append(locs[i])
            vias.append(locs[i] % 256)

    unique, counts = np.unique(vias, return_counts=True)
    # print(dict(zip(unique, counts)))
    b = np.asarray(np.where(counts > 5))
    cant_elem.append(b.size)
# puedo retornar o guardar estadísticas aquí si lo considero necesario
#o proseguir para optener los indices de los elementos que serán cambiado
#al hacer pacth
    #return error_mask_H_L_O, locs_H_L_O, vias
    new_mask = []
    locs_modif = []
    locs_ok = []
    mask_ok = []
    marca = '*'
    for i in range(256):
        count = 0
        for j, k in enumerate(error_mask_H_L_O):
            #print('vbw', k)
            if vias[j] == i:
                if count < 5:
                    locs_modif.append(locs_H_L_O[j])
                    # mask_array[j].append(k)
                    # new_mask=str(k) + marca
                    # mask_ok.append(new_mask)
                    # locs_ok.append(new_locs)
                    count = count + 1
                    #print('count_sumador', count)
                else:
                    if vias[j - 1] == vias[j]:
                        #print('es igual y guardo aquí')
                        locs_ok.append(locs[j])
    print('locs_modif saliente', len(locs_modif))
    return locs_modif

##Esta función luego de realizar el Flip  corrige  aquellas palabras que tienen
##errore stanto en el byte más significativo como en el menos de la palabra(vbw), pero solo las que caben en las 5 vias del diseño
## ePrimero hago flip y luego hago el patch
#
def FlipPatchBetter(error_mask, locs):
    print('FlipPatchBetter')
    print('error_mask', len(error_mask))
    print('locs', len(locs))
    locs_modif = StaticIndexVbw(error_mask, locs)
    print('locs_modif',len(locs_modif))

    mask_flip, locs, locs_flip,count_flip = Flip(error_mask, locs)
    #mask_flip, locs, count_flip = Flip(error_mask, locs,mask_flip)
    count_bit_conf = 0
    error_mask_patch = []
    verif = []
    locs_patch = []
    locs_LO = []
    for index, item in enumerate(mask_flip):
        #print('index inicial',index)

        if locs[index] in locs_modif:
            #print('index locs', index)
            item_new = str("".join(item[0:16].replace('0', 'x')))
            item_new = str("".join(item_new[0:16].replace('1', 'x')))
            #print('item_new', item_new)
            #print('error_mask antes', mask_flip[index] )
            mask_flip[index] = item_new
            #print('index', index)
            #print('error_mask despues', mask_flip[index])
            error_mask_patch.append(mask_flip)
            locs_patch.append(locs[index])
            verif.append(index)
            count_bit_conf += 1
            #print('count_bit_patch',count_bit_conf)
        else:
            #print('index else', index)
            mask_flip[index] = item
    error_mask_patch=mask_flip
    print('tamaño del las direcciones con Patch', len(locs_patch))
    print('tamaño del las direcciones locs_modif', len(locs_modif))
    ###Para verificar
    print('total de palabras que se le hizo Patch dentro funcion patch better', count_bit_conf)
    print('total de palabras que se le hizo Patch dentro funcion patch better', len(verif))
    print('Tamaño de la máscara que se obtiene debe ser del mismo tamaño funcion patch better', len(error_mask_patch))
    #FlipPatch(error_mask_patch,locs)


    return mask_flip, locs, locs_patch, count_bit_conf



## Esta función voltea las máscara para convertir las activaciones en LO  y luego corrijo todas las activaciones L&HO
# que según el voltaje pueden ser más de las que caben en la memoria patch, se ha hecho con el objetivo de medir la técnica
#para aprovechar el espacio no aprovechado por algunas capas de las redes para colocar las palabras clasificadas como L&HO
def ScratchPad(error_mask,locs):
    print('FlipPatch')

    mask_flip,locs,locs_flip,count_flip=Flip(error_mask,locs)
    #print('count_flip',vbw_not_flip)
    count_bit_conf_veri = 0
    count_bit_les = 0
    coun_bit_more = 0
    count_perfect_word = 0

    error_mask_patch = []
    locs_LO =[]
    locs_patch=[]
    for i, j in enumerate(mask_flip):
        # print(i)
        bit_more = j[0:8].count('0') + j[0:8].count('1')
        bit_les = j[8:16].count('0') + j[8:16].count('1')
        perfect_word = j[8:16].count('0') + j[8:16].count('1')
        if bit_les > 0 and bit_more == 0:
            count_bit_les += 1
            error_mask_patch.append(j)
            locs_LO.append(locs[i])
        elif bit_les == 0 and bit_more > 0:
            coun_bit_more += 1
            error_mask_patch.append(j)

        elif perfect_word == 0:
            count_perfect_word += 1
            error_mask_patch.append(j)
            #locs_ana.append(locs[i])

        else:
            #print('dentro de la veruficación  deben heber  palabras', j)
            #print('patch antes', j)
            count_bit_conf_veri += 1
            #print('count_bit_conf_veri',count_bit_conf_veri)
            #print(j)
            # error_mask_ana.append(j)
            j = str("".join(j[0:16].replace('0', 'x')))
            j = str("".join(j[0:16].replace('1', 'x')))
            error_mask_patch.append(j)
            #print('patch despues', j)
            locs_patch.append(locs[i])
            #locs_ana_.append(locs[i])
            #locs_bin.append(format((locs[i]), "b"))
    print('total de palabras todo en xxxxxx', count_perfect_word)
    print('total de palabras que  se le hizo Patch', count_bit_conf_veri)
    print('total de palabras fallos HO(debe estar en 0 porque se le hizo FLIP)', coun_bit_more)
    print('total de palabras fallos LO(Al sumar con las del patch + las que no s ele hizo  patch = tamaño de la máscara)', count_bit_les)
    print('tamaño mask patch' , len(error_mask_patch))
    print('tamaño locs patch en este momento son toda xxxxx', len(locs_patch))
    print('tamaño locs LO en este momento son las LO', len(locs_LO))

    return error_mask_patch,locs,locs_LO,count_bit_conf_veri
    #return error_mask_patch,locs,count_bit_conf_veri

### Esta función despaza el error d ela palabra tantas posiciónes como sean indicadas por el usuario
#shift: contador para indicar la cantidad deelementos qu ehe movido y luego incrementarlo con x delante de la palabra
#count_bit_remove: para saber la cantidad de veces que se ha corregido una palabra de la máscara
#words_fallos: cantidad de palabras de la máscara que continuan con fallos luego de hacer el Shift

#Esta función devuelve una mácara con los fallos solo en aquellas parabras clasificadas como LHO, que no caben en la Patch Cache
#Para  llamo a la función FlipPatchBetter(obtendré una máscra donde estaran correidas la spalabras que caben en la patch cache)
def VBWGoToScratch(error_mask, locs):
    print('Estoy dentro de VBWGoToScratch')

    mask_flip, locs, locs_patch, count_bit_conf = FlipPatchBetter(error_mask, locs)
    WordType(error_mask, locs)
    error_mask, locs,locs_patch= MaskVeryBadWords(mask_flip, locs)
    WordType(error_mask, locs)
    return error_mask, locs, locs_patch



#Esta función luego de realizar el Flip  corrige todas  aquellas palabras que tienen
#errore stanto en el byte más significativo como en el menos de la palabra
# es mejor pasar la máscara ya volteada es , decir la máscara que se optiene luego d eaplicar el FLIP

def ShiftMask(error_mask,p):
    shift = 0
    count_bit_remove = 0
    mask_new = []
    for i, j in enumerate(error_mask):
        posicion = p
        #print('i',i)
        #print('j', j)
        lista = list(j)
        test = lista[16 - posicion:16].count('0') + lista[16 - posicion:16].count('1')
        #print('test', test)
        if test > 0:
            a = ''.join(lista)
            mask_new.append(a)
        else:

            while posicion > 0:
                lista.pop()
                # disminuyo las posiciones
                posicion = posicion - 1
                shift += 1  ##incremento contador para agregar los elementos que he quitado como x delante

            a = 'x' * shift + ''.join(lista)
            bit_remove = a[0:16].count('0') + a[0:16].count('1')
            mask_new.append(a)
            shift = 0
            if bit_remove == 0:
                count_bit_remove += 1
        #print('a', a)
    words_fallos= len(mask_new)- count_bit_remove
        #list_words_fallos.append(list_words_fallos)

        #print(len(a))
    #print(error_mask)
    #print(mask_new)
    print('La cantidad de palabras todas de palabras fallos que permanecen',words_fallos)
    return mask_new,words_fallos
##
# def ShiftMask(error_mask,p):
#     shift = 0
#     count_bit_remove = 0
#     mask_new = []
#
#
#     for i, j in enumerate(error_mask):
#         posicion = p
#         #print('i',i)
#         #print('j', j)
#         lista = list(j)
#         while posicion > 0:
#             lista.pop()
#             # disminuyo las posiciones
#             posicion = posicion - 1
#             shift += 1  ##incremento contador para agregar los elementos que he quitado como x delante
#
#         a = 'x' * shift + ''.join(lista)
#         bit_remove = a[0:16].count('0') + a[0:16].count('1')
#         mask_new.append(a)
#         shift = 0
#         if bit_remove == 0:
#             count_bit_remove += 1
#         words_fallos= len(mask_new)- count_bit_remove
#         #list_words_fallos.append(list_words_fallos)
#         #print('a', a)
#         #print(len(a))
#     #print(error_mask)
#     #print(mask_new)
#     print('La cantidad de palabras todas de palabras fallos que permanecen',words_fallos)
#     return mask_new,words_fallos
#

def Shift(error_mask,locs):
    error_mask_shift, locs, locs_patch, word_change = ScratchPad(error_mask, locs)
    return error_mask_shift, locs, locs_patch, word_change


from Simulation import get_all_outputs

def TestBinsAllActvs(write_layer,test_dataset,Net):
    np_count = np.full(18, 0)
    bins = [-256, -128, -64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64, 128, 256]

    for i, j in enumerate(write_layer):
        print('.........................................', j)
        index = 0
        stop = False

        iterator = iter(test_dataset)
        # while index < len(test_dataset) and stop==False:
        while index < 750:
            print('index............................', index)
            image = next(iterator)[0]
            outputs = get_all_outputs(Net, image)
            print('outputs[j].size', outputs[j].size)
            output = outputs[j]
            output.flatten(order='F')
            datos = output

            counts, bin_edges = np.histogram(datos, bins)
            intervalo = []
            contar = []
            for low, hight, count in zip(bin_edges, np.roll(bin_edges, -1), counts):
                print(f"{count}")
                intervalo.append(f'{low}-{hight}')
                # rint('size',count.size)
                contar.append(count)

                np_contar = np.array(contar)
            np_count = np_count + np_contar
            print('np_count', np_count)
            index = index + 1
    return intervalo, np_count

def TestBins(write_layer,test_dataset,Net,locs_LO):
    np_count = np.full(18, 0)
    bins = [-256,-128,-64,-32,-16,-8,-4,-2,-1,0,1 ,2, 4, 8,16,32,64,128,256]

    for i, j in enumerate(write_layer):
        print('.........................................', j)
        index = 0
        stop = False

        iterator = iter(test_dataset)
        # while index < len(test_dataset) and stop==False:
        while index < 750:
            print('index............................', index)
            image = next(iterator)[0]
            outputs = get_all_outputs(Net, image)
            # print('outputs[j].size',outputs[j].size)
            output = outputs[j]
            output.flatten(order='F')
            datos = output
            actvs = output.size
            # print('tamaño de las actvs', actvs)
            locs_lo = np.array(locs_LO)
            locs_affected = locs_lo[(locs_lo < output.size)]
            # print('tamalo de lo afectado', len(locs_affected))
            locs_size = len(locs_affected)
            # print('tamaño de locs_ffected', locs_size)
            affectedValues = np.take(output, locs_affected)
            # counts, bin_edges = np.histogram(datos, bins)
            counts, bin_edges = np.histogram(affectedValues, bins)
            intervalo = []
            contar = []
            for low, hight, count in zip(bin_edges, np.roll(bin_edges, -1), counts):
                print(f"{count}")
                intervalo.append(f'{low}-{hight}')
                # rint('size',count.size)
                contar.append(count)

            np_contar = np.array(contar)
            np_count = np_count + np_contar
            #print('np_count', np_count)
            index = index + 1
    return intervalo,np_count


def TensorUpdatePosicionInicial(tensor_act,val,valores_afectados,mask_0,tensor_with_error,faults,index):
    print('Dentro de TensorUpdatePosicionInicial')
    #val = tf.experimental.numpy.array(val)
    print('val',val)
    print('type(val)',type(val))
    tensor_act = tf.tensor_scatter_nd_update(tensor_act, tf.gather_nd(tf.where(valores_afectados==True), index),                  tf.convert_to_tensor([0]*tf.size(val) ))
    print('tensor_act tf',tensor_act)
    #tensor_act = tf.tensor_scatter_nd_update(tensor_act, val, tf.convert_to_tensor([0]*np.size(val) ))
    #valores_afectados=tf.tensor_scatter_nd_update(valores_afectados,(tf.where(valores_afectados), index),tf.convert_to_tensor([False]*tf.size(val)))
    #print('valores_afectados dentro de error 0 en primera posición',valores_afectados )
    mask_0=tf.add(mask_0,tf.size(val))
    print('tensor_act funciion a 0',tensor_act)
    print('mask_0',mask_0)
    print('valores_afectados',valores_afectados)

    #tensor_with_error_run= tf.gather_nd(tensor_with_error,tf.where(valores_afectados ==True))
    v_b=tf.gather_nd(tensor_act,tf.where(valores_afectados==True))
    print('v_b',v_b)
    v_m=tf.gather_nd(tensor_with_error,tf.where(valores_afectados==True) )
    print('v_m',v_m)
    valores_afec_static_0_Error=tf.gather_nd(faults[:,0],tf.where(valores_afectados==True))
    print('valores_afec_static_0_Error', valores_afec_static_0_Error)
    valores_afec_static_1_Error=tf.gather_nd(faults[:,1],tf.where(valores_afectados==True))
    print('valores_afec_static_1_Error', valores_afec_static_0_Error)
    return tensor_act,valores_afectados,mask_0,v_b,v_m,valores_afec_static_0_Error,valores_afec_static_1_Error

def WeightsFault(model):
    print('ok')




from random import seed
from random import sample

# def DeleteTercioRamdom(error_mask,locs, index_locs_VBW, semilla=3):
#     print('tamaño del index_locs_VBW',len(index_locs_VBW ))
#     conta = 0
#     seed(semilla)
#     #cantidad_eliminar = len(index_locs_VBW)*2// 3
#     cantidad_eliminar=16324
#     indices_a_eliminar = [choice(index_locs_VBW) for _ in range(cantidad_eliminar)]
#     elementos_eliminados = []
#     for index_locs_VBW in sorted(indices_a_eliminar, reverse=True):
#         locs_eliminada = locs.pop(index_locs_VBW)
#         mask_eleiminada=error_mask.pop(index_locs_VBW)
#         elementos_eliminados.append((index_locs_VBW, mask_eleiminada,locs_eliminada))
#         conta=conta + 1
#     print('cantidad de elementos eliminados',conta)
#     print('tamaño de la máscara',len(error_mask))
#     print('tamaño de la locs', len(locs))
#     return error_mask, locs

###Esta función elimina 1/3 de las palabras HLO, ya que son demasiadas
def DeleteTercioRamdom(error_mask,locs, locs_H_L_O, semilla=3):
    print('tamaño del index_locs_VBW',len(locs_H_L_O ))
    conta = 0
    seed(semilla)
    cantidad_eliminar = len(locs_H_L_O)*2// 3
    print(cantidad_eliminar)
    indices_a_eliminar = sample(locs_H_L_O,cantidad_eliminar)
    #print(sorted(indices_a_eliminar))
    print(len(indices_a_eliminar))
    elementos_eliminados = []
    #for i in range(len(locs)):
    for i in sorted(indices_a_eliminar, reverse=True):
        #print('i',i)
        #if i in locs_H_L_O:
        locs_eliminada = locs.pop(i)
        mask_eleiminada=error_mask.pop(i)
        elementos_eliminados.append((i, mask_eleiminada,locs_eliminada))
        conta=conta + 1
    print('cantidad de elementos eliminados',conta)
    print('tamaño de la máscara',len(error_mask))
    print('tamaño de la locs', len(locs))
    return error_mask, locs




def MaskLessVBW(error_mask,locs):
    count_bit_les = 0
    coun_bit_more = 0
    count_bit_conf = 0
    count_perfect_word = 0
    perfect_word = 0
    error_mask_ana = []
    mask_Less_VBW = []
    error_mask_ana_antes = []
    locs_LO = []
    locs_HO = []
    locs_LHO = []
    for i, j in enumerate(error_mask):
        # print(i)
        bit_more = j[0:8].count('0') + j[0:8].count('1')
        bit_les = j[8:16].count('0') + j[8:16].count('1')

        if bit_les > 0 and bit_more == 0:
            count_bit_les += 1
            mask_Less_VBW.append(j)
            locs.append(locs[i])



        elif bit_les == 0 and bit_more > 0:
            coun_bit_more += 1
            mask_Less_VBW.append(j)
            locs.append(locs[i])

        elif bit_les == 0 and bit_more == 0:

            mask_Less_VBW.append(j)
            locs.append(locs[i])
            count_perfect_word += 1


        else:

            count_bit_conf += 1
            error_mask_ana.append(j)
            locs_LHO.append(locs[i])
            #locs_bin.append(format((locs[i]), "b"))
    print('count_bit_les', count_bit_les)
    print('coun_bit_more', coun_bit_more)
    print('count_bit_conf', count_bit_conf)
    print('count_perfect_word', count_perfect_word)
    print('LO', len(locs_LO))
    print('HO', len(locs_HO))
    print('LHO', len(locs_LHO))
    print('suma de perf wored y cpunt_bit_less debe ser igual qu ela máscara',count_bit_les+ count_perfect_word )
    print('Tamaño mascara final', len(error_mask_ana))
    return  locs_LO, locs_HO, locs_LHO


def Layers(net,LI,AI,arquitectura):
    indice = []
    read_write_layers = []
    layer_size =[]
    write = []
    read = []
    write_size = []
    read_size = []

    for i, j in enumerate(net.layers):
        print('indice',i)
        print(net.layers[i].__class__.__name__)
        #print(np.prod(net.layers[i].input_shape[1:]).astype(np.int64))
        #if net.layers[i].__class__.__name__!= 'Concatenate':
        print('valorar la capa',net.layers[i].__class__.__name__== 'Concatenate')
        #print('AI', AI[i])


        if i in AI and net.layers[i].__class__.__name__!= 'Concatenate' :
            indice.append(i)
            read_write_layers.append(net.layers[i].__class__.__name__+ str('_') + str('write'))
            layer_size.append(np.prod(net.layers[i].input_shape[1:]).astype(np.int64))
            write.append(net.layers[i].__class__.__name__)
            write_size.append(np.prod(net.layers[i].input_shape[1:]).astype(np.int64))
        if i in LI and net.layers[i].__class__.__name__!= 'Concatenate':
            indice.append(i)
            read_write_layers.append(net.layers[i].__class__.__name__+ str('_') + str('read'))
            read.append(net.layers[i].__class__.__name__)
            layer_size.append(np.prod(net.layers[i].input_shape[1:]).astype(np.int64))
            read_size.append(np.prod(net.layers[i].input_shape[1:]).astype(np.int64))


    df_indice = pd.DataFrame(indice)
    df_write_read_layers = pd.DataFrame(read_write_layers)
    print(df_write_read_layers)
    def_size_layers= pd.DataFrame(layer_size)
    print(def_size_layers)
    df_write_read = pd.concat([df_indice,df_write_read_layers, def_size_layers], axis=1, join='outer')
    df_write_read.columns = ['Indice','Capa', 'tamaño']
    print(df_write_read)
    df_write_read.to_excel('MoRS/Analisis_Resultados/Energía_VBW/layers_analysis_read_writing_sizes/Capas_read_write_enviar_' + str(arquitectura) + '.xlsx', sheet_name='fichero_707', index=False)


    df_write_layers= pd.DataFrame(write)
    df_write_indice = pd.DataFrame(AI)
    df_read_size = pd.DataFrame(write_size)
    df_write_layers = pd.concat([df_write_indice,df_write_layers, df_read_size], axis=1, join='outer')
    df_write_layers.columns = ['Indice','Capa', 'tamaño']
    df_write_layers.to_excel('MoRS/Analisis_Resultados/Energía_VBW/layers_analysis_read_writing_sizes/Capas_write_enviar_' + str(arquitectura) + '.xlsx', sheet_name='fichero_707', index=False)

    df_read_layers = pd.DataFrame(read)
    df_read_indice = pd.DataFrame(LI)
    df_read_size = pd.DataFrame(read_size)
    df_read_layers = pd.concat([df_read_indice,df_read_layers, df_read_size], axis=1, join='outer')
    df_read_layers.columns = ['Indice','Capa', 'tamaño']
    df_read_layers.to_excel('MoRS/Analisis_Resultados/Energía_VBW/layers_analysis_read_writing_sizes/Capas_read_enviar_' + str(arquitectura) + '.xlsx', sheet_name='fichero_707', index=False)


# def ErrorAcero(values_to_change, mask, val_afec_0_Error_to_change, val_afec_1_Error_to_change):
#     print('dentro de error a 0')
#
#     # tensor_final=tf.where(tf.math.not_equal(vb_and_mask,vm_and_mask) )
#     n_mask = mask - 1
#     print('mask', mask)
#     print('n_mask', n_mask)
#     tensor_round = tf.bitwise.bitwise_or(values_to_change, n_mask)
#     print('n_v_b_m', tensor_round)
#     # tensor=tf.bitwise.bitwise_and(n_v_b_m,val_afec_0_Error_to_change)
#     # print('n_v_b_1', tensor)
#     # tensor_round=tf.bitwise.bitwise_or(tensor,val_afec_1_Error_to_change)
#     # print( 'tensor_round',tensor_round)
#     # tensor_round  = tf.where(tf.greater_equal(tensor_round,shift), shift-tensor_round , tensor_round )
#
#     return tensor_round
#
#
# # En este caso sele resta 1 a la máscara pero debe hacerse un not_lógico a la misma para para cuando se haga el btw_and
# # colocar el retso de los bit luego de la posición con error a 0, lo demás es como lo anterior.
# def ErrorAuno(values_to_change, mask, val_afec_0_Error_to_change, val_afec_1_Error_to_change):
#     print('dentro de error a 1`')
#
#     # tensor_final=tf.where(tf.math.not_equal(vb_and_mask,vm_and_mask) )
#     n_mask = mask - 1
#     print('mask', mask)
#     print('n_mask', n_mask)
#     not_mask = np.invert(np.array([n_mask], dtype=np.uint16))
#     print('not_mask', not_mask)
#     tensor_round = tf.bitwise.bitwise_and(values_to_change, not_mask)
#     print('n_v_b_m', tensor_round)
#     # tensor=tf.bitwise.bitwise_and(n_v_b_m,val_afec_0_Error_to_change)
#     # print('n_v_b_1', tensor)
#     # tensor_round=tf.bitwise.bitwise_or(tensor,val_afec_1_Error_to_change)
#     # tensor_round  = tf.where(tf.greater_equal(tensor_round,shift), shift-tensor_round , tensor_round )
#     return tensor_round

@tf.function
def TensorUpdateCiclo(error,v_b,c,valores_afec_static_0_Error,valores_afec_static_1_Error,mask,tensor_act,valores_afectados,mask_0):
    print('dentro de update ciclo')
    values_to_change=tf.gather_nd(v_b,c)
    val_afec_0_Error_to_change=tf.gather_nd(valores_afec_static_0_Error,c)
    print('val_afec_0_Error_to_change',val_afec_0_Error_to_change)
    val_afec_1_Error_to_change=tf.gather_nd(valores_afec_static_1_Error,c)
    print('val_afec_1_Error_to_change',val_afec_1_Error_to_change)
    print('values_to_change',values_to_change)
    if error==1:
        tensor_round=ErrorAuno(values_to_change,mask)
        print('tensor_round',tensor_round)
    if error==0:
        tensor_round=ErrorAcero(values_to_change,mask)
        print('tensor_round',tensor_round)
   # index=c.numpy()
    index = tf.experimental.numpy.array(c)

    print('indice a camniar tensor act',index)
    print('valores_afectados',valores_afectados)
    val=tf.gather_nd(tf.where(valores_afectados),index)## tomo los indices del tensor original qu ese cambiaran
    print('val',val)
    tensor_act = tf.tensor_scatter_nd_update(tensor_act, val, tensor_round )
          #valores_afectados=tf.tensor_scatter_nd_update(valores_afectados,tf.gather_nd(tf.where(valores_afectados),tf.convert_to_tensor([False]*tf.size(val)))
    mask_0=tf.add(mask_0,tf.size(val))

    print('mask_0 desdes update', mask_0)
    print('tensor tensor_act',tensor_act)
    print('valores_afectados update',valores_afectados)
    #del tensor_round
    #print('tensor_round',tensor_round)
    return tensor_act,valores_afectados,mask_0,val_afec_0_Error_to_change,val_afec_1_Error_to_change
    #return tensor_act,valores_afectados,mask_0


def compilNet(Net,wgt_dir,test_dataset,*args):
        print('ok')
        Net.load_weights(wgt_dir).expect_partial()
        loss = tf.keras.losses.CategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        Net.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
        acc = Net.evaluate(test_dataset)
        #acc_list.append(acc)
        return acc




         ##Este código crea una máscara de fallos de vectores de 16 elementos en 0, según la cantidad de direcciones con fallos cargadas
    #locs = load_obj(obj_dir_locs)
    
# def while_body(i,out,mask,mask_0,tensor_original,tensor_act,v_m,v_b,tensor_with_error,values_to_run,valores_afectados,valores_afec_static_0_Error,valores_afec_static_1_Error):
#     print('dentro del body')
#     print('i',i)
#     print('out que llega al body', out)
#     print('tensor_act',tensor_act)
#     print('tensor_original',tensor_original)
#     print('mask',mask)
#     print('mask_0',mask_0)
#     print('v_m',v_m)
#     print('v_b',v_b)
#     print('tensor_with_error',tensor_with_error)
#     #hay_cambios=tf.math.not_equal(tensor_original,tensor_act)
#     #print('hay_cambios',hay_cambios)
#     #stop_condition=tf.experimental.numpy.any(valores_afectados)
#     #if stop_condition==False:
#
#     ## Este if lo cree por si hay cambios antes de llegar al ciclo por haber errores e la posición 15 pero quizás ya o sea
#     # necesraio analizar luego de experimentar  si mask_0 llega con valores a 1 es porque se hicieron cambios y se deben
#     #analizar los btw con los valores resttantes sino todo se mantiene con los valores originales de entrada.
#     if tf.reduce_sum(mask_0)>0:
#         print('entre en condicion mask_0>0')
#         print('valores donde debo correr el mask')
#
#         vb_and_mask=tf.bitwise.bitwise_and(v_b,mask)
#         print('vb_and_mask',vb_and_mask)
#         vm_and_mask=tf.bitwise.bitwise_and(v_m,mask)
#         print('vm_and_mask',vm_and_mask)
#     else:
#
#          ### esta estructura cambiaria porque ya tendre solo los valores con problemas, tomaria valores de tensores variables
#          #inicializados en 0 hasta que tome sus valores luego del recorrido por el body
#          ## el tensor control tomara un papel aca porque luego actualizaria el tensor_actu en los posiciones donde
#              #tensorcontrol sea false mas otra condicion mas o debo ir comparando el tensor actu con el original y envio los dos
#         #valores_afectados=tf.math.not_equal(tensor_original,tensor_with_error)## inicialmente no hara nada xq son iguales
#         print('entre en el else')
#         vb_and_mask=tf.bitwise.bitwise_and(tf.gather_nd(tensor_original,tf.where(valores_afectados==True)),mask)
#         print('vb_and_mask',vb_and_mask)
#         vm_and_mask=tf.bitwise.bitwise_and(tf.gather_nd(tensor_with_error,tf.where(valores_afectados==True)),mask)
#         print('vm_and_mask',vm_and_mask)
#         print('v_b',v_b)
#         print('v_m',v_m)
#
#     a=tf.math.not_equal(vb_and_mask,vm_and_mask)
#     print('estoy calculando a y definiendo xq esta dado el error')
#     print('a vb_and_mask!=vm_and_mask',a)
#     values_to_run=tf.gather_nd(v_b,tf.where(a==False))
#     values_to_run_with_error=tf.gather_nd(v_m,tf.where(a==False))
#     valores_afec_static_0_Error_to_run=tf.gather_nd(valores_afec_static_0_Error,tf.where(a==False))
#     print('valores_afec_static_0_Error_to_run',valores_afec_static_0_Error_to_run)
#     valores_afec_static_1_Error_to_run=tf.gather_nd(valores_afec_static_1_Error,tf.where(a==False))
#     print('valores_afec_static_1_Error_to_run',valores_afec_static_1_Error_to_run)
#     print('values_to_run', values_to_run)
#     print(' values_to_run_with_error', values_to_run_with_error)
#     #valores_afectados_to_run= valores_afectados
#     #u=tf.size(values_to_run)>0
#
#
#     b=tf.math.greater(v_m,v_b ) # El error es a 1 y aplico la variante con_error_a_1
#     print('b Vm>Vb',b)
#     c=tf.where(tf.logical_and(a,b)==True)
#     #c=tf.where(tf.equal(a,b))
#     print('tamaño de c',tf.size(c))
#     print(' c Indices de los valores a transformar con la variante error a 1',c)
#
#     error_a_0=tf.math.greater(v_b,v_m)
#     print('error_a_0',error_a_0)
#     index_values_error_a_0=tf.where(tf.logical_and(a,error_a_0)==True)
#     #index_values_error_a_0=tf.where(error_a_0==True)
#     print('index_values_error_a_0',index_values_error_a_0)
#     print('vb',v_b)
#
#
#
#     if tf.size(c)>0:
#         error=1
#         tensor_act,valores_afectados,mask_0= TensorUpdateCiclo(error,v_b,c,valores_afec_static_0_Error,valores_afec_static_1_Error,mask,tensor_act,valores_afectados,mask_0)
#         #values_to_change=tf.gather_nd(v_b,c)
#         #val_afec_0_Error_to_change=tf.gather_nd(valores_afec_static_0_Error,c)
#         #print('val_afec_0_Error_to_change',val_afec_0_Error_to_change)
#         #val_afec_1_Error_to_change=tf.gather_nd(valores_afec_static_1_Error,c)
#         #print('val_afec_1_Error_to_change',val_afec_1_Error_to_change)
#         #print('values_to_change',values_to_change)
#         #tensor_round=ErrorAuno(values_to_change,mask,val_afec_0_Error_to_change,val_afec_1_Error_to_change)
#         #print('tensor_round',tensor_round)
#         #index=c.numpy()
#         #print('indice a camniar tensor act',index)
#         #print('valores_afectados',valores_afectados)
#         #val=tf.gather_nd(tf.where(valores_afectados),index)## tomo los indices del tensor original qu ese cambiaran
#         #print('val',val)
#         #tensor_act = tf.tensor_scatter_nd_update(tensor_act, val, tensor_round )
#         #valores_afectados=tf.tensor_scatter_nd_update(valores_afectados,val,tf.convert_to_tensor([False]*np.size(val)))
#         #mask_0=tf.add(mask_0,tf.size(val))
#         print('mask_0 desdes update', mask_0)
#         print('tensor tensor_act',tensor_act)
#         print('valores_afectados update',valores_afectados)
#
#
#
#
#
#
#
#     ##no es el que esta en mask en esse momento, retorno new mask, new_mask_0, tensor_act
#     if tf.size(index_values_error_a_0)>0:
#         error=0
#         print('dentro detf.size(error_a_0)>0 ')
#         tensor_act,valores_afectados,mask_0= TensorUpdateCiclo(error,v_b,index_values_error_a_0,valores_afec_static_0_Error,valores_afec_static_1_Error,mask,tensor_act,valores_afectados,mask_0)
#         #out=tf.constant(tf.math.subtract(out,tf.size(error_a_0)))
#         #print('out',out)
#         #mask_0=tf.tensor_scatter_nd_update(mask_0,error_a_0,tf.convert_to_tensor([1]*np.size(error_a_0)))
#         print()
#         #out=tf.constant(tf.math.subtract(out,tf.reduce_sum(mask_0)))
#         #values_to_change=tf.gather_nd(v_b,index_values_error_a_0)
#         #val_afec_0_Error_to_change=tf.gather_nd(valores_afec_static_0_Error,index_values_error_a_0)
#         #print('val_afec_0_Error_to_change',val_afec_0_Error_to_change)
#         #val_afec_1_Error_to_change=tf.gather_nd(valores_afec_static_1_Error,index_values_error_a_0)
#         #print('val_afec_1_Error_to_change',val_afec_1_Error_to_change)
#         #print('values_to_change',values_to_change)
#         #tensor_round=ErrorAcero(values_to_change,mask,val_afec_0_Error_to_change,val_afec_1_Error_to_change)
#         #index_0=index_values_error_a_0.numpy()
#         #print('index_0',index_0)
#         #print('valores_afectados',valores_afectados)
#         #val_0=tf.gather_nd(tf.where(valores_afectados),index_0)## tomo los indices del tensor original qu ese cambiaran
#         #print('val_0',val_0)
#         #
#         #tensor_act = tf.tensor_scatter_nd_update(tensor_act, val_0, tensor_round )
#         #valores_afectados=tf.tensor_scatter_nd_update(valores_afectados,val_0,tf.convert_to_tensor([False]*np.size(val_0)))
#         ##print('tensor_round en el body',tensor_round)
#         print('tensor tensor_act',tensor_act)
#         print('valores_afectados update',valores_afectados)
#         #mask_0=tf.add(mask_0,tf.size(val_0))
#         print('mask_0 desdes update', mask_0)
#
#
#
#
#     if tf.size(values_to_run)>0:
#         #mask=mask>>1
#         mask=tf.bitwise.right_shift(mask,1)
#         print('dentro del if values to run',values_to_run)
#         print('mask',mask)
#         v_b=values_to_run
#         v_m=values_to_run_with_error
#         valores_afec_static_0_Error=valores_afec_static_0_Error_to_run
#         valores_afec_static_1_Error=valores_afec_static_1_Error_to_run
#         print('v_b',v_b)
#         print('v_m',v_m)
#         print('valores_afec_static_0_Error',valores_afec_static_0_Error)
#         print('valores_afec_static_1_Error',valores_afec_static_1_Error)
#
#
#
#     if tf.experimental.numpy.any(valores_afectados)==False:
#         #tensor=tf.bitwise.bitwise_and(tensor_act,val_afec_0_Error_to_change)
#         #print('n_v_b_1', tensor)
#         #tensor_round=tf.bitwise.bitwise_or(tensor,val_afec_1_Error_to_change)
#         print('valores_afectados',valores_afectados)
#         print('tensor_act',tensor_act)
#         i=16
#
#     print('i',i)
#
#
#     return (i+1),out,mask,mask_0,tensor_original,tensor_act,v_m,v_b,tensor_with_error,values_to_run,valores_afectados,valores_afec_static_0_Error,valores_afec_static_1_Error ## retorno los valores que aun no se han modificado por ninguna d elas variantes puesto que el bit con error
#
# i,out,mask,mask_0,tensor_original,tensor_act,v_m,v_b,tensor_with_error,values_to_run,valores_afectados,valores_afec_static_0_Error,valores_afec_static_1_Error= tf.while_loop(
#     lambda i, *_ : tf.less(i,out),  # condición -> revisar cada bit
#      while_body,  # aqui todo lo que tengas que hacer de comparar los vectores (que estos los puedes definir por fuera
#                  # o pasarlos a la misma función de while_body y no modificarlos)
# (i,out,mask,mask_0,tensor_original,tensor_act,v_m,v_b,tensor_with_error,values_to_run,valores_afectados,valores_afec_static_0_Error,valores_afec_static_1_Error)  # valores iniciales
# )



       

                