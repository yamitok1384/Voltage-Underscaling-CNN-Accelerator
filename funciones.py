#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


import numpy as np
import collections
import os

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


# def buffer_pa(buffer,buffer_size):
#     print('tamaño del buffer dentro de funciones',len(buffer))
#     print('estoy aqi')
#     address_with_errors = np.reshape(buffer[0:16777216], (-1, 16))
#     #address_with_errors_p = np.reshape(buffer[16777216:buffer_size], (-1, 16))
#     address_with_errors_p = np.reshape(buffer[0:16777216], (-1, 16))
#     address_with_errors = ["".join(i) for i in address_with_errors[0:16777216]]
#     error_mask = [y for x, y in enumerate(address_with_errors) if y.count('x') < 16]
#     locs = [x for x, y in enumerate(address_with_errors) if y.count('x') < 16]
#     address_with_errors_p = ["".join(i) for i in address_with_errors_p]
#     error_mask_p = [y for x, y in enumerate(address_with_errors_p) if y.count('x') < 16]
#     locs_p = [x for x, y in enumerate(address_with_errors_p) if y.count('x') < 16]
#     # print('error_mask_activ',error_mask)
#     # print('locs_act',locs)
#    # print('error_mask_pesos:',error_mask_p)
#    #e print('locs_p',locs_p)
#     print('tamaño address_with_errors pesos',len(address_with_errors_p))
#     print('tamaño locs pesos', len(locs_p))
#     print('tamaño address_with_errors activac', len(address_with_errors))
#     print('tamaño locs activaciones', len(locs))
#
#
#
#     del address_with_errors_p
#     return [error_mask_p, locs_p, error_mask ,locs]