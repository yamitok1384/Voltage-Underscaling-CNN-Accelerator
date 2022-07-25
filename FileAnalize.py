#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import collections
import random
from random import randint
import os
import pickle

from datetime import datetime

def analize_file(obj_dir, buffer_size):
    with open(obj_dir , 'rb') as f:
        data = f.read()
    # Path(r'VC.bin').stat()
    # file_size = Path(r'VC.bin').stat().st_size
    index = 0
    buffer = []
    res = "{0:04b}".format(int(data, 16))
    res_list = list(res)
    print('cantidad de elemtos de todo el fichero', len(res_list))
    t_error=res_list.count('0')
    #print('Cantidad de errores de todo el fihero:', t_error)

    vin_arr = []
    num_fallos = 0
    num_x = 0
    while index < buffer_size:

        for i, j in enumerate(res_list):

            if j == '1':
                vin_arr.append('x')


            else:

                vin_arr.append(0)

                num_fallos = num_fallos + 1
            index = index + 1
            if index == buffer_size:
                break
        buffer = np.asarray(vin_arr)
        #print('Cantidad de elementos por tipo :', collections.Counter(buffer))


    #print('numeros fallos', num_fallos)
    dist = collections.Counter(buffer)
    return buffer


def analize_file_uno(obj_dir, buffer_size):
    print('si estoy aquí')
    with open(obj_dir, 'rb') as f:
        data = f.read()
    # Path(r'VC.bin').stat()
    # file_size = Path(r'VC.bin').stat().st_size
    index = 0
    buffer = []
    res = "{0:04b}".format(int(data, 16))
    res_list = list(res)
    t_error = res_list.count('0')

    print('cantidad de elemtos de todo el fichero', len(res_list))
    print('Cantidad de errores de todo el fihero:', t_error)

    vin_arr = []
    num_fallos = 0
    while index < buffer_size:

        for i, j in enumerate(res_list):

            if j == '1':
                vin_arr.append('x')

            else:

                vin_arr.append(1)

                num_fallos = num_fallos + 1
            index = index + 1
            if index == buffer_size:
                break
        buffer = np.asarray(vin_arr)
        print('Cantidad de elementos por tipo :', collections.Counter(buffer))

    print('numeros fallos', num_fallos)
    dist = collections.Counter(buffer)
    return buffer




def analize_file_uno_ceros(obj_dir, buffer_size):
    with open(obj_dir, 'rb') as f:
        data = f.read()
    # Path(r'VC.bin').stat()
    # file_size = Path(r'VC.bin').stat().st_size
    index = 0
    buffer = []
    res = "{0:04b}".format(int(data, 16))
    res_list = list(res)
    t_error = res_list.count('0')
    print('Cantidad de errores de todo el fihero:', t_error)

    vin_arr = []
    num_fallos = 0
    random.seed(15)

    while index < buffer_size:

        for i, j in enumerate(res_list):

            if j == '1':
                vin_arr.append('x')

            else:

                vin_arr.append(random.randint(0, 1))

                num_fallos = num_fallos + 1
            index = index + 1
            if index == buffer_size:
                break
        buffer = np.asarray(vin_arr)

        print('Cantidad de elementos por tipo :', collections.Counter(buffer))

    #print('numeros fallos', num_fallos)
    dist = collections.Counter(buffer)
    return buffer


def analize_fil(obj_dir, buffer_size):
    with open(obj_dir + '.bin', 'rb') as f:
        data = f.read()
    # Path(r'VC.bin').stat()
    # file_size = Path(r'VC.bin').stat().st_size
    index = 0
    res = "{0:04b}".format(int(data, 16))
    res_list = list(res)
    vin_arr = []
    num_fallos = 0

    for i, j in enumerate(res_list):
        # print(i , i+1, j)

        if j == '1':
            vin_arr.append('x')

        else:
            vin_arr.append('0')
            num_fallos = num_fallos + 1
        index = index + 1

        if index == buffer_size:
            break
    vin_arr = np.asarray(vin_arr)
    address_with_errors = np.reshape(vin_arr, (-1, 16))
    # print('vin_gg', collections.Counter(vin_arr))
    locs = np.where(address_with_errors == '0')
    # address_with_errors[:,15]='1'
    # n_bits_fails = np.where(buffer == '1')
    return address_with_errors, locs

def buffer_vectores(buffer):
    buffer_size=len(buffer)
    address_with_errors = np.reshape(buffer, (-1, 16))
    address_with_errors = ["".join(i) for i in address_with_errors[0:2000]]
    error_mask = [y for x, y in enumerate(address_with_errors) if y.count('x') < 16]
    locs = [x for x, y in enumerate(address_with_errors) if y.count('x') < 16]

    del address_with_errors
    return [error_mask, locs]


def save_file(obj, obj_dir):
    with open(obj_dir + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_file(obj_dir):
    with open(obj_dir + '.pkl', 'rb') as f:
        return pickle.load(f)
