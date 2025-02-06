from PIL import Image
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import csv
#import cv2
import os


###################################################################################################
##FUNCTION NAME: GetDatasets
##DESCRIPTION:   Prepara el set de datos a ser usado por el modelo
##OUTPUTS:       Iterables para el entrenamiento, validacion y
##               testeo de modelos, opcionalmente sus tamaños
############ARGUMENTS##############################################################################
####dataset_name:     Nombre del dataset (uno de los disponiblels en
####                  https://www.tensorflow.org/datasets/catalog/overview)
####split:            Particion porcentual deseada para training/validation, ejemplo: (80,10),
####                  el resto por defecto es para testing.
####data_shape:       Dimension de los datos al cual reajustarse, ejemplo: (32,32)
####target_shape:     Numero de clases (target)
####train_batch_size: tamaño de batch para training y validation
####test_valid_size:  tamaño de batch para testing.
####return_size:      True si se desea retornar tambien el tamaño de los sets.
###################################################################################################

# def GetDatasets(dataset_name, data_shape, target_shape, train_batch_size, batch_size, size, return_size=False):
# 	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 	(train_ds, validation_ds, test_ds), info = tfds.load(dataset_name, split=["train", "test[:35%]", "test[35%:]"],
# 														 as_supervised=True,
# 														 with_info=True, shuffle_files=True)
# 	#
# 	# (train_ds, validation_ds, test_ds), info = tfds.load('cifar10',	 split=["train", "test[:35%]", "test[35%:]"],
# 	# 													 as_supervised=True, with_info=True, shuffle_files=True)
# 	def normalize_resize(image, label):
# 		image = tf.cast(image, tf.float32)
# 		image = tf.divide(image, 255)
# 		image = tf.image.resize(image, size)
# 		return image, label
#
# 	def augment(image, label):
# 		image = tf.image.random_flip_left_right(image)
# 		image = tf.image.random_saturation(image, 0.7, 1.3)
# 		image = tf.image.random_contrast(image, 0.8, 1.2)
# 		image = tf.image.random_brightness(image, 0.1)
# 		image = tf.image.random_crop(image, (data_shape[0],data_shape[1]), 3)
# 		return image, label
#
# 	train_ds = train_ds.map(normalize_resize).cache().map(augment).batch(batch_size).prefetch(buffer_size=1)
# 	validation_ds = validation_ds.map(normalize_resize).cache().batch(batch_size).prefetch(buffer_size=1)
# 	test_ds = test_ds.map(normalize_resize).cache().batch(batch_size).prefetch(buffer_size=1)
# 	return train_ds, validation_ds, test_ds

def GetDatasets(dataset_name, split, data_shape, target_shape, train_batch_size, test_batch_size,
                return_size=False):
	x_train,y_train= tfds.as_numpy(tfds.load(dataset_name, batch_size=-1, as_supervised=True, split='train[:'+str(split[0])+'%]'))
	x_valid,y_valid= tfds.as_numpy(tfds.load(dataset_name, batch_size=-1, as_supervised=True, split='train['+str(split[0])+':'+str(split[0]+split[1])+'%]'))
	x_test, y_test = tfds.as_numpy(tfds.load(dataset_name, batch_size=-1, as_supervised=True, split='train['+str(split[0]+split[1])+'%:]'))
	def to_categorical(x_, y_):
		return x_, tf.one_hot(y_, depth = target_shape)
	def normalize(x_, y_):
		return tf.cast(x_, tf.float32) / 255., y_
	def resize(image, label):
		image = tf.image.resize(image, (data_shape[0],data_shape[1]))
		return image, label
	train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
	train_dataset = train_dataset.shuffle(buffer_size=x_train.shape[0])
	train_dataset = train_dataset.map(to_categorical)
	train_dataset = train_dataset.map(normalize)
	train_dataset = train_dataset.map(resize)
	train_dataset = train_dataset.batch(train_batch_size)
	train_dataset = train_dataset.repeat()
	valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid,y_valid))
	valid_dataset = valid_dataset.map(to_categorical)
	valid_dataset = valid_dataset.map(normalize)
	valid_dataset = valid_dataset.map(resize)
	valid_dataset = valid_dataset.batch(train_batch_size)
	valid_dataset = valid_dataset.repeat()
	test_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test))
	test_dataset = test_dataset.map(to_categorical)
	test_dataset = test_dataset.map(normalize)
	test_dataset = test_dataset.map(resize)
	test_dataset = test_dataset.batch(test_batch_size)
	if return_size:
		return train_dataset,valid_dataset,test_dataset,x_train.shape[0],x_valid.shape[0],x_test.shape[0]
	else:
		return train_dataset,valid_dataset,test_dataset


###################################################################################################
##FUNCTION NAME: GetPilotNetDataset
##DESCRIPTION:   Carga el set de datos (local) para PilotNet
##OUTPUTS:       Iterables para el entrenamiento, validacion y testeo de modelos
############ARGUMENTS##############################################################################
####csv_dir:     Directorio del .csv
####train_batch_size: tamaño de batch para training y validation
####test_valid_size:  tamaño de batch para testing.
###################################################################################################

def GetPilotNetDataset(csv_dir, train_batch_size, test_batch_size):
	lines=[]
	firstline = True
	Train_car_images =[]
	Valid_car_images =[]
	Test_car_images  =[]
	Train_steering_angles =[]
	Valid_steering_angles =[]
	Test_steering_angles  =[]
	with open(csv_dir, 'r') as f:
		reader = csv.reader(f)
		contador = 0
		for row in reader:
			try:
				contador +=1
				if firstline:    #skip first line
					firstline = False
					continue
				steering_center = float(row[3])

				# create adjusted steering measurements for the side camera images
				correction = 0.2 # this is a parameter to tune
				steering_left = steering_center + correction
				steering_right = steering_center - correction

				# read in images from center, left and right cameras
				source_path1 = row[0]
				source_path2 = row[1]
				source_path3 = row[2]
				filename1 = source_path1.split('/')[-1]
				filename2 = source_path2.split('/')[-1]
				filename3 = source_path3.split('/')[-1]
				#Specific path
				path1 = 'Data/Datos/IMG/' + filename1 
				path2 = 'Data/Datos/IMG/' + filename2 
				path3 = 'Data/Datos/IMG/' + filename3 
		
				#if os.path.isfile(path1):
				img_center = np.asarray(Image.open(path1))
				#if os.path.isfile(path2):
				img_left = np.asarray(Image.open(path2))
				#if os.path.isfile(path3):
				img_right = np.asarray(Image.open(path3))
				#training:validation:testing ratio = 8:2:3
				if contador <= 8:
					Train_car_images.extend([img_center, img_left, img_right])
					Train_steering_angles.extend([steering_center, steering_left, steering_right])
				elif contador <=10:
					Valid_car_images.extend([img_center, img_left, img_right])
					Valid_steering_angles.extend([steering_center, steering_left, steering_right])
				else:
					Test_car_images.extend([img_center, img_left, img_right])
					Test_steering_angles.extend([steering_center, steering_left, steering_right])
					if contador == 13:
						contador = 0
			except:
				break
	x_train = np.array(Train_car_images)
	x_valid = np.array(Valid_car_images)
	x_test  = np.array(Test_car_images)
	y_train = np.array(Train_steering_angles)
	y_valid = np.array(Valid_steering_angles)
	y_test  = np.array(Test_steering_angles)
	Train_Samples = x_train.shape[0]
	Valid_Samples = x_valid.shape[0]
	Test_Samples  = x_test.shape[0]
	train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
	train_dataset = train_dataset.shuffle(buffer_size=Train_Samples)
	train_dataset = train_dataset.batch(train_batch_size)
	train_dataset = train_dataset.repeat()
	valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid,y_valid))
	valid_dataset = valid_dataset.batch(train_batch_size)
	valid_dataset = valid_dataset.repeat()
	test_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test))
	test_dataset = test_dataset.batch(test_batch_size)
	return train_dataset,valid_dataset,test_dataset


###################################################################################################
##FUNCTION NAME: GetIMBDDataset
##DESCRIPTION:   Carga el set de datos IMDB Datasets para SentimentalNet
##OUTPUTS:       Iterables para el entrenamiento, validacion y testeo de modelos
############ARGUMENTS##############################################################################
####train_batch_size: tamaño de batch para training y validation
####test_valid_size:  tamaño de batch para testing.
###################################################################################################

def GetIMBDDataset(train_batch_size, test_batch_size):
	top_words = 5000
	max_words = 500
	(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)
	x_train = sequence.pad_sequences(x_train, maxlen=max_words)
	x_test  = sequence.pad_sequences(x_test, maxlen=max_words)
	x_valid = x_test[0:10000]
	y_valid = y_test[0:10000]
	x_test  = x_test[10000:]
	y_test  = y_test[10000:]
	Train_Samples = x_train.shape[0]
	Valid_Samples = x_valid.shape[0]
	Test_Samples  = x_test.shape[0]
	train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
	train_dataset = train_dataset.shuffle(buffer_size=Train_Samples)
	train_dataset = train_dataset.batch(train_batch_size)
	train_dataset = train_dataset.repeat()
	valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid,y_valid))
	valid_dataset = valid_dataset.batch(train_batch_size)
	valid_dataset = valid_dataset.repeat()
	test_dataset  = tf.data.Dataset.from_tensor_slices((x_test,y_test))
	test_dataset  = test_dataset.batch(test_batch_size)
	return train_dataset,valid_dataset,test_dataset