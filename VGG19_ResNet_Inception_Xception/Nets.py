from tensorflow.keras.layers import (Activation, AveragePooling2D, BatchNormalization, Cropping2D,
									 Concatenate, Conv1D, Conv2D, Dense, DepthwiseConv2D, Dropout,
									 Embedding, Flatten, GlobalAveragePooling2D, Lambda, MaxPool2D,
									 MaxPooling1D, ReLU, Reshape, ZeroPadding2D, MaxPooling2D,SeparableConv2D)
from tensorflow.keras.models import Sequential
from tensorflow.python.keras import backend as K
import tensorflow as tf
import numpy as np

###################################################################################################
##FUNCTION NAME: GetNeuralNetworkModel
##DESCRIPTION:   Crea la red neuronal segun las especificaciones requeridas
##OUTPUTS:       tf.keras.Model: Red Neuronal especificada
############ARGUMENTS##############################################################################
####architecture:     Modelo de la Red, uno de los siguientes string: 'AlexNet','VGG16','PilotNet',
####                  'MobileNet','ZFNet','SqueezeNet','SentimentalNet','DenseNet'
####input_shape:      Dimensiones de entrada de la red, ejemplo: (228,228)
####                  el resto por defecto es para testing.
####output_shape:     Dimensiones de salida de la red, ejemplo: 10
####faulty_addresses: Lista con las direcciones de memoria que contienen errores,
####                  ejemplo: [1,1024,203405]
####masked_faults:    Lista de las fallas para las direcciones especificadas en faulty_addresses,
####                  ejemplo: ['xx1xxxxxxxxxxxxx','1xxxxxxxxxxxxxx0', '0000000000000000'],
####                  '1'/'0' = celdas con valor 1 o 0 permanente, 'x' = celda sin fallos.
####quantization:     Contruir el modelo cuantizado o no de la red, si es Falso word_size
####                  y frac_size no son usados.
####aging_active:            Contruir el modelo con fallos o no de la red, si es Falso faulty_addresses
####                  y masked_faults no son usados
####word_size:        Tamaño en bits de una activacion
####frac_size:        Numero de bits para la parte fraccionario de una activacion.
####batch_size:       Tamaño de batch para inferencia.
###################################################################################################
def GetNeuralNetworkModel(architecture, input_shape, output_shape, faulty_addresses = [],
							 masked_faults = [], quantization = True, aging_active = True,
						  word_size=None, frac_size=None, batch_size = 1):
	# Layer to quantize the values to the magnitude sign format when needed
	def preprocess(image):  # preprocess image
		return tf.image.resize(image, (200, 66))
	###############################################################################################
	##FUNCTION NAME: Quantization
	##DESCRIPTION:   Cuantiza un tensor usando un numero fijo de bits de signo, magnitud y fraccion
	##OUTPUTS:       Tensor cuantizado
	############ARGUMENTS##########################################################################
	####tensor:    Tensorflow tensor
	####active:    Habilita la cuantizacion
	###############################################################################################
	def Quantization(tensor, active = True):
		if active:
			factor = 2.0**frac_size
			max_value = ((1 << (word_size-1)) - 1)/factor
			min_value = -max_value
			tensor = tf.round(tensor*factor) / factor
			tensor = tf.math.minimum(tensor,max_value)   # Upper Saturation
			tensor = tf.math.maximum(tensor,min_value)   # Lower Saturation
		return tensor
	###############################################################################################
	##FUNCTION NAME: Aging
	##DESCRIPTION:   Aplica envejecimiento a un tensor segun su mapeo en memoria
	##OUTPUTS:       Tensor con valores alterados debido a los fallos
	############ARGUMENTS##########################################################################
	####tensor:     Tensorflow tensor
	####index_list: tensor con indices de las activaciones afectadas por los fallos
	####mod_list:   tensor con fallos a aplicar usando bitwise & |.
	####active:     Habilita el envejecimiento
	###############################################################################################
	def Aging(tensor, index_list, mod_list, active):
		def ApplyFault(tensor,faults):
			Ogdtype = tensor.dtype
			shift   = 2**(word_size-1)
			factor  = 2**frac_size
			tensor  = tf.cast(tensor*factor,dtype=tf.int32)
			tensor  = tf.where(tf.less(tensor, 0), -tensor + shift , tensor )
			tensor  = tf.bitwise.bitwise_and(tensor,faults[:,0])
			tensor  = tf.bitwise.bitwise_or(tensor,faults[:,1])
			tensor  = tf.where(tf.greater_equal(tensor,shift), shift-tensor , tensor )
			tensor  = tf.cast(tensor/factor,dtype = Ogdtype)
			return tensor
		if active:
			affectedValues = tf.gather_nd(tensor,index_list)
			newValues = ApplyFault(affectedValues,mod_list)
			tensor = tf.tensor_scatter_nd_update(tensor, index_list, newValues)
		return tensor

		#### esta función realiza el shift(despazar bits del número) hacia la izquierda para lograr que el
		### error sea escrito en algún bit menos significativo de la palabra y a la hora de escribir no afecte tanto
	def Shift(tensor, index_list, mod_list, active):
		# print('dentro de shiftl')
		bits_shift = 1
		affectedValues_array = []
		newValues_array = []

		def ApplyFault(tensor, faults):
			# print('dentro de ApplyFault',tensor)
			Ogdtype = tensor.dtype
			shift = 2 ** (word_size - 1)
			factor = 2 ** frac_size
			tensor = tf.cast(tensor * factor, dtype=tf.int32)
			signo = tf.math.sign(tensor, name=None)
			# print(tensor)
			tensor = tf.where(tf.less(tensor, 0), -tensor + shift, tensor)
			# print('tensor luego de valorar si es negativo', tensor)
			## En Signo guardo el valor del signo de los numeros
				##tensor de  1 y -1 y 0 en dependencia del sigo, será 0 si el número es cero
				# print('signo', signo)
			tensor = tf.bitwise.left_shift(tensor, bits_shift)
			tensor = tf.bitwise.bitwise_and(tensor, faults[:, 0])
			tensor = tf.bitwise.bitwise_or(tensor, faults[:, 1])
			tensor = tf.bitwise.right_shift(tensor, bits_shift)
			tensor = tensor * signo
			tensor = tf.where(tf.greater_equal(tensor, shift), shift - tensor, tensor)
			tensor = tf.cast(tensor / factor, dtype=Ogdtype)
				# print('tensor tensor_left_shift', tensor_left_shift)
			return tensor

		if active:
				# print('Experimento Base')
			affectedValues = tf.gather_nd(tensor, index_list)
			# affectedValues_array.append(affectedValues)
			# print('index_list',index_list)

			newValues = ApplyFault(affectedValues, mod_list)
			# print('mod_list', mod_list)
			# newValues_array.append(newValues)
			# diff_abs = tf.math.reduce_sum(tf.math.abs(tf.math.subtract(affectedValues_array, newValues_array)))
			# print('diferencia absoluta', diff_abs)
			tensor = tf.tensor_scatter_nd_update(tensor, index_list, newValues)

		return tensor
	###############################################################################################
	##FUNCTION NAME: GenerateAddressList
	##DESCRIPTION:   Mapea la lista de direcciones de memoria con fallos y la lista de tipos de
	##               fallos enmascarados a una tensor de indices de activaciones con fallos y
	##               un tensor de fallos.
	##OUTPUTS:       tensor de indices, tensor de fallos, numero de activaciones afectadas
	############ARGUMENTS##########################################################################
	####shape:       Dimensiones de entrada de la capa
	###############################################################################################
	def GenerateAddressList(shape):
		# Decodes the mask of faults to the specific error due to 0 and 1 static value
		def DecodeMask(mask):
			static_1_error  = int("".join(mask.replace('x','0')),2)
			static_0_Error  = int("".join(mask.replace('x','1')),2)
			return [static_0_Error,static_1_error]
		index_list   = []
		mod_list      = []
		if len(shape) == 1:
			for index in range(batch_size):
				for address,mask in zip(faulty_addresses,masked_faults):
					if address < shape[0]-1:
						index_list.append([index,address])
						mod_list.append(DecodeMask(mask))
		elif len(shape) == 2:
			for index in range(batch_size):
				for address,mask in zip(faulty_addresses,masked_faults):
					if address < shape[0]*shape[1] - 1:
						Ch1  = address//shape[0]
						Ch2  = (address - Ch1*shape[0])//shape[1]
						index_list.append([index,Ch1,Ch2])
						mod_list.append(DecodeMask(mask))
		else:
			for index in range(batch_size):
				for address,mask in zip(faulty_addresses,masked_faults):
					if address < shape[0]*shape[1]*shape[2] - 1:
						actMap = address//(shape[0]*shape[1])
						row    = (address - actMap*shape[0]*shape[1])//shape[1]
						col    = address - actMap*shape[0]*shape[1] - row*shape[1]
						index_list.append([index,row,col,actMap])
						mod_list.append(DecodeMask(mask))
		faults_count = len(index_list)
		return tf.convert_to_tensor(index_list),tf.convert_to_tensor(mod_list), faults_count
	###############################################################################################
	##FUNCTION NAME: AddCustomLayers
	##DESCRIPTION:   Agrega a una red neuronal capas de cuantizacion y/o envejecimiento
	##OUTPUTS:       Capa de red neuronal
	############ARGUMENTS##########################################################################
	####input_layer:          Capa de red neuronal a la cual agregarle cuantizacion/aging_active
	####include_aging:        True  si se desea incorporar una capa de aging_active
	####include_quantization: True  si se desea incorporar una capa de cuantizacion
	####aging_active:         False si se desea que la capa de aging_active no este activa
	###############################################################################################
	def AddCustomLayers(input_layer, include_aging, include_quantization=True, aging_active = []):
		x = input_layer
		if include_quantization:
			quantization_arguments = {'active':quantization}
			x = Lambda(Quantization, arguments = quantization_arguments)(input_layer)
		if include_aging:
			dims = x.shape.ndims if x.__class__.__name__ == 'KerasTensor' else x.output_shape.ndim
			index_list,mod_list,faults_count = GenerateAddressList(shape=x.shape[1:])
			aging_arguments = {'index_list' : tf.identity(index_list),
							   'mod_list' : tf.identity(mod_list),
							   'active': tf.identity(faults_count and aging_active)}
			x = Lambda(Aging, arguments = aging_arguments)(x)

		return x

	# if   aging_active == True:  aging_active = [True]*300
	# elif aging_active == False: aging_active = [False]*300
	#AlexNet
	if architecture=='AlexNet':
		input_layer = tf.keras.Input(input_shape)
		x = AddCustomLayers(input_layer,include_aging=True,aging_active = aging_active[0])
		x = Conv2D(filters=96, kernel_size=(11,11), strides=(4,4),name='Conv1')(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = BatchNormalization()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[1])
		x = MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[2],include_quantization=False)
		x = Conv2D(filters=256, kernel_size=(5,5), strides=(1,1),padding="same",name='Conv2')(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = BatchNormalization()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[3])
		x = MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[4],include_quantization=False)
		x = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="same",name='Conv3')(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = BatchNormalization()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[5])
		x = Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), padding="same",name='Conv4')(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = BatchNormalization()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[6])
		x = Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), padding="same",name='Conv5')(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = BatchNormalization()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[7])
		x = MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[8],include_quantization=False)
		x = Flatten()(x)
		x = Dense(4096)(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = Dropout(0.5)(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[9],include_quantization=False)
		x = Dense(4096)(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = Dropout(0.5)(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[10],include_quantization=False)
		x = Dense(output_shape)(x)
		x = AddCustomLayers(x,include_aging=False)
		x = tf.keras.activations.softmax(x)
		x = AddCustomLayers(x,include_aging=False)
		Net = tf.keras.Model(inputs=input_layer, outputs=x)
		return Net

	elif architecture=='VGG16':
		input_layer = tf.keras.Input(input_shape)
		x = AddCustomLayers(input_layer,include_aging=True,aging_active = aging_active[0])
		x = Conv2D(filters=64,kernel_size=(3,3),padding="same")(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[1],include_quantization=False)
		x = Conv2D(filters=64,kernel_size=(3,3),padding="same")(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[2],include_quantization=False)
		x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[3],include_quantization=False)
		x = Conv2D(filters=128,kernel_size=(3,3),padding="same")(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[4],include_quantization=False)
		x = Conv2D(filters=128,kernel_size=(3,3),padding="same")(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[5],include_quantization=False)
		x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[6],include_quantization=False)
		x = Conv2D(filters=256,kernel_size=(3,3),padding="same")(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[7],include_quantization=False)
		x = Conv2D(filters=256,kernel_size=(3,3),padding="same")(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[8],include_quantization=False)
		x = Conv2D(filters=256,kernel_size=(3,3),padding="same")(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[9],include_quantization=False)
		x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[10],include_quantization=False)
		x = Conv2D(filters=512,kernel_size=(3,3),padding="same")(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[11],include_quantization=False)
		x = Conv2D(filters=512,kernel_size=(3,3),padding="same")(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[12],include_quantization=False)
		x = Conv2D(filters=512,kernel_size=(3,3),padding="same")(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[13],include_quantization=False)
		x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[14],include_quantization=False)
		x = Conv2D(filters=512,kernel_size=(3,3),padding="same")(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[15],include_quantization=False)
		x = Conv2D(filters=512,kernel_size=(3,3),padding="same")(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[16],include_quantization=False)
		x = Conv2D(filters=512,kernel_size=(3,3),padding="same")(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[17],include_quantization=False)
		x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[18],include_quantization=False)
		x = Flatten()(x)
		x = Dense(4096)(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[19],include_quantization=False)
		x = Dense(4096)(x)
		x = AddCustomLayers(x,include_aging=False)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[20],include_quantization=False)
		x = Dense(output_shape)(x)
		x = AddCustomLayers(x,include_aging=False)
		x = tf.keras.activations.softmax(x)
		x = AddCustomLayers(x,include_aging=False)
		Net = tf.keras.Model(inputs=input_layer, outputs=x)
		return Net

	elif architecture == 'PilotNet':
		input_layer = tf.keras.Input(input_shape)
		x = Cropping2D(cropping=((50,20), (0,0)))(input_layer)
		x = Lambda(preprocess)(x)
		x = Lambda(lambda x: (x/ 127.0 - 1.0))(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[0])
		x = Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2))(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[1])
		x = Conv2D(filters=36, kernel_size=(5, 5),strides=(2, 2))(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[2])
		x = Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2))(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[3])
		x = Conv2D(filters=64, kernel_size=(3, 3))(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[4])
		x = Conv2D(filters=64, kernel_size=(3, 3))(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[5])
		x = Dropout(0.5)(x)
		x = Flatten()(x)
		x = Dense(units=1164)(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[6])
		x = Dense(units=100)(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[7])
		x = Dense(units=50)(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[8])
		x = Dense(units=10)(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[9])
		x = Dense(units=output_shape)(x)
		x = AddCustomLayers(x,include_aging=False)
		Net = tf.keras.Model(inputs=input_layer, outputs=x)
		return Net

	elif architecture == 'MobileNet':
		def MobilNetInitialConvBlock(inputs, filters, kernel=(3, 3), strides=(1, 1)):
			x = AddCustomLayers(inputs,include_aging=True,aging_active = aging_active[0])
			x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
			x = Conv2D(filters, kernel, padding='same', use_bias=False, strides=strides)(x)
			x = AddCustomLayers(x,include_aging=False)
			x = BatchNormalization()(x)
			x = ReLU(6.)(x)
			x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[1])
			return x

		def DepthwiseConvBlock(inputs, filters, strides=(1, 1), blockId=1):
			pad = 'same' if strides == (1, 1) else 'valid'
			x = inputs   if strides == (1, 1) else ZeroPadding2D(((0, 1), (0, 1)))(inputs)
			x = DepthwiseConv2D((3, 3), padding=pad, strides=strides, use_bias=False)(x)
			x = AddCustomLayers(x,include_aging=False)
			x = BatchNormalization()(x)
			x = ReLU(6.)(x)
			x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[2*blockId])
			x = Conv2D(filters, (1, 1), padding='same', strides=(1, 1), use_bias=False)(x)
			x = AddCustomLayers(x,include_aging=False)
			x = BatchNormalization()(x)
			x = ReLU(6.)(x)
			x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[2*blockId+1])
			return x

		input_layer = tf.keras.Input(input_shape)
		x = MobilNetInitialConvBlock(input_layer, 32, strides=(2, 2))
		x = DepthwiseConvBlock(x, 64,   blockId=1)
		x = DepthwiseConvBlock(x, 128,  blockId=2, strides=(2, 2))
		x = DepthwiseConvBlock(x, 128,  blockId=3)
		x = DepthwiseConvBlock(x, 256,  blockId=4, strides=(2, 2))
		x = DepthwiseConvBlock(x, 256,  blockId=5)
		x = DepthwiseConvBlock(x, 512,  blockId=6, strides=(2, 2))
		x = DepthwiseConvBlock(x, 512,  blockId=7)
		x = DepthwiseConvBlock(x, 512,  blockId=8)
		x = DepthwiseConvBlock(x, 512,  blockId=9)
		x = DepthwiseConvBlock(x, 512,  blockId=10)
		x = DepthwiseConvBlock(x, 512,  blockId=11)
		x = DepthwiseConvBlock(x, 1024, blockId=12, strides=(2, 2))
		x = DepthwiseConvBlock(x, 1024, blockId=13)
		x = GlobalAveragePooling2D()(x)
		x = Reshape((1, 1, 1024))(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[28])
		x = Dropout(1e-3)(x)
		x = Conv2D(output_shape, (1, 1), padding='same')(x)
		x = AddCustomLayers(x,include_aging=False)
		x = Reshape((output_shape,))(x)
		x = Activation(activation='softmax')(x)
		x = AddCustomLayers(x,include_aging=False)
		Net = tf.keras.Model(inputs=input_layer, outputs=x)
		return Net

	elif architecture == 'ZFNet':

		input_layer = tf.keras.Input(input_shape)
		x = AddCustomLayers(input_layer,include_aging=True,aging_active = aging_active[0])
		x = Conv2D(filters=96, kernel_size=(7,7), strides=(2,2), padding = 'valid')(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[1])
		x = MaxPool2D(pool_size=(3,3),strides=(2,2))(x)
		x = Lambda(lambda x: tf.image.per_image_standardization(x))(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[2])
		x = Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), padding = 'same')(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[3])
		x = MaxPool2D(pool_size=(3,3),strides=(2,2))(x)
		x = Lambda(lambda x: tf.image.per_image_standardization(x))(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[4])
		x = Conv2D(filters = 384, kernel_size=(3,3), padding='same')(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[5])
		x = Conv2D(filters = 384, kernel_size=(3,3), padding='same')(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[6])
		x = Conv2D(filters = 256, kernel_size=(3,3), padding='same')(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[7])
		x = MaxPool2D(pool_size=(3,3),strides=(2,2))(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[8],include_quantization=False)
		x = Flatten()(x)
		x = Dense(4096)(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[9])
		x = Dense(4096)(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[10])
		x = Dense(output_shape)(x)
		x = AddCustomLayers(x,include_aging=False)
		x = Activation(activation='softmax')(x)
		x = AddCustomLayers(x,include_aging=False)
		Net = tf.keras.Model(inputs=input_layer, outputs=x)
		return Net

	elif architecture == 'SqueezeNet':

		def FireBlock(inputs, fs, fe, blockId):
			#Squeeze
			s  = Conv2D(fs, 1)(inputs)
			s  = ReLU()(s)
			s  = AddCustomLayers(s,include_aging=True,aging_active = aging_active[blockId])
			#Expand
			e1 = Conv2D(fe, 1)(s)
			e1 = ReLU()(e1)
			e3 = Conv2D(fe, 3, padding = 'same')(s)
			e3 = ReLU()(e3)
			e  = Concatenate()([e1,e3])
			e = AddCustomLayers(e,include_aging=True,aging_active = aging_active[blockId+1])
			return e

		input_layer = tf.keras.Input(input_shape)
		x = AddCustomLayers(input_layer,include_aging=True,aging_active = aging_active[0])
		x = Conv2D(96,7,2,'same')(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[1])
		x = MaxPool2D(3,2,'same')(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[2], include_quantization=False)
		x = FireBlock(x, 16, 64,  3)
		x = FireBlock(x, 16, 64,  5)
		x = FireBlock(x, 32, 128, 7)
		x = MaxPool2D(3,2,'same')(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[9], include_quantization=False)
		x = FireBlock(x, 32, 128, 10)
		x = FireBlock(x, 48, 192, 12)
		x = FireBlock(x, 48, 192, 14)
		x = FireBlock(x, 64, 256, 16)
		x = MaxPool2D(3,2,'same')(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[18], include_quantization=False)
		x = FireBlock(x, 64, 256, 19)
		x = Conv2D(output_shape,1,(1,1),'same')(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[20])
		x = GlobalAveragePooling2D()(x)
		x = AddCustomLayers(x,include_aging=False)
		x = tf.keras.activations.softmax(x)
		x = AddCustomLayers(x,include_aging=False)
		Net = tf.keras.Model(inputs=input_layer, outputs=x)
		return Net

	elif architecture == 'SentimentalNet':
		top_words = 5000
		max_words = 500

		input_layer = tf.keras.Input(input_shape)
		x = Embedding(top_words, 32, input_length=max_words)(input_layer)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[0])
		x = Conv1D(filters=32, kernel_size=3, padding='same')(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[1])
		x = MaxPooling1D(pool_size=2)(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[2])
		x = Flatten()(x)
		x = Dense(250)(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[3])
		x = Dense(output_shape)(x)
		x = AddCustomLayers(x,include_aging=False)
		x = tf.keras.activations.sigmoid(x)
		x = AddCustomLayers(x,include_aging=False)
		Net = tf.keras.Model(inputs=input_layer, outputs=x)
		return Net

	elif architecture == 'DenseNet':

		def ConvBlock(inputs, growthRate, blockId):
			x = BatchNormalization(epsilon=1.001e-5)(inputs)
			x = ReLU()(x)
			x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[blockId])
			x = Conv2D(4*growthRate, 1, use_bias=False)(x)
			x = AddCustomLayers(x,include_aging=False)
			x = BatchNormalization(epsilon=1.001e-5)(x)
			x = ReLU()(x)
			x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[blockId+1])
			x = Conv2D(growthRate, 3, padding='same', use_bias=False)(x)
			x = AddCustomLayers(x,include_aging=False)
			o = Concatenate()([inputs, x])
			return o

		def DenseBlock(x, blocks, blockId):
			for i in range(blocks):
				x = ConvBlock(x, 32, blockId+2*i)
			return x

		def TransitionBlock(x, blockId):
			x = BatchNormalization(epsilon=1.001e-5)(x)
			x = ReLU()(x)
			x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[blockId])
			x = Conv2D(int(K.int_shape(x)[-1] * 0.5), 1, use_bias=False)(x)
			x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[blockId+1])
			x = AveragePooling2D(2, strides=2)(x)
			x = AddCustomLayers(x,include_aging=False)
			return x

		input_layer = tf.keras.Input(input_shape)
		x = AddCustomLayers(input_layer,include_aging=True,aging_active = aging_active[0])
		x = ZeroPadding2D(padding=((3, 3), (3, 3)))(x)
		x = Conv2D(64, 7, strides = 2, use_bias = False)(x)
		x = AddCustomLayers(x,include_aging=False)
		x = BatchNormalization(epsilon=1.001e-5)(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[1])
		x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
		x = MaxPool2D(3, strides=2)(x)
		x = DenseBlock(x, 6, 2)
		x = TransitionBlock(x, 14)
		x = DenseBlock(x, 12, 16)
		x = TransitionBlock(x, 40)
		x = DenseBlock(x, 24,  42)
		x = TransitionBlock(x, 90)
		x = DenseBlock(x, 16,  92)
		x = BatchNormalization(epsilon=1.001e-5)(x)
		x = ReLU()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[124])
		x = GlobalAveragePooling2D()(x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[125])
		x = Dense(output_shape)(x)
		x = AddCustomLayers(x,include_aging=False)
		x = tf.keras.activations.softmax(x)
		x = AddCustomLayers(x,include_aging=False)
		Net = tf.keras.Model(inputs=input_layer, outputs=x)
		return Net

	elif architecture == 'ResNetOLD':
		def res_identity(x, filters):
			x_skip = x
			f1, f2 = filters

			x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
			x = BatchNormalization()(x)
			x = Activation(tf.keras.activations.relu)(x)
			x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[4], include_quantization=False)

			x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
			x = BatchNormalization()(x)
			x = Activation(tf.keras.activations.relu)(x)
			x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[5], include_quantization=False)

			x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
			x = BatchNormalization()(x)
			x = Activation(tf.keras.activations.relu)(x)
			x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[6], include_quantization=False)

			x = tf.keras.layers.Add()([x, x_skip])
			x = Activation(tf.keras.activations.relu)(x)
			x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[7], include_quantization=False)

			return x

		def res_conv(x, s, filters):

			x_skip = x
			f1, f2 = filters

			x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
			x = BatchNormalization()(x)
			x = Activation(tf.keras.activations.relu)(x)
			x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[8])

			x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
			x = BatchNormalization()(x)
			x = Activation(tf.keras.activations.relu)(x)
			x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[9])

			x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
			x = BatchNormalization()(x)

			x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x_skip)
			x_skip = BatchNormalization()(x_skip)

			x = tf.keras.layers.Add()([x, x_skip])
			x = Activation(tf.keras.activations.relu)(x)
			x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[10])

			return x

		input_layer = tf.keras.Input(input_shape)
		x = AddCustomLayers(input_layer, include_aging=True, aging_active=aging_active[0])
		x = ZeroPadding2D(padding=(3, 3))(input_layer)


		x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
		x = BatchNormalization()(x)
		x = Activation(tf.keras.activations.relu)(x)
		x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[1])
		x = MaxPool2D((3, 3), strides=(2, 2))(x)


		x = res_conv(x, s=1, filters=(64, 256))
		x = res_identity(x, filters=(64, 256))
		x = res_identity(x, filters=(64, 256))


		x = res_conv(x, s=2, filters=(128, 512))
		x = res_identity(x, filters=(128, 512))
		x = res_identity(x, filters=(128, 512))
		x = res_identity(x, filters=(128, 512))


		x = res_conv(x, s=2, filters=(256, 1024))
		x = res_identity(x, filters=(256, 1024))
		x = res_identity(x, filters=(256, 1024))
		x = res_identity(x, filters=(256, 1024))
		x = res_identity(x, filters=(256, 1024))
		x = res_identity(x, filters=(256, 1024))


		x = res_conv(x, s=2, filters=(512, 2048))
		x = res_identity(x, filters=(512, 2048))
		x = res_identity(x, filters=(512, 2048))

		#x = GlobalAveragePooling2D()(x)
		#x = tf.keras.layers.Flatten()(x)
		#x = tf.keras.layers.Dense(output_shape, activation='softmax')(x)

		Net = tf.keras.Model(inputs=input_layer, outputs=x)
		return Net

	elif architecture == 'ResNet':
		def res_identity(x, filters):
			x_skip = x
			f1, f2 = filters

			x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
					   kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
			x = BatchNormalization()(x, training=False)
			x = Activation(tf.keras.activations.relu)(x)
			x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[4], include_quantization=False)

			x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same',
					   kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
			x = BatchNormalization()(x, training=False)
			x = Activation(tf.keras.activations.relu)(x)
			x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[5], include_quantization=False)

			x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid',
					   kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
			x = BatchNormalization()(x, training=False)
			x = Activation(tf.keras.activations.relu)(x)
			x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[6], include_quantization=False)

			x = tf.keras.layers.Add()([x, x_skip])
			x = Activation(tf.keras.activations.relu)(x)
			x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[7], include_quantization=False)

			return x

		def res_conv(x, s, filters):
			x_skip = x
			f1, f2 = filters

			x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid',
					   kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
			x = BatchNormalization()(x, training=False)
			x = Activation(tf.keras.activations.relu)(x)
			x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[8])

			x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same',
					   kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
			x = BatchNormalization()(x, training=False)
			x = Activation(tf.keras.activations.relu)(x)
			x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[9])

			x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid',
					   kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
			x = BatchNormalization()(x, training=False)

			x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid',
							kernel_regularizer=tf.keras.regularizers.l2(0.001))(x_skip)
			x_skip = BatchNormalization()(x_skip, training=False)

			x = tf.keras.layers.Add()([x, x_skip])
			x = Activation(tf.keras.activations.relu)(x)
			x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[10])

			return x

		input_layer = tf.keras.Input(input_shape)

		x = tf.keras.layers.Rescaling(scale=1./255)(input_layer)

		x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[11])
		x = ZeroPadding2D(padding=(3, 3))(x)

		x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
		x = BatchNormalization()(x, training=False)
		x = Activation(tf.keras.activations.relu)(x)
		x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[12])
		x = MaxPool2D((3, 3), strides=(2, 2))(x)

		x = res_conv(x, s=1, filters=(64, 256))
		x = res_identity(x, filters=(64, 256))
		x = res_identity(x, filters=(64, 256))

		x = res_conv(x, s=2, filters=(128, 512))
		x = res_identity(x, filters=(128, 512))
		x = res_identity(x, filters=(128, 512))
		x = res_identity(x, filters=(128, 512))

		x = res_conv(x, s=2, filters=(256, 1024))
		x = res_identity(x, filters=(256, 1024))
		x = res_identity(x, filters=(256, 1024))
		x = res_identity(x, filters=(256, 1024))
		x = res_identity(x, filters=(256, 1024))
		x = res_identity(x, filters=(256, 1024))

		x = res_conv(x, s=2, filters=(512, 2048))
		x = res_identity(x, filters=(512, 2048))
		x = res_identity(x, filters=(512, 2048))

		x = tf.keras.layers.AveragePooling2D((1, 1), padding='same')(x)
		x = tf.keras.layers.Dropout(0.1)(x)
		x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[13])
		x = tf.keras.layers.Flatten()(x)

		x = tf.keras.layers.Dense(output_shape, activation='softmax')(x)
		outputs = AddCustomLayers(x, include_aging=True, aging_active=aging_active[14])

		Net = tf.keras.Model(inputs=input_layer, outputs=x)
		return Net

	elif architecture == 'Xception':

		def entry_flow(inputs) :

			x = Conv2D(32, 3, strides = 2, padding='same')(inputs)
			x = AddCustomLayers(x, include_aging=False)
			x = BatchNormalization()(x)
			x = Activation(tf.keras.activations.relu)(x)
			x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[3])

			x = Conv2D(64,3,padding='same')(x)
			x = AddCustomLayers(x, include_aging=False)
			x = BatchNormalization()(x)
			x = Activation(tf.keras.activations.relu)(x)
			x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[5])

			previous_block_activation = x

			for size in [128, 256, 728] :

				x = Activation(tf.keras.activations.relu)(x)
				x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[6])
				x = SeparableConv2D(size, 3, padding='same')(x)
				x = AddCustomLayers(x, include_aging=False)
				x = BatchNormalization()(x)

				x = Activation(tf.keras.activations.relu)(x)
				x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[8])
				x = SeparableConv2D(size, 3, padding='same')(x)
				x = AddCustomLayers(x, include_aging=False)
				x = BatchNormalization()(x)

				x = MaxPooling2D(3, strides=2, padding='same')(x)

				residual = Conv2D(size, 1, strides=2, padding='same')(previous_block_activation)

				x = tf.keras.layers.Add()([x, residual])
				previous_block_activation = x

			return x

		def middle_flow(x, num_blocks=8) :
			previous_block_activation = x

			for _ in range(num_blocks) :

				x = Activation(tf.keras.activations.relu)(x)
				x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[10])
				x = SeparableConv2D(728, 3, padding='same')(x)
				x = AddCustomLayers(x, include_aging=False)
				x = BatchNormalization()(x)

				x = Activation(tf.keras.activations.relu)(x)
				x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[12])
				x = SeparableConv2D(728, 3, padding='same')(x)
				x = AddCustomLayers(x, include_aging=False)
				x = BatchNormalization()(x)

				x = Activation(tf.keras.activations.relu)(x)
				x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[14])
				x = SeparableConv2D(728, 3, padding='same')(x)
				x = AddCustomLayers(x, include_aging=False)
				x = BatchNormalization()(x)

				x = tf.keras.layers.Add()([x, previous_block_activation])
				previous_block_activation = x

			return x

		def exit_flow(x) :

			previous_block_activation = x

			x = Activation(tf.keras.activations.relu)(x)
			x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[16])
			x = SeparableConv2D(728, 3, padding='same')(x)
			x = AddCustomLayers(x, include_aging=False)
			x = BatchNormalization()(x)

			x = Activation(tf.keras.activations.relu)(x)
			x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[18])
			x = SeparableConv2D(1024, 3, padding='same')(x)
			x = AddCustomLayers(x, include_aging=False)
			x = BatchNormalization()(x)

			x = MaxPooling2D(3, strides=2, padding='same')(x)

			residual = Conv2D(1024, 1, strides=2, padding='same')(previous_block_activation)
			x = tf.keras.layers.Add()([x, residual])

			x = Activation(tf.keras.activations.relu)(x)
			x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[20])
			x = SeparableConv2D(728, 3, padding='same')(x)
			x = AddCustomLayers(x, include_aging=False)
			x = BatchNormalization()(x)

			x = Activation(tf.keras.activations.relu)(x)
			x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[22])
			x = SeparableConv2D(1024, 3, padding='same')(x)
			x = AddCustomLayers(x, include_aging=False)
			x = BatchNormalization()(x)

			x = GlobalAveragePooling2D()(x)
			x = BatchNormalization()(x)
			x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[24])
			x = tf.keras.layers.Flatten()(x)

			x = tf.keras.layers.Dense(output_shape, activation='softmax')(x)

			return x

		input_layer = tf.keras.Input(input_shape)
#### Para analizar datos usando la funcion getalloutput quito el resvaling
		#scale_layer = tf.keras.layers.Rescaling(scale=1./255)
		#inputs = scale_layer(input_layer)
		inputs = input_layer

		outputs = exit_flow(middle_flow(entry_flow(inputs)))
		Net = tf.keras.Model(inputs, outputs)
		return Net

	elif architecture == 'Inception':

		def conv2d_with_Batch(prev_layer , nbr_kernels , filter_size , strides = (1,1) , padding = 'valid'):
			x = Conv2D(filters = nbr_kernels, kernel_size = filter_size, strides=strides , padding=padding) (prev_layer)
			x = AddCustomLayers(x, include_aging=False)
			x = BatchNormalization()(x)
			x = Activation(activation = 'relu') (x)
			x = AddCustomLayers(x, include_aging=True, aging_active=aging_active[4])
			return x

		def stemBlock(prev_layer):
			x = conv2d_with_Batch(prev_layer, nbr_kernels = 32, filter_size = (3,3), strides = (2,2))
			x = conv2d_with_Batch(x, nbr_kernels = 32, filter_size = (3,3))
			x = conv2d_with_Batch(x, nbr_kernels = 64, filter_size = (3,3))

			x_1 = conv2d_with_Batch(x, nbr_kernels = 96, filter_size = (3,3), strides = (2,2) )
			x_2 = MaxPool2D(pool_size=(3,3) , strides=(2,2) ) (x)
			x_2 = AddCustomLayers(x_2,include_aging=True,aging_active = aging_active[10])

			x = tf.keras.layers.concatenate([x_1 , x_2], axis = 3)

			x_1 = conv2d_with_Batch(x, nbr_kernels = 64, filter_size = (1,1))
			x_1 = conv2d_with_Batch(x_1, nbr_kernels = 64, filter_size = (1,7) , padding ='same')
			x_1 = conv2d_with_Batch(x_1, nbr_kernels = 64, filter_size = (7,1), padding ='same')
			x_1 = conv2d_with_Batch(x_1, nbr_kernels = 96, filter_size = (3,3))

			x_2 = conv2d_with_Batch(x, nbr_kernels = 96, filter_size = (1,1))
			x_2 = conv2d_with_Batch(x_2, nbr_kernels = 96, filter_size = (3,3))

			x = tf.keras.layers.concatenate([x_1 , x_2], axis = 3)

			x_1 = conv2d_with_Batch(x, nbr_kernels = 192, filter_size = (3,3) , strides=2)
			x_2 = MaxPool2D(pool_size=(3,3) , strides=(2,2) ) (x)
			x_2 = AddCustomLayers(x_2,include_aging=True,aging_active = aging_active[20])

			x = tf.keras.layers.concatenate([x_1 , x_2], axis = 3)

			return x

		def reduction_A_Block(prev_layer) :
			x_1 = conv2d_with_Batch(prev_layer = prev_layer, nbr_kernels = 192, filter_size = (1,1))
			x_1 = conv2d_with_Batch(prev_layer = x_1, nbr_kernels = 224, filter_size = (3,3) , padding='same')
			x_1 = conv2d_with_Batch(prev_layer = x_1, nbr_kernels = 256, filter_size = (3,3) , strides=(2,2))

			x_2 = conv2d_with_Batch(prev_layer = prev_layer, nbr_kernels = 384, filter_size = (3,3) , strides=(2,2))

			x_3 = MaxPool2D(pool_size=(3,3) , strides=(2,2))(prev_layer)
			x_3 = AddCustomLayers(x_3,include_aging=True,aging_active = aging_active[30])

			x = tf.keras.layers.concatenate([x_1 , x_2 , x_3], axis = 3)

			return x

		def reduction_B_Block(prev_layer):
			x_1 = MaxPool2D(pool_size=(3,3) , strides=(2,2))(prev_layer)
			x_1 = AddCustomLayers(x_1,include_aging=True,aging_active = aging_active[33])

			x_2 = conv2d_with_Batch(prev_layer = prev_layer, nbr_kernels = 192, filter_size = (1,1))
			x_2 = conv2d_with_Batch(prev_layer = x_2, nbr_kernels = 192, filter_size = (3,3) , strides=(2,2) )

			x_3 = conv2d_with_Batch(prev_layer = prev_layer, nbr_kernels = 256, filter_size = (1,1) )
			x_3 = conv2d_with_Batch(prev_layer = x_3, nbr_kernels = 256, filter_size = (1,7) , padding='same')
			x_3 = conv2d_with_Batch(prev_layer = x_3, nbr_kernels = 320, filter_size = (7,1) , padding='same')
			x_3 = conv2d_with_Batch(prev_layer = x_3, nbr_kernels = 320, filter_size = (3,3) , strides=(2,2))

			x = tf.keras.layers.concatenate([x_1 , x_2 , x_3], axis = 3)
			return x

		def InceptionBlock_A(prev_layer):

			x_1 = conv2d_with_Batch(prev_layer = prev_layer, nbr_kernels = 64, filter_size = (1,1))
			x_1 = conv2d_with_Batch(prev_layer = x_1, nbr_kernels = 96, filter_size = (3,3) , strides=(1,1), padding='same' )
			x_1 = conv2d_with_Batch(prev_layer = x_1, nbr_kernels = 96, filter_size = (3,3) , strides=(1,1) , padding='same')

			x_2 = conv2d_with_Batch(prev_layer = prev_layer, nbr_kernels = 64, filter_size = (1,1))
			x_2 = conv2d_with_Batch(prev_layer = x_2, nbr_kernels = 96, filter_size = (3,3) , padding='same')

			x_3 = AveragePooling2D(pool_size=(3,3) , strides=1 , padding='same')(prev_layer)
			x_3 = AddCustomLayers(x_3,include_aging=False)
			x_3 = conv2d_with_Batch(prev_layer = x_3, nbr_kernels = 96, filter_size = (1,1) , padding='same')

			x_4 = conv2d_with_Batch(prev_layer = prev_layer, nbr_kernels = 96, filter_size = (1,1))

			output = tf.keras.layers.concatenate([x_1 , x_2 , x_3 , x_4], axis = 3)

			return output

		def InceptionBlock_B(prev_layer):

			x_1 = conv2d_with_Batch(prev_layer = prev_layer, nbr_kernels = 192, filter_size = (1,1))
			x_1 = conv2d_with_Batch(prev_layer = x_1, nbr_kernels = 192, filter_size = (7,1) , padding='same')
			x_1 = conv2d_with_Batch(prev_layer = x_1, nbr_kernels = 224, filter_size = (1,7) , padding='same')
			x_1 = conv2d_with_Batch(prev_layer = x_1, nbr_kernels = 224, filter_size = (7,1) , padding='same')
			x_1 = conv2d_with_Batch(prev_layer = x_1, nbr_kernels = 256, filter_size = (1,7), padding='same')

			x_2 = conv2d_with_Batch(prev_layer = prev_layer, nbr_kernels = 192, filter_size = (1,1))
			x_2 = conv2d_with_Batch(prev_layer = x_2, nbr_kernels = 224, filter_size = (1,7) , padding='same')
			x_2 = conv2d_with_Batch(prev_layer = x_2, nbr_kernels = 256, filter_size = (7,1), padding='same')

			x_3 = AveragePooling2D(pool_size=(3,3) , strides=1 , padding='same')(prev_layer)
			x_3 = AddCustomLayers(x_3,include_aging=False)
			x_3 = conv2d_with_Batch(prev_layer = x_3, nbr_kernels = 128, filter_size = (1,1))

			x_4 = conv2d_with_Batch(prev_layer = prev_layer, nbr_kernels = 384, filter_size = (1,1))

			output = tf.keras.layers.concatenate([x_1 , x_2 ,x_3, x_4], axis = 3)
			return output

		def InceptionBlock_C(prev_layer):

			x_1 = conv2d_with_Batch(prev_layer = prev_layer, nbr_kernels = 384, filter_size = (1,1))
			x_1 = conv2d_with_Batch(prev_layer = x_1, nbr_kernels = 448, filter_size = (3,1) , padding='same')
			x_1 = conv2d_with_Batch(prev_layer = x_1, nbr_kernels = 512, filter_size = (1,3) , padding='same')
			x_1_1 = conv2d_with_Batch(prev_layer = x_1, nbr_kernels = 256, filter_size = (1,3), padding='same')
			x_1_2 = conv2d_with_Batch(prev_layer = x_1, nbr_kernels = 256, filter_size = (3,1), padding='same')
			x_1 = tf.keras.layers.concatenate([x_1_1 , x_1_2], axis = 3)

			x_2 = conv2d_with_Batch(prev_layer = prev_layer, nbr_kernels = 384, filter_size = (1,1))
			x_2_1 = conv2d_with_Batch(prev_layer = x_2, nbr_kernels = 256, filter_size = (1,3), padding='same')
			x_2_2 = conv2d_with_Batch(prev_layer = x_2, nbr_kernels = 256, filter_size = (3,1), padding='same')
			x_2 = tf.keras.layers.concatenate([x_2_1 , x_2_2], axis = 3)

			x_3 = MaxPool2D(pool_size=(3,3),strides = 1 , padding='same')(prev_layer)
			x_3 = conv2d_with_Batch(prev_layer = x_3, nbr_kernels = 256, filter_size = 3  , padding='same')

			x_4 = conv2d_with_Batch(prev_layer = prev_layer, nbr_kernels = 256, filter_size = (1,1))

			output = tf.keras.layers.concatenate([x_1 , x_2 , x_3 , x_4], axis = 3)

			return output

		input_layer = tf.keras.Input(input_shape)

		x = stemBlock(prev_layer=input_layer)

		x = InceptionBlock_A(prev_layer=x)
		x = InceptionBlock_A(prev_layer=x)
		x = InceptionBlock_A(prev_layer=x)
		x = InceptionBlock_A(prev_layer=x)

		x = reduction_A_Block(prev_layer=x)

		x = InceptionBlock_B(prev_layer=x)
		x = InceptionBlock_B(prev_layer=x)
		x = InceptionBlock_B(prev_layer=x)
		x = InceptionBlock_B(prev_layer=x)
		x = InceptionBlock_B(prev_layer=x)
		x = InceptionBlock_B(prev_layer=x)
		x = InceptionBlock_B(prev_layer=x)

		x = reduction_B_Block(prev_layer= x)

		x = InceptionBlock_C(prev_layer=x)
		x = InceptionBlock_C(prev_layer=x)
		x = InceptionBlock_C(prev_layer=x)

		x = GlobalAveragePooling2D()(x)
		x = AddCustomLayers(x,include_aging=False)

		x = Dense(units = 1536, activation='relu') (x)
		x = Dropout(rate = 0.8) (x)
		x = AddCustomLayers(x,include_aging=True,aging_active = aging_active[60])
		x = Dense(output_shape, activation='softmax')(x)

		model = tf.keras.Model(inputs = input_layer , outputs = x , name ='Inception-V4')

		return model

	elif architecture == 'VGG19':

		input_layer = tf.keras.Input(input_shape)

		#Block 1
		x = Conv2D(filters = 64, kernel_size = (3,3), padding='same' , activation='relu') (input_layer)
		x = AddCustomLayers(x, include_aging=False)
		x = Conv2D(filters = 64, kernel_size = (3,3), padding='same' , activation='relu') (x)
		x = AddCustomLayers(x, include_aging=False)
		x = MaxPool2D(pool_size=(2,2) , strides=(2,2) , padding='same') (x)
		x = AddCustomLayers(x, include_aging=True, aging_active = aging_active[4])

		#Block 2
		x = Conv2D(filters = 128, kernel_size = (3,3), padding='same' , activation='relu') (x)
		x = AddCustomLayers(x, include_aging=False)
		x = Conv2D(filters = 128, kernel_size = (3,3), padding='same' , activation='relu') (x)
		x = AddCustomLayers(x, include_aging=False)
		x = MaxPool2D(pool_size=(2,2) , strides=(2,2) , padding='same') (x)
		x = AddCustomLayers(x, include_aging=True, aging_active = aging_active[7])

		#Block 3
		x = Conv2D(filters = 256, kernel_size = (3,3), padding='same' , activation='relu') (x)
		x = AddCustomLayers(x, include_aging=False)
		x = Conv2D(filters = 256, kernel_size = (3,3), padding='same' , activation='relu') (x)
		x = AddCustomLayers(x, include_aging=False)
		x = Conv2D(filters = 256, kernel_size = (3,3), padding='same' , activation='relu') (x)
		x = AddCustomLayers(x, include_aging=False)
		x = Conv2D(filters = 256, kernel_size = (3,3), padding='same' , activation='relu') (x)
		x = AddCustomLayers(x, include_aging=False)
		x = MaxPool2D(pool_size=(2,2) , strides=(2,2) , padding='same') (x)
		x = AddCustomLayers(x, include_aging=True, aging_active = aging_active[13])

		#Block 4
		x = Conv2D(filters = 512, kernel_size = (3,3), padding='same' , activation='relu') (x)
		x = AddCustomLayers(x, include_aging=False)
		x = Conv2D(filters = 512, kernel_size = (3,3), padding='same' , activation='relu') (x)
		x = AddCustomLayers(x, include_aging=False)
		x = Conv2D(filters = 512, kernel_size = (3,3), padding='same' , activation='relu') (x)
		x = AddCustomLayers(x, include_aging=False)
		x = Conv2D(filters = 512, kernel_size = (3,3), padding='same' , activation='relu') (x)
		x = AddCustomLayers(x, include_aging=False)
		x = MaxPool2D(pool_size=(2,2) , strides=(2,2) , padding='same') (x)
		x = AddCustomLayers(x, include_aging=True, aging_active = aging_active[18])

		#Block 5
		x = Conv2D(filters = 512, kernel_size = (3,3), padding='same' , activation='relu') (x)
		x = AddCustomLayers(x, include_aging=False)
		x = Conv2D(filters = 512, kernel_size = (3,3), padding='same' , activation='relu') (x)
		x = AddCustomLayers(x, include_aging=False)
		x = Conv2D(filters = 512, kernel_size = (3,3), padding='same' , activation='relu') (x)
		x = AddCustomLayers(x, include_aging=False)
		x = Conv2D(filters = 512, kernel_size = (3,3), padding='same' , activation='relu') (x)
		x = AddCustomLayers(x, include_aging=False)
		x = MaxPool2D(pool_size=(2,2) , strides=(2,2) , padding='same') (x)
		x = AddCustomLayers(x, include_aging=True, aging_active = aging_active[23])

		#Block 6
		x = Flatten()(x)
		x = Dense(units = 4096 , activation='relu') (x)
		x = Dropout(rate = 0.8)(x)
		x = AddCustomLayers(x, include_aging=True, aging_active = aging_active[25])
		x = Dense(units = 4096 , activation='relu') (x)
		x = Dropout(rate = 0.4)(x)
		x = AddCustomLayers(x, include_aging=True, aging_active = aging_active[27])
		x = Dense(output_shape, activation='softmax')(x)

		model = tf.keras.Model(inputs = input_layer , outputs = x , name ='VGG-19')

		return model
