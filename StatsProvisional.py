import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import backend as K
import gc
from Nets import GetNeuralNetworkModel

# GenerateFaultsList, IntroduceFaultsInWeights y WeightQuantization son versiones de funciones
# creadas en Nets.py adaptadas a pesos.
def GenerateFaultsList(shape,locs,error_mask):
	# Decodes the mask of faults to the specific error due to 0 and 1 static value
	def DecodeMask(mask):
		static1Error  = int("".join(mask.replace('x','0')),2)
		static0Error  = int("".join(mask.replace('x','1')),2)
		return [static0Error,static1Error]
	positionList   = []
	faultList      = []
	if len(shape) == 1:
		for address,mask in zip(locs,error_mask):
			if address < shape[0] - 1:
				positionList.append([address])
				faultList.append(DecodeMask(mask))
	elif len(shape) == 2:
		for address,mask in zip(locs,error_mask):
			if address < shape[0]*shape[1] - 1:
				Ch1  = address//shape[0]
				Ch2  = (address - Ch1*shape[0])//shape[1]
				positionList.append([Ch1,Ch2])
				faultList.append(DecodeMask(mask))
	else:
		for address,mask in zip(locs,error_mask):
			if address < shape[0]*shape[1]*shape[2] - 1:
				filt   = address//(shape[0]*shape[1]*shape[2])
				row    = (address - filt*shape[0]*shape[1]*shape[2])//(shape[1]*shape[2])
				col    = (address - filt*shape[0]*shape[1]*shape[2] - row*shape[1]*shape[2])//shape[2]
				Amap   = address  - filt*shape[0]*shape[1]*shape[2] - row*shape[1]*shape[2] - col*shape[2]
				positionList.append([row,col,Amap,filt])
				faultList.append(DecodeMask(mask))
	NumberOfFaults = len(positionList)
	return tf.convert_to_tensor(positionList),tf.convert_to_tensor(faultList), NumberOfFaults

def IntroduceFaultsInWeights(tensor,positionList,faultList,intSize,fractSize):
	def ApplyFault(tensor,faults):
		Ogdtype = tensor.dtype
		shift   = 2**(intSize+fractSize-1)
		factor  = 2**fractSize
		tensor  = tf.cast(tensor*factor,dtype=tf.int32)
		tensor  = tf.where(tf.less(tensor, 0), -tensor + shift , tensor )
		tensor  = tf.bitwise.bitwise_and(tensor,faults[:,0])
		tensor  = tf.bitwise.bitwise_or(tensor,faults[:,1])
		tensor  = tf.where(tf.greater_equal(tensor,shift), shift-tensor , tensor )
		tensor  = tf.cast(tensor/factor,dtype = Ogdtype)
		return tensor
	affectedValues = tf.gather_nd(tensor,positionList)
	newValues = ApplyFault(affectedValues,faultList)
	tensor = tf.tensor_scatter_nd_update(tensor, positionList, newValues)
	return tensor

def WeightQuantization(model, frac_bits, int_bits):
	def Quantization(tensor):
		factor = 2.0**frac_bits
		maxValue = ((1 << (frac_bits+int_bits)) - 1)/factor
		minValue = -maxValue - 1
		tensor = tf.round(tensor*factor) / factor
		tensor = tf.math.minimum(tensor,maxValue)   # Upper Saturation
		tensor = tf.math.maximum(tensor,minValue)   # Lower Saturation
		return tensor
	for layer in model.layers:
		weights = layer.get_weights()
		if weights:
			qWeights    = [Quantization(itm) for itm in weights]
			layer.set_weights(qWeights)

###################################################################################################
##FUNCTION NAME: CheckAccuracyAndLoss
##DESCRIPTION:   evalua la accuracy y loss de la red bajo ciertas condiciones
##OUTPUTS:       accuracy y/o loss
############ARGUMENTS##############################################################################
####architecture:     Modelo de la Red, uno de los siguientes string: 'AlexNet','VGG16','PilotNet',
####                  'MobileNet','ZFNet','SqueezeNet','SentimentalNet','DenseNet'
####test_dataset:     iterable con el conjunto de datos a inferir
####wgt_dir:          Direccion de los pesos entrenados
####act_frac_size:    Tamaño en bits parte fraccionaria de activaciones
####act_int_size:     Tamaño en bits parte entera de activaciones
####wgt_frac_size:    Tamaño en bits parte fraccionaria de pesos
####wgt_int_size:     Tamaño en bits parte entera de activaciones
####input_shape:      Dimensiones de entrada de la red
####output_shape:     Dimensiones de salida de la red
####faulty_addresses: Lista con las direcciones de memoria que contienen errores
####masked_faults:    Lista de las fallas para las direcciones especificadas en faulty_addresses
####aging_active:     True si se desea incluir el efecto de envejecimiento para las activaciones
####batch_size:       Tamaño de batch de inferencia
####weights_faults:   True si se desea aplicar los fallos ademas en el buffer de memoria
###################################################################################################
def CheckAccuracyAndLoss(architecture, test_dataset, wgt_dir, act_frac_size, act_int_size, wgt_frac_size, wgt_int_size, input_shape, output_shape,
						 faulty_addresses = [], masked_faults = [], aging_active = False , batch_size = 1, verbose = 1, weights_faults = False):
	qNet = GetNeuralNetworkModel(architecture,input_shape,output_shape,faulty_addresses,masked_faults,aging_active=aging_active,
								 word_size=(1+act_frac_size+act_int_size), frac_size=act_frac_size, batch_size = batch_size)
	#Load Weights
	qNet.load_weights(wgt_dir).expect_partial()
	#Quantize Weights
	WeightQuantization(model = qNet, frac_bits = wgt_frac_size, int_bits = wgt_int_size)
	if weights_faults:
		weights = qNet.get_weights()
		for index,itm in enumerate(weights):
			positionList,faultList,NumberOfFaults = GenerateFaultsList(shape=itm.shape,locs=faulty_addresses,error_mask=masked_faults)
			if NumberOfFaults > 0:
				weights[index] = IntroduceFaultsInWeights(itm,positionList,faultList,wgt_int_size,wgt_frac_size)
		qNet.set_weights(weights)
	# Params
	if architecture == 'Sentimental':
		loss = 'binary_crossentropy'
	else:
		loss = tf.keras.losses.CategoricalCrossentropy()
	optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
	if architecture == 'PilotNet':
		qNet.compile(optimizer=optimizer, loss='mse',)
		loss = qNet.evaluate(test_dataset,verbose=verbose)
		outputs = (None,loss)
	else:
		qNet.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
		(loss,acc) = qNet.evaluate(test_dataset,verbose=verbose)
		outputs = (loss,acc)

	# Cleaning Memory
	del qNet
	gc.collect()
	K.clear_session()
	tf.compat.v1.reset_default_graph()
	return outputs



def get_All_Outputs(model, input_data, learning_phase=False):
	outputs = [layer.output for layer in model.layers]  # exclude Input
	layers_fn = K.function([model.input], outputs)
	return layers_fn([input_data])



###################################################################################################
##FUNCTION NAME: ActivationStats
##DESCRIPTION:   Printea estadisticas de las activaciones
##OUTPUTS:       void
############ARGUMENTS##############################################################################
####network:           tf.keras.model: modelo de red
####test_dataset:      iterable con el conjunto de datos a inferir
####frac_bits:         Numero de bits parte fraccionaria activaciones
####int_bits:          Tamaño en bits parte entera de activaciones
####dataset_size:      Tamaño del dataset
###################################################################################################
def ActivationStats(network, test_dataset, frac_bits, int_bits, dataset_size):
	def getAllOutputs(model, input_data, learning_phase=False):
		outputs = [layer.output for layer in model.layers] # exclude Input
		layers_fn = K.function([model.input], outputs)
		return layers_fn([input_data])

	maxvalue   = ((1 << (frac_bits+int_bits)) - 1)/(2.0**frac_bits)
	minvalue   = -maxvalue - 1
	iterator   = iter(test_dataset)
	minimumMMU = minimumBuffer  = 999
	maximumMMU = maximumBuffer  = -999
	MMUMeans      = []
	bufferMeans   = []
	MMUIndexes    = []
	bufferIndexes = []
	saturatedMMUCount = saturatedBufferCount = MMUActivationCount = bufferActivationCount = 0
	for index in range(len(network.layers)):
		if index == len(network.layers) - 1:
			bufferIndexes.append(index)
		elif network.layers[index+1].__class__.__name__ in ['Conv2D','Conv1D','MaxPooling2D','MaxPooling1D','GlobalAveragePooling2D','Dense','AveragePooling2D','DepthwiseConv2D']:
			bufferIndexes.append(index)
		elif network.layers[index].__class__.__name__ in ['Conv2D','Conv1D','Dense','DepthwiseConv2D']:
			MMUIndexes.append(index)
	index  = 1
	while index <= dataset_size:
		image       = next(iterator)[0]
		activations = getAllOutputs(network,image)
		bufferActs = [activations[i] for i in bufferIndexes]
		bufferActs = np.concatenate( [itm.flatten() for itm in bufferActs] , axis=0 )
		MMUActs    = [activations[i] for i in MMUIndexes]
		MMUActs    = np.concatenate( [itm.flatten() for itm in MMUActs] , axis=0 )
		tmp1 = np.max(MMUActs)
		tmp2 = np.min(MMUActs)
		tmp3 = np.max(bufferActs)
		tmp4 = np.min(bufferActs)
		if tmp1 > maximumMMU:
			maximumMMU = tmp1
		if tmp2 < minimumMMU:
			minimumMMU = tmp2
		if tmp3 > maximumBuffer:
			maximumBuffer = tmp3
		if tmp4 < minimumBuffer:
			minimumBuffer = tmp4
		MMUMeans.append(np.mean(MMUActs))
		bufferMeans.append(np.mean(bufferActs))
		saturatedMMUCount     += np.sum(MMUActs > maxvalue) + np.sum(MMUActs < minvalue)
		saturatedBufferCount  += np.sum(bufferActs > maxvalue) + np.sum(bufferActs < minvalue)
		MMUActivationCount    += len(MMUActs)
		bufferActivationCount += len(bufferActs)
		index = index + 1
	print('mean value (MMU):',np.mean(MMUMeans))
	print('mean value (Buffer):',np.mean(bufferMeans))
	print('maximum (MMU):',maximumMMU)
	print('minimum (MMU):',minimumMMU)
	print('maximum (Buffer):',maximumBuffer)
	print('minimum (Buffer):',minimumBuffer)
	print('saturation ratio (MMU):',saturatedMMUCount/MMUActivationCount)
	print('saturation ratio (Buffer):',saturatedBufferCount/bufferActivationCount)

###################################################################################################
##FUNCTION NAME: QuantizationEffect
##DESCRIPTION:   Testea el efecto de cuantificar en la accuracy para distintas distribuciones de bits
##OUTPUTS:       dataframe resumen
############ARGUMENTS##############################################################################
#### mismos argumentos que CheckAccuracyAndLoss
###################################################################################################
def QuantizationEffect(architecture, dataset, wgt_dir, input_shape, output_shape, batch_size, verbose=0):
	df = pd.DataFrame({'Experiment':[],'bits':[],'acc':[],'loss':[]})
	NQBE = 16        # bits assigned to the parts that are assumed non quantized
	bits = range(0,15) # bits to test in quantized parts.
	print('Activation fraction part')
	for bit in bits:
		loss, acc = CheckAccuracyAndLoss(architecture, dataset, wgt_dir, act_frac_size = bit, act_int_size = NQBE, wgt_frac_size = NQBE, wgt_int_size = NQBE,
										input_shape = input_shape, output_shape = output_shape, batch_size = batch_size, verbose = verbose)
		print(bit,' bits results: ','acc: ',acc, 'loss: ',loss)
		df = df.append(pd.DataFrame({'Experiment':['Activation fraction part'],'bits':[bit],'acc':[acc],'loss':[loss]}))
	print('Weights fraction part')
	for bit in bits:
		loss, acc = CheckAccuracyAndLoss(architecture, dataset, wgt_dir, act_frac_size = NQBE, act_int_size = NQBE, wgt_frac_size = bit, wgt_int_size = NQBE,
										input_shape = input_shape, output_shape = output_shape, batch_size = batch_size, verbose = verbose)
		print(bit,' bits results: ','acc: ',acc, 'loss: ',loss)
		df = df.append(pd.DataFrame({'Experiment':['Weights fraction part'],'bits':[bit],'acc':[acc],'loss':[loss]}))
	print('Activation integer part')
	for bit in bits:
		loss, acc = CheckAccuracyAndLoss(architecture, dataset, wgt_dir, act_frac_size = NQBE, act_int_size = bit, wgt_frac_size = NQBE, wgt_int_size = NQBE,
										input_shape = input_shape, output_shape = output_shape, batch_size = batch_size, verbose = verbose)
		print(bit,' bits results: ','acc: ',acc, 'loss: ',loss)
		df = df.append(pd.DataFrame({'Experiment':['Activation integer part'],'bits':[bit],'acc':[acc],'loss':[loss]}))
	print('Weights integer part')
	for bit in bits:
		loss, acc = CheckAccuracyAndLoss(architecture, dataset, wgt_dir, act_frac_size = NQBE, act_int_size = NQBE, wgt_frac_size = NQBE, wgt_int_size = bit,
										input_shape = input_shape, output_shape = output_shape, batch_size = batch_size, verbose = verbose)
		print(bit,' bits results: ','acc: ',acc, 'loss: ',loss)
		df = df.append(pd.DataFrame({'Experiment':['Weights integer part'],'bits':[bit],'acc':[acc],'loss':[loss]}))
	return df


###################################################################################################
##FUNCTION NAME: GetReadAndWrites
##DESCRIPTION:   obtiene las estadisticas de escrituras y lecturas del buffer
##OUTPUTS:       diccionario con escrituras y lecturas
############ARGUMENTS##############################################################################
####network:          tf.keras.model: modelo de red
####layer_indices:    indices de capas a considerar
####addresses:        numero de direcciones del buffer
####samples:          numero de imagenes a simular
####CNN_gating:       True si se desea aplicar la tecnica CNN gating
####network_name:     'None' o 'MobileNet'
###################################################################################################
read_layers = []
write_layers = []
size_layers=[]
write_read_layers = []
def GetReadAndWrites(network,layer_indices,addresses,samples,CNN_gating=False, network_name = False):
	PE_columns = 16
	PE_rows    = 16
	Data = {}
	Data['Writes'] = np.zeros(addresses,dtype=np.int64)
	Data['Reads']  = np.zeros(addresses,dtype=np.int64)
	Data['offset'] = 0
	def get_next_address(outsize):
		buffer_divitions = list(range(0,addresses+1,addresses//8))
		next_address = 0 if not CNN_gating else min([itm for itm in buffer_divitions if itm >= (Data['offset']+outsize)%addresses],default=addresses)
		Data['offset'] = next_address
	def Buffer_Writing(layer):
		print('escrituras')
		print(layer.__class__.__name__)
		#write_layers.append(layer.__class__.__name__ + str('_') + str('write'))
		write_read_layers.append(layer.__class__.__name__ + str('_') + str('write'))
		out_size = np.prod(layer.output_shape[0][1:]).astype(np.int64) if type(layer.output_shape[0])==tuple else np.prod(layer.output_shape[1:]).astype(np.int64)
		actual_values = np.take(Data['Writes'],range(Data['offset'],Data['offset']+out_size),mode='wrap')
		np.put(Data['Writes'],range(Data['offset'],Data['offset']+out_size),1+actual_values,mode='wrap')
		size_layers.append(out_size)
		#print('tamaño de la capa de escritura', (out_size))
		get_next_address(out_size)
	def Buffer_Reading(layer):
		#read_layers.append(layer.__class__.__name__ + str('_') + str('read'))
		write_read_layers.append(layer.__class__.__name__ + str('_') + str('read'))
		print(layer.__class__.__name__)
		if layer.__class__.__name__ in ['MaxPooling2D','GlobalAveragePooling2D','MaxPooling1D','AveragePooling2D','BatchNormalization']:
			in_size = np.prod(layer.input_shape[1:]).astype(np.int64)
			print('in_size',in_size)
			actual_values = np.take(Data['Reads'],range(Data['offset'],Data['offset']+in_size),mode='wrap')
			np.put(Data['Reads'],range(Data['offset'],Data['offset']+in_size),1+actual_values,mode='wrap')
			size_layers.append(in_size.size)
			print('tamaño de la capa de lectura', in_size.size)
		elif len(layer.input_shape) == 4:
			print('layer.input_shape) == 4',layer.__class__.__name__)
			#Convolution
			(input_height,input_width,input_channels) = layer.input_shape[1:]
			padding = layer.padding
			#Create Activation Map Reads Count
			if padding == 'valid':
				print('paddin==valid')
				input_map = np.ones((input_height,input_width,input_channels),dtype=np.int64)

				#print('input_map',input_map)
			else:
				print('no es valid')
				height_zero_pads = np.ceil(layer.kernel_size[0]/2).astype(np.int64)
				width_zero_pads = np.ceil(layer.kernel_size[1]/2).astype(np.int64)
				input_map = np.ones((input_height+height_zero_pads,input_width+width_zero_pads,input_channels),dtype=np.int64)
			(output_height,output_width,output_channels) = layer.output_shape[1:]

			#input_map = np.ones(( output_height,output_width,output_channels), dtype=np.int64)
			#print('input_map', input_map)
			#Iterate over outputs and update the reads from Activation Map
			# Kernel_X1 = 0
			# Kernel_X2 = layer.kernel_size[0]
			# Kernel_Y1 = 0
			# Kernel_Y2 = layer.kernel_size[1]
			# (Stride_X,Stride_Y)  = layer.strides
			# for y in range(output_height):
			# 	for x in range(output_width):
			# 		input_map[Kernel_Y1:Kernel_Y2,Kernel_X1:Kernel_X2,:] += 1
			# 		Kernel_X1 += Stride_X
			# 		Kernel_X2 += Stride_X
			# 	Kernel_X1 = 0
			# 	Kernel_X2 = layer.kernel_size[0]
			# 	Kernel_Y1 += Stride_Y
			# 	Kernel_Y2 += Stride_Y

			if layer.__class__.__name__ in ['Conv2D']:
				print(layer.__class__.__name__)
				#print('input_map in conv 2d', input_map)
				#print('output_channels',output_channels)

				input_map = input_map*(np.ceil(output_channels/PE_columns).astype(np.int64))
				#print('input_map', input_map)
				#print('suma de los canales input_map', np.sum(input_map))


			if padding == 'same':
				# Eliminate the zero activations
				input_map = np.delete(input_map,list(range(np.ceil(height_zero_pads/2).astype(int))),0)
				input_map = np.delete(input_map,list(range(-1,-1 -height_zero_pads//2,-1)),0)
				input_map = np.delete(input_map,list(range(np.ceil(width_zero_pads/2).astype(int))),1)
				input_map = np.delete(input_map,list(range(-1,-1 -width_zero_pads//2,-1)),1)
			input_map = input_map.reshape(-1,input_map.shape[-1]).flatten()
			size_layers.append(input_map.size)
			#print('tamaño de la capa de lectura', input_map.size)

			actual_values = np.take(Data['Reads'],range(Data['offset'],Data['offset']+input_map.size),mode='wrap')
			np.put(Data['Reads'],range(Data['offset'],Data['offset']+input_map.size),input_map+actual_values,mode='wrap')
		elif len(layer.input_shape) == 3:
			print('layer.input_shape) == 3', layer.__class__.__name__)
			#Convolution 1D
			(input_width,input_channels) = layer.input_shape[1:]
			padding = layer.padding
			#Create Activation Map Reads Count
			if padding == 'valid':
				input_map = np.zeros((input_width,input_channels),dtype=np.int64)
			else:
				width_zero_pads = np.ceil(layer.kernel_size[0]/2).astype(np.int64)
				input_map = np.zeros((input_width+width_zero_pads,input_channels),dtype=np.int64)
			(output_width,output_channels) = layer.output_shape[1:]
			#Iterate over outputs and update the reads from Activation Map
			# Kernel_X1 = 0
			# Kernel_X2 = layer.kernel_size[0]
			# Stride_X  = layer.strides[0]
			# for x in range(output_width):
			# 	input_map[Kernel_X1:Kernel_X2,:] += 1
			# 	Kernel_X1 += Stride_X
			# 	Kernel_X2 += Stride_X
			if layer.__class__.__name__ in ['Conv1D']:
				print('ayer.__class__.__name__ in Conv1D', layer.__class__.__name__)
				input_map = input_map*(np.ceil(output_channels/PE_columns).astype(np.int64))
			if padding == 'same':
				# Eliminate the zero activations
				input_map = np.delete(input_map,list(range(np.ceil(width_zero_pads/2).astype(int))),0)
				input_map = np.delete(input_map,list(range(-1,-1 -width_zero_pads//2,-1)),0)
			input_map = input_map.reshape(-1,input_map.shape[-1]).flatten()
			size_layers.append(input_map.size)
			#print('tamaño de la capa de lectura', input_map.size)
			actual_values = np.take(Data['Reads'],range(Data['offset'],Data['offset']+input_map.size),mode='wrap')
			np.put(Data['Reads'],range(Data['offset'],Data['offset']+input_map.size),input_map+actual_values,mode='wrap')
		else:
			#FC
			print('FC', layer.__class__.__name__)

			input_size  = layer.input_shape[-1]
			output_size = layer.output_shape[-1]
			size_layers.append(input_size)
			#print('tamaño capasa', (input_size))
			actual_values = np.take(Data['Reads'],range(Data['offset'],Data['offset']+input_size),mode='wrap')
			np.put(Data['Reads'],range(Data['offset'],Data['offset']+input_size),input_size*np.ceil(output_size/(PE_columns*PE_rows)).astype(np.int64)+actual_values,mode='wrap')
	def sim_layer(layer,bid):
		if bid == 0:
			Buffer_Writing(layer)
		else:
			Buffer_Reading(layer)
	def sim_concat(layer,bid):
		if bid == 0:
			Buffer_Writing(layer[0])
			Buffer_Writing(layer[1])
		else:
			Buffer_Reading(layer[0])
	def sim_expand(layer,bid):
		if bid == 0:
			Buffer_Writing(layer[0])
			Buffer_Writing(layer[1])
		else:
			Buffer_Reading(layer[0])
			Buffer_Reading(layer[1])
	def handle_layer(layer,bid):
		if isinstance(layer,np.ndarray) and network_name == 'DenseNet':
			return sim_concat(layer, bid)
		elif isinstance(layer, np.ndarray) and network_name == 'SqueezeNet':
			return sim_expand(layer, bid)
		else:
			return sim_layer(layer,bid)
	def simulate_single_inference(layers,first_buffer):
		if first_buffer == 0:
			#print('procesando: ',layers[0].name)
			Buffer_Writing(layers[0])
			#print('Lecturas/Escrituras/offset: ',np.max(Data['Reads']),np.max(Data['Writes']),Data['offset'])
			for index,layer in enumerate(layers[1:]):
				print('procesando: ',layer.name)
				handle_layer(layer,(first_buffer+index+1)%2)
				#print('Lecturas/Escrituras/offset: ',np.max(Data['Reads']),np.max(Data['Writes']),Data['offset'])
		else:
			for index,layer in enumerate(layers[1:]):
				print('procesando: ',layer.name)
				handle_layer(layer,(first_buffer+index)%2)
				#print('Lecturas/Escrituras/offset: ',np.max(Data['Reads']),np.max(Data['Writes']),Data['offset'])
	layers = [np.take(network.layers,index) for index in layer_indices]
	for image in range(samples):
		if (image+1)%25 == 0:
			print('procesados: ',image+1)
		simulate_single_inference(layers,image%2)
	df_read_layers = pd.DataFrame(read_layers)
	df_write_layers = pd.DataFrame(write_layers)

	df_write_layers = pd.DataFrame(write_read_layers)
	def_size_layers= pd.DataFrame(size_layers)
	df_Block = pd.concat([df_write_layers, def_size_layers], axis=1, join='outer')
	df_Block.columns = ['Capa', 'tamaño']
	df_Block.to_excel('MoRS/Analisis_Resultados/Energía_VBW/layers_analysis_read_writing_sizes/Capas_read_write_size_test_StatsProvisional_entre_16' + str(network_name) + '.xlsx', sheet_name='fichero_707', index=False)
	return Data