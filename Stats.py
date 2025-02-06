import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import backend as K
import gc
from Nets_original import GetNeuralNetworkModel
#from Nets_test_shift import GetNeuralNetworkModel
#from NetsVecino import GetNeuralNetworkModel


def color(val):
	color = 'red' if 'write' in val  else 'blue'
	return 'color: %s' % color
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

				#print(mask)
				faultList.append(DecodeMask(mask))
		#print('hola')

	# elif len(shape) == 2:
	# 	for address,mask in zip(locs,error_mask):
	# 		if address < shape[0]*shape[1] - 1:
	# 			Ch1  = address//shape[0]
	# 			Ch2  = (address - Ch1*shape[0])//shape[1]
	# 			positionList.append([Ch1,Ch2])
	# 			faultList.append(DecodeMask(mask))
	# else:
	# 	for address, mask in zip(locs, error_mask):
	# 		if address < shape[0] * shape[1] * shape[2] - 1:
	# 			filt = address // (shape[0] * shape[1] * shape[2])
	# 			row = (address - filt * shape[0] * shape[1] * shape[2]) // (shape[1] * shape[2])
	# 			col = (address - filt * shape[0] * shape[1] * shape[2] - row * shape[1] * shape[2]) // shape[2]
	# 			Amap = address - filt * shape[0] * shape[1] * shape[2] - row * shape[1] * shape[2] - col * shape[2]
	# 			positionList.append([row, col, Amap, filt])
	# 			faultList.append(DecodeMask(mask))
	# 			print('faultList', faultList)
	NumberOfFaults = len(positionList)
	#print(faultList)
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
	print('estoy en WeightQuantization')
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
						 faulty_addresses = [], masked_faults = [],aging_active = False, batch_size = 1, verbose = 1, weights_faults = False):
	qNet = GetNeuralNetworkModel(architecture,input_shape,output_shape,faulty_addresses,masked_faults,aging_active=aging_active,
								 word_size=(1+act_frac_size+act_int_size), frac_size=act_frac_size, batch_size = batch_size)

	#Load Weights
	qNet.load_weights(wgt_dir).expect_partial()
	pesos_antes = qNet.get_weights()
	#print(len(pesos_antes))
	#print('pesos antes',pesos_antes)
	#print('qnet.load',qNet.load_weights(wgt_dir).expect_partial())
	#Quantize Weights
	WeightQuantization(model = qNet, frac_bits = wgt_frac_size, int_bits = wgt_int_size)
	if weights_faults:
		print('inyectando fallos en pesos ')
		num_weights = []
		num_fallos = []
		num_direc_afect = []
		name_layer = []
		acc_layer = []
		for i,layer in enumerate(qNet.layers):
			weights = layer.get_weights()
			if weights:
				layer_name = qNet.layers[i].__class__.__name__
				name_layer.append(layer_name)
				vectorized_weights = []
				for itm in weights:
					vectorized_weights.extend(itm.flatten(order='F'))
				vectorized_weights = np.array(vectorized_weights)
				num_weights.append(vectorized_weights.size)
				indexList, faultList, NumberOfFaults = GenerateFaultsList(shape=vectorized_weights.shape, locs=faulty_addresses, error_mask=masked_faults)
				num_fallos.append(NumberOfFaults)
				if NumberOfFaults > 0:
					faulty_weights = IntroduceFaultsInWeights(vectorized_weights, indexList, faultList, wgt_int_size,   wgt_frac_size)
					direc_afect = np.count_nonzero(vectorized_weights != faulty_weights)
					num_direc_afect.append(direc_afect)
					for index, weight in enumerate(weights):
						weights[index] = faulty_weights[0:weight.size].numpy().reshape(weight.shape, order='F')
						faulty_weights = faulty_weights[weight.size:]
				else:
					num_direc_afect.append(0)
				layer.set_weights(weights)
	# DFrame_num_weights = pd.DataFrame(num_weights)
	# DFrame_num_fallos = pd.DataFrame(num_fallos)
	# DFrame_num_direc_afect = pd.DataFrame(num_direc_afect)
	# DFrame_layer = pd.DataFrame(name_layer)
	# DFrame_acc_layer = pd.DataFrame(acc_layer)

	# statistics = pd.concat([DFrame_layer, DFrame_num_weights, DFrame_num_fallos, DFrame_num_direc_afect], axis=1,join='outer')
	# statistics.columns = ['DFrame_layer', 'num_weights', 'num_fallos', 'num_direc_afect']
	# print(statistics)
	# statistics.to_excel(str(architecture) + '_statist_weigts.xlsx', sheet_name='fichero_707', index=False)
	# parameters
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
		loss,acc  = qNet.evaluate(test_dataset,verbose=verbose)
		outputs = (loss, acc)

		#Activa_Relu(qNet,test_dataset)
		iterator = iter(test_dataset)
		image = next(iterator)[0]
		activations = get_All_Outputs(qNet, image)
		Relu_inp = activations[3]
		Ri_l = Relu_inp.ravel().tolist()# Lo convierto en lista para crear el dataset
		Relu_out = activations[3]

		capa_batch_norma = activations[6]
	# Cleaning Memory
	del qNet
	gc.collect()
	K.clear_session()
	tf.compat.v1.reset_default_graph()
	return outputs


###################################################################################################
##FUNCTION PAARA SABER LA ESNTRDAS Y SALIDAD DE UNA CAPA ESPECÍFICA
##FUNCTION NAME: InputOutput
##DESCRIPTION:   Muestra las entradas y salidad de una capa determinada
##OUTPUTS:       void
############ARGUMENTS##############################################################################
####network:           tf.keras.model: modelo de red
####input_data:   np.array conteniendo las activaciones de entrada de la red
####frac_bits:         Numero de bits parte fraccionaria activaciones
####int_bits:          Tamaño en bits parte entera de activaciones
####dataset_size:      Tamaño del dataset

# def get_All_Outputs(model, test_dataset, learning_phase=False):
# 	iterator = iter(test_dataset)
# 	outputs = [layer.output for layer in model.layers]  # exclude Input
# 	#layers_fn_inp = K.function([model.input], [model.layers[4].output])
# 	layers_fn = K.function([model.input], [model.layers[5].output])
# 	image = next(iterator)[0]
# 	#out.append(model.layers[5].output)
# 	out=layers_fn([image])
# 	print(out)
# 	return layers_fn([image])
# def Activa_Relu(qNet,test_dataset):
# 	print('soy q_net dentro d ela función Activa_Relu ',qNet)
# 	iterator = iter(test_dataset)
# 	image = next(iterator)[0]
# 	activations = get_All_Outputs(qNet, image)
# 	Relu_inp = activations[4]
# 	Ri_l=Relu_inp.ravel().tolist()
# 	Relu_out = activations[5]
# 	Ro_l=Relu_out.ravel().tolist()
#
# 	# print('entradas de la Relu', Ri_l)
# 	# print('salidas de la relu',Ro_l )
# 	print('np.isnan input', np.isnan(Ri_l))
# 	print('np.isnan output', np.isnan(Ro_l))
# 	i=pd.DataFrame(Ri_l)
# 	o=pd.DataFrame(Ro_l)
# 	df = pd.concat([i,o], axis=1, join='outer')
# 	df.columns = ['Relu_inp','Relu_out']
# 	print(df)
# 	df.to_csv('Input_ouout_Relu.csv',  index=False, encoding= 'utf-8')
# 	return df


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
	activation_Relu=[]
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
def QuantizationEffect(architecture, dataset, wgt_dir, input_shape, output_shape, batch_size, f, verbose=0):
	print('Estoy aquí en el efecto d ecuantización')
	df = pd.DataFrame({'Experiment':[],'bits':[],'acc':[],'loss':[]})
	NQBE = 0        # bits assigned to the parts that are assumed non quantized
	bits = range(0,15) # bits to test in quantized parts.
	print('Activation fraction part')
	for bit in bits:
		loss, acc = CheckAccuracyAndLoss(architecture, dataset, wgt_dir, act_frac_size = bit, act_int_size = NQBE, wgt_frac_size = NQBE, wgt_int_size = NQBE, 
										input_shape = input_shape, output_shape = output_shape, batch_size = batch_size, aging_active= aging_active,verbose = verbose)
		print(bit,' bits results: ','acc: ',acc, 'loss: ',loss)
		df = df.append(pd.DataFrame({'Experiment':['Activation fraction part'],'bits':[bit],'acc':[acc],'loss':[loss]}))
	print('Weights fraction part')
	for bit in bits:
		loss, acc = CheckAccuracyAndLoss(architecture, dataset, wgt_dir, act_frac_size = NQBE, act_int_size = NQBE, wgt_frac_size = bit, wgt_int_size = NQBE, 
										input_shape = input_shape, output_shape = output_shape, batch_size = batch_size,aging_active= aging_active, verbose = verbose)
		print(bit,' bits results: ','acc: ',acc, 'loss: ',loss)
		df = df.append(pd.DataFrame({'Experiment':['Weights fraction part'],'bits':[bit],'acc':[acc],'loss':[loss]}))
	print('Activation integer part')
	for bit in bits:
		loss, acc = CheckAccuracyAndLoss(architecture, dataset, wgt_dir, act_frac_size = NQBE, act_int_size = bit, wgt_frac_size = NQBE, wgt_int_size = NQBE, 
										input_shape = input_shape, output_shape = output_shape, batch_size = batch_size,aging_active= aging_active, verbose = verbose)
		print(bit,' bits results: ','acc: ',acc, 'loss: ',loss)
		df = df.append(pd.DataFrame({'Experiment':['Activation integer part'],'bits':[bit],'acc':[acc],'loss':[loss]}))
	print('Weights integer part')
	for bit in bits:
		loss, acc = CheckAccuracyAndLoss(architecture, dataset, wgt_dir, act_frac_size = NQBE, act_int_size = NQBE, wgt_frac_size = NQBE, wgt_int_size = bit, 
										input_shape = input_shape, output_shape = output_shape, batch_size = batch_size, aging_active= aging_active,verbose = verbose)
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
write_read_layers = []
write_layers = []
read_layers = []
array_read_by_block = []
array_write_by_block =[]
array_read_by_block =[]
write_layers=[]
read_layers=[]
write_layer_name=[]
read_layer_name=[]

array_write_read_block =[]
formula_test = []
array_read_block_newForm=[]
count_write =0
def GetReadAndWrites(network,layer_indices,addresses,samples,CNN_gating=False, network_name = False, ):
	print('network', network_name)
	print(layer_indices,layer_indices)
	print('addresses_entrada', addresses)

	PE_columns = 16
	PE_rows    = 16
	Data = {}
	data_read_write = {}
	data_read_write_temp ={}

	Data['Writes'] = np.zeros(addresses,dtype=np.int64)
	Data['Reads']  = np.zeros(addresses,dtype=np.int64)
## Agrargué estas variables para no perturbar el código anterior########################################
	data_read_write['Writes'] = np.zeros(addresses, dtype=np.int64)
	data_read_write['Reads'] = np.zeros(addresses, dtype=np.int64)
	data_read_write_temp['Reads'] = np.zeros(addresses, dtype=np.int64)
	Data['offset'] = 0
	data_read_write['offset']=0
	data_read_write_temp['offset'] = 0



	def get_next_address(outsize):
		buffer_divitions = list(range(0,addresses+1,addresses//8))
		print('buffer_divitions',buffer_divitions)
		next_address = 0 if not CNN_gating else min([itm for itm in buffer_divitions if itm >= (data_read_write['offset']+outsize)%addresses],default=addresses)
		data_read_write['offset'] = next_address


	def Buffer_Writing(layer):

		out_size = np.prod(layer.output_shape[0][1:]).astype(np.int64) if type(layer.output_shape[0])==tuple else np.prod(layer.output_shape[1:]).astype(np.int64)
		actual_values = np.take(data_read_write['Writes'],range(data_read_write['offset'],data_read_write['offset']+out_size),mode='wrap')
		np.put(data_read_write['Writes'], range(data_read_write['offset'], data_read_write['offset'] + out_size), 1 + actual_values, mode='wrap')
###Para las escrituras, esto no tiene problema porque es simple y lo da bien#############################################3
		#np.put(data_read_write['Writes'], range(data_read_write['offset'], data_read_write['offset'] + out_size), 1 + actual_values, mode='wrap')
		print('suma de las escrituras' ,np.sum(data_read_write['Writes']))
		#write_layer_name.append(layer.__class__.__name__)
		write_read_layers.append(layer.__class__.__name__ + str('_') + str('write'))
		write_block = (out_size / 16)
		write_layers.append(write_block)
		array_write_read_block.append(write_block)
		#array_write_by_block.append(write_block)
		get_next_address(out_size)
	def Buffer_Reading(layer):
		print(' capa', layer.__class__.__name__ )
		if layer.__class__.__name__ in ['MaxPooling2D','GlobalAveragePooling2D','MaxPooling1D','AveragePooling2D','BatchNormalization']:
			in_size = np.prod(layer.input_shape[1:]).astype(np.int64)
			#actual_values = np.take(Data['Reads'],range(Data['offset'],Data['offset']+in_size),mode='wrap')
			#np.put(Data['Reads'],range(Data['offset'],Data['offset']+in_size),1+actual_values,mode='wrap')
####Utilizo mi variable para guardar los valores nuevos que deseo en este caso me va contando la cantidad de veces que se lee un index ########
			actual_values = np.take(data_read_write['Reads'], range(data_read_write['offset'], data_read_write['offset'] + in_size), mode='wrap')
			print('actual_values', actual_values)
			np.put(data_read_write['Reads'], range(data_read_write['offset'], data_read_write['offset'] + in_size), 1 + actual_values, mode='wrap')
			read_block = in_size/ 16
			array_write_read_block.append(read_block)
			# write_read_layers.append(layer.__class__.__name__)
			#array_read_by_block.append(read_block)
			write_read_layers.append(layer.__class__.__name__+  str('_') +  str('read'))


			print('data_read_writeReads',data_read_write['Reads'])
#####Aquí es donde creo que está el tema ###########################################3
		elif len(layer.input_shape) == 4:
			print('layer==4',layer.__class__.__name__)
			#print('filtros', layer.filters)
			#Convolution
			(input_height,input_width,input_channels) = layer.input_shape[1:]
			padding = layer.padding
			#Create Activation Map Reads Count
			# valid sin ceros de relleno
			if padding == 'valid':
				input_map = np.zeros((input_height,input_width,input_channels),dtype=np.int64)
			else:
				height_zero_pads = np.ceil(layer.kernel_size[0]/2).astype(np.int64)
				width_zero_pads = np.ceil(layer.kernel_size[1]/2).astype(np.int64)
				input_map = np.zeros((input_height+height_zero_pads,input_width+width_zero_pads,input_channels),dtype=np.int64)


			(output_height,output_width,output_channels) = layer.output_shape[1:]
			#Iterate over outputs and update the reads from Activation Map
			Kernel_X1 = 0
			Kernel_X2 = layer.kernel_size[0]
			Kernel_Y1 = 0
			Kernel_Y2 = layer.kernel_size[1]
			(Stride_X,Stride_Y)  = layer.strides
########## En este caso los bloques de lescutras se clacula teniendo en cuenta el tamaño de
#del kernel , la cantidad de canales  así como las dimenciones de la imagen
			# read_by_block = ((input_width / 16 * (layer.kernel_size[1])) * input_channels) * (input_height / Stride_Y)
			# array_read_by_block.append(read_by_block)
			# read_layers.append(layer.__class__.__name__)
			# print('layer==4',layer.__class__.__name__)
			# print('read_by_block', read_by_block)
			for y in range(output_height):
				for x in range(output_width):
					input_map[Kernel_Y1:Kernel_Y2,Kernel_X1:Kernel_X2,:] += 1
					#print('input_map', input_map)
					Kernel_X1 += Stride_X
					Kernel_X2 += Stride_X
				Kernel_X1 = 0
				Kernel_X2 = layer.kernel_size[0]
				Kernel_Y1 += Stride_Y
				Kernel_Y2 += Stride_Y
			if layer.__class__.__name__ in ['Conv2D']:
				input_map = input_map*(np.ceil(output_channels/PE_columns).astype(np.int64))
			if padding == 'same':
				# Eliminate the zero activations
				input_map = np.delete(input_map,list(range(np.ceil(height_zero_pads/2).astype(int))),0)
				input_map = np.delete(input_map,list(range(-1,-1 -height_zero_pads//2,-1)),0)
				input_map = np.delete(input_map,list(range(np.ceil(width_zero_pads/2).astype(int))),1)
				input_map = np.delete(input_map,list(range(-1,-1 -width_zero_pads//2,-1)),1)
			input_map = input_map.reshape(-1,input_map.shape[-1]).flatten()


			#actual_values = np.take(Data['Reads'],range(Data['offset'],Data['offset']+input_map.size),mode='wrap')
	        #np.put(Data['Reads'],range(Data['offset'],Data['offset']+input_map.size),input_map+actual_values,mode='wrap')
	#######Esta es la fórmula que definimos para obtener los valores, luego que tenngo el total de elementos lo divido entre el numero de columnas  ##################################################################
	#y esto se divide entre esl tamaño del kernel entre el stride y todo entre el numero de canales

			actual_values = np.take(data_read_write['Reads'], range(data_read_write['offset'], data_read_write['offset'] + input_map.size), mode='wrap')
			np.put(data_read_write['Reads'], range(data_read_write['offset'], data_read_write['offset'] + input_map.size), ((((input_map + actual_values)/(layer.kernel_size[1]**2/Stride_Y**2)))), mode='wrap')
			read_block = np.sum(data_read_write['Reads']) / 16
			array_write_read_block.append(read_block)
			# write_read_layers.append(layer.__class__.__name__)
			#array_read_by_block.append(read_block)
			write_read_layers.append(layer.__class__.__name__+  str('_') +  str('read'))
			print('data_read_writeReads', data_read_write['Reads'])
			# read_layers.append(read_block)
			# read_layer_name.append(layer.__class__.__name__)
# La linea habilitada es para realizar la lectura y escritura por bloques de 16 por capas , si deso obtenerlas para todas las direcciones del buffer acumulandolas la comento
# lo que hace es restablecer lso valores a 0 para no acumular los datos
		#np.put(data_read_write['Reads'], range(data_read_write['offset'], data_read_write['offset'] + input_map.size), 0, mode='wrap')

		elif len(layer.input_shape) == 3:
			print('layer==1', layer.__class__.__name__)
			print('filtros', layer.filters)
			#Convolution 1D
			(input_width,input_channels) = layer.input_shape[1:]
			padding = layer.padding
			#Create Activation Map Reads Count
			#valid sin ceros de relleno
			if padding == 'valid':
				input_map = np.zeros((input_width,input_channels),dtype=np.int64)
				width_zero_pads = np.ceil(layer.kernel_size[0]/2).astype(np.int64)
				input_map = np.zeros((input_width+width_zero_pads,input_channels),dtype=np.int64)
			(output_width,output_channels) = layer.output_shape[1:]
			#Iterate over outputs and update the reads from Activation Map
			Kernel_X1 = 0
			Kernel_X2 = layer.kernel_size[0]
			Stride_X  = layer.strides[0]
#############La fórmula varía porque el shape==3
			# read_by_block = ((input_width / 16 ) * input_channels)
			# array_read_by_block.append(read_by_block)
			# read_layers.append(layer.__class__.__name__)
			# print('read_by_block', read_by_block)
			for x in range(output_width):
				input_map[Kernel_X1:Kernel_X2,:] += 1
				Kernel_X1 += Stride_X
				Kernel_X2 += Stride_X
			if layer.__class__.__name__ in ['Conv1D']: 
				input_map = input_map*(np.ceil(output_channels/PE_columns).astype(np.int64))
			#same con ceros de relleno
			if padding == 'same':
				# Eliminate the zero activations
				input_map = np.delete(input_map,list(range(np.ceil(width_zero_pads/2).astype(int))),0)
				input_map = np.delete(input_map,list(range(-1,-1 -width_zero_pads//2,-1)),0)
				input_map = input_map.reshape(-1,input_map.shape[-1]).flatten()
			#np.put(Data['Reads'],range(Data['offset'],Data['offset']+input_map.size),input_map+actual_values,mode='wrap')
#Agregué esta fórmula
			#print('filtros', layer.filters)
			# actual_values = np.take(Data['Reads'], range(Data['offset'], Data['offset'] + input_map.size), mode='wrap')
			# np.put(Data['Reads'], range(Data['offset'], Data['offset'] + input_map.size), input_map + actual_values,  mode='wrap')
			actual_values = np.take(data_read_write['Reads'],range(data_read_write['offset'], data_read_write['offset'] + input_map.size), 	mode='wrap')
			np.put(data_read_write['Reads'], range(data_read_write['offset'], data_read_write['offset'] + input_map.size), (((input_map + actual_values))/(layer.kernel_size[1]**2/Stride_X**2)), mode='wrap')
			print('data_read_write :Reads', data_read_write['Reads'])
			read_block = np.sum(data_read_write['Reads']) / 16
			print('data_read_writeReads',data_read_write['Reads'])
			array_write_read_block.append(read_block)
			# read_layers.append(read_block)
			# read_layer_name.append(layer.__class__.__name__)
			#array_read_by_block.append(read_block)
			write_read_layers.append(layer.__class__.__name__+  str('_') + str('read'))
# La linea habilitada es para realizar la lectura y escritura por bloques de 16 por capas , si deso obtenerlas para todas las direcciones del buffer acumulandolas la comento
# lo que hace es restablecer lso valores a 0 para no acumular los datos
		#np.put(data_read_write['Reads'],  range(data_read_write['offset'], data_read_write['offset'] + input_map.size),  0, mode='wrap')

		else:

			#FC
			input_size  = layer.input_shape[-1]
			output_size = layer.output_shape[-1]
			# read_by_block = (input_size / 16)
			# print('read_by_block', read_by_block)
			# array_read_by_block.append(read_by_block)
			# a = input_size * np.ceil(output_size / (PE_columns * PE_rows)).astype(np.int64)
			# # formula_test.append(a)
			#actual_values = np.take(data_read_write['Reads'],range(Data['offset'],Data['offset']+input_size),mode='wrap')
			#np.put(Data['Reads'],range(Data['offset'],Data['offset']+input_size),input_size*np.ceil(output_size/(PE_columns*PE_rows)).astype(np.int64)+actual_values,mode='wrap')
		#Aquí mantuve la formula original porque sonsidero qu ees así para estas capas
			actual_values = np.take(data_read_write['Reads'], range(data_read_write['offset'], data_read_write['offset'] + input_size),mode='wrap')
			print('actual_values en Dense', actual_values)
			np.put(data_read_write['Reads'], range(data_read_write['offset'], data_read_write['offset'] + input_size),   input_size * np.ceil(output_size / (PE_columns * PE_rows)).astype(np.int64) + actual_values, mode='wrap')
			print('data_read_write_Reads', data_read_write['Reads'])
			read_block = np.sum(data_read_write['Reads'])/16
			print('suma diccionario d electura', np.sum(data_read_write['Reads']))
			array_write_read_block.append(read_block)
			write_read_layers.append(layer.__class__.__name__ +  str('_') +  str('read'))
# La linea habilitada es para realizar la lectura y escritura por bloques de 16 por capas , si deso obtenerlas para todas las direcciones del buffer acumulandolas la comento
# lo que hace es restablecer lso valores a 0 para no acumular los datos
	#np.put(data_read_write['Reads'], range(data_read_write['offset'], data_read_write['offset'] + input_size),  0, mode='wrap')

			#print('array_read_by_block', array_read_by_block)
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
			print('bid',bid)
			return sim_layer(layer,bid)
	def simulate_single_inference(layers,first_buffer):
		if first_buffer == 0:
			Buffer_Writing(layers[0])
			for index,layer in enumerate(layers[1:]):
				print('procesando index d ela capa : ', layer.__class__.__name__)
				print('procesando index d ela capa : ',index)
				handle_layer(layer,(first_buffer+index+1)%2)
		else:
			for index,layer in enumerate(layers[1:]):
				handle_layer(layer,(first_buffer+index)%2)
	layers = [np.take(network.layers,index) for index in layer_indices]
	for image in range(samples):
		if (image+1)%25 == 0:
			print('procesados: ',image+1)
		simulate_single_inference(layers,image%2)
	# print('network', network_name)


	# df_write_read_layers = pd.DataFrame(write_read_layers)
	# df_write_read_block= pd.DataFrame(array_write_read_block)
	# df_write_Block = pd.concat([df_write_read_layers, df_write_read_block], axis=1, join='outer')
	# df_write_Block.columns = ['Capa', 'Write_read_block']
	# df_write_Block.to_excel('read_and_write_newForm_PE_16' + str(network_name) + '.xlsx', sheet_name='fichero_707', index=False)
	#

#el codigo original debuelve data pero yo devuelvo la variable que cree nueva
	return data_read_write

