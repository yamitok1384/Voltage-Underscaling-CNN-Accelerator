from numba import cuda
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.python.keras import backend as K
from datetime import datetime
import time
import pandas as pd
#############Debugging###################################################################################
DEBUG = True
#####Cuda&MMU configuration##############################################################################
#Cuda
threadsperblock = 512
#Codigo para 25 activaciones por ciclos
# #Matrix Multiplication Unit dimensions
# ROWS      	   = 25
# COLUMNS   	   = 25
# #DRAM2buffer Bandwith
# acts_per_cycle = 25


#Codigo modificado para 16 activaciones por ciclos
#Matrix Multiplication Unit dimensions
ROWS      	   = 16
COLUMNS   	   = 16
#DRAM2buffer Bandwith
acts_per_cycle = 16
#####load/save functions#################################################################################
def save_obj(obj, obj_dir ):
	with open( obj_dir + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(obj_dir):
	with open( obj_dir + '.pkl', 'rb') as f:
		return pickle.load(f)


###################################################################################################
##FUNCTION NAME: update_high_cycle_count
##DESCRIPTION:   Actualiza el numero de ciclos en alto de cada celda de memoria
##OUTPUTS:       void
############ARGUMENTS##############################################################################
####data:   np.array representando la memoria
####stats:  np.array representando el acumulado de los ciclos en alto de cada celda
####cycles: numero de ciclos desde la ultima actualizacion
###################################################################################################

@cuda.jit
def update_high_cycle_count(data,stats,cycles):
	pos = cuda.grid(1)
	if pos < stats.size and data[pos] == 1:
		stats[pos] += cycles            


###################################################################################################
##FUNCTION NAME: get_all_outputs
##DESCRIPTION:   Obtiene todas las activaciones de cada capa de la red para una entrada determinada
##OUTPUTS:       Lista de np.arrays con los valores de activaciones por capa
############ARGUMENTS##############################################################################
####model:       tf.keras.model: modelo de una red neuronal
####input_data:  np.array conteniendo las activaciones de entrada de la red.
###################################################################################################

def get_all_outputs(model, input_data, learning_phase=False):
		outputs   = [layer.output for layer in model.layers] # exclude Input
		layers_fn = K.function([model.input], outputs)
		return layers_fn([input_data])
###################################################################################################
##FUNCTION NAME: simulate_one_image
##DESCRIPTION:   Realiza la simulacion de la utilizacion de los buffers del acelerador,
##               para 1 imagen.
##OUTPUTS:       void
############ARGUMENTS##############################################################################
####layers:       Lista de las capas de interes (aquellas que tienen impacto en el buffer)
####activations:  Lista de np.arrays con los valores de activaciones por capa
####config:       Diccionario con la configuracion de simulacion
####first_buffer: 0 si se inicia en el buffer 1.
###################################################################################################
cycles_by_layer=[]
layer_list=[]
def simulate_one_image(network_type,layers, activations, config, buffer, first_buffer):
	###################################################################################################
    ##FUNCTION NAME: write_in_buffer
    ##DESCRIPTION:   escribe en la direccion offset del buffer las activaciones (en binario)
    ##OUTPUTS:       void
    ############ARGUMENTS##############################################################################
    ####activations: np.array de Activaciones a escribir
    ####buffer:      np.array simulando el buffer
    ####offset:      direccion de inicio de escritura
    ###################################################################################################
	def write_in_buffer(activations, buffer, offset):
		#Function to convert integer representation to 0 and 1
		def integer_to_binary(integer_array):
			binary_repr_vector = np.vectorize(np.binary_repr)
			binary_array = np.where(integer_array<0,binary_repr_vector(-integer_array + 2**(config['word size']-1), width=config['word size']),
											        binary_repr_vector(integer_array, width=config['word size']))
			binary_array = np.array(list(''.join(binary_array)),dtype=int)
			return binary_array
		#Function to apply dvth mitigation common techniques
		def invert_and_shift_technique(binary_array):
			array_after_invertion = 1 - binary_array if config['invert bits'] else binary_array
			values         = np.split(array_after_invertion,len(array_after_invertion))
			shifted_values = np.roll(values,config['bitshift'])
			return np.concatenate(shifted_values)
		integer_repr = np.floor(activations*2**config['frac size']).astype(np.int32)
		binary_rep   = integer_to_binary(integer_repr)
		binary_rep_after_techniques = invert_and_shift_technique(binary_rep)
		buffer[offset:offset + len(binary_rep_after_techniques)] = binary_rep_after_techniques
	###################################################################################################
    ##FUNCTION NAME: hardware_utilization
    ##DESCRIPTION:   Dada las caracteristicas de una capa obtiene su utilizacion del Procesing Element
    ##               array
    ##OUTPUTS:       Diccionario con los parametros del PE array
    ############ARGUMENTS##############################################################################
    ####layer: tf.keras.layers: capa de la red en cuestion
    ###################################################################################################
	def hardware_utilization(layer):
		parameters = {}
		if layer.__class__.__name__ in ['Conv2D','Conv1D','DepthwiseConv2D']:
			map_size                       = layer.output_shape[1] if layer.__class__.__name__ == 'Conv1D' else layer.output_shape[1]*layer.output_shape[2]
			parameters['Procesing Cycles'] = np.prod(layer.kernel_size) if layer.__class__.__name__ == 'DepthwiseConv2D' else np.prod(layer.kernel_size)*layer.input_shape[-1]
			parameters['Columns Used']     = 1 if layer.__class__.__name__ == 'DepthwiseConv2D' else min(layer.filters, COLUMNS, parameters['Procesing Cycles'])
			parameters['Rows Used']        = min(ROWS, map_size)
			parameters['Initial Delay']    = parameters['Columns Used'] + parameters['Rows Used'] + 2
			parameters['Total cycles']     = int(parameters['Initial Delay'] + parameters['Procesing Cycles']*np.ceil(map_size/parameters['Rows Used'])*np.ceil(layer.output_shape[-1]/parameters['Columns Used']))							 
		else:
			parameters['Columns Used']     = min(layer.output_shape[-1],COLUMNS)
			parameters['Rows Used']        = min(int(np.ceil(layer.output_shape[-1]/parameters['Columns Used'])),ROWS)
			parameters['Initial Delay']    = 2
			parameters['Procesing Cycles'] = layer.input_shape[-1]
			parameters['Total cycles']     = int(parameters['Initial Delay'] + parameters['Procesing Cycles']*np.ceil(layer.output_shape[-1]/(parameters['Columns Used']*parameters['Rows Used'])))
		return parameters
    ###################################################################################################
    ##FUNCTION NAME: DRAM_bypass
    ##DESCRIPTION:   LLamada cuando la capa es muy grande para ser llevada a buffer
    ##OUTPUTS:       void
    ############ARGUMENTS##############################################################################
    ####buffer:       np.array usado como buffer de memoria
    ####cnn_gated:    True si se utiliza la tecnica de cnn_gated
    ####layer:cycles: numero de ciclos requerido por la capa
    ###################################################################################################
	def DRAM_bypass(buffer, cnn_gated, layer_cycles):
		if cnn_gated:
				buffer['Data'][:] = 2
		#Update Stats
		buffer['HighCyclesCount'][buffer['Data']==1] += layer_cycles
		buffer['OffCyclesCount'][buffer['Data']==2]  += layer_cycles
	###################################################################################################
    ##FUNCTION NAME: OnOff_buffer
    ##DESCRIPTION:   realiza el apagado/encendido de bancos para preparar la escritura de la capa actual
    ##OUTPUTS:       localizacion de memoria para la escritura de la capa subsiguiente
    ############ARGUMENTS##############################################################################
    ####buffer:       np.array usado como buffer de memoria
    ####out_size:      Numero de activaciones de la capa actual
    ###################################################################################################
	def OnOff_buffer(buffer, out_size):
		#Wake up sleeping areas.
		tmp = np.random.randint(2,size=buffer['Data'].size).astype(np.int8)
		buffer['Data']   = np.where(buffer['Data']==2,tmp,buffer['Data'])
		# Get Banks partition
		buffer_divitions = list(range(0,buffer['Number of Addresses']+1,buffer['Number of Addresses']//config['Number of switchable sections']))
		# Bounds of the used area
		lower_bound = buffer['offset']
		lower_bound = max([itm for itm in buffer_divitions if itm <= buffer['offset']])
		upper_bound = min([itm for itm in buffer_divitions if itm >= (buffer['offset']+out_size)%buffer['Number of Addresses']],default=buffer['Number of Addresses'])
		if buffer['offset'] + out_size <= buffer['Number of Addresses']:
			buffer['Data'][:lower_bound*config['word size']] = 2
			buffer['Data'][upper_bound*config['word size']:] = 2
		else:
			buffer['Data'][upper_bound*config['word size']:lower_bound*config['word size']] = 2
		return upper_bound
	###################################################################################################
    ##FUNCTION NAME: write_Loop
    ##DESCRIPTION:   Loop de escritura de la capa actual en buffer
    ##OUTPUTS:       numero de ciclos empleados
    ############ARGUMENTS##############################################################################
    ####buffer:        np.array usado como buffer de memoria
    ####layer:         tf.keras.layer: Capa a simular
    ####activations:   np.array de activaciones a escribir
    ####manual_offset: offset manual para la escritura
    ####params:        diccionario de parametros del PE array.
    ###################################################################################################
	def write_Loop(buffer, layer, activations, manual_offset, params):
		# Debug timers
		t0 = 0
		t1 = 0
		layer_type = layer.__class__.__name__ if layer else 'CpuLayer'
		if DEBUG: print('------Current Address: ',buffer['offset'])
		if config['CNN-Gated']:
			next_bank     = OnOff_buffer(buffer,activations.size)
		initial_state     = buffer['Data'].copy()
		wrap = True if buffer['offset'] + manual_offset + activations.size > buffer['Number of Addresses'] else False
		if DEBUG: print('------Wrap: ',wrap)
		start_address     = (buffer['offset']+manual_offset)*config['word size']
		end_address       = start_address + activations.size*config['word size']
		#Work in a copy of the data
		high_cycles_count = np.zeros(end_address-start_address,dtype=np.uint32)	
		data              = np.take(buffer['Data'],range(start_address,end_address),mode='wrap')
		#Cuda parameter
		blockspergrid = (high_cycles_count.size + (threadsperblock - 1)) // threadsperblock
		#Update stats after the initial delay
		cycles = params['Initial Delay']
		update_high_cycle_count[blockspergrid, threadsperblock](data,high_cycles_count,params['Initial Delay'])
		#Acceleration of GPU Memory Transfer using small representation
		tmp_counter = 0
		if params['Procesing Cycles'] < 255:
			tmp_high_cycles_count = np.zeros(end_address-start_address,dtype=np.uint8)
			max_value = 255
		else:
			tmp_high_cycles_count = np.zeros(end_address-start_address,dtype=np.uint16)
			max_value = 65535
		if layer_type in ['Conv2D','Conv1D','DepthwiseConv2D']:
			for filter_patch in range(0,layer.output_shape[-1],params['Columns Used']):
				for activations_patch in range(0,activations.shape[0],params['Rows Used']):
					cycles      += params['Procesing Cycles']
					tmp_counter += params['Procesing Cycles']
					t0_tmp = time.time()    
					update_high_cycle_count[blockspergrid, threadsperblock](data,tmp_high_cycles_count,params['Procesing Cycles'])
					t0 += time.time() - t0_tmp
					# For the simulation speed purposes its assumed that the accelerator wait for a entire PE array activations (125) to be deposited in the local FIFO buffers and write all them in the same cycle
					t1_tmp = time.time()
					processed_outputs = activations[activations_patch:activations_patch+params['Rows Used'],filter_patch:filter_patch+params['Columns Used']]
					act_offset = activations_patch*layer.output_shape[-1]
					for out_slice in processed_outputs:
						write_in_buffer(out_slice,data,(act_offset+filter_patch)*config['word size'])
						act_offset += layer.output_shape[-1]
					t1 += time.time() - t1_tmp
					# Preventing possible overflow
					if tmp_counter + params['Procesing Cycles'] > max_value:
						high_cycles_count += tmp_high_cycles_count
						tmp_counter = 0
						tmp_high_cycles_count[:] = 0
		elif layer_type == 'Dense':
			for output_patch in range(0,activations.shape[0],params['Columns Used']*params['Rows Used']):
				cycles      += params['Procesing Cycles']
				tmp_counter += params['Procesing Cycles']
				t0_tmp = time.time()
				update_high_cycle_count[blockspergrid, threadsperblock](data,tmp_high_cycles_count,params['Procesing Cycles'])
				t0 += time.time() - t0_tmp
				# Its assumed that the accelerator can compute a neuron in each PE in parallel, again for simulation speed purposes the (125) outputs of a patch are writed in the same cycle they are processed.
				t1_tmp = time.time()
				processed_outputs = activations[output_patch:output_patch+params['Columns Used']*params['Rows Used']]
				write_in_buffer(processed_outputs,data,output_patch*config['word size'])
				t1 += time.time() - t1_tmp
				# Preventing possible overflow
				if tmp_counter + params['Procesing Cycles'] > max_value:
					high_cycles_count += tmp_high_cycles_count
					tmp_counter = 0
					tmp_high_cycles_count[:] = 0
		else:
			for activations_patch in range(0, activations.size, acts_per_cycle):
				cycles += params['Procesing Cycles']
				tmp_counter += params['Procesing Cycles']
				t0_tmp = time.time()
				update_high_cycle_count[blockspergrid, threadsperblock](data,tmp_high_cycles_count,params['Procesing Cycles'])
				t0 += time.time() - t0_tmp
				t1_tmp = time.time()
				processed_outputs = activations[activations_patch:activations_patch + acts_per_cycle]
				write_in_buffer(processed_outputs,data,activations_patch*config['word size'])
				t1 += time.time() - t1_tmp
				# Preventing possible overflow
				if tmp_counter + params['Procesing Cycles'] > max_value:
					high_cycles_count += tmp_high_cycles_count
					tmp_counter = 0
					tmp_high_cycles_count[:] = 0
		#Pass Leftovers
		high_cycles_count += tmp_high_cycles_count
		if DEBUG: print('Time spended in cuda: ',t0,'Time spended updating buffer: ',t1)
		#Update buffer stats
		if wrap:
			buffer['HighCyclesCount'][start_address:] += high_cycles_count[:buffer['Number of Addresses']*config['word size']-start_address]
			buffer['HighCyclesCount'][:end_address-buffer['Number of Addresses']*config['word size']] += high_cycles_count[buffer['Number of Addresses']*config['word size']-start_address:]
			buffer['HighCyclesCount'][end_address-buffer['Number of Addresses']*config['word size']:start_address][buffer['Data'][end_address-buffer['Number of Addresses']*config['word size']:start_address]==1] += cycles
			buffer['Data'][start_address:] = data[:buffer['Number of Addresses']*config['word size']-start_address]
			buffer['Data'][:end_address-buffer['Number of Addresses']*config['word size']] = data[buffer['Number of Addresses']*config['word size']-start_address:]
			buffer['OffCyclesCount'][buffer['Data']==2] += cycles
		else:
			buffer['HighCyclesCount'][start_address:end_address] += high_cycles_count
			buffer['HighCyclesCount'][:start_address][buffer['Data'][:start_address]==1] += cycles 
			buffer['HighCyclesCount'][end_address:][buffer['Data'][end_address:]==1]     += cycles
			buffer['Data'][start_address:end_address] = data
			buffer['OffCyclesCount'][buffer['Data']==2] += cycles
		buffer['Flips'][buffer['Data'] != initial_state] += 1
		if config['CNN-Gated']:
			buffer['offset'] = next_bank
		if DEBUG: print('------End Address: ',buffer['offset'])
		return cycles
	##############Layer Simulation#######################################################################
	#Clasic Convolution
	def sim_conv(buffer, layer, layer_outputs, write_buffer_id):
		t0     = datetime.now()
		params = hardware_utilization(layer)
		# Dont Simulate the second buffer write routine.
		if write_buffer_id == 1:
			if DEBUG: print('----Skiped due to unninsterested buffer')
			buffer['HighCyclesCount'][buffer['Data']==1] += params['Total cycles']
			buffer['OffCyclesCount'][buffer['Data']==2]  += params['Total cycles']
			return params['Total cycles']
		#Convert acts from (B,H,W,C) to (H*W,C)
		activations = layer_outputs[0].reshape(-1,layer_outputs[0].shape[-1])
		layer_size = activations.size*config['word size']
		if DEBUG: print('----layer size: ',layer_size//16)
		#Check if layer must be bypassed to DRAM
		if layer_size > buffer['Number of Addresses']*config['word size']:
			if DEBUG: print('----Bypassed')
			DRAM_bypass(buffer,config['CNN-Gated'],params['Total cycles'])
			return params['Total cycles']
		else:
			cycles = write_Loop(buffer,layer,activations,0,params)
			if DEBUG: print('----elapsed layer simulation time: ',str(datetime.now() - t0).split('.')[0])
			return cycles
	# Dense layer
	def sim_FC(buffer, layer, layer_outputs, write_buffer_id):
		t0     = datetime.now()
		params = hardware_utilization(layer)
		# Dont Simulate the second buffer write routine.
		if write_buffer_id == 1:
			if DEBUG: print('----Skiped due to unninsterested buffer')
			buffer['HighCyclesCount'][buffer['Data']==1] += params['Total cycles']
			buffer['OffCyclesCount'][buffer['Data']==2]  += params['Total cycles']
			return params['Total cycles']
		# Convert acts from (B,N) to (N) 
		activations = layer_outputs[0]
		layer_size = activations.size*config['word size']
		if DEBUG: print('----layer size: ',layer_size//16)
		cycles = write_Loop(buffer,layer,activations,0,params)
		if DEBUG: print('----elapsed layer simulation time: ',str(datetime.now() - t0).split('.')[0])
		return cycles
	#layer computed in cpu (Input,embedding)
	def sim_CPULayer(buffer, layer, layer_outputs, write_buffer_id):
		t0     = datetime.now()
		params = {'Initial Delay': 0, 'Procesing Cycles': 1}
		# Dont Simulate the second buffer write routine.
		if write_buffer_id == 1:
			if DEBUG: print('----Skiped due to unninsterested buffer')
			buffer['HighCyclesCount'][buffer['Data']==1] += int(np.ceil(layer_outputs.size/acts_per_cycle))
			buffer['OffCyclesCount'][buffer['Data']==2]  += int(np.ceil(layer_outputs.size/acts_per_cycle))
			return int(np.ceil(layer_outputs.size/acts_per_cycle))
		#Convert acts from (B,H,W,C) to (H*W,C)
		activations = layer_outputs[0].flatten()
		layer_size = activations.size*config['word size']
		if DEBUG: print('----layer size: ',layer_size//16)
		#Check if layer must be bypassed to DRAM
		if layer_size > buffer['Number of Addresses']*config['word size']:
			if DEBUG: print('----Bypassed')
			DRAM_bypass(buffer,config['CNN-Gated'],int(np.ceil(layer_outputs.size/acts_per_cycle)))
			return int(np.ceil(layer_outputs.size/acts_per_cycle))
		else:
			cycles = write_Loop(buffer,layer,activations,0,params)
			if DEBUG: print('----elapsed layer simulation time: ',str(datetime.now() - t0).split('.')[0])
			return cycles
	#Expand (SqueezeNet layer)
	def sim_expand(buffer, layers, layer_outputs, write_buffer_id):
		t0        = datetime.now()
		params1x1 = hardware_utilization(layers[0])
		params3x3 = hardware_utilization(layers[1])
		# Dont Simulate the second buffer write routine.
		if write_buffer_id == 1:
			if DEBUG: print('----Skiped due to unninsterested buffer')
			buffer['HighCyclesCount'][buffer['Data']==1] += params1x1['Total cycles'] + params3x3['Total cycles']
			buffer['OffCyclesCount'][buffer['Data']==2]  += params1x1['Total cycles'] + params3x3['Total cycles']
			return params1x1['Total cycles'] + params3x3['Total cycles']
		# Convert acts from (B,H,W,C) to (H*W,C) 
		out_channels = layer_outputs[0].shape[-1]
		activations  = layer_outputs[0].reshape(-1,out_channels)
		activations1x1 = activations[:,:out_channels//2]
		activations3x3 = activations[:,out_channels//2:]
		layer_size = (activations1x1.size + activations3x3.size)*config['word size']
		if DEBUG: print('----layer size: ',layer_size//16)
		#Check if layer must be bypassed to DRAM
		if layer_size > buffer['Number of Addresses']*config['word size']:
			if DEBUG: print('----Bypassed')
			DRAM_bypass(buffer,config['CNN-Gated'],params1x1['Total cycles'] + params3x3['Total cycles'])
			return params1x1['Total cycles'] + params3x3['Total cycles']
		else:
			cycles =  write_Loop(buffer,layers[0],activations1x1,0,params1x1)
			cycles += write_Loop(buffer,layers[1],activations3x3,activations1x1.size if (not config['CNN-Gated']) and (config['write mode'] == 'default') else 0,params3x3)
			if DEBUG: print('----elapsed layer simulation time: ',str(datetime.now() - t0).split('.')[0])
			return cycles
	#Concatenation of Conv (DenseNet layer)
	def sim_concat_conv(buffer, layer, layer_outputs, write_buffer_id):
		t0     = datetime.now()
		params = hardware_utilization(layer)
		# Dont Simulate the second buffer write routine.
		if write_buffer_id == 1:
			if DEBUG: print('----Skiped due to unninsterested buffer')
			buffer['HighCyclesCount'][buffer['Data']==1] += params['Total cycles'] + int(np.ceil(layer_outputs[1].size/acts_per_cycle))
			buffer['OffCyclesCount'][buffer['Data']==2]  += params['Total cycles'] + int(np.ceil(layer_outputs[1].size/acts_per_cycle))
			return params['Total cycles'] + int(np.ceil(layer_outputs[1].size/acts_per_cycle))
		#Convert acts from (B,H,W,C) to (H*W,C) and (H*W*C)
		activations1 = layer_outputs[0][0].reshape(-1,layer_outputs[0][0].shape[-1])
		activations2 = layer_outputs[1][0].flatten()
		layer_size1 = activations1.size*config['word size']
		layer_size2 = activations2.size*config['word size']
		if DEBUG: print('----layer size1: ',layer_size1//16,'layer size2: ',layer_size2//16)
		#Check if layer must be bypassed to DRAM
		if layer_size1 + layer_size2 > buffer['Number of Addresses']*config['word size']:
			if DEBUG: print('----Bypassed')
			DRAM_bypass(buffer,config['CNN-Gated'],params['Total cycles'] + int(np.ceil(layer_outputs[1].size/acts_per_cycle)) )
			return params['Total cycles'] + int(np.ceil(layer_outputs[1].size/acts_per_cycle))
		else:
			cycles = write_Loop(buffer,layer,activations1,0,params)
			cycles += write_Loop(buffer,None,activations2,activations1.size if (not config['CNN-Gated']) and (config['write mode'] == 'default') else 0,{'Initial Delay': 0,'Procesing Cycles': 1})
			if DEBUG: print('----elapsed layer simulation time: ',str(datetime.now() - t0).split('.')[0])
			return cycles
	#########Dispatcher to the simulation of the corresponding layer#####################################
	def handle_layer(buffer, layer, layer_outputs, out_buffer_id):
		if isinstance(layer_outputs,list):
			if DEBUG: print('--now processing: ConvConcat', layer.name)
			return sim_concat_conv(buffer, layer, layer_outputs, out_buffer_id)
		elif isinstance(layer, np.ndarray):
			if DEBUG: print('--now processing: Expand', layer[0].name,layer[1].name)
			return sim_expand(buffer, layer, layer_outputs, out_buffer_id)
		elif layer.__class__.__name__ in ['Conv2D','Conv1D','DepthwiseConv2D']:
			if DEBUG: print('--now processing: Conv', layer.name)
			return sim_conv(buffer, layer, layer_outputs, out_buffer_id)
		elif layer.__class__.__name__ == 'Dense':
			if DEBUG: print('--now processing Dense', layer.name)
			return sim_FC(buffer, layer, layer_outputs, out_buffer_id)
		else:
			if DEBUG: print('--now processing CPULayer', layer.name)
			return sim_CPULayer(buffer, layer, layer_outputs,out_buffer_id)
	#########Simulation layer by layer####################################################################
	cycles = 0
	# We only simulate buffer 2 (buffer 1 behaves statistically equal)
	if first_buffer == 1 and config['CNN-Gated']:
		buffer['Data'][:] = 2
	for index,layer in enumerate(layers):
		tmp = cycles
		cycles_temp=handle_layer(buffer, layer, activations[index], (first_buffer + index) % 2)
		layer_list.append(layer.__class__.__name__)
		cycles_by_layer.append(cycles_temp)
		print(cycles_by_layer)
		cycles += handle_layer(buffer,layer,activations[index],(first_buffer+index)%2)
		print('layer', layer.__class__.__name__)
		print('cycles layer by layer', cycles)
		df_layers_list = pd.DataFrame(layer_list)
		df_cycles_by_layer = pd.DataFrame(cycles_by_layer)
		df_layer_by_cycles = pd.concat([df_layers_list, df_cycles_by_layer], axis=1, join='outer')
		df_layer_by_cycles.columns = ['Capa', 'cycles']
		df_layer_by_cycles.to_excel('df_layer_by_cycles_' + str(network_type) + '.xlsx', sheet_name='fichero_707', index=False)

		if DEBUG: print('layer processed in: ',cycles-tmp,' cycles')
	return cycles
###################################################################################################
##FUNCTION NAME: buffer_simulation
##DESCRIPTION:   Realiza la simulacion de uso del buffer primario
##OUTPUTS:       Diccionario de estadisticas, numero de ciclos
############ARGUMENTS##############################################################################
####network:            tf.keras.model: modelo de la red a simular
####dataset:            iterarble del conjunto de datos a usar en inferencia
####integer_bits:       numero de bits parte entera de las activaciones
####fractional_bits:    numero de bits parte fraccionaria de las activaciones
####bit_invertion:      True si se desea aplicar inversion de bits cada 2 inferencias
####bit_shifting:       True si se desea aplicar desplazamiento de bits cada 2 inferencias
####write_mode:         True si se desea rotacion de direcciones luego de cada capa, activado
####                    por defecto si CNN_gating está activo
####CNN_gating:         True si se desea aplicar la tecnica de CNN gating
####buffer_size:        Tamaño en bytes del buffer
####start_from:         Imagen desde la cual empezar la inferencia
####results_dir:        Directorio para almacenar resultados
####layer_indexes:      indices de las capas de interes en simular
####activation_indixes: indices de las capas con las activaciones almacenadas en buffer
####save_results:       True si se desea guardar los resultados periodicamente en results_dir
####network_type:       'None' o 'MobileNet' (se realiza una simulacion especial para esta red)
###################################################################################################
def buffer_simulation(network, dataset, samples, integer_bits, fractional_bits, bit_invertion,
                      bit_shifting, write_mode, CNN_gating, buffer_size, start_from, results_dir,
					  layer_indexes, activation_indixes, save_results = True, network_type = None):
	###############Stats vars#############################################################################################################
	# Get initial state
	if start_from == 0:
		#Context
		config = {}
		# Data Representation
		config['word size']   = (1+integer_bits+fractional_bits)
		config['frac size']   = fractional_bits
		# Duty Techniques
		config['invert bits'] = 0
		config['bitshift']    = 0
		config['CNN-Gated']   = CNN_gating
		config['write mode']  = write_mode
		config['Number of switchable sections'] = 8
		#Buffer
		buffer = {}
		buffer['Number of Addresses'] = buffer_size//2
		buffer['Data']                = np.zeros(buffer_size*8,dtype=np.int8)
		buffer['HighCyclesCount']     = np.zeros(buffer_size*8,dtype=np.uint32)
		buffer['OffCyclesCount']      = np.zeros(buffer_size*8,dtype=np.uint32)
		buffer['LowCyclesCount']      = np.zeros(buffer_size*8,dtype=np.uint32)
		buffer['Flips']               = np.zeros(buffer_size*8,dtype=np.uint32)
		buffer['offset']              = 0
		# Initial variables
		cycles                        = 0
	else:
		buffer  = load_obj(results_dir + 'buffer')
		cycles  = load_obj(results_dir + 'cycles')
		config  = load_obj(results_dir + 'config')
	print('buffer sections: ',list(range(0,buffer['Number of Addresses']+1,buffer['Number of Addresses']//config['Number of switchable sections'])))
	################Simulation Loop########################################################################################################
	layers = [np.take(network.layers,index) for index in layer_indexes]
	X = [x for x,y in dataset]
	print('Simulation Started, time:',datetime.now().strftime("%H:%M:%S"),'cycles: ',cycles, 'offset: ',buffer['offset'])
	for index in range(start_from,samples):
		# Simulate the next image
		activacions = get_all_outputs(network,X[index])
		activacions = [activacions[itm] if type(itm) != tuple else [activacions[subitm] for subitm in itm] for itm in activation_indixes]
		if network_type == 'MobileNet':
			activacions[-1] = activacions[-1].reshape((1,1,1,activacions[-1].size))
		# una variable [] donde guardaré los ciclos de cada iteración sin sumar
		# una variable [arreglo] donde guardaré el nombre de las capas de cada iteración sin sumar
		# cycles_capa.append(cycles)

		cycles += simulate_one_image(network_type,layers,activacions,config, buffer, index%2)
		print('cycles',cycles)

		buffer['LowCyclesCount'] = cycles -  buffer['HighCyclesCount']  -  buffer['OffCyclesCount']
		# Update duty strategies
		if index % 2 == 0 and bit_invertion:
			print('bit_invertion',bit_invertion)
			config['invert bits'] = 1 - config['invert bits']
		if index % 2 == 0 and bit_shifting:
			print('bit_shifting',bit_shifting)
			config['bitshift'] = (config['bitshift'] + 1) % config['word size']
		# Save Results
		if index+1 % 25 == 0 and save_results:
			save_obj(buffer,results_dir+'buffer %s images'%index)
			save_obj(cycles, results_dir+'cycles %s images'%index)
			save_obj(config, results_dir+'config %s images'%index)
		if save_results:
			save_obj(buffer, results_dir+'buffer')
			save_obj(cycles, results_dir+'cycles')
			save_obj(config, results_dir+'config')
		print('procesed images:',index+1,' time:',datetime.now().strftime("%H:%M:%S"),'cycles: ',cycles, 'offset: ',buffer['offset'])
	return buffer,cycles