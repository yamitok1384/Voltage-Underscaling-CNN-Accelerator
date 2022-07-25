import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
import os
from Training import GetDatasets
from Nets import GetNeuralNetworkModel
from Simulation import buffer_simulation
from Stats import WeightQuantization

if __name__=='__main__':
	# Set weight directory
	cwd = os.getcwd()
	wgt_dir = os.path.join(cwd, 'Data')
	wgt_dir = os.path.join(wgt_dir, 'Trained Weights')
	wgt_dir = os.path.join(wgt_dir, 'AlexNet')
	wgt_dir = os.path.join(wgt_dir, 'Colorectal Dataset')
	wgt_dir = os.path.join(wgt_dir,'Weights')
	trainBatchSize = testBatchSize = 1
	#Select Datasets, and shape
	train_batchSize = test_batchSize = 1
	_,_,test_set = GetDatasets('colorectal_histology',(80,5,15),(227,227), 8, train_batchSize, test_batchSize)
	#Select Network and input dimensions
	QNet  = GetNeuralNetworkModel('AlexNet',(227,227,3),8, quantization = True, aging_active=False, word_size = 16, frac_size = 11)
	#Set parameters (loss,optimizer,metrics)
	loss = tf.keras.losses.CategoricalCrossentropy()
	optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
	metrics = ['accuracy']
	QNet.compile(optimizer=optimizer, loss=loss, metrics=metrics)
	QNet.load_weights(wgt_dir).expect_partial()
	# Select bit representation
	WeightQuantization(model = QNet, frac_bits = 11, int_bits = 4)

	#Indicate the layers and activation indexes
	LI = [0,3,9,11,17,19,25,31,37,40,45,50]
	AI = [2,8,10,16,18,24,30,36,38,44,49,53]
	#Indicate data representation, start point, tehcniques, directory, buffer size.
	buffer_simulation(QNet, test_set, integer_bits = 4, fractional_bits = 11, samples = 5, start_from = 0,
                      bit_invertion = False, bit_shifting = False, CNN_gating = False,
                      buffer_size = 2*290400, write_mode ='default', save_results = False,
                      results_dir = 'Data/',
                      layer_indexes = LI , activation_indixes = AI)