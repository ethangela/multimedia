import ast
import glob
import math
import tensorflow as tf
import logging
import numpy as np
import os
import pandas as pd
import time
from tensorflow.keras import layers, Model, metrics
from tensorflow.keras.layers import LSTM, Dense, Input, ReLU, Concatenate, Subtract, BatchNormalization

LSTM_NUM_TIMESTEPS 	= 15 	# Also max sentence length
LSTM_INPUT_DIM 		= 768
LSTM_HIDDEN_UNITS 	= 128

VID_GLOBAL_FEAT 	= 2048 	# Dimension size
VID_LOCAL_FEAT 		= 2048
VID_TEMPORAL_FEAT 	= 2

DENSE_LAYER_1 		= 40
DENSE_OUTPUT_FEAT 	= 20

TRAINING_DATA_PATH 		= os.environ["TRAINING_DATA_PATH"]
VALIDATION_DATA_PATH 	= os.environ["VALIDATION_DATA_PATH"]
CHECKPOINT_PATH 		= os.environ["MODEL_CHECKPOINT_PATH"]
BATCH_SIZE 				= int(os.environ.get("BATCH_SIZE", 1000))

EXECUTION_TIME 			= int(time.time())

logging.basicConfig(level = logging.INFO)

logging.info("Setting batch size to {0}".format(BATCH_SIZE))

def df_processing(df):
	temporal_enc_arr 	= None
	global_enc_arr 		= None
	local_enc_arr 		= None
	language_enc_arr 	= None

	for _, row in df.iterrows():
		temporal_enc = ast.literal_eval(row["temporal_enc"])
		global_enc = ast.literal_eval(row["global_enc"])
		local_enc = ast.literal_eval(row["local_enc"])
		language_enc = ast.literal_eval(row["language_enc"])

		temporal_tf = np.array(temporal_enc, dtype = np.float32) 
		global_tf = np.array(global_enc, dtype = np.float32)
		local_tf = np.array(local_enc, dtype = np.float32)
		language_tf = np.array(language_enc, dtype = np.float32)

		if language_tf.shape[0] < LSTM_NUM_TIMESTEPS:
			# Need to pad array to meet max LSTM timesteps
			# Padding done only at the bottom of the array
			current_rows = language_tf.shape[0]
			language_tf = np.pad(language_tf, [(0, LSTM_NUM_TIMESTEPS - current_rows), (0, 0)], mode = 'constant', constant_values = 0)  

		temporal_tf = np.expand_dims(temporal_tf, axis=0) 
		global_tf = np.expand_dims(global_tf, axis=0) 
		local_tf = np.expand_dims(local_tf, axis=0) 
		language_tf = np.expand_dims(language_tf, axis=0) 

		temporal_enc_arr = np.vstack((temporal_tf, temporal_enc_arr)) if temporal_enc_arr is not None else temporal_tf
		global_enc_arr = np.vstack((global_tf, global_enc_arr)) if global_enc_arr is not None else global_tf
		local_enc_arr = np.vstack((local_tf, local_enc_arr)) if local_enc_arr is not None else local_tf
		language_enc_arr = np.vstack((language_tf, language_enc_arr)) if language_enc_arr is not None else language_tf
	
	return (temporal_enc_arr, global_enc_arr, local_enc_arr, language_enc_arr)

def get_training_data_generator():
	csv_files = glob.glob(TRAINING_DATA_PATH + "/*.csv")
	logging.info("Reading training files: {0}".format(csv_files))

	while True:
		for each_file in csv_files:
			df = pd.read_csv(each_file)
			split_chunks = math.ceil(len(df) / BATCH_SIZE)
			df_split = np.array_split(df, split_chunks)

			for each_df_split in df_split:
				(temporal_enc_arr, global_enc_arr, local_enc_arr, language_enc_arr) = df_processing(df = each_df_split)
				yield ((temporal_enc_arr, global_enc_arr, local_enc_arr, language_enc_arr), (np.zeros((len(each_df_split), DENSE_OUTPUT_FEAT), dtype = np.float32)))

def get_validation_data_generator():
	csv_files = glob.glob(VALIDATION_DATA_PATH + "/*.csv")
	logging.info("Reading validation files: {0}".format(csv_files))

	while True:
		for each_file in csv_files:
			df = pd.read_csv(each_file)
			split_chunks = math.ceil(len(df) / BATCH_SIZE)
			df_split = np.array_split(df, split_chunks)

			for each_df_split in df_split:
				(temporal_enc_arr, global_enc_arr, local_enc_arr, language_enc_arr) = df_processing(df = each_df_split)
				yield ((temporal_enc_arr, global_enc_arr, local_enc_arr, language_enc_arr), (np.zeros((len(each_df_split), DENSE_OUTPUT_FEAT), dtype = np.float32)))

def get_training_batch():
	csv_files = glob.glob(TRAINING_DATA_PATH + "/*.csv")
	logging.info("Reading validation files: {0}".format(csv_files))

	training_df = pd.concat([pd.read_csv(each_file) for each_file in csv_files])
	(temporal_enc_arr, global_enc_arr, local_enc_arr, language_enc_arr) = df_processing(df = training_df)

	labels = np.zeros((len(validation_df), DENSE_OUTPUT_FEAT), dtype = np.float32)
	return ((temporal_enc_arr, global_enc_arr, local_enc_arr, language_enc_arr), labels)

def get_validation_batch():
	csv_files = glob.glob(VALIDATION_DATA_PATH + "/*.csv")
	logging.info("Reading validation files: {0}".format(csv_files))

	validation_df = pd.concat([pd.read_csv(each_file) for each_file in csv_files])
	(temporal_enc_arr, global_enc_arr, local_enc_arr, language_enc_arr) = df_processing(df = validation_df)

	labels = np.zeros((len(validation_df), DENSE_OUTPUT_FEAT), dtype = np.float32)
	return ((temporal_enc_arr, global_enc_arr, local_enc_arr, language_enc_arr), labels)

def get_model():
	sentence_embedding_input 	= tf.keras.Input(shape = (LSTM_NUM_TIMESTEPS, LSTM_INPUT_DIM,), dtype = tf.float32)
	video_global_features 		= tf.keras.Input(shape = (VID_GLOBAL_FEAT,), dtype = tf.float32)
	video_local_features 		= tf.keras.Input(shape = (VID_LOCAL_FEAT,), dtype = tf.float32)
	video_temporal_features 	= tf.keras.Input(shape = (VID_TEMPORAL_FEAT,), dtype = tf.float32)

	# Sentence network
	lstm_1 = LSTM(LSTM_HIDDEN_UNITS, return_sequences = True, return_state = True)
	seq_out, hidden_out, carry_out = lstm_1(sentence_embedding_input)
	sentence_out 	= Dense(DENSE_OUTPUT_FEAT, activation=tf.nn.softmax)(hidden_out)
	sentence_out_norm = BatchNormalization()(sentence_out)

	# Video feature network
	merged_features = Concatenate()([video_global_features, video_local_features, video_temporal_features])
	dense_1			= Dense(DENSE_LAYER_1, activation=tf.nn.softmax)(merged_features)
	relu_1 			= ReLU()(dense_1)
	vid_feat_out 	= Dense(DENSE_OUTPUT_FEAT, activation=tf.nn.softmax)(dense_1)
	vid_feat_out_norm = BatchNormalization()(vid_feat_out)

	# Loss computation
	subtract_1 		= Subtract()([vid_feat_out_norm, sentence_out_norm])

	model = Model(inputs  = [video_temporal_features, video_global_features, video_local_features, sentence_embedding_input], 
				  outputs = subtract_1)	
	return model

if __name__ == "__main__":
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

	config = tf.ConfigProto()
	sess = tf.Session(config=config)

	train_data_gen 			= tf.data.Dataset.from_generator(generator = get_training_data_generator, 
															 output_types = ((tf.float32, tf.float32, tf.float32, tf.float32), tf.float32),
															 output_shapes = 	((	
													 					tf.TensorShape([None, VID_TEMPORAL_FEAT]), 
													 					tf.TensorShape([None, VID_GLOBAL_FEAT]), 
													 					tf.TensorShape([None, VID_LOCAL_FEAT]), 
													 					tf.TensorShape([None, LSTM_NUM_TIMESTEPS, LSTM_INPUT_DIM])
											 						), 
												 					tf.TensorShape([None, DENSE_OUTPUT_FEAT])
												 					)
												 				)

	validation_data_gen 	= tf.data.Dataset.from_generator(generator = get_validation_data_generator, 
															 output_types = ((tf.float32, tf.float32, tf.float32, tf.float32), tf.float32),
															 output_shapes = 	((	
													 					tf.TensorShape([None, VID_TEMPORAL_FEAT]), 
													 					tf.TensorShape([None, VID_GLOBAL_FEAT]), 
													 					tf.TensorShape([None, VID_LOCAL_FEAT]), 
													 					tf.TensorShape([None, LSTM_NUM_TIMESTEPS, LSTM_INPUT_DIM])
											 						), 
												 					tf.TensorShape([None, DENSE_OUTPUT_FEAT])
												 					)
												 				)

	lm_net 		= get_model()

	lm_net.compile(optimizer = tf.keras.optimizers.SGD(),
	               loss = tf.keras.losses.mean_squared_error,
	               metrics = [metrics.mean_squared_error])

	checkpoint_path 	= os.path.join(CHECKPOINT_PATH, str(EXECUTION_TIME))
	os.mkdir(checkpoint_path)

	callbacks 			= [
						    tf.keras.callbacks.ModelCheckpoint(
						        filepath = os.path.join(checkpoint_path, "lm_{epoch:02d}_{val_loss:.8f}.hdf5"),
						        save_best_only = False,
						        verbose = 1)
						]

	history 			= lm_net.fit(x = train_data_gen,
									 steps_per_epoch = math.ceil(562639 / BATCH_SIZE),
									 validation_data = validation_data_gen,
									 validation_steps = math.ceil(56281 / BATCH_SIZE),
 									 epochs = 200,
									 verbose = 1,
				                     callbacks = callbacks,
				                     use_multiprocessing = True,
				                     max_queue_size=10,
				                     workers = 100)
