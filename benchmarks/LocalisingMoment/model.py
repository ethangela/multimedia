import ast
import glob
import tensorflow as tf
import logging
import numpy as np
import os
import pandas as pd
from tensorflow.keras import layers, Model, metrics
from tensorflow.keras.layers import LSTM, Dense, Input, ReLU, Concatenate, Flatten, Subtract, Lambda

LSTM_NUM_TIMESTEPS 	= 15 	# Also max sentence length
LSTM_INPUT_DIM 		= 768
LSTM_HIDDEN_UNITS 	= 128

VID_GLOBAL_FEAT 	= 2048 	# Dimension size
VID_LOCAL_FEAT 		= 2048
VID_TEMPORAL_FEAT 	= 2

DENSE_LAYER_1 		= 40
DENSE_OUTPUT_FEAT 	= 20

TRAINING_DATA_PATH 	= os.environ["TRAINING_DATA_PATH"]
CHECKPOINT_PATH 	= os.environ["MODEL_CHECKPOINT_PATH"]

logging.basicConfig(level = logging.INFO)

def data_generator():
	csv_files = glob.glob(TRAINING_DATA_PATH + "/*.csv")
	logging.info("Reading files: {0}".format(csv_files))

	while True:
		for each_file in csv_files:
			df = pd.read_csv(each_file)
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

				yield (temporal_tf, global_tf, local_tf, language_tf), (np.zeros(DENSE_OUTPUT_FEAT, dtype = np.float32))

def get_model():
	sentence_embedding_input 	= tf.keras.Input(shape = (LSTM_NUM_TIMESTEPS, LSTM_INPUT_DIM,), dtype = tf.float32)
	video_global_features 		= tf.keras.Input(shape = (VID_GLOBAL_FEAT,), dtype = tf.float32)
	video_local_features 		= tf.keras.Input(shape = (VID_LOCAL_FEAT,), dtype = tf.float32)
	video_temporal_features 	= tf.keras.Input(shape = (VID_TEMPORAL_FEAT,), dtype = tf.float32)

	# Sentence network
	lstm_1 = LSTM(LSTM_HIDDEN_UNITS, return_sequences = True, return_state = True)
	seq_out, hidden_out, carry_out = lstm_1(sentence_embedding_input)
	sentence_out 	= Dense(DENSE_OUTPUT_FEAT, activation=tf.nn.softmax)(hidden_out)

	# Video feature network
	merged_features = Concatenate()([video_global_features, video_local_features, video_temporal_features])
	dense_1			= Dense(DENSE_LAYER_1, activation=tf.nn.softmax)(merged_features)
	relu_1 			= ReLU()(dense_1)
	vid_feat_out 	= Dense(DENSE_OUTPUT_FEAT, activation=tf.nn.softmax)(dense_1)

	# Loss computation
	subtract_1 		= Subtract()([vid_feat_out, sentence_out])

	model = Model(inputs  = [video_temporal_features, video_global_features, video_local_features, sentence_embedding_input], 
				  outputs = subtract_1)	
	return model

if __name__ == "__main__":
	lm_net 		= get_model()

	# train_gen 	= data_generator(csv_dir = TRAINING_DATA_PATH)
	train_data 	= tf.data.Dataset.from_generator(data_generator, ((tf.float32, tf.float32, tf.float32, tf.float32), tf.float32)).batch(1)

	lm_net.compile(optimizer = tf.keras.optimizers.SGD(),
	               loss = tf.keras.losses.mean_squared_error,
	               metrics = [metrics.mean_squared_error])

	callbacks 	= [
				    tf.keras.callbacks.ModelCheckpoint(
				        filepath = os.path.join(CHECKPOINT_PATH, "lm_{epoch:02d}.hdf5"),
				        save_best_only = True,
				        monitor = "val_loss",
				        verbose = 1)
				]

	history 	= lm_net.fit(x = train_data,
							 batch_size = 1000,
							 steps_per_epoch = 100,
							 verbose = 1,
							 epochs = 100,
		                     callbacks = callbacks,
		                     workers = 10)
