import tensorflow as tf
from tensorflow.keras import layers, Model, metrics
from tensorflow.keras.layers import LSTM, Dense, Input, concatenate, Flatten

LSTM_NUM_TIMESTEPS 	= 10
LSTM_INPUT_DIM 		= 728
LSTM_HIDDEN_UNITS 	= 128

VID_GLOBAL_FEAT 	= 2048 	# Dimension size
VID_LOCAL_FEAT 		= 2048
VID_TEMPORAL_FEAT 	= 2

def get_model():
	sentence_embedding_input 	= tf.keras.Input(shape = (LSTM_NUM_TIMESTEPS, LSTM_INPUT_DIM), dtype = tf.float32)

	video_global_features 		= tf.keras.Input(shape = (VID_GLOBAL_FEAT), dtype = tf.float32)
	video_local_features 		= tf.keras.Input(shape = (VID_LOCAL_FEAT), dtype = tf.float32)
	video_temporal_features 	= tf.keras.Input(shape = (VID_TEMPORAL_FEAT), dtype = tf.float32)

	# Sentence network
	lstm_1 = LSTM(LSTM_HIDDEN_UNITS, return_sequences=True, return_state=True)
	seq_out, hidden_out, carry_out = lstm_1(sentence_embedding_input)
	sentence_out 	= Dense(32, activation=tf.nn.softmax)(hidden_out)

	# Video feature network
	merged_features = tf.concat([video_global_features, video_local_features, video_temporal_features], axis = 1)
	dense_1			= Dense(50, activation=tf.nn.softmax)(merged_features)
	relu_1 			= tf.nn.relu(dense_1)
	vid_feat_out 	= Dense(32, activation=tf.nn.softmax)(dense_1)

	# Loss computation
	l2_norm_loss	 = tf.norm(vid_feat_out - sentence_out, ord='euclidean', axis = 1)

	model = Model(inputs = [sentence_embedding_input, video_global_features, video_local_features, video_temporal_features], 
				  outputs = [l2_norm_loss])	

if __name__ == "__main__":
	lm_net 	= get_model()

	lm_net.compile(optimizer = tf.keras.optimizers.SGD(),
	               loss = tf.keras.losses.MeanSquaredError(),
	               metrics=[metrics.mean_squared_error])

	callbacks = [
				    tf.keras.callbacks.ModelCheckpoint(
				        filepath='mymodel_{epoch:02d}_{val_loss:.2f}.hdf5',
				        save_best_only=True,
				        monitor='val_loss',
				        verbose=1)
				]

	history = model.fit(x_train, y_train,
						epochs = 1000,
	                    batch_size = 100,
	                    callbacks = callbacks,
	                    validation_data = (x_val, y_val))
