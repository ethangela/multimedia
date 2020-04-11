import ast
import copy
import tensorflow as tf 
import h5py
import logging
import numpy as np
import os
import pandas as pd
from benchmarks.LocalisingMoment.data_preparation import get_global_encoding, get_language_encoding, get_temporal_encoding
from benchmarks.LocalisingMoment.model import compile_model
from bert_embedding import BertEmbedding

MODEL_WEIGHTS_PATH 	= os.environ["MODEL_WEIGHTS_PATH"]
TEST_CSV 			= os.environ["TEST_CSV"]
FEATURES_FILE 		= os.environ["FEATURES_FILE"]

LSTM_NUM_TIMESTEPS 	= 15
TEST_TRAILS 		= 10
POSITIVE_SAMPLES 	= 10
NEGATIVE_SAMPLES 	= 20
K_BEST				= 10
DENSE_OUTPUT_FEAT 	= 20
IOU 				= 0.5
KEY_FRAME_THRESHOLD = 0.7

bert_embedding 		= BertEmbedding()
logging.basicConfig(level=logging.INFO)

"""
Test is done using Recall@K with IoU = m

Our approach is a modification of TALL: Temporal Activity Localization via Language Query [Gao et al.]

We want to determine the average percentage of videos within top K selected videos that
1) Fits the correct video class from user search intent
2) Has IoU >= m

We do the following:
1) Select a random video from the test set (without replacement)
2) Select P other videos of similar class at random
3) Select N other videos of different classes at random
4) Iterate through each frame of a video to calculate its error from user input
5) Define a threshold by which error < threshold is considered a key frame
6) Obtain video IoU value by comparing with Ground truth
7) Obtain video score: Summation of all error to perform video ranking
8) Identify #videos with IoU > m and correct classes from baseline
"""

def select_incorrect_class_df(df: pd.DataFrame, candidate_row, size: int) -> pd.DataFrame:
	candidate_row_class 	= candidate_row["classname"]
	negative_example_df 	= df[df["classname"] != candidate_row_class]
	shuffled_df 			= negative_example_df.sample(frac = 1)
	return shuffled_df[:NEGATIVE_SAMPLES]

def select_same_class_df(df: pd.DataFrame, candidate_row, size: int) -> pd.DataFrame:
	candidate_row_class 	= candidate_row["classname"]
	positive_example_df 	= df[df["classname"] == candidate_row_class]
	shuffled_df 			= positive_example_df.sample(frac = 1)
	return shuffled_df[:POSITIVE_SAMPLES]

def get_video_error(model, global_encoding, temporal_encoding, frame_encodings, sentence_encoding):
	frame_errors = []
	for each_frame_encoding in frame_encodings:
		each_frame_encoding = np.array(each_frame_encoding, dtype = np.float32)
		each_frame_encoding = np.expand_dims(each_frame_encoding, axis=0) 

		loss, _ = model.evaluate(	x = (temporal_encoding, global_encoding, each_frame_encoding, sentence_encoding), 
									y = (np.zeros((1, DENSE_OUTPUT_FEAT), dtype = np.float32)))
		frame_errors.append(loss)
	return frame_errors

def get_key_frames(video_errors: [float], threshold: float) -> [int]:
	video_errors_np 	= np.array(video_errors, dtype = np.float32)
	normalized_errrors 	= (video_errors_np - min(video_errors_np)) / (max(video_errors_np) - min(video_errors_np))
	return [int(i >= KEY_FRAME_THRESHOLD) for i in normalized_errrors]

def evaluate_df(candidate_df: pd.DataFrame, model: tf.keras.Model, image_features: h5py._hl.files.File, sentence_encoding) -> pd.DataFrame:
	evaluation_df 	= copy.copy(candidate_df)
	evaluation_df 	= pd.concat([evaluation_df, pd.DataFrame(columns = ["predicted_error", "key_frame_labels"])])

	sentence_encoding 	= np.expand_dims(sentence_encoding, axis=0) 
	for indx, row in evaluation_df.iterrows():
		video_name 			= row["segmented_video_id"].replace('.mp4', '')

		global_encoding 	= get_global_encoding(features = image_features, row = row)
		global_encoding 	= np.array(global_encoding, dtype = np.float32)
		global_encoding 	= np.expand_dims(global_encoding, axis=0) 

		frame_encodings 	= image_features[video_name]

		temporal_encoding 	= get_temporal_encoding(row = row)
		temporal_encoding 	= np.array(temporal_encoding, dtype = np.float32)
		temporal_encoding 	= np.expand_dims(temporal_encoding, axis=0) 

		video_errors 		= get_video_error(	model = model, global_encoding = global_encoding, 
												temporal_encoding = temporal_encoding, frame_encodings = frame_encodings, 
												sentence_encoding = sentence_encoding)

		evaluation_df["predicted_error"][indx] 		= sum(video_errors)
		evaluation_df["key_frame_labels"][indx] 	= get_key_frames(video_errors = video_errors, threshold = KEY_FRAME_THRESHOLD)
	return evaluation_df

def get_iou_df(evaluation_df: pd.DataFrame, ground_truth_row):
	iou_df 		= copy.copy(evaluation_df)
	iou_df 		= pd.concat([iou_df, pd.DataFrame(columns = ["iou"])])
	
	for indx, row in iou_df.iterrows():
		predicted_np 			= np.array(evaluation_df["key_frame_labels"][indx], dtype = int)

		ground_truth_labels 	= ast.literal_eval(ground_truth_row["ground_truth"])
		ground_truth_np 		= np.array(ground_truth_labels, dtype = int)[:len(predicted_np)]

		intersection 			= 0
		for i in range(len(predicted_np)):
			intersection 		+= 1 if predicted_np[i] == ground_truth_np[i] == 1 else 0

		union 					= sum(ground_truth_np) + sum(predicted_np) - intersection
		iou_df["iou"][indx] 	= intersection / union
	return iou_df

def get_correct_vid_selection_count(best_k_df: pd.DataFrame, ground_truth_row):
	correct_selections = 0
	for _, row in best_k_df.iterrows():
		# Assert correct class selection
		is_correct = (row["classname"] == ground_truth_row["classname"]) and (row["iou"] >= IOU)
		correct_selections += int(is_correct)
	return correct_selections

def init_test(df: pd.DataFrame, model: tf.keras.Model) -> np.ndarray:
	image_features = h5py.File(FEATURES_FILE, "r")
	results = []

	for _, row in df[:TEST_TRAILS].iterrows():
		logging.info("Evaluating video {0} of class {1}".format(row["unique_clip_name"], row["classname"]))

		sentence_encoding 			= get_language_encoding(row = row)
		sentence_encoding 			= np.array(sentence_encoding, dtype = np.float32)
		words 						= sentence_encoding.shape[0]
		sentence_encoding_padded 	= np.pad(sentence_encoding, [(0, LSTM_NUM_TIMESTEPS - words), (0, 0)], mode = 'constant', constant_values = 0)

		positive_sample_df 	= select_same_class_df(df = df, candidate_row = row, size = POSITIVE_SAMPLES)
		negative_sample_df 	= select_incorrect_class_df(df = df, candidate_row = row, size = NEGATIVE_SAMPLES)
		union_df 			= pd.concat([row, positive_sample_df, negative_sample_df])

		evaluation_df 		= evaluate_df(	candidate_df = union_df, model = model, 
											image_features = image_features, sentence_encoding = sentence_encoding_padded)

		iou_df 				= get_iou_df(evaluation_df = evaluation_df, ground_truth_row = row)

		# Take top K videos
		best_k_df 			= iou_df.sort_values(by = "predicted_error", ascending = True)[0 : K_BEST]

		correct_selections 	= get_correct_vid_selection_count(best_k_df = best_k_df, ground_truth_row = row)
		total_choices 		= len(best_k_df)

		logging.info("IoU: {0}, Top K: {1}, Selection {2} / {3}".format(IOU, K_BEST, 
																		correct_selections, total_choices))
		results.append((correct_selections, total_choices))
	return results

if __name__ == "__main__":
	logging.info("Loading model weights from {0}".format(MODEL_WEIGHTS_PATH))
	model 	= compile_model()
	model.load_weights(MODEL_WEIGHTS_PATH)

	df 		= pd.read_csv(TEST_CSV)
	results = init_test(df = df, model = model)
