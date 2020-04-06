import ast
import h5py
import multiprocessing
import numpy as np
import os
import pandas as pd
from bert_embedding import BertEmbedding
from editing.writer import DatasetWriter
from functools import partial

MAX_THREAD_POOL 		= int(os.environ.get("THREAD_POOL", multiprocessing.cpu_count()))
FEATURES_FILE 			= os.environ["FEATURES_FILE"]
SEGMENT_METADATA_CSV 	= os.environ["METADATA_CSV"]
SEGMENTED_CLIPS_ROOT 	= os.environ["SEGMENTED_CLIPS_ROOT"] 	# Write out to root dir here
SEGMENTED_CLIPS_PATH 	= os.environ.get("SEGMENTED_CLIPS_PATH", "_localizing_moments_bm")

IMAGE_FEATURES 	= h5py.File(FEATURES_FILE, "r")
bert_embedding 	= BertEmbedding()
data_writer 	= DatasetWriter()

def get_temporal_encoding(row) -> pd.Series:
	"""
	Gets normalized temporal encoding for start / end of key frames
	"""
	video_duration 	= row["segment_time_end"] - row["segment_time_start"]
	frame_labels 	= ast.literal_eval(row["ground_truth"])
	key_frame_indx 	= [indx for indx, val in enumerate(frame_labels) if val == 1]

	(min_frames, max_frames) 		= (0, len(frame_labels))
	(min_key_indx, max_key_indx) 	= (min(key_frame_indx), max(key_frame_indx))

	return 	[
				(min_key_indx - min_frames) / (max_frames - min_frames),
				(max_key_indx - min_frames) / (max_frames - min_frames),
			]

def get_global_encoding(features: h5py._hl.files.File, row) -> pd.Series:
	"""
	global encoding is the mean over all frame features
	"""
	video_name 		= row["segmented_video_id"].replace('.mp4', '')
	feat 			= np.array(features[video_name])
	global_feat 	= np.mean(feat, axis = 1)
	return pd.Series([global_feat])

def get_local_encoding(features: h5py._hl.files.File, row) -> pd.Series:
	"""
	local encoding is the mean over all key frames
	"""
	video_name 				= row["segmented_video_id"].replace('.mp4', '')
	feat 					= np.array(features[video_name])
	(feat_row, feat_cols) 	= feat.shape
	feat_cp 				= np.zeros((feat_row, feat_cols))

	frame_labels 	= ast.literal_eval(row["ground_truth"])
	frame_labels_np = np.array(frame_labels)

	for i in range(feat_cols):
		feat_cp[:, i] = feat[:, i] * frame_labels_np[:feat_row]

	local_feat 		= np.mean(feat_cp, axis = 1)
	return pd.Series([local_feat])

def get_language_encoding(row) -> pd.Series:
	"""
	Bert encoding for sentence
	"""
	sentence 			= row["text"]
	sentence_tokens 	= bert_embedding([sentence])
	embeddings 			= []
	for each_sentence_token in sentence_tokens:
		embeddings.append(list(each_sentence_token[1]))
	return embeddings

def data_preprocessing(features: h5py._hl.files.File, segment_df: pd.DataFrame) -> pd.DataFrame:
	"""
	Processing of data done in accordance to 
	Paper: Localizing Moments in Video with Natural Language [Hendricks et al.]
	"""
	df = pd.DataFrame(columns = ["temporal_enc", "global_enc", "local_enc", "language_enc"])
	df["temporal_enc"] 	= segment_df.apply(lambda row: get_temporal_encoding(row = row), axis = 1)
	df["global_enc"] 	= segment_df.apply(lambda row: get_global_encoding(features = features, row = row), axis = 1)
	df["local_enc"] 	= segment_df.apply(lambda row: get_local_encoding(features = features, row = row), axis = 1)
	df["language_enc"] 	= segment_df.apply(lambda row: get_language_encoding(row = row), axis = 1)
	return df

def execute(segment_df: pd.DataFrame) -> None:
	df = data_preprocessing(features = IMAGE_FEATURES, segment_df = segment_df)
	output_file = os.path.join(SEGMENTED_CLIPS_ROOT, SEGMENTED_CLIPS_PATH, "data.csv")
	data_writer.writeCsv(df = df, location = output_file)

if __name__ == "__main__":
	segment_metadata_df = pd.read_csv(SEGMENT_METADATA_CSV)
	execute(segment_df = segment_metadata_df)
	# segment_metadata_df_splits = np.array_split(segment_metadata_df, MAX_THREAD_POOL)
	# with multiprocessing.Pool(processes = MAX_THREAD_POOL) as pool:
	# 	pool.map(execute, segment_metadata_df_splits)
