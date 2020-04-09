import ast
import glob
import h5py
import logging
import multiprocessing
import numpy as np
import os
import pandas as pd
from bert_embedding import BertEmbedding
from editing.writer import DatasetWriter
from functools import partial
from tqdm import tqdm

MAX_THREAD_POOL 		= int(os.environ.get("THREAD_POOL", multiprocessing.cpu_count()))
FEATURES_FILE 			= os.environ["FEATURES_FILE"]
SEGMENT_METADATA_CSV 	= os.environ["METADATA_CSV"]
SEGMENTED_CLIPS_ROOT 	= os.environ["SEGMENTED_CLIPS_ROOT"] 	# Write out to root dir here
SEGMENTED_CLIPS_PATH 	= os.environ.get("SEGMENTED_CLIPS_PATH", "_localizing_moments_bm")

bert_embedding 	= BertEmbedding()
data_writer 	= DatasetWriter()

logging.basicConfig(level=logging.INFO)
tqdm.pandas()

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
	global_feat 	= np.mean(feat, axis = 0)
	return global_feat.tolist()

def get_local_encoding(features: h5py._hl.files.File, row) -> pd.Series:
	"""
	local encoding is the mean over all key frames
	"""
	video_name 				= row["segmented_video_id"].replace('.mp4', '')
	feat 					= np.array(features[video_name])
	(feat_row, feat_cols) 	= feat.shape
	feat_cp 				= np.zeros((feat_row, feat_cols))

	frame_labels 		= ast.literal_eval(row["ground_truth"])
	frame_labels_row 	= len(frame_labels)
	frame_labels_np 	= np.array(frame_labels)

	for i in range(feat_cols):
		if feat_row > frame_labels_row:
			feat_cp[:, i] = feat[:frame_labels_row, i] * frame_labels_np
		else:
			feat_cp[:, i] = feat[:, i] * frame_labels_np[:feat_row]

	feat_cp_wo_zeros 	= feat_cp[~np.all(feat_cp == 0, axis = 1)]
	local_feat 			= np.mean(feat_cp_wo_zeros, axis = 0)
	if local_feat is None:
		print(feat_cp)
	return local_feat.tolist()

def get_language_encoding(row) -> pd.Series:
	"""
	Bert encoding for sentence
	"""
	sentence 			= row["text"]
	sentence_tokens 	= bert_embedding([sentence])
	embeddings 			= []
	for each_sentence_token in sentence_tokens[0][1]:
		embeddings.append(each_sentence_token.tolist())
	return embeddings

def data_preprocessing(features: h5py._hl.files.File, segment_df: pd.DataFrame) -> pd.DataFrame:
	"""
	Processing of data done in accordance to 
	Paper: Localizing Moments in Video with Natural Language [Hendricks et al.]
	"""
	df = pd.DataFrame(columns = ["temporal_enc", "global_enc", "local_enc", "language_enc"])
	df["temporal_enc"] 	= segment_df.progress_apply(lambda row: get_temporal_encoding(row = row), axis = 1)
	df["global_enc"] 	= segment_df.progress_apply(lambda row: get_global_encoding(features = features, row = row), axis = 1)
	df["local_enc"] 	= segment_df.progress_apply(lambda row: get_local_encoding(features = features, row = row), axis = 1)
	df["language_enc"] 	= segment_df.progress_apply(lambda row: get_language_encoding(row = row), axis = 1)
	return df

def execute(segment_df: pd.DataFrame) -> None:
	image_features 	= h5py.File(FEATURES_FILE, "r")
	df = data_preprocessing(features = image_features, segment_df = segment_df)
	output_file = os.path.join(SEGMENTED_CLIPS_ROOT, SEGMENTED_CLIPS_PATH, "{0}_lm_data.csv".format(os.getpid()))
	data_writer.writeCsv(df = df, location = output_file)
	image_features.close()

if __name__ == "__main__":
	prior_segment_metadata_files = glob.glob(os.path.join(SEGMENTED_CLIPS_ROOT, SEGMENTED_CLIPS_PATH) + "/*lm_data.csv")
	logging.warning("Deleting prior files: {0}".format(prior_segment_metadata_files))
	for prior_files in prior_segment_metadata_files:
		os.remove(prior_files)

	segment_metadata_df = pd.read_csv(SEGMENT_METADATA_CSV)
	# execute(segment_df = segment_metadata_df)
	segment_metadata_df_splits = np.array_split(segment_metadata_df, MAX_THREAD_POOL)
	with multiprocessing.Pool(processes = MAX_THREAD_POOL) as pool:
		pool.map(execute, segment_metadata_df_splits)
