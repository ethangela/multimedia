import ast
import copy
import h5py
import logging
import numpy as np
import os
import pandas as pd 
from editing.writer import DatasetWriter
from numpy import random

FEATURES_FILE 			= os.environ["FEATURES_FILE"]
SEGMENT_METADATA_CSV 	= os.environ["METADATA_CSV"]
PERMUTATION_ROOT 		= os.environ["PERMUTATION_ROOT"]
SHIFT_MAX_VAL 			= os.environ.get("MAX_SHIFT_INDX", 20)

logging.basicConfig(level=logging.INFO)

data_writer = DatasetWriter()

def get_shift() -> int:
	"""
	Returns a |random value| <= SHIFT_MAX_VAL
	"""
	return random.randint(low = -1 *SHIFT_MAX_VAL, high = SHIFT_MAX_VAL)

def shift_video(video_feature, shift: int) -> np.ndarray:
	return np.roll(video_feature, shift, axis=0)

def shift_ground_truth(ground_truth: [int], shift: int) -> np.ndarray:
	return np.roll(ground_truth, shift, axis=0)

def execute(df: pd.DataFrame):
	video_features_all = h5py.File(FEATURES_FILE, "r")
	video_feature_out_path = os.path.join(PERMUTATION_ROOT, "_shift.hdf5")
	video_feature_copied = h5py.File(video_feature_out_path, "w")

	df_copied = copy.copy(df)
	df_out_path = os.path.join(PERMUTATION_ROOT, "_shift.csv")
	results = []

	logging.info("Gittering videos")

	for indx, row in df.iterrows():
		shift_val = get_shift()
		
		video_name = row["segmented_video_id"].replace('.mp4', '')
		video_feature = video_features_all[video_name]
		video_feature_shifted = shift_video(video_feature = video_feature, shift = shift_val)
		video_feature_copied.create_dataset(video_name, data=video_feature_shifted)

		ground_truth = ast.literal_eval(row["ground_truth"])
		ground_truth_shifted = shift_ground_truth(ground_truth = ground_truth, shift = shift_val)
		df_copied.loc(indx, "ground_truth") = str(ground_truth_shifted.tolist())

	logging.info("Write out CSV")
	df_out_path = os.path.join(PERMUTATION_ROOT, "_shift.csv")
	data_writer.writeCsv(df = df_copied, location = df_out_path)

if __name__ == "__main__":
	df = pd.read_csv(SEGMENT_METADATA_CSV)
	execute(df = df)