"""
Sample TVR-XML dataset can be found at https://github.com/JKOK005/TVRetrieval/tree/master/data

This script converts our CSV structure into the format supported by TVR-XML. Repo in above ^^
This is needed for fine tuning the language model using our vocal.

Reference: TVR: A Large-Scale Dataset for Video-Subtitle Moment Retrieval [Lei et al.]
"""

import copy
import glob
import json
import logging
import os
import numpy as np
import pandas as pd 
from editing.writer import DatasetWriter
from typing import Dict

SEGMENTED_CLIPS_ROOT 	= os.environ["SEGMENTED_CLIPS_ROOT"] 	# Write out to root dir here
TRAIN_VAL_PATH 			= os.environ.get("TRAIN_VAL_PATH", "_train_val")
TVR_PATH 				= os.environ.get("TVR_PATH", "_tvr")

data_writer = DatasetWriter()

def convert_to_tvr_subtitle(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Not implementing this part atm. Subtitles are not important for our application
	"""
	pass

def convert_to_tvr_query(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
	tvr_df 	= copy.copy(df)
	tvr_filter_relevant_columns_df 	= tvr_df[["segmented_video_id", "text"]]
	tvr_filter_relevant_columns_df 	= tvr_filter_relevant_columns_df.rename(columns = {	"segmented_video_id" 	: "vid_name", 
													 									"text" : "desc"})

	assigned_constants = {	"duration" 	: 4, 
							"type" 		: data_type
						}

	tvr_filter_relevant_columns_df 	= tvr_filter_relevant_columns_df.assign(**assigned_constants)
	tvr_filter_relevant_columns_df["ts"] = tvr_filter_relevant_columns_df.apply(lambda row: [0,4])  
	tvr_filter_relevant_columns_df["desc_id"] = np.arange(len(tvr_df))
	tvr_filter_relevant_columns_df["vid_name"] = tvr_filter_relevant_columns_df["vid_name"].str.replace(r'.mp4', '')
	return tvr_filter_relevant_columns_df

if __name__ == "__main__":
	# For training dataset
	train_files 	= glob.glob(os.path.join(SEGMENTED_CLIPS_ROOT, TRAIN_VAL_PATH) + "/*train.csv")
	train_df 		= pd.concat(map(pd.read_csv, train_files))
	train_tvr_df 	= convert_to_tvr_query(df = train_df, data_type = "t")
	train_file_path = os.path.join(SEGMENTED_CLIPS_ROOT, TVR_PATH, "tvr_train.jsonl")
	data_writer.writeJsonL(df = train_tvr_df, location = train_file_path)

	# For validation dataset
	validation_files 		= glob.glob(os.path.join(SEGMENTED_CLIPS_ROOT, TRAIN_VAL_PATH) + "/*validation.csv")
	validation_df 			= pd.concat(map(pd.read_csv, validation_files))
	validation_tvr_df 		= convert_to_tvr_query(df = validation_df, data_type = "v")
	validation_file_path 	= os.path.join(SEGMENTED_CLIPS_ROOT, TVR_PATH, "tvr_validation.jsonl")
	data_writer.writeJsonL(df = validation_tvr_df, location = validation_file_path)