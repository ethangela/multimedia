import glob
import json
import logging
import math
import os
import pandas as pd 
from editing.writer import DatasetWriter

SEGMENTED_CLIPS_ROOT 	= os.environ["SEGMENTED_CLIPS_ROOT"] 	# Write out to root dir here
SEGMENTED_CLIPS_PATH 	= os.environ.get("SEGMENTED_CLIPS_PATH", "_splits")
TRAIN_VAL_PATH 			= os.environ.get("TRAIN_VAL_PATH", "_train_val")
CHATETTE_JSON_PATH 		= os.environ["CHATETTE_JSON"]
TEST_VAL_RATIO 			= 0.8

logging.basicConfig(level=logging.INFO)

data_writer = DatasetWriter()

def split_df(df: pd.DataFrame) -> [pd.DataFrame, pd.DataFrame]:
	"""
	Splits the dataset based on given ratio into training & validation dataset
	Splits are done by grouping on classname
	"""
	sampling_interval = math.ceil(1 / (1 - TEST_VAL_RATIO))
	test_df = df[df.index % sampling_interval != 0]
	validation_df = df[df.index % sampling_interval == 0]
	return [test_df, validation_df]

def build_video_df(video_root: str) -> pd.DataFrame:
	video_data = []
	for (directory, sub_dir, files) in os.walk(video_root):
		for each_file in files:
			if ".MP4" in each_file.upper():
				video_id = each_file.split("/")[-1]
				video_realpath = os.path.join(directory, each_file)
				video_data.append((video_realpath, video_id))

	return pd.DataFrame(video_data, columns=["full_video_path", "youtube_id"])

def extract_chatette_df(chatette_json: str) -> pd.DataFrame:
	with open(chatette_json, "r") as f:
		chatette_json_data = json.loads(f.read())
	text_list = chatette_json_data["rasa_nlu_data"]["common_examples"]
	df = pd.DataFrame.from_dict(text_list)
	return df.drop("entities", axis=1)

if __name__ == "__main__":
	segmented_video_df 			= build_video_df(video_root = SEGMENTED_CLIPS_ROOT)
	segment_metadata_files 		= glob.glob(os.path.join(SEGMENTED_CLIPS_ROOT, SEGMENTED_CLIPS_PATH) + "/*_segment_metadata.csv")
	segment_metadata_df 		= pd.concat(map(pd.read_csv, segment_metadata_files))
	chatette_df 				= extract_chatette_df(chatette_json = CHATETTE_JSON_PATH)

	video_merged_df 			= pd.merge(	left = segmented_video_df, right = segment_metadata_df,
											how = "inner", left_on = "youtube_id", right_on = "unique_clip_name")

	chatette_video_merged_df 	= pd.merge(	left = video_merged_df, right = chatette_df,
											how = "inner", left_on = "classname", right_on = "intent")

	[train_df, validation_df] 	= split_df(df = chatette_video_merged_df)

	# Write out training / validation df to SEGMENTED_CLIPS_ROOT/TRAIN_VAL_PATH
	train_path 				= os.path.join(SEGMENTED_CLIPS_ROOT, TRAIN_VAL_PATH, "train.csv")
	validation_path 		= os.path.join(SEGMENTED_CLIPS_ROOT, TRAIN_VAL_PATH, "validation.csv")
	data_writer.writeCsv(df = train_df, location = train_path)
	data_writer.writeCsv(df = validation_df, location = validation_path)
