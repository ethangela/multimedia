"""
Script to identify relevant portions in a video clip, crop out the video and write it to separate storage
"""

import logging
import os
import pandas as pd

RESOURCE_FILE 		= os.environ["RESOURCE_FILE"] 	# CSV file
UNEDITED_CLIPS_ROOT = os.environ["CLIPS_ROOT"] 		# To raw video files
VIDEO_METADATA_PATH = os.environ["METADATA_PATH"]

def build_video_df(video_root: str) -> pd.DataFrame:
	"""
	Iteratively traverses video_root to gather all video clips in a DF

	cols:	video_realpath | youtube_id

	Parameters:
	video_root (str) 	- Root directory for all raw, unedited video clips
	"""
	video_data = []
	for (directory, sub_dir, files) in os.walk(video_root):
		for each_file in files:
			if ".MP4" in each_file.upper():
				video_wo_extension = each_file.replace(".mp4", "")
				video_id = video_wo_extension[2:]
				video_realpath = os.path.join(directory, each_file)
				video_data.append((video_realpath, video_id))

	return pd.DataFrame(video_data, columns=["video_realpath", "youtube_id"])

def merge_video_metadata(video_df: pd.DataFrame, video_metadata: pd.DataFrame) -> pd.DataFrame:
	"""
	Only video IDs present in video_df will be kept
	Other information will be discarded

	Cols: video_realpath | youtube_id | classname | start | end
	"""
	merged_df = pd.merge(video_df, video_metadata, how="inner", on="youtube_id")
	return merged_df.drop(["subset", "label"], axis=1)

if __name__ == "__main__":
	video_metadata_df 	= pd.read_csv(RESOURCE_FILE)
	video_df 			= build_video_df(video_root = UNEDITED_CLIPS_ROOT)
	video_merged_df 	= merge_video_metadata(video_df = video_df, video_metadata = video_metadata_df)
	video_merged_df.to_csv(VIDEO_METADATA_PATH)