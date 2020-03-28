"""
Script to identify relevant portions in a video clip, crop out the video and write it to separate storage
"""

import copy
import logging
import os
import pandas as pd
from commons.executions import multiple_executions_wrapper
from editing.segmentation import VideoSegmentation
from tqdm import tqdm

RESOURCE_FILE 		= os.environ["RESOURCE_FILE"] 	# CSV file
UNEDITED_CLIPS_ROOT = os.environ["CLIPS_ROOT"] 		# To raw video files
VIDEO_METADATA_PATH = os.environ["METADATA_PATH"]

logging.basicConfig(level=logging.INFO)
tqdm.pandas()

segmenter_obj 		= VideoSegmentation()

def build_video_df(video_root: str) -> pd.DataFrame:
	"""
	Iteratively traverses video_root to gather all video clips in a DF

	cols:	full_video_path | youtube_id

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

	return pd.DataFrame(video_data, columns=["full_video_path", "youtube_id"])

def merge_video_metadata(video_df: pd.DataFrame, video_metadata: pd.DataFrame) -> pd.DataFrame:
	"""
	Only video IDs present in video_df will be kept
	Other information will be discarded

	Cols: full_video_path | youtube_id | unique_clip_name | classname | start | end
	"""
	merged_df = pd.merge(video_df, video_metadata, how="inner", on="youtube_id")
	merged_df["_vid_indx"] = merged_df.groupby("youtube_id").cumcount()
	merged_df["unique_clip_name"] = merged_df.progress_apply(lambda row: "{0}_{1}.mp4".format(row["youtube_id"], row["_vid_indx"]), axis=1)
	return merged_df.drop(["subset", "label", "_vid_indx"], axis=1)

def inject_video_duration(full_video_path: str) -> int:
	full_video 	= segmenter_obj.readAsVideo(video_path = full_video_path)
	return full_video.duration

@multiple_executions_wrapper
def with_video_duration_df(video_df: pd.DataFrame) -> pd.DataFrame:
	"""
	New columns would be 
	Cols: full_video_path | youtube_id | unique_clip_name | classname | start | end | full_video_duration
	"""
	logging.info("PID {0}: Extracting full video duration".format(os.getpid()))
	_video_df = copy.copy(_video_df)
	_video_df["full_video_duration"] = _video_df.progress_apply(lambda row: inject_video_duration(row), axis =1)
	return _video_df

@multiple_executions_wrapper
def filter_invalid_clips(video_df: pd.DataFrame) -> pd.DataFrame:
	"""
	Removes clip from the DF if start / end time is beyond duration
	"""
	logging.info("PID {0}: Filtering invalid clips".format(os.getpid()))
	_video_df = copy.copy(_video_df)
	return _video_df[ (_video_df["start"] <= _video_df["full_video_duration"]) & (_video_df["end"] <= _video_df["full_video_duration"]) ]

if __name__ == "__main__":
	video_metadata_df 	= pd.read_csv(RESOURCE_FILE)
	video_df 			= build_video_df(video_root = UNEDITED_CLIPS_ROOT)
	video_merged_df 	= merge_video_metadata(video_df = video_df, video_metadata = video_metadata_df)
	video_merged_df.to_csv(VIDEO_METADATA_PATH)