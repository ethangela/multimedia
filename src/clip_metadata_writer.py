"""
Script to identify relevant portions in a video clip, crop out the video and write it to separate storage
"""

import copy
import glob
import logging
import numpy as np
import os
import multiprocessing
import pandas as pd
from commons.executions import multiple_executions_wrapper
from editing.segmentation import VideoSegmentation
from editing.writer import DatasetWriter
from tqdm import tqdm

MAX_THREAD_POOL 		= int(os.environ.get("THREAD_POOL", multiprocessing.cpu_count()))
RESOURCE_FILE 			= os.environ["RESOURCE_FILE"] 	# CSV file
UNEDITED_CLIPS_ROOT 	= os.environ["CLIPS_ROOT"] 		# To raw video files
VIDEO_METADATA_ROOT 	= os.environ["METADATA_ROOT"]
VIDEO_METADATA_PATH 	= os.environ.get("METADATA_PATH", "_metadata")

logging.basicConfig(level=logging.INFO)
tqdm.pandas()

segmenter_obj 	= VideoSegmentation()
data_writer 	= DatasetWriter()

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

@multiple_executions_wrapper
def merge_video_metadata(video_df: pd.DataFrame, video_metadata: pd.DataFrame) -> pd.DataFrame:
	"""
	Only video IDs present in video_df will be kept
	Other information will be discarded

	Cols: full_video_path | youtube_id | unique_clip_name | classname | start | end
	"""
	merged_df = pd.merge(video_df, video_metadata, how="inner", on="youtube_id")
	return merged_df

def inject_unique_clip_name(video_df: pd.DataFrame) -> pd.DataFrame:
	video_df["_vid_indx"] = video_df.groupby("youtube_id").cumcount()
	video_df["unique_clip_name"] = video_df.progress_apply(lambda row: "{0}_{1}.mp4".format(row["youtube_id"], row["_vid_indx"]), axis=1)
	return video_df.drop(["subset", "label", "_vid_indx"], axis=1)

@multiple_executions_wrapper
def inject_video_duration(row) -> int:
	full_video 	= segmenter_obj.readAsVideo(video_path = row["full_video_path"])
	return full_video.duration

@multiple_executions_wrapper
def with_video_duration_df(video_df: pd.DataFrame) -> pd.DataFrame:
	"""
	New columns would be 
	Cols: full_video_path | youtube_id | unique_clip_name | classname | start | end | full_video_duration
	"""
	logging.info("PID {0}: Extracting full video duration".format(os.getpid()))
	_video_df = copy.copy(video_df)
	_video_df["full_video_duration"] = _video_df.progress_apply(lambda row: inject_video_duration(row), axis =1)
	return _video_df

@multiple_executions_wrapper
def filter_invalid_clips(video_df: pd.DataFrame) -> pd.DataFrame:
	"""
	Removes clip from the DF if start / end time is beyond duration
	"""
	logging.info("PID {0}: Filtering invalid clips".format(os.getpid()))
	_video_df = copy.copy(video_df)
	return _video_df[ (_video_df["start"] <= _video_df["full_video_duration"]) & (_video_df["end"] <= _video_df["full_video_duration"]) ]

def exec_segment_metadata(metadata_df: pd.DataFrame) -> None:
	video_df 		= build_video_df(video_root = UNEDITED_CLIPS_ROOT)
	video_merged_df = merge_video_metadata(video_df = video_df, video_metadata = metadata_df)
	if video_merged_df.empty:
		logging.info("PID {0}: Video merged DF is empty".format(os.getpid()))
	else:
		unique_clip_df 				= inject_unique_clip_name(video_df = video_merged_df)
		video_merged_w_duration_df 	= with_video_duration_df(video_df = unique_clip_df)
		final_df 					= filter_invalid_clips(video_df = video_merged_w_duration_df)
		metadata_file 				= os.path.join(VIDEO_METADATA_ROOT, VIDEO_METADATA_PATH, "{0}_metadata.csv".format(os.getpid()))
		data_writer.writeCsv(df = final_df, location = metadata_file)
	return

if __name__ == "__main__":
	# Delete all files within VIDEO_METADATA_PATH directory for idempotency
	prior_segment_metadata_files = glob.glob(os.path.join(VIDEO_METADATA_ROOT, VIDEO_METADATA_PATH) + "/*_metadata.csv")
	logging.warning("Deleting prior files: {0}".format(prior_segment_metadata_files))
	for prior_files in prior_segment_metadata_files:
		os.remove(prior_files)

	video_metadata_df 			= pd.read_csv(RESOURCE_FILE)
	video_metadata_df_splits 	= np.array_split(video_metadata_df, MAX_THREAD_POOL)
	with multiprocessing.Pool(processes = MAX_THREAD_POOL) as pool:
		pool.map(exec_segment_metadata, video_metadata_df_splits)
