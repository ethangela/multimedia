"""
Script to segment video clips
"""

import copy
import glob
import logging
import os
import multiprocessing
import numpy as np
import pandas as pd
from commons.executions import multiple_executions_wrapper
from editing.segmentation import VideoSegmentation
from editing.writer import DatasetWriter
from functools import partial
from pathlib import Path
from tqdm import tqdm

MAX_THREAD_POOL 		= int(os.environ.get("THREAD_POOL", multiprocessing.cpu_count()))
CLIP_ADDITIONAL_WIDTH 	= int(os.environ.get("CLIP_ADDITIONAL_WIDTH", 1))
UNEDITED_CLIPS_ROOT 	= os.environ["CLIPS_ROOT"] 		
SEGMENTED_CLIPS_ROOT 	= os.environ["SEGMENTED_CLIPS_ROOT"] 	# Write out to root dir here
SEGMENTED_CLIPS_PATH 	= os.environ.get("SEGMENTED_CLIPS_PATH", "_splits")
VIDEO_METADATA_PATH 	= os.environ["METADATA_PATH"]

logging.basicConfig(level=logging.INFO)
tqdm.pandas()

segmenter_obj 	= VideoSegmentation()
data_writer 	= DatasetWriter()

def get_segment_bounds(duration:int, start: int, end: int) -> (int, int):
	"""
	Defines video segment to crop, with (start, end) as the center of the segmented clip
	We attempt to add CLIP_ADDITIONAL_WIDTH seconds to the start and end of the video
	If this bounds exceeds the video's duration, we peg the value to the duration
	"""
	segment_start 	= max(0, start - CLIP_ADDITIONAL_WIDTH)
	segment_end 	= min(duration, end + CLIP_ADDITIONAL_WIDTH)
	return (segment_start, segment_end)

@multiple_executions_wrapper
def extract_segment_ground_truth(row) -> np.ndarray:
	video 			= segmenter_obj.readAsVideo(video_path=row["full_video_path"])
	segment_clip 	= segmenter_obj.segment(video=video, start_time=row["segment_time_start"], end_time=row["segment_time_end"])

	(segment_frames_start, segment_frames_end) 			= segmenter_obj.getFrameIndex(video=video, start_time=row["segment_time_start"], 
																						end_time=row["segment_time_end"])
	(key_segment_frames_start, key_segement_frames_end) = segmenter_obj.getFrameIndex(video=video, start_time=row["start"], 
																						end_time=row["end"])

	encoded_arr 	= segmenter_obj.encodeToArr(clip_frames=(segment_frames_start, segment_frames_end), 
												truth_frames=(key_segment_frames_start, key_segement_frames_end))
	del video.reader
	del video
	return encoded_arr

@multiple_executions_wrapper
def extract_segment_timings(row) -> pd.Series:
	key_segment_time_start 	= row["start"]
	key_segment_time_end 	= row["end"]
	video 					= segmenter_obj.readAsVideo(video_path=row["full_video_path"])
	video_duration 			= video.duration
	(segment_time_start, segment_time_end) = get_segment_bounds(duration=video_duration, start=key_segment_time_start, end=key_segment_time_end)
	del video.reader
	del video
	return pd.Series([segment_time_start, segment_time_end])

@multiple_executions_wrapper
def extract_segment_path(row) -> str:
	class_name 			= row["classname"].strip().replace(" ", "_")	
	segment_video_path 	= os.path.join(SEGMENTED_CLIPS_ROOT, class_name, row["unique_clip_name"])
	return segment_video_path

def get_segemnt_clip_metadata(metadata_df: pd.DataFrame) -> pd.DataFrame:
	"""
	Function creates a new column in the df (segmented_clips_path) which identifies the path to write our cropped video clips
	When writing, directory structure after UNEDITED_CLIPS_ROOT is preserved
	"""	
	_metadata_df = copy.copy(metadata_df)
	_metadata_df = pd.concat([_metadata_df, pd.DataFrame(columns=["segmented_clips_path", "segment_time_start", "segment_time_end", "ground_truth"])])

	_metadata_df["segmented_clips_path"] = _metadata_df.progress_apply(lambda row: extract_segment_path(row), axis=1)
	logging.info("PID: {0} Segment path extraction completed".format(os.getpid()))

	_metadata_df[["segment_time_start", "segment_time_end"]] = _metadata_df.progress_apply(lambda row: extract_segment_timings(row), axis=1)
	logging.info("PID: {0} Segment timing extraction completed".format(os.getpid()))

	_metadata_df["ground_truth"] = _metadata_df.progress_apply(lambda row: extract_segment_ground_truth(row), axis=1)
	logging.info("PID: {0} Segment ground truth extraction completed".format(os.getpid()))
	return _metadata_df

def exec_segment_clip_metadata(metadata_df: pd.DataFrame) -> None:
	if metadata_df.empty:
		logging.info("PID {0}: Empty dataframe received".format(os.getpid()))
	else:
		# Write out segment metadata
		segment_metadata_df = get_segemnt_clip_metadata(metadata_df = metadata_df)
		segment_metadata_path = os.path.join(SEGMENTED_CLIPS_ROOT, SEGMENTED_CLIPS_PATH, "{0}_segment_metadata.csv".format(os.getpid()))
		logging.info("Writing segment metadata file: {0}".format(segment_metadata_path))
		data_writer.writeCsv(df = segment_metadata_df, location = segment_metadata_path)
	return

if __name__ == "__main__":
	# Delete all files within SEGMENTED_CLIPS_PATH directory for idempotency
	prior_segment_metadata_files = glob.glob(os.path.join(SEGMENTED_CLIPS_ROOT, SEGMENTED_CLIPS_PATH) + "/*segment_metadata.csv")
	logging.warning("Deleting prior files: {0}".format(prior_segment_metadata_files))
	for prior_files in prior_segment_metadata_files:
		os.remove(prior_files)

	# Write split files to SEGMENTED_CLIPS_PATH
	video_metadata_df 			= pd.read_csv(VIDEO_METADATA_PATH)
	video_metadata_df_splits 	= np.array_split(video_metadata_df, MAX_THREAD_POOL)
	with multiprocessing.Pool(processes = MAX_THREAD_POOL) as pool:
		pool.map(exec_segment_clip_metadata, video_metadata_df_splits)
