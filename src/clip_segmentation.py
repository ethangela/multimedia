"""
Script to segment video clips
"""

import copy
import logging
import os
import numpy as np
import pandas as pd
from editing.segmentation import VideoSegmentation
from editing.writer import DatasetWriter
from pathlib import Path

CLIP_DURATION 			= int(os.environ.get("CLIP_DURATION", 4))
UNEDITED_CLIPS_ROOT 	= os.environ["CLIPS_ROOT"] 		
SEGMENTED_CLIPS_ROOT 	= os.environ["SEGMENTED_CLIPS_ROOT"] 	# Write out to root dir here
VIDEO_METADATA_PATH 	= os.environ["METADATA_PATH"]

logging.basicConfig(filename='clip_segmentation.log',level=logging.DEBUG)

segmenter_obj = VideoSegmentation()

def create_dir(path: str) -> None:
	"""
	Attempts to create the directory specified if it does not exist
	Nested directories are supported
	"""
	_path = Path(path)
	_path.mkdir(parents=True, exist_ok=True)
	return 

def get_segment_bounds(start: int, end: int) -> (int, int):
	"""
	Strictly segments a clip on "CLIP_DURATION" seconds, containing start - end seconds of the clip
	"""
	key_frames_duration = end - start
	assert key_frames_duration <= CLIP_DURATION
	
	segment_bounds 		= (CLIP_DURATION - key_frames_duration) / 2
	segment_start 		= start - segment_bounds
	segment_end 		= end + segment_bounds

	if segment_start < 0:
		segment_end 	+= -1 *segment_start
		segment_start 	= 0

	assert segment_end - segment_start == CLIP_DURATION
	return (segment_start, segment_end)

def extract_segment_ground_truth(row) -> np.ndarray:
	video 			= segmenter_obj.readAsVideo(video_path=row["full_video_path"])
	segment_clip 	= segmenter_obj.segment(video=video, start_time=row["segment_time_start"], end_time=row["segment_time_end"])

	(segment_frames_start, segment_frames_end) 			= segmenter_obj.getFrameIndex(video=video, start_time=row["segment_time_start"], 
																						end_time=row["segment_time_end"])
	(key_segment_frames_start, key_segement_frames_end) = segmenter_obj.getFrameIndex(video=video, start_time=row["start"], 
																						end_time=row["end"])

	encoded_arr 	= segmenter_obj.encodeToArr(clip_frames=(segment_frames_start, segment_frames_end), 
												truth_frames=(key_segment_frames_start, key_segement_frames_end))
	return encoded_arr

def extract_segment_timings(row) -> pd.Series:
	"""
	"""
	key_segment_time_start 	= row["start"]
	key_segment_time_end 	= row["end"]
	(segment_time_start, segment_time_end) = get_segment_bounds(start=key_segment_time_start, end=key_segment_time_end)
	return pd.Series([segment_time_start, segment_time_end])

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

	_metadata_df["segmented_clips_path"] = _metadata_df.apply(lambda row: extract_segment_path(row), axis=1)
	_metadata_df[["segment_time_start", "segment_time_end"]] = _metadata_df.apply(lambda row: extract_segment_timings(row), axis=1)
	_metadata_df["ground_truth"] = _metadata_df.apply(lambda row: extract_segment_ground_truth(row), axis=1)
	return _metadata_df
		
if __name__ == "__main__":
	video_metadata_df 	= pd.read_csv(VIDEO_METADATA_PATH)
	segment_metadata_df = get_segemnt_clip_metadata(metadata_df=video_metadata_df)
	
	# Write out segment metadata
	segment_metadata_path = os.path.join(VIDEO_METADATA_PATH, "segment_metadata.csv")
	logging.info("Writing segment metadata file: {0}".format(segment_metadata_path))
	segment_metadata_df.to_csv(segment_metadata_path)

	# Begin writing segmented to file
	