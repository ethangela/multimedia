import os
import glob
import logging
import multiprocessing
import numpy as np
import pandas as pd
from commons.executions import multiple_executions_wrapper
from editing.segmentation import VideoSegmentation
from editing.writer import DatasetWriter

MAX_THREAD_POOL 		= int(os.environ.get("THREAD_POOL", multiprocessing.cpu_count()))
SEGMENTED_CLIPS_ROOT 	= os.environ["SEGMENTED_CLIPS_ROOT"] 	# Write out to root dir here
SEGMENTED_CLIPS_PATH 	= os.environ.get("SEGMENTED_CLIPS_PATH", "_splits")

logging.basicConfig(level=logging.INFO)

segmenter_obj 	= VideoSegmentation()
data_writer 	= DatasetWriter()

@multiple_executions_wrapper
def write_segment_clip(segment_metadata_df: pd.DataFrame) -> None:
	for _, row in segment_metadata_df.iterrows():
		full_video 	= segmenter_obj.readAsVideo(video_path = row["full_video_path"])
		clip 		= segmenter_obj.segment(video = full_video, start_time = row["segment_time_start"], end_time = row["segment_time_end"])
		data_writer.writeVideo(clip = clip, location = row["segmented_clips_path"])
		logging.info("PID {0} - Write to {1}".format(os.getpid(), row["segmented_clips_path"]))
	return

def exec_write_segment_clip(segment_metadata_df: pd.DataFrame) -> None:
	try:
		write_segment_clip(segment_metadata_df = segment_metadata_df)
	except Exception as ex:
		logging.error("PID {0} - Ignoring failed writes to {1}".format(os.getpid(), row["segmented_clips_path"]))
	return

if __name__ == "__main__":
	# Begin writing segments to dfs
	segment_metadata_files 		= glob.glob(os.path.join(SEGMENTED_CLIPS_ROOT, SEGMENTED_CLIPS_PATH) + "/*_segment_metadata.csv")
	segment_metadata_df 		= pd.concat(map(pd.read_csv, segment_metadata_files))
	segment_metadata_df_splits 	= np.array_split(segment_metadata_df, MAX_THREAD_POOL)
	with multiprocessing.Pool(processes = MAX_THREAD_POOL) as pool:
		pool.map(exec_write_segment_clip, segment_metadata_df_splits)