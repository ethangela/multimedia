import os
import glob
import logging

MAX_THREAD_POOL 		= int(os.environ.get("THREAD_POOL", multiprocessing.cpu_count()))
SEGMENTED_CLIPS_ROOT 	= os.environ["SEGMENTED_CLIPS_ROOT"] 	# Write out to root dir here
SEGMENTED_CLIPS_PATH 	= os.environ.get("SEGMENTED_CLIPS_PATH", "_splits")

logging.basicConfig(level=logging.INFO)

def write_segment_clip(segment_metadata_df: pd.DataFrame) -> None:
	for _, row in segment_metadata_df.iterrows():
		full_video 	= segmenter_obj.readAsVideo(video_path = row["full_video_path"])
		clip 		= segmenter_obj.segment(video = full_video, start_time = row["segment_time_start"], end_time = row["segment_time_end"])
		data_writer.writeVideo(clip = clip, location = row["segmented_clips_path"])
		logging.info("PID {0} - Write to {1}".format(os.getpid(), row["segmented_clips_path"]))
	return

def exec_write_segment_clip(path_to_df: str) -> None:
	segment_metadata_df = pd.read_csv(path_to_df)
	write_segment_clip(segment_metadata_df = segment_metadata_df)
	return

if __name__ == "__main__":
	# Begin writing segments to dfs
	segment_metadata_files = glob.glob(os.path.join(SEGMENTED_CLIPS_ROOT, SEGMENTED_CLIPS_PATH) + "/*segment_metadata.csv")
	with multiprocessing.Pool(processes = MAX_THREAD_POOL) as pool:
		pool.map(exec_write_segment_clip, segment_metadata_files)