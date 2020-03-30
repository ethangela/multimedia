import glob
import os
import pandas as pd 
from clip_metadata_writer import build_video_df, merge_video_metadata

SEGMENTED_CLIPS_ROOT 	= os.environ["SEGMENTED_CLIPS_ROOT"] 	# Write out to root dir here
SEGMENTED_CLIPS_PATH 	= os.environ.get("SEGMENTED_CLIPS_PATH", "_splits")

def split_df(df: pd.DataFrame) -> [pd.DataFrame, pd.DataFrame]:
	pass

def extract_segmented_metadata(segment_clips_path: str) -> pd.DataFrame:
	pass

def extract_chatette_data(chatette_json: str) -> pd.DataFrame:
	pass

if __name__ == "__main__":
	segmented_video_df 		= build_video_df(video_root = SEGMENTED_CLIPS_ROOT)
	segment_metadata_files 	= glob.glob(os.path.join(SEGMENTED_CLIPS_ROOT, SEGMENTED_CLIPS_PATH) + "/*_segment_metadata.csv")
	segment_metadata_df 	= pd.concat(map(pd.read_csv, segment_metadata_files))
	video_merged_df 		= merge_video_metadata(	video_df = segmented_video_df, video_metadata = segment_metadata_df,
													how="inner", left_on="youtube_id", right_on="unique_clip_name")

