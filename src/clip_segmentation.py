"""
Script to segment video clips
"""

import logging
import os
import pandas as pd
from editing.segmentation import VideoSegmentation
from editing.writer import DatasetWriter

CLIP_DURATION 			= int(os.environ.get("CLIP_DURATION", 4))
UNEDITED_CLIPS_ROOT 	= os.environ["CLIPS_ROOT"] 		
SEGMENTED_CLIPS_ROOT 	= os.environ["SEGMENT_ROOT"] 	
VIDEO_METADATA_PATH 	= os.environ["METADATA_PATH"]

logging.basicConfig(filename='clip_segmentation.log',level=logging.DEBUG)

if __name__ == "__main__":
	video_metadata_df = pd.read_csv(VIDEO_METADATA_PATH)
