import moviepy
import numpy as np
from moviepy.editor import *
from typing import Dict

class VideoSegmentation(object):
	def encodeToArr(self, clip_frames: (int, int), truth_frames: (int, int)) -> np.ndarray:
		"""
		Creates an encoded vector of length clip_frames_end - clip_frames_start index

		Frames that are labelled as true will be encoded to 1
		All other frames will be encoded to 0

		eg: 
			clip_frames 	= (1, 9)
			truth_frames 	= (4, 6)
			------------------------
			returns array 	= [0, 0, 0, 1, 1, 1, 0, 0, 0]

		Parameters:
		clip_frames: (int, int) 	- (clip_frames_start, clip_frames_end) pair denoting indices for frames of interest
		truth_frames: (int, int) 	- (truth_start, truth_end) pair denoting indices for frames of interest to be labelled as 1
		"""
		(clip_frames_start, clip_frames_end) = clip_frames
		(truth_frames_start, truth_frames_end) = truth_frames
		assert clip_frames_start <= truth_frames_start <= truth_frames_end <= clip_frames_end

		arr = np.zeros(clip_frames_end -clip_frames_start +1, dtype=int)
		arr[truth_frames_start -clip_frames_start : truth_frames_end -clip_frames_start +1] = 1
		return arr

	def getFrameIndex(self, video: moviepy.video.io.VideoFileClip.VideoFileClip, start_time: int, end_time: int) -> (int, int):
		"""
		Gets frame positioning for key frames of interest relative to the start of the video

		Parameters:
		video (moviepy.video.io.VideoFileClip.VideoFileClip) - Loaded video clip
		start_time (int) - Start time for key frame (relative to t = 0)
		end_time (int) - End time for key frame (relative to t = 0)
		
		Returns:
		(int, int) - (start_frame_index, end_frame_index)
		"""
		total_frames 	= int(video.fps * video.duration)
		start_index 	= max(0, int((start_time / video.duration) * total_frames))
		end_index 		= min(total_frames, int((end_time / video.duration) * total_frames))
		return (start_index, end_index)

	def segment(self, video: moviepy.video.io.VideoFileClip.VideoFileClip, start_time: int, end_time: int) -> moviepy.video.io.VideoFileClip.VideoFileClip:
		return video.subclip(start_time, end_time)

	def readAsVideo(self, video_path: str) -> moviepy.video.io.VideoFileClip.VideoFileClip:
		return VideoFileClip(video_path)