import pandas as pd

class DatasetWriter(object):
	def writeVideo(self, clip: moviepy.video.io.VideoFileClip.VideoFileClip, location: str, *, params: Dict = {}) -> None:
		clip.write_videofile(filename=location, **params)
		return 

	def writeCsv(self, df: pd.DataFrame, location: str) -> None:
		df.to_csv(location, sep=",", header=True)
		return