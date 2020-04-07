import moviepy
import pandas as pd
from moviepy.editor import *
from typing import Dict
from pathlib import Path

class DatasetWriter(object):
	def _createParents(self, path: str) -> None:
		path = Path(path)
		path.parent.mkdir(parents=True, exist_ok=True)
		return

	def writeVideo(self, clip: moviepy.video.io.VideoFileClip.VideoFileClip, location: str, *, params: Dict = {}) -> None:
		self._createParents(path = location)
		clip.write_videofile(filename=location, **params)
		return 

	def writeCsv(self, df: pd.DataFrame, location: str) -> None:
		self._createParents(path = location)
		df.to_csv(location, sep=",", header=True, index=False)
		return

	def writeJsonL(self, df: pd.DataFrame, location: str) -> None:
		self._createParents(path = location)
		df.to_json(location, orient = "records", lines = True)
		return

	def writeHdf5(self, df: pd.DataFrame, location: str) -> None:
		self._createParents(path = location)
		with pd.HDFStore(location, mode="w") as f:
			f.append(key="/dataset", value=df, format="table", data_columns=df.columns)
		# df.to_hdf(location, key='/dataset', mode='w')