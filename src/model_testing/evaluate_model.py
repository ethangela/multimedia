"""
Point PYTHOPATH to video_evaluate package
"""

import argparse
import copy
import h5py
import logging
import numpy as np
import os
import pandas as pd
from benchmarks.LocalisingMoment.data_preparation import get_language_encoding
from model_testing.commons import *
from evaluate import Solver
from configs import get_config

TEST_CSV 			= os.environ["TEST_CSV"]

TEST_TRAILS 		= None
POSITIVE_SAMPLES 	= None
NEGATIVE_SAMPLES 	= None
K_BEST				= None
IOU 				= None
KEY_FRAME_THRESHOLD = None

"""
Test is done using Recall@K with IoU = m

Our approach is a modification of TALL: Temporal Activity Localization via Language Query [Gao et al.]

We want to determine the average percentage of videos within top K selected videos that
1) Fits the correct video class from user search intent
2) Has IoU >= m

We do the following:
1) Select a random video from the test set (without replacement)
2) Select P other videos of similar class at random
3) Select N other videos of different classes at random
4) Iterate through each frame of a video to calculate its error from user input
5) Define a threshold by which error < threshold is considered a key frame
6) Obtain video IoU value by comparing with Ground truth
7) Obtain video score: Summation of all error to perform video ranking
8) Identify #videos with IoU > m and correct classes from baseline
"""

logging.basicConfig(level=logging.INFO)

def evaluate_df(candidate_df: pd.DataFrame) -> pd.DataFrame:
	evaluation_df 	= copy.copy(candidate_df)
	evaluation_df 	= pd.concat([evaluation_df, pd.DataFrame(columns = ["predicted_error", "key_frame_labels"])])

	test_config = get_config(mode='test')
	solver = Solver(test_config)
	solver.build()

	for indx, row in evaluation_df.iterrows():
		video_name 			= row["unique_clip_name"]
		frames, score 		= solver.test(indx, video_name)	
		evaluation_df["key_frame_labels"][indx] = frames
		evaluation_df["score"][indx] = score
	return evaluation_df

def init_test(df: pd.DataFrame) -> np.ndarray:
	results 		= []
	shuffle_df 		= df.sample(frac = 1)

	for _, row in shuffle_df[:TEST_TRAILS].iterrows():
		logging.info("Evaluating video {0} : class {1} : text {2}".format(row["unique_clip_name"], 
																			row["classname"], row["text"]))

		positive_sample_df 	= select_same_class_df(df = df, candidate_row = row, size = POSITIVE_SAMPLES)
		negative_sample_df 	= select_incorrect_class_df(df = df, candidate_row = row, size = NEGATIVE_SAMPLES)
		union_df 			= pd.concat([positive_sample_df, negative_sample_df])

		evaluation_df 		= evaluate_df(	candidate_df = union_df)

		iou_df 				= get_iou_df(evaluation_df = evaluation_df, ground_truth_row = row)

		# Take top K videos
		best_k_df 			= iou_df.sort_values(by = "score", ascending = True)[0 : K_BEST]

		correct_selections 	= get_correct_vid_selection_count(best_k_df = best_k_df, ground_truth_row = row)
		total_choices 		= len(best_k_df)

		logging.info("Max IOU {0}".format(iou_df["iou"].max()))
		logging.info("IoU: {0}, Top K: {1}, Selection {2} / {3}".format(IOU, K_BEST, 
																		correct_selections, total_choices))
		results.append((correct_selections, total_choices))
	return results

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Localising moments benchmark test')

	parser.add_argument('--test_trails', type=int, default=1 ,help='number of testings')
	parser.add_argument('--positive_samples', type=int, default=10 ,help='number of positive classes')
	parser.add_argument('--negative_samples', type=int, default=20 ,help='number of negative classes')
	parser.add_argument('--k_best', type=int, default=10 ,help='to select top K clips based on lowest error')
	parser.add_argument('--iou', type=float, default=0.3 ,help='iou threshold')
	parser.add_argument('--key_frame_threshold', type=float ,default=0.7 ,help='threshold (after normalization) to determine if it is a key frame or not')

	args = parser.parse_args()
	logging.info(args)

	TEST_TRAILS 		= args.test_trails
	POSITIVE_SAMPLES 	= args.positive_samples
	NEGATIVE_SAMPLES 	= args.negative_samples
	K_BEST				= args.k_best
	IOU 				= args.iou
	KEY_FRAME_THRESHOLD = args.key_frame_threshold

	df 		= pd.read_csv(TEST_CSV)
	results = init_test(df = df)

	logging.info("{0}".format(results))

	acc_correct_selections 	= sum([i[0] for i in results])
	acc_total_choices 		= sum([i[1] for i in results])

	logging.info("Average percentage for R@{0} : {1} %".format(K_BEST, (acc_correct_selections / acc_total_choices) * 100))