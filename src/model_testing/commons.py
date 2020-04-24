import ast
import copy
import numpy as np
import pandas as pd

def filter_frame_count(df: pd.DataFrame, count: int) -> pd.DataFrame:
	_df = copy.copy(df)
	return _df[_df.apply(lambda row: len(ast.literal_eval(row["ground_truth"])) == count, axis = 1)]

def select_incorrect_class_df(df: pd.DataFrame, candidate_row, size: int) -> pd.DataFrame:
	candidate_row_class 	= candidate_row["classname"]
	negative_example_df 	= df[df["classname"] != candidate_row_class]
	shuffled_df 			= negative_example_df.sample(frac = 1)
	return shuffled_df[:size]

def select_same_class_df(df: pd.DataFrame, candidate_row, size: int) -> pd.DataFrame:
	candidate_row_class 	= candidate_row["classname"]
	positive_example_df 	= df[df["classname"] == candidate_row_class]
	shuffled_df 			= positive_example_df.sample(frac = 1)
	return shuffled_df[:size]

def get_iou_df(evaluation_df: pd.DataFrame, ground_truth_row):
	iou_df 		= copy.copy(evaluation_df)
	iou_df 		= pd.concat([iou_df, pd.DataFrame(columns = ["iou"])])
	
	for indx, row in iou_df.iterrows():
		predicted_np 			= np.array(evaluation_df["key_frame_labels"][indx], dtype = int)

		ground_truth_labels 	= ast.literal_eval(ground_truth_row["ground_truth"])
		ground_truth_np 		= np.array(ground_truth_labels, dtype = int)

		intersection 			= 0
		for i in range(min(len(predicted_np), len(ground_truth_np))):
			intersection 		+= 1 if predicted_np[i] == ground_truth_np[i] == 1 else 0

		union 					= sum(ground_truth_np) + sum(predicted_np) - intersection
		iou_df["iou"][indx] 	= intersection / union
	return iou_df

def get_correct_vid_selection_count(best_k_df: pd.DataFrame, ground_truth_row, IoU: float):
	correct_selections = 0
	for _, row in best_k_df.iterrows():
		# Assert correct class selection
		is_correct = (row["classname"] == ground_truth_row["classname"]) and (row["iou"] >= IoU)
		correct_selections += int(is_correct)
	return correct_selections