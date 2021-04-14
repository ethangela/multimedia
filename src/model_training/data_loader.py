# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
import h5py
import numpy as np
import pickle 
import pandas as pd
import ast
from torch.nn.utils.rnn import pad_sequence

class VideoData(Dataset):
    def __init__(self, root, mode):
        self.root = root
        
        frame_add = self.root + '/train.hdf5'
        self.v_feature = h5py.File(frame_add, 'r')
        
        if mode.lower() == 'train':
            list_add = self.root + '/video_list.pkl'
            with open(list_add, 'rb') as f:
                self.video_list = pickle.load(f)
            df_add = self.root + '/train.csv'
            self.df = pd.read_csv(df_add)
        else:
            list_add = self.root + '/video_test_list_random.pkl'
            with open(list_add, 'rb') as f:
                self.video_list = pickle.load(f)
            df_add = self.root + '/validation.csv'
            self.df = pd.read_csv(df_add) 
        
    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        idx, video_name = self.video_list[index]
        video_name = video_name[:-4]
        
        v = torch.Tensor(np.array(self.v_feature[video_name]))
        gt = torch.FloatTensor(ast.literal_eval(self.df.iloc[idx]['ground_truth'])[:120])
        return v, gt



class TextData(Dataset):
    def __init__(self, root, mode):
        self.root = root     
        if mode.lower() == 'train':
            list_add = self.root + '/video_list.pkl'
            with open(list_add, 'rb') as f:
                self.video_list = pickle.load(f)
            text_add = self.root + '/hacs_text_features.hdf5'
        else:
            list_add = self.root + '/video_test_list.pkl'
            with open(list_add, 'rb') as f:
                self.video_list = pickle.load(f)
            text_add = self.root + '/hacs_text_test_features.hdf5'
         
        self.q_feature = h5py.File(text_add, 'r')
            
    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        idx, video_name = self.video_list[index]
        key = str(self.video_list[index])
        q = torch.Tensor(np.array(self.q_feature[key]))       
        return q
       

       
def collate_fn(train_data):
    data_length = [len(data) for data in train_data]
    train_data = pad_sequence(train_data, padding_value=0)
    return train_data, data_length



def get_loader(root, mode, batch_size):
    if mode.lower() == 'train':
        return DataLoader(VideoData(root, mode), batch_size), DataLoader(TextData(root, mode), batch_size, collate_fn=collate_fn)
    else:
        return VideoData(root, mode), TextData(root, mode)


if __name__ == '__main__':
    pass
