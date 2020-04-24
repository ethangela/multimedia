# -*- coding: utf-8 -*-
import ast
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

import numpy as np
import json
from tqdm import tqdm, trange
import pickle
import os 

from summarizer import Encoder, LSTMDecoder, MLPDecoder
from configs import get_config


class Solver(object):
    def __init__(self, config=None, video_loader=None, text_loader=None):
        self.config = config
        self.video_loader = video_loader
        self.text_loader = text_loader

    # Build Modules
    def build(self):       
        self.encoder = Encoder(
            self.config.v_input_size,
            self.config.v_hidden_size,
            self.config.q_input_size,
            self.config.q_hidden_size,
            self.config.decoder_mode,
            self.config.attention_mode).cuda() #GPU setting
                            
        if self.config.decoder_mode == 'LSTM':
            self.lstm_decoder = LSTMDecoder(
                self.config.lstm_input_size, 
                self.config.lstm_hidden_size,
                self.config.mlp_hidden_size1,
                self.config.mlp_hidden_size2).cuda() #GPU setting
                
            self.model = nn.ModuleList([
                self.encoder, self.lstm_decoder])
                
        if self.config.decoder_mode == 'MLP':
            self.mlp_decoder = MLPDecoder(
                self.config.mlp_input_size,
                self.config.mlp_hidden_size1,
                self.config.mlp_hidden_size2).cuda() #GPU setting
                
            self.model = nn.ModuleList([
                self.encoder, self.mlp_decoder])
                
        self.model.eval()
    
    # Test session 
    def test(self, index, video_name):        
        # Load image feature
        frame_address = self.config.video_root_dir + '/test.hdf5'
        v_feature_file = h5py.File(frame_address, 'r')
        v_feature = torch.Tensor(np.array(v_feature_file[video_name[:-4]]))
        v_feature = v_feature.unsqueeze(1)
        v_feature_ = Variable(v_feature).cuda() #GPU setting
        
        # Load text feature
        text_address = self.config.video_root_dir + '/test_hacs_text_features.hdf5' 
        q_feature_file = h5py.File(text_address, 'r')
        
        df_address = self.config.video_root_dir + '/test.csv'
        df = pd.read_csv(df_address)
        query_video_name = df.iloc[index]['segmented_video_id']
        
        q_feature = torch.Tensor(np.array(q_feature_file[str((index, query_video_name))]))
        q_feature = q_feature.unsqueeze(1)
        q_feature_ = Variable(q_feature).cuda() #GPU setting
        
        # Load weights
        checkpoint = self.config.save_dir
        print(f'Load parameters from {checkpoint}')
        file = os.listdir(checkpoint)
        checkpoint_file_path = os.path.join(checkpoint, file[0])
        self.model.load_state_dict(torch.load(checkpoint_file_path))
        self.model.eval()
           
        #---- test ----#       
        if self.config.decoder_mode == 'LSTM':
            #[v_seq_len, batch_size, v_hidden_size+q_hidden_size]
            con_h = self.encoder(v_feature_.detach(), q_feature_.detach()) 
            #[batch_size, 120]
            predicts = self.lstm_decoder(con_h)
            
        if self.config.decoder_mode == 'MLP':
            #[v_seq_len, batch_size, v_hidden_size+q_hidden_size]
            con_h = self.encoder(v_feature_.detach(), q_feature_.detach()) 
            #[batch_size, 120]
            predicts = self.mlp_decoder(con_h)
            
        # Model results
        sigmoid = nn.Sigmoid()
        predicts_ = sigmoid(predicts)
        predicts_ = predicts_.tolist()
        
        # Vectors
        print(predicts_) 
        vectors = list(map(lambda x: 0 if x < 0.98 else 1, predicts_[0]))

        # Confidence scores
        start_time = list(map(lambda i: i==1, vectors)).index(True)
        end_time = list(map(lambda i: i==0, vectors[start_time:])).index(True) + start_time 
        
        #gt_start_time = list(map(lambda i: i==1, gt)).index(True)
        #gt_end_time = list(map(lambda i: i==0, gt[gt_start_time:])).index(True) + gt_start_time
        #score = 1 - (abs(start_time-gt_start_time) + abs(end_time-gt_end_time))/(gt_end_time-gt_start_time)
        
        score = end_time - start_time + 1
        
        return vectors, score
        
        

if __name__ == '__main__':
    test_config = get_config(mode='test')
    print(test_config)
    solver = Solver(test_config)
    index = 3
    query = 'lady doing a shaking belly'
    video_name = 'LqbrIAQ05rM_0.mp4'
    solver.build()
    vectors, scores = solver.test(index, video_name)
    print(vectors)
    print(scores)

