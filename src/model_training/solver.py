# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

import numpy as np
import json
from tqdm import tqdm, trange
import pickle
import os

from summarizer3 import Encoder, LSTMDecoder, MLPDecoder


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
        
        
        if self.config.mode == 'train':
            # Build Optimizers 
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)

            # Indication 
            self.model.train()

    
    # Build Loss
    def BCE_loss(self):
        return nn.BCEWithLogitsLoss()
    
    # Train 
    def train(self):
        step = 0
        for epoch_i in trange(self.config.n_epochs, desc='Epoch', ncols=80):
            loss_history = []

            for batch_i, features_and_groundtruth in enumerate(zip(self.video_loader, self.text_loader)):
                
                # video features, groundtruth and query_features
                video_features_and_ground_truth, query_features_and_length = features_and_groundtruth
                
                # video features and groundtruth 
                video_features, ground_truth = video_features_and_ground_truth
                video_features = video_features.permute(1,0,2)
                video_features_ = Variable(video_features).cuda() #GPU setting 
                ground_truth_ = Variable(ground_truth).cuda()
                
                # query features 
                query_features, length = query_features_and_length
                query_features_ = Variable(query_features).cuda() #GPU setting
                query_features_detach = pack_padded_sequence(query_features_.detach(), length, enforce_sorted=False)


                #---- train ----#
                if self.config.verbose:
                    tqdm.write(str(epoch_i) + ': training for ' +str(batch_i))
           
                if self.config.decoder_mode == 'LSTM':
                    # [v_seq_len, batch_size, v_hidden_size+q_hidden_size]
                    con_h = self.encoder(video_features_.detach(), query_features_detach)
                    # [batch_size, v_seq_len=120]
                    predicts = self.lstm_decoder(con_h)
                    
                if self.config.decoder_mode == 'MLP':
                    # [v_seq_len, batch_size, v_hidden_size+q_hidden_size]
                    con_h = self.encoder(video_features_.detach(), query_features_detach)
                    # [batch_size, v_seq_len=120]
                    predicts = self.mlp_decoder(con_h)
                
                # loss calculation 
                entropy_loss_function = self.BCE_loss() 
                entropy_loss = entropy_loss_function(predicts, ground_truth_)
                tqdm.write(f'entropy loss {entropy_loss.item():.3f}')
                
                # zero the parameter gradients
                self.optimizer.zero_grad()
                
                # backward propagation 
                entropy_loss.backward()
                
                # Gradient cliping
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.config.clip)
                
                # parameters update 
                self.optimizer.step()
                
                # batch loss record 
                loss_history.append(entropy_loss.data)               
                
                # tesorboard plotting 
                if self.config.verbose:
                    tqdm.write('Plotting...')
                print('entropy_loss at step {} is {}'.format(step, entropy_loss.data))

                step += 1
            
            # average loss per epoch  
            e_loss = torch.stack(loss_history).mean()

            # tesorboard plotting 
            if self.config.verbose:
                tqdm.write('Plotting...')
            print('avg_epoch_loss at epoch {} is {}'.format(epoch_i, e_loss))

            # Save parameters at checkpoint
            if os.path.isdir(self.config.save_dir) is False:
                os.mkdir(self.config.save_dir)
            ckpt_path = str(self.config.save_dir) + f'_epoch-{epoch_i}.pkl'
            tqdm.write(f'Save parameters at {ckpt_path}')
            torch.save(self.model.state_dict(), ckpt_path)

            # evaluate
            #self.evaluate(epoch_i)
            #self.model.train()

if __name__ == '__main__':
    pass
