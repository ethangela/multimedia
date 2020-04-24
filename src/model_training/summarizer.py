# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
from lstmcell import StackedLSTMCell


class sLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        """Scoring LSTM"""
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),  # bidirection => scalar
            nn.Sigmoid())

    def forward(self, features, init_hidden=None):
        """
        Args:
            features: [seq_len, batch_size, input_size=2048] (Resnet features)
        Return:
            scores [seq_len, batch_size, 1]
        """
        self.lstm.flatten_parameters()

        # [seq_len, batch_size, hidden_size * 2]
        features, (h_n, c_n) = self.lstm(features)

        # [seq_len, batch_size, 1]
        scores = self.out(features)

        return scores


class eLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        """Encoder LSTM"""
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

        self.linear_mu = nn.Linear(hidden_size, hidden_size)
        self.linear_var = nn.Linear(hidden_size, hidden_size)

    def forward(self, frame_features):
        """
        Args:
            frame_features: [seq_len, batch_size, input_size=2048](weighted Resnet features)
        Return:
            output features (h_t) from the last layer of the LSTM, for each t:
            [seq_len, batch_size, num_directions*hidden_size = 1*hidden_size]
        """
        self.lstm.flatten_parameters()
        features, (h_last, c_last) = self.lstm(frame_features)

        return features
        

class qLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        """Encoder LSTM"""
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

        self.linear_mu = nn.Linear(hidden_size, hidden_size)
        self.linear_var = nn.Linear(hidden_size, hidden_size)

    def forward(self, query_features):
        """
        Args:
            query_features: [seq_len, batch_size, input_size=768] (Bert_embedding features)
        Return:
            last hidden
                h_last [num_layers=2, batch_size, hidden_size]
                c_last [num_layers=2, batch_size, hidden_size]
        """
        self.lstm.flatten_parameters()
        _, (h_last, c_last) = self.lstm(query_features)

        return (h_last, c_last)


class Encoder(nn.Module):
    def __init__(self, v_input_size, v_hidden_size, q_input_size, q_hidden_size, decoder_mode, attention_mode, num_layers=2):
        """Decoder LSTM"""
        super().__init__()
        self.s_lstm = sLSTM(v_input_size, v_hidden_size, num_layers)
        self.e_lstm = eLSTM(v_input_size, v_hidden_size, num_layers)
        self.q_lstm = qLSTM(q_input_size, q_hidden_size, num_layers)
        self.softplus = nn.Softplus()
        self.v_norm = nn.BatchNorm1d(v_hidden_size)
        self.q_norm = nn.BatchNorm1d(q_hidden_size)
        self.decoder_mode = decoder_mode
        self.attention_mode = attention_mode
        
    def reparameterize(self, mu, log_variance):
        """Sample z via reparameterization trick
        Args:
            mu: [num_layers, batch_size, hidden_size]
            log_var: [num_layers, batch_size, hidden_size]
        Return:
            h: [num_layers, batch_size, hidden_size]
        """
        std = torch.exp(0.5 * log_variance)

        # e ~ N(0,1)
        epsilon = Variable(torch.randn(std.size())).cuda() # GPU setting 

        # [num_layers, batch_size, hidden_size]
        return mu + epsilon * std

    def forward(self, v_features, packed_q_features):
        """
        Args:
            v_features: [v_seq_len, batch_size, v_hidden_size]
            packed_q_features: [q_seq_len, batch_size, q_hidden_size]
        Return:
            concatenated_features: [v_seq_len, batch_size, v_hidden_size+q_hidden_size]
        """
        #------ v_features ------#
        # [v_seq_len, batch_size, 1]
        if self.attention_mode:
            # [v_seq_len, batch_size, 1]
            scores = self.s_lstm(v_features)
            
            # [v_seq_len, batch_size, v_hidden_size]
            v_features = v_features * scores
        
        # [v_seq_len, batch_size, v_hidden_size]
        h = self.e_lstm(v_features)
        v_seq_len = h.size()[0]
        batch_size = h.size()[1]

        # [num_layers, batch_size, v_hidden_size]
        #h_mu = self.e_lstm.linear_mu(h)
        #h_log_variance = torch.log(self.softplus(self.e_lstm.linear_var(h)))
        #h = self.reparameterize(h_mu, h_log_variance)


        #------ q_features ------#
        # [v_seq_len, batch_size, q_hidden_size]
        hq, cq = self.q_lstm(packed_q_features)
        hq = hq[-1]
        hq = hq.unsqueeze(0)
        hq = hq.expand(*((v_seq_len, batch_size, -1)))

        # [num_layers, batch_size, q_hidden_size]
        #hq_mu = self.q_lstm.linear_mu(hq)
        #hq_log_variance = torch.log(self.softplus(self.q_lstm.linear_var(hq)))
        #hq = self.reparameterize(hq_mu, hq_log_variance)
        
        
        #------ concatenation ------# 
        # [v_seq_len, batch_size, v_hidden_size+q_hidden_size]
        con_h = torch.cat([h, hq], 2)
        
        return con_h


class dLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2): # [input_size = v_hidden_size+q_hidden_size]
        """Decoder LSTM"""
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, con_h):
        """
        Args:
            con_h: [v_seq_len, batch_size, v_hidden_size+q_hidden_size]
        Return:
            out_features: [v_seq_len, batch_size, num_directions*hidden_size = 1*hidden_size]
        """
        self.lstm.flatten_parameters()
        # [v_seq_len, batch_size, num_directions*hidden_size = 1*hidden_size]
        features, (h_last, c_last) = self.lstm(con_h)
        
        return features


class LSTMDecoder(nn.Module):
    def __init__(self, con_input_size, con_hidden_size, hidden_size1, hidden_size2, num_layers=2):
        super().__init__()
        self.d_lstm = dLSTM(con_input_size, con_hidden_size, num_layers)
        #self.predict = nn.Linear(con_hidden_size, 1)
        input_size = con_input_size + con_hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size1,hidden_size2)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_size2,1)

    def forward(self, con_h):
        """
        Args:
            con_h: [v_seq_len, batch_size, con_input_size]
        Return:
            predicts: [batch_size, 120]
        """
             
        #------ prediction ------#
        # [v_seq_len, batch_size, con_hidden_size]
        dout = self.d_lstm(con_h)  
        
        #[v_seq_len, batch_size, con_input_size+con_hidden_size]
        mlp_input = torch.cat([con_h, dout], 2)
        
        # [v_seq_len, batch_size, 1]
        mout = nn.functional.relu(self.drop1(self.fc1(mlp_input)))
        mout = nn.functional.relu(self.drop2(self.fc2(mout)))
        predicts = self.fc3(mout)       
 
        # [batch_size, v_seq_len=120]
        predicts = predicts.squeeze(-1)
        predicts = predicts.permute(1,0)
        
        return predicts


class MLPDecoder(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super().__init__()
        self.fc1 = nn.Linear(input_size,hidden_size1)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size1,hidden_size2)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_size2,1)

    def forward(self, con_h):
        """
        Args:
            con_h: [v_seq_len, batch_size, v_hidden_size+q_hidden_size]
        Return:
            predicts: [batch_size, v_seq_len]
        """
        
        #------ prediction ------#
        # [batch_size, v_seq_len=120]
        dout = nn.functional.relu(self.drop1(self.fc1(con_h)))
        dout = nn.functional.relu(self.drop2(self.fc2(dout)))
        predicts = self.fc3(dout)
        predicts = predicts.squeeze(-1)
        predicts = predicts.permute(1,0)
        
        return predicts
        
    
if __name__ == '__main__':

    pass
