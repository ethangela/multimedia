# multimedia
source code for paper "Key Moments Search Over Video, Done Right" as the project work for CS6240 Multimedia Analysis 

###### 1.Requirements for environment: pytorch
###### 2.Script to reproduce results: run /src/model_training/train.py --decoder_mode='LSTM' --attention_mode='True'
###### 3.Script to evaluate: run /src/model_evaluation/evaluate.py --decoder_mode='MLP' --attention_mode='False'
####### comments: we designed two decoder networks ('LSTM' and 'MLP') with freedom to apply attention mechanisms ('True' or 'False'), therefore there are two options for each of two arargument presented above  

