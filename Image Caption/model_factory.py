################################################################################
# CSE 151B: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin
# Winter 2022
################################################################################
from curses import keyname
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Build and return the model here based on the configuration.
class baseline(nn.Module):
    def __init__(self, vocab, embed_size, vocab_size, config_data,hidden_size=512, num_layers=1,max_seq=20):
        super(baseline, self).__init__()
        resnet_temp=models.resnet50(pretrained=True)
        self.resnet_modify=torch.nn.Sequential(*(list(resnet_temp.children())[:-1]))
        for param in self.resnet_modify.parameters():
            param.requires_grad = False
        self.fc1=nn.Linear(2048,embed_size)
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.config_data=config_data
        self.max_seq=max_seq
        self.vocab=vocab
        # lstm 
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size,num_layers=num_layers,batch_first=True)
        self.fc2 = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x, captions): 
            # batch size
            feature = self.resnet_modify(x)
            feature=feature.reshape(feature.size(0),-1)
            feature = self.fc1(feature)
            feature=feature.unsqueeze(1)
            h_state=None
            embedding= self.embed(captions)
            embeddings = torch.cat( (feature, embedding[:,:-1,:]), dim=1)
            hiddens,_=self.lstm(embeddings)
            outputs = self.fc2(hiddens)
            return outputs
    def sample(self, x, states=None):
            result = []
            candidate=list(range(0,self.vocab_size))
            states=None
#             plt.imshow(x.cpu().reshape((3,256,256)).permute(1,2,0))
            features = self.resnet_modify(x)
            features=features.reshape(features.size(0),-1)
            features = self.fc1(features)
            inputs = features.unsqueeze(1)
            if( not self.config_data['generation']['deterministic']):
                for _ in range(self.max_seq):
                    hiddens,states=self.lstm(inputs,states)
                    output=self.fc2(hiddens.squeeze(1))
                    output=(output/self.config_data['generation']['temperature'])
                    prob=torch.softmax(output,dim=1)
                    predict=torch.multinomial(prob,num_samples=1)
                    result.append(predict.item())
                    inputs=self.embed(predict)
    #                 print("input: ", inputs.shape)
                    if self.vocab.idx2word[predict.item()]=="<end>":
                        break
                return [self.vocab.idx2word[idx] for idx in result if (idx!=1) and (idx!=2)]
            
#             determin
            else:
                for _ in range(self.max_seq):
                    hiddens,states=self.lstm(inputs,states)
                    output=self.fc2(hiddens.squeeze(1))
                    predict=output.argmax(1)
                    result.append(predict.item())
                    inputs=self.embed(predict).unsqueeze(1)
    #                 print("input: ", inputs.shape)
                    if self.vocab.idx2word[predict.item()]=="<end>":
                        break
                return [self.vocab.idx2word[idx] for idx in result]
            
            
class RNN(nn.Module):
    def __init__(self, vocab, embed_size, vocab_size, config_data,hidden_size=512, num_layers=1,max_seq=20):
        super(RNN, self).__init__()
        resnet_temp=models.resnet50(pretrained=True)
        self.resnet_modify=torch.nn.Sequential(*(list(resnet_temp.children())[:-1]))
        for param in self.resnet_modify.parameters():
            param.requires_grad = False
        self.fc1=nn.Linear(2048,embed_size)
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.config_data=config_data
        self.max_seq=max_seq
        self.vocab=vocab
        # lstm 
        self.rnn = nn.RNN(input_size=self.embed_size, hidden_size=self.hidden_size,num_layers=num_layers,batch_first=True,nonlinearity='relu')
        self.fc2 = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x, captions): 
            feature = self.resnet_modify(x)
            feature=feature.reshape(feature.size(0),-1)
            feature = self.fc1(feature)
            feature=feature.unsqueeze(1)
            h_state=None
            embedding= self.embed(captions)
            embeddings = torch.cat( (feature, embedding[:,:-1,:]), dim=1)
            hiddens,_=self.rnn(embeddings)
            outputs = self.fc2(hiddens)
            return outputs
    def sample(self, x, states=None):
            result = []
            candidate=list(range(0,self.vocab_size))
            states=None
#             plt.imshow(x.cpu().reshape((3,256,256)).permute(1,2,0))
            features = self.resnet_modify(x)
            features=features.reshape(features.size(0),-1)
            features = self.fc1(features)
            inputs = features.unsqueeze(1)
            if( not self.config_data['generation']['deterministic']):
                for _ in range(self.max_seq):
                    hiddens,states=self.rnn(inputs,states)
                    output=self.fc2(hiddens.squeeze(1))
                    output=(output/self.config_data['generation']['temperature'])
                    prob=torch.softmax(output,dim=1)
                    predict=torch.multinomial(prob,num_samples=1)
                    result.append(predict.item())
                    inputs=self.embed(predict)
    #                 print("input: ", inputs.shape)
                    if self.vocab.idx2word[predict.item()]=="<end>":
                        break
                return [self.vocab.idx2word[idx] for idx in result if (idx!=1) and (idx!=2)]
            
#             determin
            else:
                for _ in range(self.max_seq):
                    hiddens,states=self.rnn(inputs,states)
                    output=self.fc2(hiddens.squeeze(1))
                    predict=output.argmax(1)
                    result.append(predict.item())
                    inputs=self.embed(predict).unsqueeze(1)
    #                 print("input: ", inputs.shape)
                    if self.vocab.idx2word[predict.item()]=="<end>":
                        break
                return [self.vocab.idx2word[idx] for idx in result]
            
class A2(nn.Module):
    def __init__(self, vocab, embed_size, vocab_size, config_data,hidden_size=512, num_layers=1,max_seq=20):
        super(A2, self).__init__()
        resnet_temp=models.resnet50(pretrained=True)
        self.resnet_modify=torch.nn.Sequential(*(list(resnet_temp.children())[:-1]))
        for param in self.resnet_modify.parameters():
            param.requires_grad = False
        self.fc1=nn.Linear(2048,embed_size)
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.config_data=config_data
        self.max_seq=max_seq
        self.vocab=vocab
        # lstm 
        self.lstm = nn.LSTM(input_size=self.embed_size*2, hidden_size=self.hidden_size,num_layers=num_layers,batch_first=True)
        self.fc2 = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x, captions): 
            feature = self.resnet_modify(x)
            feature=feature.reshape(feature.size(0),-1)
            feature = self.fc1(feature)
            feature=feature.unsqueeze(1)
            padding=torch.tensor([0]*captions.shape[0]).to(device)
            padding=self.embed(padding).unsqueeze(1)
            first=torch.cat((padding,feature),2)
            embedding= self.embed(captions)
            feature=feature.repeat(1,captions.shape[1],1)
            embedding= torch.cat((embedding[:,:,:],feature), dim=2)
            embeddings = torch.cat( (first, embedding[:,:-1,:]), dim=1)
            hiddens,_=self.lstm(embeddings)
            outputs = self.fc2(hiddens)
            return outputs
    def sample(self, x, states=None):
            result = []
            candidate=list(range(0,self.vocab_size))
            states=None
#             plt.imshow(x.cpu().reshape((3,256,256)).permute(1,2,0))
            features = self.resnet_modify(x)
            features=features.reshape(features.size(0),-1)
            features = self.fc1(features)
            features=features.unsqueeze(1)
            padding=torch.tensor([0]).to(device)
            padding=self.embed(padding).unsqueeze(1)
            inputs=torch.cat((padding,features),2)
            if( not self.config_data['generation']['deterministic']):
                for _ in range(self.max_seq):
                    hiddens,states=self.lstm(inputs,states)
                    output=self.fc2(hiddens.squeeze(1))
                    output=(output/self.config_data['generation']['temperature'])
                    prob=torch.softmax(output,dim=1)
                    predict=torch.multinomial(prob,num_samples=1) 
                    result.append(predict.item())
                    inputs=self.embed(predict)
                    inputs=torch.cat((inputs,features),2)
    #                 print("input: ", inputs.shape)
                    if self.vocab.idx2word[predict.item()]=="<end>":
                        break
                return [self.vocab.idx2word[idx] for idx in result if (idx!=1) and (idx!=2)]
            
#             determin
            else:
                for _ in range(self.max_seq):
                    hiddens,states=self.lstm(inputs,states)
                    output=self.fc2(hiddens.squeeze(1))
                    predict=output.argmax(1)
                    result.append(predict.item())
                    inputs=self.embed(predict).unsqueeze(1)
                    inputs=torch.cat((inputs,features),2)
    #                 print("input: ", inputs.shape)
                    if self.vocab.idx2word[predict.item()]=="<end>":
                        break
                return [self.vocab.idx2word[idx] for idx in result]            
def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    if model_type=="baseline":
               model=baseline(vocab=vocab,embed_size=embedding_size,hidden_size=hidden_size,vocab_size=len(vocab),config_data=config_data)
               return model
    elif model_type=="RNN":
               model=RNN(vocab=vocab,embed_size=embedding_size,hidden_size=hidden_size,vocab_size=len(vocab),config_data=config_data)
               return model
    else:
               model=A2(vocab=vocab,embed_size=embedding_size,hidden_size=hidden_size,vocab_size=len(vocab),config_data=config_data)
               return model

#     raise NotImplementedError("Model Factory Not Implemented")
