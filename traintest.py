import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import random

from arch import Encoder, Decoder, Seq2Seq


def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        src = batch['src']
        trg = batch['trg']
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch['src']
            trg = batch['trg']

            output = model(src, trg)

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


# if __name__ == "__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#     INPUT_DIM = len(SRC.vocab)
#     OUTPUT_DIM = len(TRG.vocab)
#     ENC_EMB_DIM = 256
#     DEC_EMB_DIM = 256
#     HID_DIM = 512
#     N_LAYERS = 2
#     ENC_DROPOUT = 0.5
#     DEC_DROPOUT = 0.5

#     enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
#     dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

#     model = Seq2Seq(enc, dec, device).to(device)

#     def init_weights(m):
#         for name, param in m.named_parameters():
#             nn.init.uniform_(param.data, -0.08, 0.08)
            
#     model.apply(init_weights)

#     def count_parameters(model):
#         return sum(p.numel() for p in model.parameters() if p.requires_grad)

#     print(f'The model has {count_parameters(model):,} trainable parameters')

#     optimizer = optim.Adam(model.parameters())

#     criterion = nn.CrossEntropyLoss()