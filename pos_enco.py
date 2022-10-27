import torch
import torch.nn as nn
import numpy as np
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                np.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                np.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        # pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        #add constant to embedding
        # seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len], \
        requires_grad=False).cuda()
        return x

a = PositionalEncoder(4,56)
print(a.pe[[1,2,4]])