import math

import torch, copy
import torch.nn as nn
import torch.nn.functional as F

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])    
    


class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward = 2048, dropout = 0.1, activation = "relu", layer_norm_eps = 1e-5):
        super(TransformerEncoderLayer, self).__init__()
        # d_model is emb_size
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)


    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)


    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2, attn = self.self_attn(src, src, src, attn_mask = src_mask,
                              key_padding_mask = src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn



class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, device, norm = None):
        super(TransformerEncoder, self).__init__()
        self.device = device
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm


    def forward(self, src, mask = None, src_key_padding_mask = None):
        r"""Pass the input through the encoder layers in turn.
        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = src
        attn_output = torch.zeros((src.shape[1], src.shape[0], src.shape[0]), device = self.device) # batch, seq_len, seq_len
        
        for mod in self.layers:
            output, attn = mod(output, src_mask = mask, src_key_padding_mask = src_key_padding_mask)
            attn_output += attn
            
        if self.norm is not None:
            output = self.norm(output)

        return output, attn_output

class PositionalEncoding(nn.Module):

    def __init__(self, seq_len, d_model, dropout = 0.1):
        super(PositionalEncoding, self).__init__()
        max_len = max(5000, seq_len)
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[: , 0 : -1]
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    # Input: seq_len x batch_size x dim, Output: seq_len, batch_size, dim
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
    
    
class Permute(torch.nn.Module):
    def forward(self, x):
        return x.permute(1, 0)
    
    

class MultitaskTransformerModel(nn.Module):

    def __init__(self, task_type, device, nclasses, seq_len, batch, input_size, emb_size, nhead, nhid, nhid_tar, nhid_task, nlayers, dropout = 0.1):
        super(MultitaskTransformerModel, self).__init__()
        # from torch.nn import TransformerEncoder, TransformerEncoderLayer
        
        self.trunk_net = nn.Sequential(
            nn.Linear(input_size, emb_size),
            nn.BatchNorm1d(batch),
            PositionalEncoding(seq_len, emb_size, dropout),
            nn.BatchNorm1d(batch)
        )
        
        # encoder_layers = transformer_encoder_class.TransformerEncoderLayer(emb_size, nhead, nhid, out_channel, filter_height, filter_width, dropout)
        # encoder_layers = TransformerEncoderLayer(emb_size, nhead, nhid, dropout)
        # self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        encoder_layers = TransformerEncoderLayer(emb_size, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers, device)
        
        self.batch_norm = nn.BatchNorm1d(batch)
        
        # Task-aware Reconstruction Layers
        self.tar_net = nn.Sequential(
            nn.Linear(emb_size, nhid_tar),
            nn.BatchNorm1d(batch),
            nn.Linear(nhid_tar, nhid_tar),
            nn.BatchNorm1d(batch),
            nn.Linear(nhid_tar, input_size),
        )

        if task_type == 'classification':
            # Classification Layers
            self.class_net = nn.Sequential(
                nn.Linear(emb_size, nhid_task),
                nn.ReLU(),
                Permute(),
                nn.BatchNorm1d(batch),
                Permute(),
                nn.Dropout(p = 0.3),
                nn.Linear(nhid_task, nhid_task),
                nn.ReLU(),
                Permute(),
                nn.BatchNorm1d(batch),
                Permute(),
                nn.Dropout(p = 0.3),
                nn.Linear(nhid_task, nclasses)
            )
        else:
            # Regression Layers
            self.reg_net = nn.Sequential(
                nn.Linear(emb_size, nhid_task),
                nn.ReLU(),
                Permute(),
                nn.BatchNorm1d(batch),
                Permute(),
                nn.Linear(nhid_task, nhid_task),
                nn.ReLU(),
                Permute(),
                nn.BatchNorm1d(batch),
                Permute(),
                nn.Linear(nhid_task, 1),
            )
            

        
    def forward(self, x, task_type):
        x = self.trunk_net(x.permute(1, 0, 2))
        x, attn = self.transformer_encoder(x)
        x = self.batch_norm(x)
        # x : seq_len x batch x emb_size

        if task_type == 'reconstruction':
            output = self.tar_net(x).permute(1, 0, 2)
        elif task_type == 'classification':
            output = self.class_net(x[-1])
        elif task_type == 'regression':
            output = self.reg_net(x[-1])
        return output, attn
