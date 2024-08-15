# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/. 

import numpy as np
import torch as t
import random
from common.torch.ops import default_device
from models.nbeats import GenericBasis, NBeats, NBeatsBlock
from typing import Tuple, Optional


##
## TCN from github.com/locuslab/TCN; Bai et al 2018
## original license:
'''
MIT License

Copyright (c) 2018 CMU Locus Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

from torch.nn.utils import weight_norm


class Chomp1d(t.nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(t.nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.conv1 = weight_norm(t.nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation))
        
        self.chomp1 = Chomp1d(padding)
        #self.bnorm1 = t.nn.BatchNorm1d(n_outputs)  ## try bn (instead of wt norm?) if not using window norm?
        self.relu1 = t.nn.ReLU()
        self.dropout1 = t.nn.Dropout(dropout)

        self.conv2 = weight_norm(t.nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        
        self.chomp2 = Chomp1d(padding)
        #self.bnorm2 = t.nn.BatchNorm1d(n_outputs)
        self.relu2 = t.nn.ReLU()
        self.dropout2 = t.nn.Dropout(dropout)

        self.net = t.nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                    self.conv2, self.chomp2, self.relu2, self.dropout2)
        #self.net = t.nn.Sequential(self.conv1, self.chomp1, self.bnorm1, self.relu1,
        #                         self.conv2, self.chomp2, self.bnorm2, self.relu2)
        
        self.downsample = t.nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = t.nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


## num_inputs is # features at each time step
## num_channels is a list of output (hidden) dim at each block; list length = # blocks
class TemporalConvNet(t.nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = t.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



## encoder only
class TCN_encoder(t.nn.Module):
    def __init__(self, n_features, hidden_dim_list, kernel_size, dropout, temporal: bool = True, n_embed: int = 0, embed_dim: int = 0):
        super(TCN_encoder, self).__init__()

        n_features = n_features + embed_dim + 1  ## +1 because including the target var

        self.tcn = TemporalConvNet(n_features, hidden_dim_list, kernel_size=kernel_size, dropout=dropout)

        self.combine_chans = t.nn.Conv1d(hidden_dim_list[-1], 1, 1) if temporal else None  ## see below

        if n_embed > 0:
            self.embed = t.nn.Embedding(n_embed,embed_dim)
        else:
            self.embed = None

        self.init_weights()

    def init_weights(self):
        if self.combine_chans is not None:
            self.combine_chans.weight.data.normal_(0, 0.01)

    def forward(self, x_input: t.Tensor, static_cat: t.Tensor, forecast_target: Optional[t.Tensor] = None) -> Tuple[t.Tensor, t.Tensor]:
        
        covars = x_input

        ## include category embedding vector as covars
        if self.embed is not None:
            embed_vecs = self.embed(static_cat) ## [1 cat per batch] -> [batch, embed dim]
            aug_covars = t.concat((covars,embed_vecs.unsqueeze(1).expand(-1,covars.shape[1],-1)),dim=2)
        else:
            aug_covars = covars

        y1 = self.tcn(aug_covars.transpose(1,2)) ## [batch, time, features] -> [batch, channels, time] for conv1d
        if self.combine_chans is not None:
            ## tcn return shape is [batch, channels, time]; combine channels into 1 and return [batch, time]
            return self.combine_chans(y1).squeeze(1)
        else:
            ## tcn return shape is [batch, channels, time]; return [batch, channels] at final time
            return y1[:, :, -1] 



##
## same idea, but using LSTM; TCN encoder seems to work slightly better (but has more moving parts)
##
class LSTM_test(t.nn.Module):

        def __init__(self, n_features: int, input_size: int, output_size: int, layer_size: int,  
                     n_embed: int = 0, embed_dim: int = 0,
                     decoder_extra_layers: int = 0,
                     lstm_layers: int = 1, lstm_hidden: int = 0,
                     temporal: bool = True, decode: bool = False):
        
            super().__init__()
            self.backcast_size = input_size
            self.forecast_size = output_size

            if lstm_hidden == 0:
                lstm_hidden = layer_size

            self.lstm = t.nn.LSTM(n_features+embed_dim+1, 
                                  lstm_hidden, num_layers=lstm_layers, batch_first=True)

            if n_embed > 0:
                 self.embed = t.nn.Embedding(n_embed,embed_dim)
            else:
                 self.embed = None

            self.downsample = t.nn.Linear(lstm_hidden, 1) if temporal else None

            if decode:
                enc_dim = input_size if temporal else lstm_hidden
                layers = [t.nn.Linear(enc_dim, layer_size)]
                layers.append(t.nn.ReLU())
                for i in range(decoder_extra_layers):
                    layers.append(t.nn.Linear(layer_size, layer_size))
                    layers.append(t.nn.ReLU())
                layers.append(t.nn.Linear(layer_size, input_size + output_size))
                self.block = t.nn.Sequential(*layers)
            else:
                self.block = None

        def forward(self, x_input: t.Tensor, static_cat: t.Tensor, forecast_target: Optional[t.Tensor] = None) -> Tuple[t.Tensor, t.Tensor]:
            covars = x_input

            if self.embed is not None:
                embed_vecs = self.embed(static_cat) ## [1 cat per batch] -> [batch, embed dim]
                aug_covars = t.concat((covars,embed_vecs.unsqueeze(1).expand(-1,covars.shape[1],-1)),dim=2)
            else:
                aug_covars = covars

            out, _ = self.lstm(aug_covars) ## out shape is [batch, seq len, hidden dim]

            if self.downsample is not None:
                out = self.downsample(out).squeeze(2) #[batch, len, hidden] -> [batch, len]
            else:
                out = out[:,-1,:] #[batch, len, hidden] -> [batch, hidden] at last timepoint

            if self.block is not None:
                theta = self.block(out)
                return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]
            else:
                return out







## previous efforts
'''

## this is an encoder-decoder architecture using TemporalConvNet as the encoder + a linear decoder
## decoder input = channel vector at the last timepoint 
class TCN(t.nn.Module):
    def __init__(self, n_features, in_seq_len, out_seq_len, hidden_dim_list, kernel_size, dropout, 
                 temporal_decoder,
                 n_embed: int = 0, embed_dim: int = 0):
        super(TCN, self).__init__()

        ##
        ## +1 if including the target var
        ##
        n_features = n_features + embed_dim + 1 

        self.backcast_size = in_seq_len
        self.forecast_size = out_seq_len

        self.tcn = TemporalConvNet(n_features, hidden_dim_list, kernel_size=kernel_size, dropout=dropout)

        self.combine_chans = t.nn.Conv1d(hidden_dim_list[-1], 1, 1) if temporal_decoder else None  ## see below

        ##
        ## try using a fancier structured decoder? like https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/models/tcn.py
        ##
        if temporal_decoder:
            self.linear = t.nn.Linear(in_seq_len, in_seq_len+out_seq_len)
        else:
            self.linear = t.nn.Linear(hidden_dim_list[-1], in_seq_len+out_seq_len)

        if n_embed > 0:
            self.embed = t.nn.Embedding(n_embed,embed_dim)
        else:
            self.embed = None

        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)
        if self.combine_chans is not None:
            self.combine_chans.weight.data.normal_(0, 0.01)

    def forward(self, x_input: t.Tensor, static_cat: t.Tensor, forecast_target: Optional[t.Tensor] = None) -> Tuple[t.Tensor, t.Tensor]:
        
        #covars = x_input[:,:,1:] ## first variable is past values of target; remove it?
        covars = x_input

        ## include category embedding vector as covars
        if self.embed is not None:
            embed_vecs = self.embed(static_cat) ## [1 cat per batch] -> [batch, embed dim]
            aug_covars = t.concat((covars,embed_vecs.unsqueeze(1).expand(-1,covars.shape[1],-1)),dim=2)
        else:
            aug_covars = covars

        y1 = self.tcn(aug_covars.transpose(1,2)) ## [batch, time, features] -> [batch, channels, time] for conv1d
        
        if self.combine_chans is not None:
            ## tcn return shape is [batch, channels, time]; combine channels into 1 and use value at each time
            theta = self.linear(self.combine_chans(y1).squeeze(1))
        else:
            ## tcn return shape is [batch, hidden, time]; use last hidden
            theta = self.linear(y1[:, :, -1]) 

        return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]







class Encoder(t.nn.Module):

    def __init__(self, n_features: int, hidden_dim: int, n_layers: int):
        
        super().__init__()
        self.lstm = t.nn.LSTM(n_features, hidden_dim, num_layers=n_layers, batch_first=True)

    ## automatically loops through sequence
    ## x_input has shape (batch size, seq len, # features)
    def forward(self, x_input: t.Tensor) -> Tuple[t.Tensor, t.Tensor]: 
        
        #lstm outputs each hidden state generated as it (internally) loops
        #but we only need hidden and cell state for the last one        
        _, (h_n, c_n) = self.lstm(x_input)
        
        ## return full set of hidden states for input to decoder
        return (h_n, c_n)
    

class Decoder(t.nn.Module):
    
    def __init__(self, hidden_dim: int, output_dim: int, n_layers: int):

        super().__init__()
        ## using lstm instead of single cell to make multiple layers easier
        self.lstm = t.nn.LSTM(output_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.linear = t.nn.Linear(hidden_dim, output_dim)  

    ## hanldles one seq item at a time
    ## shape of seq item = (batch size, seq_len, output_dim) where seq_len = 1
    ## hidden_tuple is (h_0, c_0), comes from encoder output or previous decoder cell
    def forward(self, x_input: t.Tensor, hidden_tuple: Tuple[t.Tensor, t.Tensor]) -> Tuple[t.Tensor, Tuple[t.Tensor, t.Tensor]]:
        
        _, (h_1, c_1) = self.lstm(x_input, hidden_tuple) ## h_1 has shape (# layers, batch size, hidden dim)
        ## linear takes shape (batch size, hidden_dim), outputs shape (batch size, output_dim)
        output = self.linear(h_1[-1]) ## if more than one layer, using the final layer only

        ## output and hidden state is sent to the next decoder step 
        return output, (h_1, c_1)
    


## targ_len is the forecast length
##   (since targ_len is always the same, maybe a simpler method would be more efficient?)
class EncDec(t.nn.Module):

    def __init__(self, n_features: int, targ_len: int, hidden_dim: int, n_layers: int):
        
        super().__init__()
        self.encoder = Encoder(n_features, hidden_dim, n_layers)
        self.decoder = Decoder(hidden_dim, 1, n_layers) ## target variable is 1d
        self.targ_len = targ_len
        self.forcing_rate = 1.0

    ## x_input shape is (batch size, input sequence len, # features)
    ## forecast_target is (batch size, forecast length)
    def forward(self, x_input: t.Tensor, static_cat: t.Tensor, forecast_target: Optional[t.Tensor] = None) -> Tuple[t.Tensor, t.Tensor]:
        ## first "feature" is past values of the forecast target; don't use it as regressor
        past_targets = x_input[:,:,[0]]
        covars = x_input[:,:,1:] 

        ## initialize predictions, which are univariate, so shape = (batch size, forecast length, 1)
        forecast = t.zeros(covars.shape[0], self.targ_len, 1, device=default_device())

        ## don't need to reset encoder hidden because it defaults to 0 when called
        h_n, c_n = self.encoder(covars) ## final encoder state is initial decoder state

        ## initial decoder input is last known target value, shape (batch size, 1, 1)
        decoder_input = past_targets[:, [-1], :]

        ## decoder outputs once for each item in target sequence
        for i in range(self.targ_len):
            ## pass previous decoder hidden state at each step
            decoder_output, (h_n, c_n) = self.decoder(decoder_input, (h_n, c_n))
            forecast[:,i] = decoder_output ## preds shape = (batch size, targ_len, 1)
            if (forecast_target is not None) and (random.random() < self.forcing_rate):
                decoder_input = forecast_target[:, [i]].unsqueeze(1) ## "teacher forcing"; add empty time dim
            else:
                decoder_input = decoder_output.unsqueeze(1)  ## next input is previous output; add empty time dim

        ## after each batch, decrease the forcing rate
        self.forcing_rate = self.forcing_rate * 0.99

        ## this model doesn't generate a backcast
        backcast = t.zeros(covars.shape[0], 1, device=default_device())
        return backcast, forecast.squeeze(2)


class ConvBlock(t.nn.Module):

        def __init__(self, n_features: int, input_size: int, output_size: int, kernel_size: int, layer_size: int):
        
            super().__init__()
            self.backcast_size = input_size
            self.forecast_size = output_size
            self.conv_block = t.nn.Sequential(
                t.nn.Conv1d(n_features, n_features, kernel_size),
                t.nn.ReLU(),
                t.nn.Conv1d(n_features, 1, kernel_size), ## downsample
                #t.nn.Conv1d(n_features, n_features, kernel_size), ## don't downsample
                t.nn.ReLU())
            self.lin_block = t.nn.Sequential(
                t.nn.Linear(input_size - 2*kernel_size + 2, layer_size), ## with downsampling
                #t.nn.Linear(n_features*(input_size - 2*kernel_size + 2), layer_size),  ## otherwise
                t.nn.ReLU(),
                t.nn.Linear(layer_size, layer_size),
                t.nn.ReLU(),
                t.nn.Linear(layer_size, input_size + output_size)
            )

        def forward(self, x_input: t.Tensor, static_cat: t.Tensor, forecast_target: Optional[t.Tensor] = None) -> Tuple[t.Tensor, t.Tensor]:
            covars = x_input[:,:,1:] ## first variable is past values of target; not using it
            theta = self.conv_block(covars.transpose(2,1)) ## [batch, channels, seq len]
            theta = self.lin_block(theta.squeeze(1)) ## if downsampling
            #theta = self.lin_block(theta.flatten(1)) ## otherwise
            return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]


class LSTM_with_backcast(t.nn.Module):

        def __init__(self, n_features: int, input_size: int, output_size: int, layer_size: int, decoder_extra_layers: int = 0, lstm_layers: int = 1, lstm_hidden: int = 0):
        
            super().__init__()
            self.backcast_size = input_size
            self.forecast_size = output_size
            if lstm_hidden == 0:
                lstm_hidden = layer_size
            self.lstm = t.nn.LSTM(n_features, lstm_hidden, num_layers=lstm_layers, batch_first=True)

            layers = []
            layers.append(t.nn.Linear(lstm_hidden, layer_size))
            layers.append(t.nn.ReLU())
            for i in range(decoder_extra_layers):
                 layers.append(t.nn.Linear(layer_size, layer_size))
                 layers.append(t.nn.ReLU())
            layers.append(t.nn.Linear(layer_size, input_size + output_size))
            self.block = t.nn.Sequential(*layers)

        def forward(self, x_input: t.Tensor, static_cat: t.Tensor, forecast_target: Optional[t.Tensor] = None) -> Tuple[t.Tensor, t.Tensor]:
            covars = x_input[:,:,1:] ## first variable is past values of target; not using it
            out, _ = self.lstm(covars)
            theta = self.block(out[:,-1,:]) ## out shape is [batch, seq len, hidden dim]
            return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]



class LSTM_back_cat(t.nn.Module):

        def __init__(self, n_features: int, input_size: int, output_size: int, layer_size: int,  
                     n_embed: int = 0, embed_dim: int = 0,
                     decoder_extra_layers: int = 0,
                     lstm_layers: int = 1, lstm_hidden: int = 0):
        
            super().__init__()
            self.backcast_size = input_size
            self.forecast_size = output_size

            if lstm_hidden == 0:
                lstm_hidden = layer_size
            self.lstm = t.nn.LSTM(n_features, lstm_hidden, num_layers=lstm_layers, batch_first=True)

            if n_embed > 0:
                 self.embed = t.nn.Embedding(n_embed,embed_dim)
            else:
                 self.embed = None

            layers = []
            layers.append(t.nn.Linear(lstm_hidden+embed_dim, layer_size))
            layers.append(t.nn.ReLU())
            for i in range(decoder_extra_layers):
                 layers.append(t.nn.Linear(layer_size, layer_size))
                 layers.append(t.nn.ReLU())
            layers.append(t.nn.Linear(layer_size, input_size + output_size))
            self.block = t.nn.Sequential(*layers)

        def forward(self, x_input: t.Tensor, static_cat: t.Tensor, forecast_target: Optional[t.Tensor] = None) -> Tuple[t.Tensor, t.Tensor]:
            covars = x_input[:,:,1:] ## first variable is past values of target; not using it
            out, _ = self.lstm(covars) ## out shape is [batch, seq len, hidden dim]
            last_out = out[:,-1,:] ## this slice is [batch, hidden dim]

            ## append the embedding vectors to the hidden context generated by the lstm?
            if self.embed is not None:
                 embed_vecs = self.embed(static_cat) ## [1 cat per batch] -> [batch, embed dim]
                 context = t.concat((embed_vecs,last_out),dim=1)
            else:
                 context = last_out
            theta = self.block(context)
            return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]



class LSTM_back_cat2(t.nn.Module):

        def __init__(self, n_features: int, input_size: int, output_size: int, layer_size: int,  
                     n_embed: int = 0, embed_dim: int = 0,
                     decoder_extra_layers: int = 0,
                     lstm_layers: int = 1, lstm_hidden: int = 0):
        
            super().__init__()
            self.backcast_size = input_size
            self.forecast_size = output_size

            if lstm_hidden == 0:
                lstm_hidden = layer_size
            self.lstm = t.nn.LSTM(n_features+embed_dim, lstm_hidden, num_layers=lstm_layers, batch_first=True)

            if n_embed > 0:
                 self.embed = t.nn.Embedding(n_embed,embed_dim)
            else:
                 self.embed = None

            layers = []
            layers.append(t.nn.Linear(lstm_hidden, layer_size))
            layers.append(t.nn.ReLU())
            for i in range(decoder_extra_layers):
                 layers.append(t.nn.Linear(layer_size, layer_size))
                 layers.append(t.nn.ReLU())
            layers.append(t.nn.Linear(layer_size, input_size + output_size))
            self.block = t.nn.Sequential(*layers)

        def forward(self, x_input: t.Tensor, static_cat: t.Tensor, forecast_target: Optional[t.Tensor] = None) -> Tuple[t.Tensor, t.Tensor]:
            covars = x_input[:,:,1:] ## first variable is past values of target; not using it

            ## this is probably a bad idea? but repeat the embedding values and treat them as time-invariant covars?
            if self.embed is not None:
                embed_vecs = self.embed(static_cat) ## [1 cat per batch] -> [batch, embed dim]
                aug_covars = t.concat((covars,embed_vecs.unsqueeze(1).expand(-1,covars.shape[1],-1)),dim=2)
            else:
                aug_covars = covars

            out, _ = self.lstm(aug_covars) ## out shape is [batch, seq len, hidden dim]
            theta = self.block(out[:,-1,:]) ## this slice is [batch, hidden dim]
            return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]






## predict future values of each exog covar using NBeats, then forecast target using multiple regression
## (note, they are not really predictions; will not be evaluated against actual future values; 
## only evaluated in terms of their usefulness in forecasting the variable of interest)
class PredictAndRegress(t.nn.Module):

    def __init__(self, n_features: int, input_size: int, output_size: int, stacks: int, layer_size: int,
                 n_embed: int = 0, embed_dim: int = 0):

        super().__init__()
        self.n_features = n_features
        self.backcast_size = input_size
        self.forecast_size = output_size
        ## separate model for each feature
        self.nb_models = t.nn.ModuleList(
                            [NBeats(t.nn.ModuleList([NBeatsBlock(input_size=input_size,
                                                    theta_size=input_size + output_size,
                                                    basis_function=GenericBasis(backcast_size=input_size,
                                                                                forecast_size=output_size),
                                                    layers=4,
                                                    layer_size=layer_size)
                                                for _ in range(stacks)]),
                                                False)
                            for i in range(n_features)])
        
        ## regression: features at each timepoint -> single value at each timepoint
        ##  (same regression weights at every timepoint)
        n_r = n_features + embed_dim
        n_h = (n_r + 2) // 2
        self.mlp_regress = t.nn.Sequential(t.nn.Linear(n_r,n_h),t.nn.ReLU(),t.nn.Linear(n_h,1))

        if n_embed > 0:
            self.embed = t.nn.Embedding(n_embed,embed_dim)
        else:
            self.embed = None


    def forward(self, x_input: t.Tensor, static_cat: t.Tensor, forecast_target: Optional[t.Tensor] = None) -> Tuple[t.Tensor, t.Tensor]:
        
        x = x_input[:,:,1:] ## first variable is past values of target; not using it

        ## call a model to "forecast" each covar, final shape is [batch, time, # covars]
        xf = t.stack(
                    [m(x[:,:,i], t.ones_like(x[:,:,i])) ## ones = fake mask (for now)
                    for (i,m) in enumerate(self.nb_models)], 
                    dim=2)
        
        ## concatenate in time dimension
        x_with_f = t.cat([x,xf],dim=1)

        ## include category embedding vector as regressors
        if self.embed is not None:
            embed_vecs = self.embed(static_cat) ## [1 cat per batch] -> [batch, embed dim]
            aug_covars = t.concat((x_with_f,embed_vecs.unsqueeze(1).expand(-1,x_with_f.shape[1],-1)),dim=2)
        else:
            aug_covars = x_with_f

        ## regression network acts on each timepoint
        ## input shape is [batch, time, # covars], so ouput shape is [batch, time, 1]
        theta = self.mlp_regress(aug_covars).squeeze(2)
        return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]

'''