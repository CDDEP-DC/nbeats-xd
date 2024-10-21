# modified from github.com/ServiceNow/N-BEATS
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  
#
# original header:
# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
N-BEATS Model.
"""
from typing import Tuple, Optional

import numpy as np
import torch as t


## basis functions (see paper)
## currently only using GenericBasis

class GenericBasis(t.nn.Module):
    """
    Generic basis function.
    """
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: t.Tensor):
        return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]

'''
class TrendBasis(t.nn.Module):
    """
    Polynomial function to model trend.
    """
    def __init__(self, degree_of_polynomial: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.polynomial_size = degree_of_polynomial + 1  # degree of polynomial with constant term
        self.backcast_time = t.nn.Parameter(
            t.tensor(np.concatenate([np.power(np.arange(backcast_size, dtype=np.float) / backcast_size, i)[None, :]
                                     for i in range(self.polynomial_size)]), dtype=t.float32),
            requires_grad=False)
        self.forecast_time = t.nn.Parameter(
            t.tensor(np.concatenate([np.power(np.arange(forecast_size, dtype=np.float) / forecast_size, i)[None, :]
                                     for i in range(self.polynomial_size)]), dtype=t.float32), requires_grad=False)

    def forward(self, theta: t.Tensor):
        backcast = t.einsum('bp,pt->bt', theta[:, self.polynomial_size:], self.backcast_time)
        forecast = t.einsum('bp,pt->bt', theta[:, :self.polynomial_size], self.forecast_time)
        return backcast, forecast


class SeasonalityBasis(t.nn.Module):
    """
    Harmonic functions to model seasonality.
    """
    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.frequency = np.append(np.zeros(1, dtype=np.float32),
                                   np.arange(harmonics, harmonics / 2 * forecast_size,
                                             dtype=np.float32) / harmonics)[None, :]
        backcast_grid = -2 * np.pi * (
                np.arange(backcast_size, dtype=np.float32)[:, None] / forecast_size) * self.frequency
        forecast_grid = 2 * np.pi * (
                np.arange(forecast_size, dtype=np.float32)[:, None] / forecast_size) * self.frequency
        self.backcast_cos_template = t.nn.Parameter(t.tensor(np.transpose(np.cos(backcast_grid)), dtype=t.float32),
                                                    requires_grad=False)
        self.backcast_sin_template = t.nn.Parameter(t.tensor(np.transpose(np.sin(backcast_grid)), dtype=t.float32),
                                                    requires_grad=False)
        self.forecast_cos_template = t.nn.Parameter(t.tensor(np.transpose(np.cos(forecast_grid)), dtype=t.float32),
                                                    requires_grad=False)
        self.forecast_sin_template = t.nn.Parameter(t.tensor(np.transpose(np.sin(forecast_grid)), dtype=t.float32),
                                                    requires_grad=False)

    def forward(self, theta: t.Tensor):
        params_per_harmonic = theta.shape[1] // 4
        backcast_harmonics_cos = t.einsum('bp,pt->bt', theta[:, 2 * params_per_harmonic:3 * params_per_harmonic],
                                          self.backcast_cos_template)
        backcast_harmonics_sin = t.einsum('bp,pt->bt', theta[:, 3 * params_per_harmonic:], self.backcast_sin_template)
        backcast = backcast_harmonics_sin + backcast_harmonics_cos
        forecast_harmonics_cos = t.einsum('bp,pt->bt',
                                          theta[:, :params_per_harmonic], self.forecast_cos_template)
        forecast_harmonics_sin = t.einsum('bp,pt->bt', theta[:, params_per_harmonic:2 * params_per_harmonic],
                                          self.forecast_sin_template)
        forecast = forecast_harmonics_sin + forecast_harmonics_cos

        return backcast, forecast
'''


class NBeatsBlock(t.nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    as in original model, but with optional dropout
    """
    def __init__(self,
                 input_size: int,
                 theta_size: int,
                 basis_function: t.nn.Module,
                 layers: int,
                 layer_size: int,
                 dropout_rate: Optional[float] = None):
        """
        N-BEATS block.

        :param input_size: Insample size.
        :param theta_size:  Number of parameters for the basis function.
        :param basis_function: Basis function which takes the parameters and produces backcast and forecast.
        :param layers: Number of layers.
        :param layer_size: Layer size.
        """
        super().__init__()
        self.layers = t.nn.ModuleList([t.nn.Linear(in_features=input_size, out_features=layer_size)] +
                                      [t.nn.Linear(in_features=layer_size, out_features=layer_size)
                                       for _ in range(layers - 1)])
        self.basis_parameters = t.nn.Linear(in_features=layer_size, out_features=theta_size)
        self.basis_function = basis_function

        if dropout_rate is not None:
            self.dropout = t.nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        block_input = x
        for layer in self.layers:
            block_input = t.relu(layer(block_input))
            ## try dropout here? (after every layer except basis_parameters)
            #if self.dropout is not None:
            #    block_input = self.dropout(block_input)

        ## try dropout here? (once, before final layer of res block)
        if self.dropout is not None:
            block_input = self.dropout(block_input)

        basis_parameters = self.basis_parameters(block_input)
        return self.basis_function(basis_parameters)


class NBeats_stack(t.nn.Module):
    """
    doubly-residual stack of nbeatsblock, as in original paper
    option to force positive output (so this can be used to forecast variance, below)
    """
    def __init__(self, blocks: t.nn.ModuleList, force_positive: bool = False):
        super().__init__()
        self.blocks = blocks
        self.force_positive = force_positive

    def forward(self, residuals: t.Tensor, forecast: t.Tensor, input_mask: t.Tensor):
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * input_mask
            forecast = forecast + block_forecast

        if self.force_positive:
            forecast = t.nan_to_num(t.nn.functional.softplus(forecast))
        else:
            forecast = t.nan_to_num(forecast)

        return (residuals, forecast)


class NBeats_orig(t.nn.Module):
    """
    N-Beats Model. original univariate
    """
    def __init__(self, blocks: t.nn.ModuleList):
        super().__init__()
        self.nbeats = NBeats_stack(blocks)

    def forward(self, x: t.Tensor, input_mask: t.Tensor, static_cat: Optional[t.Tensor] = None, forecast_target: Optional[t.Tensor] = None) -> t.Tensor:
        residuals = x.flip(dims=(1,)) ## passing x to the network in reverse (does this make a difference?)
        input_mask = input_mask.flip(dims=(1,))
        forecast = x[:, -1:]  ### the last element in each series is the "level"; each block's forecast adds to it 
        (residuals, forecast) = self.nbeats(residuals, forecast, input_mask)
        return forecast


class NBeats_wnorm(t.nn.Module):
    """
    N-Beats Model, but with optional windowed normalization; univariate
    """
    def __init__(self, blocks: t.nn.ModuleList, use_norm: bool = True, norm_mean: bool = False):
        super().__init__()
        self.nbeats = NBeats_stack(blocks)
        self.use_norm = use_norm ## seems to improve forecasts
        self.norm_mean = norm_mean

    def forward(self, x_raw: t.Tensor, input_mask: t.Tensor, static_cat: Optional[t.Tensor] = None, forecast_target: Optional[t.Tensor] = None) -> t.Tensor:
        ## normalize by window
        if self.use_norm:
            if self.norm_mean:
                norm = x_raw.mean(dim=1,keepdim=True)
            else:
                norm = x_raw.median(dim=1,keepdim=True).values

            x = x_raw / norm
        else:
            x = x_raw.clone()

        residuals = x.flip(dims=(1,)) ## passing x to the network in reverse (does this make a difference?)
        input_mask = input_mask.flip(dims=(1,))
        forecast = x[:, -1:]  ### the last element in each series is the "level"; each block's forecast adds to it 
        (residuals, forecast) = self.nbeats(residuals, forecast, input_mask)

        if self.use_norm:
            return forecast * norm ## denormalize
        else:
            return forecast


class NBeats_var(t.nn.Module):
    """
    univariate version with error variance
    """
    def __init__(self, blocks: t.nn.ModuleList, blocks_var: t.nn.ModuleList, use_norm: bool = False, force_positive: bool = False, norm_mean: bool = False):
        super().__init__()
        self.nbeats = NBeats_stack(blocks, force_positive) ## force forecast to be positive?
        self.nbeats_var = NBeats_stack(blocks_var, True) ## always force positive variance
        self.use_norm = use_norm ## variance estimation maybe doesn't get along with windowed normalization?
        self.norm_mean = norm_mean
        
    def forward(self, x_raw: t.Tensor, input_mask: t.Tensor, static_cat: Optional[t.Tensor] = None, forecast_target: Optional[t.Tensor] = None) -> t.Tensor:
        ## normalize by window
        if self.use_norm:
            if self.norm_mean:
                norm = x_raw.mean(dim=1,keepdim=True) + 1e-6 
            else:
                norm = x_raw.median(dim=1,keepdim=True).values + 1e-6

            x = x_raw / norm
        else:
            x = x_raw.clone()

        residuals = x.flip(dims=(1,)) ## passing x to the network in reverse (does this make a difference?)
        input_mask = input_mask.flip(dims=(1,))
        forecast = x[:, -1:]  ### the last element in each series is the "level"; each block's forecast adds to it 
        (residuals, forecast) = self.nbeats(residuals, forecast, input_mask)

        ## try using final residuals to train error var forecast?
        residuals = t.square(residuals)
        ## should variance forcast add to final residual, or to zero?
        variance_forecast = t.zeros_like(forecast)
        (residuals, variance_forecast) = self.nbeats_var(residuals, variance_forecast, input_mask)

        ## note: must denormalize after softplus, not before
        if self.use_norm:
            forecast = forecast * norm
            variance_forecast = variance_forecast * norm * norm

        ## return both; the loss fn will decide what to do with them
        return (forecast, variance_forecast)


class NB_decoder(t.nn.Module):
    """
    NBeats model, but operates on encoded context derived from the target and exogenous covariates
    point forecasts, no error variance
    """

    def __init__(self, blocks: t.nn.ModuleList, exog_block: t.nn.Module, use_norm: bool = True, norm_mean: bool = False):
        super().__init__()
        self.nbeats = NBeats_stack(blocks)
        self.exog_block = exog_block  ## just an encoder
        self.use_norm = use_norm ## seems to improve forecasts
        self.norm_mean = norm_mean

    def forward(self, all_vars: t.Tensor, input_mask: t.Tensor, static_cat: t.Tensor, forecast_target: Optional[t.Tensor] = None) -> t.Tensor:
        ## first variable is the forecast target; rest are exogenous covariates
        x_raw = all_vars[:,:,0]
        covars = all_vars.clone() ## exog block also gets the target var

        ## normalize target by window median
        ## covars should be normalized at the dataset level (by window doesn't make sense)
        if self.use_norm:
            if self.norm_mean:
                norm = x_raw.mean(dim=1,keepdim=True)
            else:
                norm = x_raw.median(dim=1,keepdim=True).values

            x = x_raw / norm
            covars[:,:,0] = x ## replace with normalized
        else:
            x = x_raw.clone()

        ## exog block return sequence = input sequence to nbeats
        residuals = self.exog_block(covars, static_cat, forecast_target)

        residuals = residuals.flip(dims=(1,)) ## passing x to the network in reverse (does this make a difference?)
        input_mask = input_mask.flip(dims=(1,))
        forecast = x[:, -1:]  ### the last element in each series is the "level"; each block's forecast adds to it 
        (residuals, forecast) = self.nbeats(residuals, forecast, input_mask)

        if self.use_norm:
            return forecast * norm ## denormalize
        else:
            return forecast


class NB_dec_var(t.nn.Module):
    """
    NBeats model, but operates on encoded context derived from the target and exogenous covariates
    estimates error variance
    """

    def __init__(self, blocks: t.nn.ModuleList, blocks_var: t.nn.ModuleList, exog_block: t.nn.Module, use_norm: bool = False, force_positive: bool = False, norm_mean: bool = False):
        super().__init__()
        self.nbeats = NBeats_stack(blocks, force_positive) ## force forecast to be positive?
        self.nbeats_var = NBeats_stack(blocks_var, True) ## always force positive variance
        self.exog_block = exog_block  ## just an encoder
        self.use_norm = use_norm ## variance estimation maybe doesn't get along with windowed normalization?
        self.norm_mean = norm_mean

    def forward(self, all_vars: t.Tensor, input_mask: t.Tensor, static_cat: t.Tensor, forecast_target: Optional[t.Tensor] = None) -> t.Tensor:
        ## first variable is the forecast target; rest are exogenous covariates
        x_raw = all_vars[:,:,0]
        covars = all_vars.clone() ## exog block also gets the target var

        ## normalize by window
        ## covars should be normalized at the dataset level (by window doesn't make sense)
        if self.use_norm:
            if self.norm_mean:
                norm = x_raw.mean(dim=1,keepdim=True) + 1e-6 
            else:
                norm = x_raw.median(dim=1,keepdim=True).values + 1e-6

            x = x_raw / norm
            covars[:,:,0] = x ## replace with normalized
        else:
            x = x_raw.clone()

        ## exog block return sequence = input sequence to nbeats
        residuals = self.exog_block(covars, static_cat, forecast_target)

        residuals = residuals.flip(dims=(1,)) ## passing x to the network in reverse (does this make a difference?)
        input_mask = input_mask.flip(dims=(1,))
        forecast = x[:, -1:]  ### the last element in each series is the "level"; each block's forecast adds to it 
        (residuals, forecast) = self.nbeats(residuals, forecast, input_mask)

        ## try using final residuals to train error var forecast?
        residuals = t.square(residuals)
        ## should variance forcast add to final residual, or to zero?
        variance_forecast = t.zeros_like(forecast)
        (residuals, variance_forecast) = self.nbeats_var(residuals, variance_forecast, input_mask)

        ## note: must denormalize after softplus, not before
        if self.use_norm:
            forecast = forecast * norm 
            variance_forecast = variance_forecast * norm * norm

        ## return both; the loss fn will decide what to do with them
        return (forecast, variance_forecast)


class NB2stage(t.nn.Module):
    """
    as NB_dec_var, but uses separate stacks for target var and exogenous predictors
    """

    def __init__(self, blocks1: t.nn.ModuleList, blocks2: t.nn.ModuleList, blocks_var: t.nn.ModuleList, exog_block: t.nn.Module, use_norm: bool = False, force_positive: bool = False, norm_mean: bool = False):
        super().__init__()
        self.nbeats1 = NBeats_stack(blocks1, False)
        self.nbeats2 = NBeats_stack(blocks2, force_positive) ## force forecast to be positive?
        self.nbeats_var = NBeats_stack(blocks_var, True) ## always force positive variance
        self.exog_block = exog_block  ## just an encoder
        self.use_norm = use_norm ## variance estimation maybe doesn't get along with windowed normalization?
        self.norm_mean = norm_mean

    def forward(self, all_vars: t.Tensor, input_mask: t.Tensor, static_cat: t.Tensor, forecast_target: Optional[t.Tensor] = None) -> t.Tensor:
        ## first variable is the forecast target; rest are exogenous covariates
        x_raw = all_vars[:,:,0]

        ## normalize by window
        ## covars should be normalized at the dataset level (by window doesn't make sense)
        if self.use_norm:
            if self.norm_mean:
                norm = x_raw.mean(dim=1,keepdim=True) + 1e-6 
            else:
                norm = x_raw.median(dim=1,keepdim=True).values + 1e-6
                
            x = x_raw / norm
        else:
            x = x_raw.clone()

        residuals = x.flip(dims=(1,)) ## passing x to the network in reverse (does this make a difference?)
        input_mask = input_mask.flip(dims=(1,))
        forecast = x[:, -1:]  ### the last element in each series is the "level"; each block's forecast adds to it 
        (residuals, forecast) = self.nbeats1(residuals, forecast, input_mask)

        ## pass residuals from stage 1 along with covars to exog block
        covars = all_vars.clone() 
        covars[:,:,0] = residuals
        ## exog block return sequence = input sequence to nbeats
        residuals = self.exog_block(covars, static_cat, forecast_target)
        ## continue adding to forecast
        (residuals, forecast) = self.nbeats2(residuals, forecast, input_mask)

        ## try using final residuals to train error var forecast?
        residuals = t.square(residuals)
        ## should variance forcast add to final residual, or to zero?
        variance_forecast = t.zeros_like(forecast)
        (residuals, variance_forecast) = self.nbeats_var(residuals, variance_forecast, input_mask)

        ## note: must denormalize after softplus, not before
        if self.use_norm:
            forecast = forecast * norm 
            variance_forecast = variance_forecast * norm * norm

        ## return both; the loss fn will decide what to do with them
        return (forecast, variance_forecast)








# other attempts
'''
class NBeatsX(t.nn.Module):
    """
    N-Beats Model, with exog predictors (applied last)
    """
    def __init__(self, blocks: t.nn.ModuleList, exog_block: t.nn.Module, use_norm: bool = True):
        super().__init__()
        self.blocks = blocks
        self.exog_block = exog_block  ## some kind of encoder-decoder block
        self.use_norm = use_norm

    ## forecast_target is only to help train some types of exog models
    def forward(self, all_vars: t.Tensor, input_mask: t.Tensor, static_cat: t.Tensor, forecast_target: Optional[t.Tensor] = None) -> t.Tensor:
        x_raw = all_vars[:,:,0] ## first variable is the forecast target; rest are exogenous covariates

        ## normalize target by window median
        ## covars should be normalized at the dataset level (by window doesn't make sense)
        if self.use_norm:
            norm = x_raw.median(dim=1,keepdim=True).values
            x = x_raw / norm
        else:
            x = x_raw.clone()

        residuals = x
        forecast = x[:, -1:]  ### the last element in each series is the "level"; each block's forecast adds to it 

        ## including the (normalized) target along with the covars in case it's needed
        ## but don't use it as a regressor; let the rest of the model handle autoregression
        covars = all_vars.clone()
        covars[:,:,0] = x ## replace with normalized

        residuals = residuals.flip(dims=(1,)) ## passing x to the network in reverse (does this make a difference?)
        input_mask = input_mask.flip(dims=(1,))
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * input_mask
            forecast = forecast + block_forecast

        ## append resids to covars? (seems to make things worse)
        #z = t.concat([covars, residuals.flip(dims=(1,)).unsqueeze(2)], dim=2)
        #_, block_forecast = self.exog_block(z, static_cat, forecast_target)
        
        ## simply add to forecast at the end
        _, block_forecast = self.exog_block(covars, static_cat, forecast_target)
        forecast = forecast + block_forecast

        if self.use_norm:
            return forecast * norm ## denormalize
        else:
            return forecast
    

class NBeatsXr(t.nn.Module):
    """
    N-Beats Model, with exog predictors (applied first, potentially applying backcast to residuals)
    """
    def __init__(self, blocks: t.nn.ModuleList, exog_block: t.nn.Module, use_norm: bool = True):
        super().__init__()
        self.blocks = blocks
        self.exog_block = exog_block  ## some kind of encoder-decoder block
        self.use_norm = use_norm

    ## forecast_target is only to help train some types of exog models
    def forward(self, all_vars: t.Tensor, input_mask: t.Tensor, static_cat: t.Tensor, forecast_target: Optional[t.Tensor] = None) -> t.Tensor:
        x_raw = all_vars[:,:,0] ## first variable is the forecast target; rest are exogenous covariates

        ## normalize target by window median
        ## covars should be normalized at the dataset level (by window doesn't make sense)
        if self.use_norm:
            norm = x_raw.median(dim=1,keepdim=True).values
            x = x_raw / norm
        else:
            x = x_raw.clone()

        residuals = x
        forecast = x[:, -1:]  ### the last element in each series is the "level"; each block's forecast adds to it 

        ## including the (normalized) target along with the covars in case it's needed
        ## but don't use it as a regressor; let the rest of the model handle autoregression
        covars = all_vars.clone()
        covars[:,:,0] = x ## replace with normalized

        ## if exog_block generates a backcast, can try regression on covars first
        backcast, block_forecast = self.exog_block(covars, static_cat, forecast_target)
        residuals = (residuals - backcast) * input_mask
        forecast = forecast + block_forecast

        residuals = residuals.flip(dims=(1,)) ## passing x to the network in reverse (does this make a difference?)
        input_mask = input_mask.flip(dims=(1,))
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * input_mask
            forecast = forecast + block_forecast

        if self.use_norm:
            return forecast * norm ## denormalize
        else:
            return forecast
    

## exog-only, for testing
class ExogOnly(t.nn.Module):
    def __init__(self, exog_block: t.nn.Module, use_norm: bool = True):
        super().__init__()
        self.exog_block = exog_block
        self.use_norm = use_norm

    def forward(self, all_vars: t.Tensor, input_mask: t.Tensor, static_cat: t.Tensor, forecast_target: Optional[t.Tensor] = None) -> t.Tensor:
        x_raw = all_vars[:,:,0] ## first variable is the forecast target; rest are exogenous covariates

        ## normalize target by window median
        ## covars should be normalized at the dataset level (by window doesn't make sense)
        if self.use_norm:
            norm = x_raw.median(dim=1,keepdim=True).values
            x = x_raw / norm
        else:
            x = x_raw.clone()

        forecast = x[:, -1:]  ### the last element in each series is the "level"; each block's forecast adds to it 
        covars = all_vars.clone()
        covars[:,:,0] = x ## replace with normalized

        _, block_forecast = self.exog_block(covars, static_cat, forecast_target)
        forecast = forecast + block_forecast

        if self.use_norm:
            return forecast * norm ## denormalize
        else:
            return forecast

'''