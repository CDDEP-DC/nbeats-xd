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
Shortcut functions to create N-BEATS models.
"""
#import gin
import numpy as np
import torch as t
from typing import Optional

from models.nbeats import GenericBasis, NBeats_wnorm, NBeatsBlock, NBeats_var, NB_decoder, NB_dec_var, NB2stage

def generic_block(input_size: int, output_size: int, stacks: int, layers: int, layer_size: int, dropout: Optional[float] = None):
    return t.nn.ModuleList([NBeatsBlock(input_size=input_size,
                                            theta_size=input_size + output_size,
                                            basis_function=GenericBasis(backcast_size=input_size,
                                                                        forecast_size=output_size),
                                            layers=layers,
                                            layer_size=layer_size, dropout_rate=dropout)
                                for _ in range(stacks)])

#@gin.configurable()
def generic(input_size: int, output_size: int, stacks: int, layers: int, layer_size: int, 
            use_norm: bool = True, dropout: Optional[float] = None):
    """
    Create N-BEATS generic model. univariate, no error variance
    """
    return NBeats_wnorm(generic_block(input_size,output_size,stacks,layers,layer_size,dropout),
                  use_norm)


def generic_var(input_size: int, output_size: int, stacks: int, layers: int, layer_size: int,
                use_norm: bool = False, dropout: Optional[float] = None):
    """
    Create N-BEATS generic univariate model with error variance forecasting
    """
    return NBeats_var(generic_block(input_size, output_size, stacks, layers, layer_size, dropout),
                        ## use same architecture for variance
                        generic_block(input_size, output_size, stacks, layers, layer_size, dropout),
                        use_norm)


def generic_decoder(enc_dim: int, output_size: int, stacks: int, layers: int, layer_size: int, 
                    exog_block: t.nn.Module, use_norm: bool = True, dropout: Optional[float] = None):
    """
    Create N-BEATS decoder model for covariates, no error variance
    """
    return NB_decoder(generic_block(enc_dim,output_size,stacks,layers,layer_size,dropout),
                      exog_block, use_norm)


def generic_dec_var(enc_dim: int, output_size: int, stacks: int, layers: int, layer_size: int,
                    exog_block: t.nn.Module, use_norm: bool = False, dropout: Optional[float] = None,
                    force_positive: bool = False):
    """
    Create N-BEATS decoder model for covariates with error variance forecasting
    """
    return NB_dec_var(generic_block(enc_dim, output_size, stacks, layers, layer_size, dropout),
                        ## use same architecture for variance
                        generic_block(enc_dim, output_size, stacks, layers, layer_size, dropout),
                        exog_block, use_norm, force_positive)


def generic_2stage(enc_dim: int, output_size: int, stacks: int, layers: int, layer_size: int,
                    exog_block: t.nn.Module, use_norm: bool = False, dropout: Optional[float] = None,
                    force_positive: bool = False):
    """
    Create N-BEATS 2-stage decoder model for covariates with error variance forecasting
    """
    return NB2stage(generic_block(enc_dim, output_size, stacks, layers, layer_size, dropout),
                        ## use same architecture for 2nd stage and for variance
                        generic_block(enc_dim, output_size, stacks, layers, layer_size, dropout),
                        generic_block(enc_dim, output_size, stacks, layers, layer_size, dropout),
                        exog_block, use_norm, force_positive)





'''

def generic_exog(input_size: int, output_size: int,
            stacks: int, layers: int, layer_size: int,
            exog_block: t.nn.Module, exog_first: bool = False, use_norm: bool = True,
            exog_only: bool = False, dropout: Optional[float] = None):
    """
    Create N-BEATS generic model with a separate model for exogenous covariates
    """
    if exog_only:
        return ExogOnly(exog_block, use_norm)
    elif exog_first:
        return NBeatsXr(generic_block(input_size,output_size,stacks,layers,layer_size,dropout),
                        exog_block, use_norm)
    else:
        return NBeatsX(generic_block(input_size,output_size,stacks,layers,layer_size,dropout),
                        exog_block, use_norm)



                        
#@gin.configurable()
def interpretable(input_size: int,
                  output_size: int,
                  trend_blocks: int,
                  trend_layers: int,
                  trend_layer_size: int,
                  degree_of_polynomial: int,
                  seasonality_blocks: int,
                  seasonality_layers: int,
                  seasonality_layer_size: int,
                  num_of_harmonics: int):
    """
    Create N-BEATS interpretable model.
    """
    trend_block = NBeatsBlock(input_size=input_size,
                              theta_size=2 * (degree_of_polynomial + 1),
                              basis_function=TrendBasis(degree_of_polynomial=degree_of_polynomial,
                                                        backcast_size=input_size,
                                                        forecast_size=output_size),
                              layers=trend_layers,
                              layer_size=trend_layer_size)
    seasonality_block = NBeatsBlock(input_size=input_size,
                                    theta_size=4 * int(
                                        np.ceil(num_of_harmonics / 2 * output_size) - (num_of_harmonics - 1)),
                                    basis_function=SeasonalityBasis(harmonics=num_of_harmonics,
                                                                    backcast_size=input_size,
                                                                    forecast_size=output_size),
                                    layers=seasonality_layers,
                                    layer_size=seasonality_layer_size)

    return NBeats(t.nn.ModuleList(
        [trend_block for _ in range(trend_blocks)] + [seasonality_block for _ in range(seasonality_blocks)]))


'''