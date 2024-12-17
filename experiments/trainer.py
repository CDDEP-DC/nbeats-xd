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
Models training logic.
"""
from typing import Iterator

#import gin
import numpy as np
import torch as t
from torch import optim

from common.torch.losses import smape_2_loss, mape_loss, mase_loss
from common.torch.snapshots import SnapshotManager
from common.torch.ops import default_device, to_tensor, divide_no_nan
from torch.distributions.studentT import StudentT
#from torch.distributions.cauchy import Cauchy
#from torch.distributions.laplace import Laplace
#from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma

##
## training loop for point forecasts; loss based on distance
##

#@gin.configurable
def trainer(snapshot_manager: SnapshotManager,
            model: t.nn.Module,
            training_set: Iterator,
            loss_name: str,
            iterations: int,
            learning_rate: float = 0.001,
            pretrained_model_file = None,
            **kwargs):

    timeseries_frequency = kwargs.get("timeseries_frequency", 0)

    model = model.to(default_device())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    training_loss_fn = __loss_fn(loss_name)

    lr_decay_step = iterations // 3
    if lr_decay_step == 0:
        lr_decay_step = 1

    ## if a model file was provided, don't try to restore from snapshot folder
    if pretrained_model_file is None:
        iteration = snapshot_manager.restore(model, optimizer)
    else:
        model.load_state_dict(t.load(pretrained_model_file))
        iteration = 0

    #
    # Training Loop
    #
    snapshot_manager.enable_time_tracking()
    for i in range(iteration + 1, iterations + 1):
        model.train()
        x, x_mask, static_c, y, y_mask = next(training_set)
        optimizer.zero_grad()
        ## for some models, training is helped by peeking at the answer
        forecast = model(x, x_mask, static_c, y*y_mask)
        training_loss = training_loss_fn(x, timeseries_frequency, forecast, y, y_mask)

        if np.isnan(float(training_loss)):
            break

        training_loss.backward()
        t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate * 0.5 ** (i // lr_decay_step)

        snapshot_manager.register(iteration=i,
                                  training_loss=float(training_loss),
                                  validation_loss=np.nan, model=model,
                                  optimizer=optimizer)
    return model


def __loss_fn(loss_name: str):
    def loss(x, freq, forecast, target, target_mask):
        if loss_name == 'MAPE':
            return mape_loss(forecast, target, target_mask)
        elif loss_name == 'MASE':
            return mase_loss(x, freq, forecast, target, target_mask)
        elif loss_name == 'SMAPE':
            return smape_2_loss(forecast, target, target_mask)
        elif loss_name == "SL1":
            return t.nn.functional.smooth_l1_loss(forecast*target_mask, target*target_mask)
        else:
            raise Exception(f'Unknown loss function: {loss_name}')

    return loss





## likelihood-based loss fn:
def t_nll_loss(forecast_mu, target, variance):
    n = 5.0
    eps = 1e-6
    s = t.sqrt(variance + eps)
    m = StudentT(n, forecast_mu, s)
    return -1.0 * t.mean(t.sum(m.log_prob(target),dim=1))

## gamma is a continuous approximation to the negative binomial dist.
## which is like an overdispersed Poisson, so it works for count data
## (note: the approximation is bad for small counts, especially zeros)
## (also note: this predicts a probability distribution, not the future
## value directly. Because the dist is skewed, the likelihood is maximized
## when the predicted mean and median are higher than the observed data.)
def gamma_nll_loss(forecast_mu, target, variance):
    adj_targ = target + 0.5 ## prevents 0's in the data from hurting the model as much
    ## alpha = mu^2/var; beta = mu/var
    eps = 1e-4
    b = forecast_mu / (variance + eps)
    a = forecast_mu * b
    m = Gamma(a+eps,b+eps) ## not good if a or b are too close to 0
    p = t.nan_to_num(m.log_prob(adj_targ),-1e6) ## nans are bad, but don't let them crash the run
    return -1.0 * t.mean(t.sum(p,dim=1))
    
#def cauchy_nll_loss(forecast_mu, target, s):
#    m = Cauchy(forecast_mu, s)
#    return -1.0 * t.mean(t.sum(m.log_prob(target),dim=1))

#def laplace_nll_loss(forecast_mu, target, variance):
#    eps = 1e-6
#    s = t.sqrt(variance + eps) 
#    m = Laplace(forecast_mu, s)
#    return -1.0 * t.mean(t.sum(m.log_prob(target),dim=1))

## t loss with additional penalty for largest error in each forecast horizon
def t_nll_pen(forecast_mu, target, variance):
    n = 5.0
    eps = 1e-6
    s = t.sqrt(variance + eps)
    m = StudentT(n, forecast_mu, s)
    ## note, using mean instead of sum as reduction:
    mean_LL = -1.0 * t.mean(m.log_prob(target))

    se = t.square(forecast_mu - target)
    W = t.nn.functional.softmax(se, dim=1)
    ## differentializable approximation:
    largest_err = t.sum(W * se, dim=1)
    ## normalizing the squared error by the square of the forecast (make the scale match mean_LL above):
    normalizer = t.sum(W * t.square(forecast_mu), dim=1) + eps
    ## x 4 because normalizer is probably too high (like having st dev = the forecast value)
    penalty = 4.0 * t.mean(largest_err/normalizer)

    return mean_LL + penalty



def __ll_fn(loss_name: str):
    if loss_name == 't_nll':
        return t_nll_loss
    elif loss_name == 'norm_nll':
        return t.nn.functional.gaussian_nll_loss
    elif loss_name == 'gamma_nll':
        return gamma_nll_loss
    elif loss_name == 't_pen':
        return t_nll_pen
    else:
        raise Exception(f'Unknown loss function: {loss_name}')

##
## training loop for mean + variance forecasts
##

def trainer_var(snapshot_manager: SnapshotManager,
            model: t.nn.Module,
            training_set: Iterator,
            loss_name: str,
            iterations: int,
            learning_rate: float = 0.001,
            pretrained_model_file = None,
            **kwargs):

    model = model.to(default_device())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    training_loss_fn = __ll_fn(loss_name)

    lr_decay_step = iterations // 3
    if lr_decay_step == 0:
        lr_decay_step = 1

    ## if a model file was provided, don't try to restore from snapshot folder
    if pretrained_model_file is None:
        iteration = snapshot_manager.restore(model, optimizer)
    else:
        model.load_state_dict(t.load(pretrained_model_file))
        iteration = 0

    #
    # Training Loop
    #
    snapshot_manager.enable_time_tracking()
    for i in range(iteration + 1, iterations + 1):
        model.train()
        x, x_mask, static_c, y, y_mask = next(training_set)
        optimizer.zero_grad()

        forecast_mu, forecast_var = model(x, x_mask, static_c, y*y_mask)
        
        ## TODO: implement mask for loss fn (currently not masking)
        training_loss = training_loss_fn(forecast_mu, y, forecast_var)

        if np.isnan(float(training_loss)):
            print("argh")
            break

        training_loss.backward()
        t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate * 0.5 ** (i // lr_decay_step)

        snapshot_manager.register(iteration=i,
                                  training_loss=float(training_loss),
                                  validation_loss=np.nan, model=model,
                                  optimizer=optimizer)
    return model





##
## like trainer_var, but writes validation loss to snapshot
##

def trainer_validation(snapshot_manager: SnapshotManager,
            model: t.nn.Module,
            training_set: Iterator,
            loss_name: str,
            iterations: int,
            learning_rate: float = 0.001,
            pretrained_model_file = None,
            **kwargs):

    validation_input = kwargs.get("validation_input",None)
    validation_data = kwargs.get("validation_data",None)

    if validation_data is not None:
        validation_data = to_tensor(validation_data)
    
    model = model.to(default_device())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    training_loss_fn = __ll_fn(loss_name)

    lr_decay_step = iterations // 3
    if lr_decay_step == 0:
        lr_decay_step = 1

    ## if a model file was provided, don't try to restore from snapshot folder
    if pretrained_model_file is None:
        iteration = snapshot_manager.restore(model, optimizer)
    else:
        model.load_state_dict(t.load(pretrained_model_file))
        iteration = 0

    #
    # Training Loop
    #
    snapshot_manager.enable_time_tracking()
    for i in range(iteration + 1, iterations + 1):
        model.train()
        x, x_mask, static_c, y, y_mask = next(training_set)
        optimizer.zero_grad()

        forecast_mu, forecast_var = model(x, x_mask, static_c, y*y_mask)
        
        ## TODO: implement mask for loss fn (currently not masking)
        training_loss = training_loss_fn(forecast_mu, y, forecast_var)

        if np.isnan(float(training_loss)):
            print("argh")
            break

        training_loss.backward()
        t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate * 0.5 ** (i // lr_decay_step)

        validation_loss = np.nan
        if validation_input is not None:
            x, x_mask, static_c = validation_input
            model.eval()
            with t.no_grad():
                forecast_mu, forecast_var = model(x, x_mask, static_c)
                validation_loss = training_loss_fn(forecast_mu, validation_data, forecast_var)

        snapshot_manager.register(iteration=i,
                                  training_loss=float(training_loss),
                                  validation_loss=validation_loss, model=model,
                                  optimizer=optimizer)
    return model


