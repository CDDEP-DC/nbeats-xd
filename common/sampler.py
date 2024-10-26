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
Timeseries sampler
"""
import numpy as np
import torch as t
from torch.utils.data import IterableDataset
from common.torch.ops import to_tensor, default_device
#import gin
from typing import Optional



##
## NOTE: this version assumes all timeseries are same length
## never samples past the beginning/end
## mask is disabled (returns a dummy mask)
##
## returns a torch IterableDataset
## this is then passed to torch DataLoader to handle batching
##
## if using timeseries of differing lengths, will need to re-enable mask
## or use original batch_sampler() below
##

## note, if cut weights are provided, they are assumed to apply to [min_cut, max_cut] inclusive
## ok to provide weights equal in length to time series, but they will be truncated and re-weighted
## (assumed that each weight indicates probability of cutting at (=after) the indicated position)

def ts_dataset(timeseries: np.ndarray, static_cat: Optional[np.ndarray],
                 insample_size: int,
                 outsample_size: int,
                 window_sampling_limit: int,
                 cut_weights = None):
    if timeseries.ndim == 2:
        return TimeDatasetUV(timeseries,insample_size,outsample_size,window_sampling_limit,cut_weights)
    else:
        return TimeDatasetMV(timeseries,static_cat,insample_size,outsample_size,window_sampling_limit,cut_weights)

##
## NOTE: to_tensor() puts training data in GPU memory to avoid extra copying later
## if there are many GB of data, don't do this
##

## dataset version, univariate
class TimeDatasetUV(IterableDataset):
    def __init__(self,
                 ts_np: np.ndarray,
                 insample_size: int,
                 outsample_size: int,
                 window_sampling_limit: int,
                 cut_weights = None):

        self.timeseries = to_tensor(ts_np) ## shape is [series, time], all the same length
        self.static_cat = t.zeros(ts_np.shape[0],dtype=int,device=default_device()) ## currently no category data in univar case
        self.insample_size = insample_size
        self.outsample_size = outsample_size
        self.n_series, self.ts_len = ts_np.shape
        self.mask = to_tensor(np.ones(1)) ## dummy mask, model expects it
        self.rng = np.random.default_rng()
        min_cut = max(self.insample_size, self.ts_len - window_sampling_limit)
        max_cut = self.ts_len - self.outsample_size
        self.cut_options = range(min_cut, max_cut+1)
        if cut_weights is not None:
            if len(cut_weights) > len(self.cut_options):
                cut_weights = cut_weights[:max_cut]
            if len(cut_weights) > len(self.cut_options):
                cut_weights = cut_weights[(min_cut-1):]
            cut_weights = cut_weights / np.sum(cut_weights,dtype=float)
        self.cut_weights = cut_weights

    def __iter__(self):
        """
        Returns a single sampled window=
         Insample: [insample size]
         Insample mask: [1]  -- won't sample past the start
         Outsample: [outsample size] 
         Outsample mask: [1] -- won't sample past the end
        """
        while True:
            i = self.rng.integers(self.n_series)
            #cut_point = self.rng.integers(self.min_cut, self.max_cut, endpoint=True)
            cut_point = self.rng.choice(self.cut_options, p=self.cut_weights)
            insample = self.timeseries[i, (cut_point - self.insample_size):cut_point]
            outsample = self.timeseries[i, cut_point:(cut_point + self.outsample_size)]
            yield insample, self.mask, self.static_cat[i], outsample, self.mask 

    def last_insample_window(self):
        """
        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        return self.timeseries[:, -self.insample_size:], to_tensor(np.ones((1,1))), self.static_cat


## multivariate
class TimeDatasetMV(IterableDataset):
    def __init__(self,
                 ts_np: np.ndarray, static_cat: Optional[np.ndarray],
                 insample_size: int,
                 outsample_size: int,
                 window_sampling_limit: int,
                 cut_weights = None):

        self.timeseries = to_tensor(ts_np) ## shape is [series, time, variables], all the same length
        if static_cat is not None:
            self.static_cat = t.tensor(static_cat, dtype=int).to(default_device()) ## shape is [series] -- only one categorical var for now
        else:
            self.static_cat = t.zeros(ts_np.shape[0],dtype=int,device=default_device())
        self.insample_size = insample_size
        self.outsample_size = outsample_size
        self.n_series, self.ts_len, self.n_variables = ts_np.shape
        self.mask = to_tensor(np.ones(1)) ## dummy mask, model expects it
        self.rng = np.random.default_rng()
        min_cut = max(self.insample_size, self.ts_len - window_sampling_limit)
        max_cut = self.ts_len - self.outsample_size
        self.cut_options = range(min_cut, max_cut+1)
        if cut_weights is not None:
            if len(cut_weights) > len(self.cut_options):
                cut_weights = cut_weights[:max_cut]
            if len(cut_weights) > len(self.cut_options):
                cut_weights = cut_weights[(min_cut-1):]
            cut_weights = cut_weights / np.sum(cut_weights,dtype=float)
        self.cut_weights = cut_weights

    def __iter__(self):
        """
        Returns a single sampled window=
         Insample: [insample size, # variables]
         Insample mask: [1]  -- won't sample past the start
         Outsample: [outsample size]  (only the first variable, for now)
         Outsample mask: [1] -- won't sample past the end
        """
        while True:
            i = self.rng.integers(self.n_series)
            #cut_point = self.rng.integers(self.min_cut, self.max_cut, endpoint=True)
            cut_point = self.rng.choice(self.cut_options, p=self.cut_weights)
            insample = self.timeseries[i, (cut_point - self.insample_size):cut_point, :] ## all vars, incl first
            outsample = self.timeseries[i, cut_point:(cut_point + self.outsample_size), 0] ## first variable only
            yield insample, self.mask, self.static_cat[i], outsample, self.mask

    def last_insample_window(self):
        """
        :return: Last insample window of all timeseries. Shape "timeseries, insample size, # variables"
        """
        return self.timeseries[:, -self.insample_size:, :], to_tensor(np.ones((1,1))), self.static_cat







## sampler below is the original version from NBEATS repo
## handles the batching itself

## batch sampler sometimes samples past ts start/end, resulting in truncated samples
## don't know if that's good or bad

def batch_sampler(timeseries: np.ndarray,
                 insample_size: int,
                 outsample_size: int,
                 window_sampling_limit: int,
                 batch_size: int = 1024):
    if timeseries.ndim == 2:
        return TimeseriesSampler(timeseries, insample_size, outsample_size, window_sampling_limit, batch_size)
    else:
        return TimeseriesSamplerMV(timeseries, insample_size, outsample_size, window_sampling_limit, batch_size)

#@gin.configurable
class TimeseriesSampler:
    def __init__(self,
                 timeseries: np.ndarray,
                 insample_size: int,
                 outsample_size: int,
                 window_sampling_limit: int,
                 batch_size: int = 1024):
        """
        Timeseries sampler.

        :param timeseries: Timeseries data to sample from. Shape: timeseries, timesteps
        :param insample_size: Insample window size. If timeseries is shorter then it will be 0-padded and masked.
        :param outsample_size: Outsample window size. If timeseries is shorter then it will be 0-padded and masked.
        :param window_sampling_limit: Size of history the sampler should use.
        :param batch_size: Number of sampled windows.
        """
        self.timeseries = [ts for ts in timeseries]
        self.window_sampling_limit = window_sampling_limit
        self.batch_size = batch_size
        self.insample_size = insample_size
        self.outsample_size = outsample_size

    def __iter__(self):
        """
        Batches of sampled windows.

        :return: Batches of:
         Insample: "batch size, insample size"
         Insample mask: "batch size, insample size"
         Outsample: "batch size, outsample size"
         Outsample mask: "batch size, outsample size"
        """
        while True:
            insample = np.zeros((self.batch_size, self.insample_size))
            insample_mask = np.zeros((self.batch_size, self.insample_size))
            outsample = np.zeros((self.batch_size, self.outsample_size))
            outsample_mask = np.zeros((self.batch_size, self.outsample_size))
            sampled_ts_indices = np.random.randint(len(self.timeseries), size=self.batch_size)
            for i, sampled_index in enumerate(sampled_ts_indices):
                sampled_timeseries = self.timeseries[sampled_index]
                cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                              high=len(sampled_timeseries),
                                              size=1)[0]

                insample_window = sampled_timeseries[max(0, cut_point - self.insample_size):cut_point]
                insample[i, -len(insample_window):] = insample_window
                insample_mask[i, -len(insample_window):] = 1.0
                outsample_window = sampled_timeseries[
                                   cut_point:min(len(sampled_timeseries), cut_point + self.outsample_size)]
                outsample[i, :len(outsample_window)] = outsample_window
                outsample_mask[i, :len(outsample_window)] = 1.0
            yield map(to_tensor, (insample, insample_mask, outsample, outsample_mask))

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.insample_size))
        insample_mask = np.zeros((len(self.timeseries), self.insample_size))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.insample_size:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return map(to_tensor, (insample, insample_mask))
    

##
## multivariate version
##
class TimeseriesSamplerMV:
    def __init__(self,
                 timeseries: np.ndarray,
                 insample_size: int,
                 outsample_size: int,
                 window_sampling_limit: int,
                 batch_size: int = 1024):
        """
        Timeseries sampler.

        :param timeseries: Timeseries data to sample from. Shape: [timeseries, timesteps, variables]
        :param insample_size: Insample window size. If timeseries is shorter then it will be 0-padded and masked.
        :param outsample_size: Outsample window size. If timeseries is shorter then it will be 0-padded and masked.
        :param window_sampling_limit: Size of history the sampler should use.
        :param batch_size: Number of sampled windows.
        """
        self.timeseries = [ts for ts in timeseries]  ## this allows timeseries of different lengths
        self.window_sampling_limit = window_sampling_limit
        self.batch_size = batch_size
        self.insample_size = insample_size
        self.outsample_size = outsample_size
        self.n_variables = self.timeseries[0].shape[1]

    def __iter__(self):
        """
        Batches of sampled windows.

        :return: Batches of:
         Insample: [batch size, insample size, # variables]
         Insample mask: [batch size, insample size]  (for now)
         Outsample: [batch size, outsample size]  (only the first variable, for now)
         Outsample mask: [batch size, outsample size]  (for now)
        """
        while True:
            insample = np.zeros((self.batch_size, self.insample_size, self.n_variables))
            insample_mask = np.zeros((self.batch_size, self.insample_size))
            outsample = np.zeros((self.batch_size, self.outsample_size))
            outsample_mask = np.zeros((self.batch_size, self.outsample_size))
            sampled_ts_indices = np.random.randint(len(self.timeseries), size=self.batch_size)
            for i, sampled_index in enumerate(sampled_ts_indices):
                sampled_timeseries = self.timeseries[sampled_index]
                cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                              high=len(sampled_timeseries),
                                              size=1)[0]

                insample_window = sampled_timeseries[max(0, cut_point - self.insample_size):cut_point]
                insample[i, -len(insample_window):] = insample_window
                insample_mask[i, -len(insample_window):] = 1.0
                outsample_window = sampled_timeseries[
                                   cut_point:min(len(sampled_timeseries), cut_point + self.outsample_size),
                                   0] ## first variable only
                
                outsample[i, :len(outsample_window)] = outsample_window
                outsample_mask[i, :len(outsample_window)] = 1.0
            yield map(to_tensor, (insample, insample_mask, outsample, outsample_mask))

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size, # variables"
        """
        insample = np.zeros((len(self.timeseries), self.insample_size, self.n_variables))
        insample_mask = np.zeros((len(self.timeseries), self.insample_size))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.insample_size:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return map(to_tensor, (insample, insample_mask))
    
