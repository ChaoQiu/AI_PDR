from abc import ABC, abstractmethod
import h5py
import random
import numpy as np
import json
import math
import quaternion
import os
import warnings
from os import path as osp
import sys

from math_util import gyro_integration


class CompiledSequence(ABC):
    """
    An abstract interface for compiled sequence.
    """
    def __init__(self, **kwargs):
        super(CompiledSequence, self).__init__()

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def get_feature(self):
        pass

    @abstractmethod
    def get_target(self):
        pass

    @abstractmethod
    def get_aux(self):
        pass

    def get_meta(self):
        return "No info available"


def load_cached_sequences(seq_type, root_dir, data_list, cache_path, **kwargs):
    grv_only = kwargs.get('grv_only', True)

    if cache_path is not None and cache_path not in ['none', 'invalid', 'None']:
        if not osp.isdir(cache_path):
            os.makedirs(cache_path)
        if osp.exists(osp.join(cache_path, 'config.json')):
            info = json.load(open(osp.join(cache_path, 'config.json')))
            if info['feature_dim'] != seq_type.feature_dim or info['target_dim'] != seq_type.target_dim:
                warnings.warn('The cached dataset has different feature or target dimension. Ignore')
                cache_path = 'invalid'
            if info.get('aux_dim', 0) != seq_type.aux_dim:
                warnings.warn('The cached dataset has different auxiliary dimension. Ignore')
                cache_path = 'invalid'
            if info.get('grv_only', 'False') != str(grv_only):
                warnings.warn('The cached dataset has different flag in "grv_only". Ignore')
                cache_path = 'invalid'
        else:
            info = {'feature_dim': seq_type.feature_dim, 'target_dim': seq_type.target_dim,
                    'aux_dim': seq_type.aux_dim, 'grv_only': str(grv_only)}
            json.dump(info, open(osp.join(cache_path, 'config.json'), 'w'))

    features_all, targets_all, aux_all = [], [], []
    for i in range(len(data_list)):
        if cache_path is not None and osp.exists(osp.join(cache_path, data_list[i] + '.hdf5')):
            with h5py.File(osp.join(cache_path, data_list[i] + '.hdf5')) as f:
                feat = np.copy(f['feature'])
                targ = np.copy(f['target'])
                aux = np.copy(f['aux'])
        else:
            seq = seq_type(osp.join(root_dir, data_list[i]), **kwargs)
            feat, targ, aux = seq.get_feature(), seq.get_target(), seq.get_aux()
            print(seq.get_meta())
            if cache_path is not None and osp.isdir(cache_path):
                with h5py.File(osp.join(cache_path, data_list[i] + '.hdf5'), 'x') as f:
                    f['feature'] = feat
                    f['target'] = targ
                    f['aux'] = aux
        features_all.append(feat)
        targets_all.append(targ)
        aux_all.append(aux)
    return features_all, targets_all, aux_all


def select_orientation_source(data_path, max_ori_error=20.0, grv_only=True, use_ekf=True):

    ori_names = ['gyro_integration', 'game_rv']
    ori_sources = [None, None, None]

    with open(osp.join(data_path, 'info.json')) as f:
        info = json.load(f)
        ori_errors = np.array(
            [info['gyro_integration_error'], info['grv_ori_error'], info['ekf_ori_error']])
        init_gyro_bias = np.array(info['imu_init_gyro_bias'])

    with h5py.File(osp.join(data_path, 'data.hdf5')) as f:
        ori_sources[1] = np.copy(f['synced/game_rv'])
        if grv_only or ori_errors[1] < max_ori_error:
            min_id = 1
        else:
            if use_ekf:
                ori_names.append('ekf')
                ori_sources[2] = np.copy(f['pose/ekf_ori'])
            min_id = np.argmin(ori_errors[:len(ori_names)])
            # Only do gyro integration when necessary.
            if min_id == 0:
                ts = f['synced/time']
                gyro = f['synced/gyro_uncalib'] - init_gyro_bias
                ori_sources[0] = gyro_integration(ts, gyro, ori_sources[1][0])

    return ori_names[min_id], ori_sources[min_id], ori_errors[min_id]
