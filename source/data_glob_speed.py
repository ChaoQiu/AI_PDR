import json
import random
from os import path as osp
import pandas as pd

import h5py
import numpy as np
import quaternion
from numpy.core.defchararray import isdigit
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset
import math
from math_util import low_pass_filter
from data_utils import CompiledSequence, select_orientation_source, load_cached_sequences
import matplotlib.pyplot as plt

class GlobSpeedSequence(CompiledSequence):
    """
    Features :- raw angular rate and acceleration (includes gravity).
    """
    feature_dim = 6
    target_dim = 2
    aux_dim = 8

    def __init__(self, data_path=None, **kwargs):
        super().__init__(**kwargs)
        self.ts, self.features, self.targets, self.orientations, self.gt_pos = None, None, None, None, None
        self.info = {}

        self.grv_only = kwargs.get('grv_only', False)
        self.max_ori_error = kwargs.get('max_ori_error', 20.0)
        self.w = kwargs.get('interval', 1)
        self.no_gt=kwargs.get('no_gt', False)
        if data_path is not None:
            self.load(data_path)

    def Nrotate(self, angle, valuex, valuey):
        valuex = np.array(valuex)
        valuey = np.array(valuey)
        nRotatex = (valuex - 0) * math.cos(angle) - (valuey - 0) * math.sin(angle) + 0
        nRotatey = (valuex - 0) * math.sin(angle) + (valuey - 0) * math.cos(angle) + 0
        return nRotatex, nRotatey


    def load(self, data_path):
        if data_path[-1] == '/':
            data_path = data_path[:-1]
        data_name=osp.split(data_path)[-1]
        print(data_name)
        if data_name.endswith('.csv'):

            self.info['path'] = data_name
            self.info['ori_source'] = 'game rotation vector'
            self.info['source_ori_error'] = 0.0
            self.info['device'] = 'pixel'
            if self.no_gt:
                imu_data = pd.read_csv(osp.join(data_path),
                                   usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                   names=['time', 'acc_x', 'acc_y', 'acc_z', 'gyro_x',
                                          'gyro_y', 'gyro_z', 'rv_x', 'rv_y', 'rv_z', 'rv_w'])
                ts = imu_data.iloc[:, 0].values / 1000
                self.targets = np.zeros((len(ts)-200, 2))
                self.gt_pos = np.zeros((len(ts), 3))

            else:
                imu_data = pd.read_csv(osp.join(data_path),
                                   usecols=[0, 1, 2, 3, 4 ,5, 6, 7, 8, 9, 10, 18 ,19],
                                   names=['time', 'acc_x', 'acc_y', 'acc_z', 'gyro_x',
                                          'gyro_y', 'gyro_z', 'rv_x', 'rv_y', 'rv_z', 'rv_w', 'pos_x', 'pos_y'])
                imu_data=imu_data[1000:-1000]
                if osp.split(data_path)[-1].split('_')[-1]==None or not isdigit(osp.split(data_path)[-1].split('_')[-1][:-4].lstrip('-')):
                    degree_offset=0
                else:
                    degree_offset = int(osp.split(data_path)[-1].split('_')[-1][:-4])
                    # degree_offset=0
                g_pos_x, g_pos_y = self.Nrotate(math.radians(degree_offset), imu_data.iloc[:, 11:12].values,
                                                imu_data.iloc[:, 12:13].values)
                g_pos = np.concatenate([g_pos_x, g_pos_y, np.zeros([g_pos_x.shape[0], 1])], axis=1)
                ts = imu_data.iloc[:, 0].values/1000
                dt = (ts[self.w:] - ts[:-self.w])[:, None]
                glob_v = (g_pos[self.w:] - g_pos[:-self.w]) / dt
                self.gt_pos = g_pos
                self.targets = glob_v[:, :2]

            ori1 = imu_data.iloc[:, 7:10]
            ori2 = imu_data.iloc[:, 10]
            ori = pd.concat([ori2, ori1], axis=1)
            imu_ori_q = quaternion.from_float_array(ori)

            imu_acc = imu_data.iloc[:, 1:4].values
            imu_gro = imu_data.iloc[:, 4:7].values

            gyro_q = quaternion.from_float_array(np.concatenate([np.zeros([imu_gro.shape[0], 1]), imu_gro], axis=1))
            acce_q = quaternion.from_float_array(np.concatenate([np.zeros([imu_acc.shape[0], 1]), imu_acc], axis=1))
            glob_gyro = quaternion.as_float_array(imu_ori_q * gyro_q * imu_ori_q.conj())[:, 1:]
            glob_acce = quaternion.as_float_array(imu_ori_q * acce_q * imu_ori_q.conj())[:, 1:]
            # glob_acce=low_pass_filter(glob_acce)
            # glob_gyro=low_pass_filter(glob_gyro)
            self.features = np.concatenate([glob_gyro, glob_acce], axis=1)
            self.orientations = quaternion.as_float_array(imu_ori_q)
            self.ts = ts


        else:
            with open(osp.join(data_path, 'info.json')) as f:
                self.info = json.load(f)

            self.info['path'] = osp.split(data_path)[-1]

            self.info['ori_source'], ori, self.info['source_ori_error'] = select_orientation_source(
                data_path, self.max_ori_error, self.grv_only)


            with h5py.File(osp.join(data_path, 'data.hdf5')) as f:
                gyro_uncalib = f['synced/gyro_uncalib']
                acce_uncalib = f['synced/acce']
                gyro = gyro_uncalib - np.array(self.info['imu_init_gyro_bias'])
                acce = np.array(self.info['imu_acce_scale']) * (acce_uncalib - np.array(self.info['imu_acce_bias']))

                ts = np.copy(f['synced/time'])
                tango_pos = np.copy(f['pose/tango_pos'])
                init_tango_ori = quaternion.quaternion(*f['pose/tango_ori'][0])

            # Compute the IMU orientation in the Tango coordinate frame.

            ori_q = quaternion.from_float_array(ori)
            rot_imu_to_tango = quaternion.quaternion(*self.info['start_calibration'])
            init_rotor = init_tango_ori * rot_imu_to_tango * ori_q[0].conj()
            ori_q = init_rotor * ori_q

            dt = (ts[self.w:] - ts[:-self.w])[:, None]
            glob_v = (tango_pos[self.w:] - tango_pos[:-self.w]) / dt

            gyro_q = quaternion.from_float_array(np.concatenate([np.zeros([gyro.shape[0], 1]), gyro], axis=1))
            acce_q = quaternion.from_float_array(np.concatenate([np.zeros([acce.shape[0], 1]), acce], axis=1))
            glob_gyro = quaternion.as_float_array(ori_q * gyro_q * ori_q.conj())[:, 1:]
            glob_acce = quaternion.as_float_array(ori_q * acce_q * ori_q.conj())[:, 1:]

            start_frame = self.info.get('start_frame', 0)
            self.ts = ts[start_frame:]
            self.features = np.concatenate([glob_gyro, glob_acce], axis=1)[start_frame:]
            self.targets = glob_v[start_frame:, :2]
            self.orientations = quaternion.as_float_array(ori_q)[start_frame:]
            self.gt_pos = tango_pos[start_frame:]

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        print(self.ts.shape)
        print(self.orientations.shape)
        return np.concatenate([self.ts[:, None], self.orientations, self.gt_pos], axis=1)

    def get_meta(self):
        return '{}: device: {}, ori_error ({}): {:.3f}'.format(
            self.info['path'], self.info['device'], self.info['ori_source'], self.info['source_ori_error'])


class DenseSequenceDataset(Dataset):
    def __init__(self, seq_type, root_dir, data_list, cache_path=None, step_size=10, window_size=200,
                 random_shift=0, transform=None, **kwargs):
        super().__init__()
        self.feature_dim = seq_type.feature_dim
        self.target_dim = seq_type.target_dim
        self.aux_dim = seq_type.aux_dim
        self.window_size = window_size
        self.step_size = step_size
        self.random_shift = random_shift
        self.transform = transform

        self.data_path = [osp.join(root_dir, data) for data in data_list]
        self.index_map = []
        self.ts, self.orientations, self.gt_pos = [], [], []

        self.features, self.targets, aux = load_cached_sequences(
            seq_type, root_dir, data_list, cache_path, interval=1, **kwargs)

        # Optionally smooth the sequence
        feat_sigma = kwargs.get('feature_sigma,', -1)
        targ_sigma = kwargs.get('target_sigma,', -1)
        if feat_sigma > 0:
            self.features = [gaussian_filter1d(feat, sigma=feat_sigma, axis=0) for feat in self.features]
        if targ_sigma > 0:
            self.targets = [gaussian_filter1d(targ, sigma=targ_sigma, axis=0) for targ in self.targets]

        for i in range(len(data_list)):
            self.ts.append(aux[i][:, 0])
            self.orientations.append(aux[i][:, 1:5])
            self.gt_pos.append(aux[i][:, -3:])
            self.index_map += [[i, j] for j in range(window_size, self.targets[i].shape[0], step_size)]

        if kwargs.get('shuffle', True):
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        if self.random_shift > 0:
            frame_id += random.randrange(-self.random_shift, self.random_shift)
            frame_id = max(self.window_size, min(frame_id, self.targets[seq_id].shape[0] - 1))

        feat = self.features[seq_id][frame_id - self.window_size:frame_id]
        targ = self.targets[seq_id][frame_id]

        if self.transform is not None:
            feat, targ = self.transform(feat, targ)

        return feat.astype(np.float32).T, targ.astype(np.float32), seq_id, frame_id

    def __len__(self):
        return len(self.index_map)


class StridedSequenceDataset(Dataset):
    def __init__(self, seq_type, root_dir, data_list, cache_path=None, step_size=10, window_size=200,
                 random_shift=0, transform=None, **kwargs):
        super(StridedSequenceDataset, self).__init__()
        self.feature_dim = seq_type.feature_dim
        self.target_dim = seq_type.target_dim
        self.aux_dim = seq_type.aux_dim
        self.window_size = window_size
        self.step_size = step_size
        self.random_shift = random_shift
        self.transform = transform
        self.interval = kwargs.get('interval', window_size)

        self.data_path = [osp.join(root_dir, data) for data in data_list]
        self.index_map = []
        self.ts, self.orientations, self.gt_pos = [], [], []
        self.features, self.targets, aux = load_cached_sequences(
            seq_type, root_dir, data_list, cache_path, interval=self.interval, **kwargs)
        for i in range(len(data_list)):
            self.ts.append(aux[i][:, 0])
            self.orientations.append(aux[i][:, 1:5])
            self.gt_pos.append(aux[i][:, -3:])
            self.index_map += [[i, j] for j in range(0, self.targets[i].shape[0], step_size)]

        if kwargs.get('shuffle', True):
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        if self.random_shift > 0:
            frame_id += random.randrange(-self.random_shift, self.random_shift)
            frame_id = max(self.window_size, min(frame_id, self.targets[seq_id].shape[0] - 1))

        feat = self.features[seq_id][frame_id:frame_id + self.window_size]
        targ = self.targets[seq_id][frame_id]

        if self.transform is not None:
            feat, targ = self.transform(feat, targ)

        return feat.astype(np.float32).T, targ.astype(np.float32), seq_id, frame_id

    def __len__(self):
        return len(self.index_map)


class SequenceToSequenceDataset(Dataset):
    def __init__(self, seq_type, root_dir, data_list, cache_path=None, step_size=100, window_size=400,
                 random_shift=0, transform=None, **kwargs):
        super(SequenceToSequenceDataset, self).__init__()
        self.seq_type = seq_type
        self.feature_dim = seq_type.feature_dim
        self.target_dim = seq_type.target_dim
        self.aux_dim = seq_type.aux_dim
        self.window_size = window_size
        self.step_size = step_size
        self.random_shift = random_shift
        self.transform = transform

        self.data_path = [osp.join(root_dir, data) for data in data_list]
        self.index_map = []

        self.features, self.targets, aux = load_cached_sequences(
            seq_type, root_dir, data_list, cache_path, **kwargs)

        # Optionally smooth the sequence
        feat_sigma = kwargs.get('feature_sigma,', -1)
        targ_sigma = kwargs.get('target_sigma,', -1)
        if feat_sigma > 0:
            self.features = [gaussian_filter1d(feat, sigma=feat_sigma, axis=0) for feat in self.features]
        if targ_sigma > 0:
            self.targets = [gaussian_filter1d(targ, sigma=targ_sigma, axis=0) for targ in self.targets]

        max_norm = kwargs.get('max_velocity_norm', 3.0)
        self.ts, self.orientations, self.gt_pos, self.local_v = [], [], [], []
        for i in range(len(data_list)):
            self.features[i] = self.features[i][:-1]
            self.targets[i] = self.targets[i]
            self.ts.append(aux[i][:-1, :1])
            self.orientations.append(aux[i][:-1, 1:5])
            self.gt_pos.append(aux[i][:-1, 5:8])

            velocity = np.linalg.norm(self.targets[i], axis=1)  # Remove outlier ground truth data
            bad_data = velocity > max_norm
            for j in range(window_size + random_shift, self.targets[i].shape[0], step_size):
                if not bad_data[j - window_size - random_shift:j + random_shift].any():
                    self.index_map.append([i, j])

        if kwargs.get('shuffle', True):
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        # output format: input, target, seq_id, frame_id
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        if self.random_shift > 0:
            frame_id += random.randrange(-self.random_shift, self.random_shift)
            frame_id = max(self.window_size, min(frame_id, self.targets[seq_id].shape[0] - 1))

        feat = np.copy(self.features[seq_id][frame_id - self.window_size:frame_id])
        targ = np.copy(self.targets[seq_id][frame_id - self.window_size:frame_id])

        if self.transform is not None:
            feat, targ = self.transform(feat, targ)

        return feat.astype(np.float32), targ.astype(np.float32), seq_id, frame_id

    def __len__(self):
        return len(self.index_map)

    def get_test_seq(self, i):
        return self.features[i].astype(np.float32)[np.newaxis,], self.targets[i].astype(np.float32)
