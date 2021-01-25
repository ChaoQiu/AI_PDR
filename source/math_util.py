import math

import numpy as np
import quaternion
from scipy import signal

def adjust_angle_array(angles):

    new_angle = np.copy(angles)
    angle_diff = angles[1:] - angles[:-1]

    diff_cand = angle_diff[:, None] - np.array([-math.pi * 4, -math.pi * 2, 0, math.pi * 2, math.pi * 4])
    min_id = np.argmin(np.abs(diff_cand), axis=1)

    diffs = np.choose(min_id, diff_cand.T)
    new_angle[1:] = np.cumsum(diffs) + new_angle[0]
    return new_angle


def orientation_to_angles(ori):

    if ori.dtype != quaternion.quaternion:
        ori = quaternion.from_float_array(ori)

    rm = quaternion.as_rotation_matrix(ori)
    angles = np.zeros([ori.shape[0], 3])
    angles[:, 0] = adjust_angle_array(np.arctan2(rm[:, 0, 1], rm[:, 1, 1]))
    angles[:, 1] = adjust_angle_array(np.arcsin(-rm[:, 2, 1]))
    angles[:, 2] = adjust_angle_array(np.arctan2(-rm[:, 2, 0], rm[:, 2, 2]))

    return angles


def angular_velocity_to_quaternion_derivative(q, w):
    omega = np.array([[0, -w[0], -w[1], -w[2]],
                      [w[0], 0, w[2], -w[1]],
                      [w[1], -w[2], 0, w[0]],
                      [w[2], w[1], -w[0], 0]]) * 0.5
    return np.dot(omega, q)


def gyro_integration(ts, gyro, init_q):

    output_q = np.zeros((gyro.shape[0], 4))
    output_q[0] = init_q
    dts = ts[1:] - ts[:-1]
    for i in range(1, gyro.shape[0]):
        output_q[i] = output_q[i - 1] + angular_velocity_to_quaternion_derivative(output_q[i - 1], gyro[i - 1]) * dts[
            i - 1]
        output_q[i] /= np.linalg.norm(output_q[i])
    return output_q


def interpolate_quaternion_linear(data, ts_in, ts_out):

    assert np.amin(ts_in) <= np.amin(ts_out), 'Input time range must cover output time range'
    assert np.amax(ts_in) >= np.amax(ts_out), 'Input time range must cover output time range'
    pt = np.searchsorted(ts_in, ts_out)
    d_left = quaternion.from_float_array(data[pt - 1])
    d_right = quaternion.from_float_array(data[pt])
    ts_left, ts_right = ts_in[pt - 1], ts_in[pt]
    d_out = quaternion.quaternion_time_series.slerp(d_left, d_right, ts_left, ts_right, ts_out)
    return quaternion.as_float_array(d_out)


def icp_fit_transformation(source, target):

    assert source.shape == target.shape
    center_source = np.mean(source, axis=0)
    center_target = np.mean(target, axis=0)
    m = source.shape[1]
    source_zeromean = source - center_source
    target_zeromean = target - center_target
    W = np.dot(source_zeromean.T, target_zeromean)
    U, S, Vt = np.linalg.svd(W)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = center_target.T - np.dot(R, center_source.T)

    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t
    return T, R, t


def dot_product_arr(v1, v2):
    if v1.ndim == 1:
        v1 = np.expand_dims(v1, axis=0)
    if v2.ndim == 1:
        v2 = np.expand_dims(v2, axis=0)
    assert v1.shape[0] == v2.shape[0], '{} {}'.format(v1.shape, v2.shape)
    dp = np.matmul(np.expand_dims(v1, axis=1), np.expand_dims(v2, axis=2))
    return np.squeeze(dp, axis=(1, 2))


def quaternion_from_two_vectors(v1, v2):

    one_dim = False
    if v1.ndim == 1:
        v1 = np.expand_dims(v1, axis=0)
        one_dim = True
    if v2.ndim == 1:
        v2 = np.expand_dims(v2, axis=0)
    assert v1.shape == v2.shape
    v1n = v1 / np.linalg.norm(v1, axis=1)[:, None]
    v2n = v2 / np.linalg.norm(v2, axis=1)[:, None]
    w = np.cross(v1n, v2n)
    q = np.concatenate([1.0 + dot_product_arr(v1n, v2n)[:, None], w], axis=1)
    q /= np.linalg.norm(q, axis=1)[:, None]
    if one_dim:
        return q[0]
    return q


def align_3dvector_with_gravity_legacy(data, gravity, local_g_direction=np.array([0, 1, 0])):

    assert data.ndim == 2, 'Expect 2 dimensional array input'
    assert data.shape[1] == 3, 'Expect Nx3 array input'
    assert data.shape[0] == gravity.shape[0], '{}, {}'.format(data.shape[0], gravity.shape[0])
    epsilon = 1e-03
    gravity_normalized = gravity / np.linalg.norm(gravity, axis=1)[:, None]
    output = np.copy(data)
    for i in range(data.shape[0]):
        # Be careful about two singular conditions where gravity[i] and local_g_direction are parallel.
        gd = np.dot(gravity_normalized[i], local_g_direction)
        if gd > 1. - epsilon:
            continue
        if gd < -1. + epsilon:
            output[i, [1, 2]] *= -1
            continue
        q = quaternion.from_float_array(quaternion_from_two_vectors(gravity[i], local_g_direction))
        output[i] = (q * quaternion.quaternion(1.0, *data[i]) * q.conj()).vec
    return output


def get_rotation_compensate_gravity(gravity, local_g_direction=np.array([0, 1, 0])):
    assert np.linalg.norm(local_g_direction) == 1.0
    gravity_normalized = gravity / np.linalg.norm(gravity, axis=1)[:, None]
    local_gs = np.stack([local_g_direction] * gravity.shape[0], axis=1).T
    dp = dot_product_arr(local_gs, gravity_normalized)
    flag_arr = np.zeros(dp.shape[0], dtype=np.int)
    flag_arr[dp < 0.0] = -1
    qs = quaternion_from_two_vectors(gravity_normalized, local_gs)
    return qs

def low_pass_filter(data):
    # 采样频率（sample frequency）Hz
    SF = 200
    # 截止频率（cut-off frequency）Hz
    CCF_1 = 25
    CCF_2 = 25
    # 阶数 N
    N_1 = 12
    N_2 = 12
    # 参数Wn
    Wn_1 = (2 * CCF_1) / SF
    Wn_2 = (2 * CCF_2) / SF
    b1, a1 = signal.butter(N_1, Wn_1, 'lowpass')
    b2, a2 = signal.butter(N_2, Wn_2, 'lowpass')

    data_temp = np.transpose(data)
    data_filtered = signal.filtfilt(b1, a1, data_temp)
    return np.transpose(data_filtered)