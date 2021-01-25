# 滤波程序，对6轴（_1）和真值（_2）都进行了低通滤波
# 将真值（_2）独立列出来是为了后面看是否需要调整真值滤波器
# 目前感觉真值本身频率就很低，滤波25Hz基本没变化，不知道要不要修改，就先和6轴（_1）的滤波器设置一样
import os
import pandas as pd
import numpy as np
from scipy import signal

# 采样频率（sample frequency）Hz
SF = 200
# 截止频率（cut-off frequency）Hz
CCF_1 = 25
CCF_2 = 25
# 阶数 N
N_1 = 12
N_2 = 12
# 参数Wn
Wn_1 = (2*CCF_1)/SF
Wn_2 = (2*CCF_2)/SF
b1, a1 = signal.butter(N_1, Wn_1, 'lowpass')
b2, a2 = signal.butter(N_2, Wn_2, 'lowpass')

data_file_path = 'C:\\Users\\14799\\Desktop\\测试\\shuju'
deal_num = 0
for root, ds, fs in os.walk(data_file_path):
    for f in fs:
        if f.endswith('.csv'):
            data_csv = pd.read_csv(os.path.join(root, f), header=None)
            data_arr = np.array(data_csv)
            # 将csv读取结果转换为python二维矩阵
            data_time = data_arr[:, 0:1]
            data_filter_1 = data_arr[:, 1:7]
            data_ignore_1 = data_arr[:, 7:11]
            data_filter_2 = data_arr[:, 11:13]
            data_ignore_2 = data_arr[:, 13:18]
            # 转置后进行滤波
            data_temp = np.transpose(data_filter_1)
            data_filtered = signal.filtfilt(b1, a1, data_temp)
            data_filter_1 = np.transpose(data_filtered)

            data_temp = np.transpose(data_filter_2)
            data_filtered = signal.filtfilt(b2, a2, data_temp)
            data_filter_2 = np.transpose(data_filtered)
            # 合并二维矩阵后输出
            data_temp = np.hstack((data_time, np.hstack((data_filter_1, np.hstack((data_ignore_1, np.hstack((data_filter_2, data_ignore_2))))))))
            path_out = os.path.dirname(os.path.join(root, f)) + '\\filtered-' + f
            np.savetxt(path_out, data_temp, delimiter=',')
            deal_num += 1
            print(deal_num, path_out)

print('The End.')
