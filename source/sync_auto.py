from sync_point import *
import os


def sync_auto(data_path,name):
    with open(os.path.join(data_path,"offset.txt")) as f:
        offset_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
    print(offset_list)
    for file in os.listdir(data_path):
        if file.isdigit():
            kudou_imu = ''
            kudou2_imu = ''
            shouchi_imu = ''
            shoubai_imu = ''
            path1 = os.path.join(data_path, file)
            for path_name in os.listdir(path1):
                if path_name.endswith('Pixel23.csv'):
                    kudou2_imu = os.path.join(path1, path_name)
                elif path_name.endswith('Pixel.csv'):
                    kudou_imu = os.path.join(path1, path_name)
                elif path_name.endswith('Pixel 3a.csv'):
                    shouchi_imu = os.path.join(path1, path_name)
                elif path_name.endswith('Pixel 3.csv'):
                    shoubai_imu = os.path.join(path1, path_name)
                elif path_name.endswith('_'):
                    path2 = os.path.join(path1, path_name)
                    for sub_path in os.listdir(os.path.join(path1, path_name)):
                        if sub_path.endswith('pbstream_x1y1x2y2.csv'):
                            slam = os.path.join(path2, sub_path)
            if kudou_imu!='':
                sync_slam_pos_and_imu_data(slam, kudou_imu, os.path.join(data_path, ('synced\kudou_'+name + str(file)) + '.csv'), 200,int(offset_list[int(file)-1]))
            if kudou2_imu != '':
                sync_slam_pos_and_imu_data(slam, kudou2_imu,
                                           os.path.join(data_path,
                                                        ('synced\kudou2_' + name + str(file)) + '.csv'), 200,
                                           int(offset_list[int(file) - 1]))
            if shouchi_imu!='':
                sync_slam_pos_and_imu_data(slam, shouchi_imu, os.path.join(data_path, ('synced\shouchi_' +name + str(file)) + '.csv'), 200,int(offset_list[int(file)-1]))
            if shoubai_imu != '':
                sync_slam_pos_and_imu_data(slam, shoubai_imu,
                                           os.path.join(data_path, ('synced\shoubai_' + name + str(file)) + '.csv'),
                                           200, int(offset_list[int(file) - 1]))


if __name__ == "__main__":
    data_path = r'C:\Users\qiuchao.DESKTOP-MEE38AN\Desktop\0104'
    name="cbk"
    sync_auto(os.path.join(data_path,name),name)



