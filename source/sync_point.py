import csv
import sys


def _postposcess_slam(slam_raw_path: str):
    res = []
    f = open(slam_raw_path)
    csv_reader = csv.reader(f)
    for row in csv_reader:
        
        row[0] = int(row[0]) / 1000000
        row_string = str(row).replace('[','').replace(']','').replace("'",'')
        temp = row_string.split(',')
        res.append(temp)
    f.close()
    return res


def _postposcess_imu(imu_raw_path: str):
    res = []
    f = open(imu_raw_path)
    csv_reader = csv.reader(f)
    for row in csv_reader:
        
        row[0] = int(row[0])
        row_string = str(row).replace(' ','').replace('[','').replace(']','').replace("'",'')
        temp = row_string.split(',')
        res.append(temp)
    f.close()
    return res


def _slam_and_imu_sync(slam_post_list, imu_post_list, offset = 0):
    p = 0
    res = []
    print("offet,",offset)
    for row_imu in imu_post_list:
        time1 = float(row_imu[0])
        # if p >= len(slam_post_list):
        #         break
        imu_row = slam_post_list[p]
        time2 = float(imu_row[0])-offset
        while time2 <= time1:
            # print("time1:", time1, "time2:", time2)
            p += 1
            # if p >= len(slam_post_list):
            #     break
            imu_row = slam_post_list[p]
            time2 = float(imu_row[0])-offset
        p = p - 1
        tp = str(slam_post_list[p][1:])        
        t_str= str(str(row_imu) +',' +tp).replace(' ','').replace("'",'').replace('[','').replace(']','')
        # print('time 1: %f, time 2: %f imu time - slam time %f' %(time1, time2,time1-time2))
        res.append(t_str)
    return res


def  timestamp_sync(data, freq:int):
    time = 0
    offset = 1/freq
    res = []
    for i in range(0, len(data)):
        tt = data[i].split(',')
        t = float(tt[0])
        while(time<t):
            res.append(str(time)+','+str(tt[1:]).replace(' ','').replace("'",'').replace('[','').replace(']',''))
            time += offset
    return res



def sync_slam_pos_and_imu_data(slam_csv_path: str, imu_csv_path: str, out_file_path: str = 'defalut.csv', freq = 100, offset = 0):
    f = open(out_file_path, 'w+')
    print('start post sync processing...')
    slam_post = _postposcess_slam(slam_csv_path)
    imu_post = _postposcess_imu(imu_csv_path)
    print('temp processing done...')
    print('start syncing...')
    res =  _slam_and_imu_sync(slam_post, imu_post, offset)
    for row in res:
        f.write(row+'\n')
    print('done. point data saved in %s' % (out_file_path))


if __name__ == "__main__":
    slam_path = r'C:\Users\qiuchao.DESKTOP-MEE38AN\Desktop\10-17\20201017_164254_\_2020-10-17-16-43-24.bag.pbstream_x1y1x2y2.csv'
    imu_path = r'C:\Users\qiuchao.DESKTOP-MEE38AN\Desktop\新建文件夹\20201119_163545_\IMU-5-20-Pixel 3a.csv'
    out_path = r'C:\Users\qiuchao.DESKTOP-MEE38AN\Desktop\新建文件夹\kudou3.csv'
    # offset(ms) 1237
    offset = -16287
    freq = 200
    sync_slam_pos_and_imu_data(slam_path, imu_path, out_path, freq, offset)
