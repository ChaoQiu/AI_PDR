# 测试说明

---
### Requirements
python3, numpy, scipy, pandas, h5py, numpy-quaternion, matplotlib, torch, torchvision, tensorboardX, numba, plyfile, 
tqdm, scikit-learn

### Data 
#####Input：
每个csv文件为一个运动序列。共有13列数据，从左到右依次为：

|Time|Acc_x	|Acc_y	|Acc_z	|Gyro_x|	Gyro_y|	Gyro_z|	Grv_x|	Grv_y|	Grv_z|	Grv_w|	Pos_x|	Pos_y|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
1.60499E+12	|-0.19272274|	-0.25891206	|9.653498	|0.004885129|	-0.055565238|	-0.07149251|	-0.003427971|	0.017806085	|-0.7135098|	0.7004106	|-0.030632585	|-0.028743373|
1.60499E+12	|-0.3731718	|-0.26997933	|9.94875|	0.010914105|	-0.059184622	|-0.05671732|	-0.00361717	|0.017590584	|-0.71371627	|0.70020473|	-0.030615892	|-0.028751183|
1.60499E+12	|-0.37322932	|-0.29848543	|9.982276	|0.03363286	|-0.026876029	|-0.034452584	|-0.003655628	|0.017403742	|-0.71385366	|0.70006907	|-0.030615892	|-0.028751183


1.```Time```:时间戳。 注：采样率为**`200Hz`**，单位为`毫秒`。（采样率不一致会导致网络无法学习）

2.```Acc_*```: 加速度计的输出，由安卓手机提供-```Android Sensor.TYPE_ACCELEROMETER```

3.```Gyro_*```: 陀螺仪的输出，由安卓手机提供-```Android Sensor.TYPE_GYROSCOPE_UNCALIBRATED```

4.```Grv_*```: 游戏旋转向量，由安卓手机提供-```Android Sensor.TYPE_GAME_ROTATION_VECTOR```

5.```Pos_*```: 真实轨迹的坐标，单位为`米`，用作对比。（可以没有，需要在参数中设置无真值)
*   如果测试数据包含真实轨迹，并且需要对预测轨迹和真实进行坐标旋转对齐，那么可以在csv文件的最后添加“_度数"，代码会根据该度数进行对齐。eg.`kudou_hjh4_4.csv` 最后`_4`表示将真实轨迹逆时针旋转4度。如果是负数，那么就是顺时针旋转真实轨迹。

#####Output:
1.```.png```:图片，包括轨迹对比，误差分析等。轨迹的初始方向不定。

2.```.npy```: numpy文件格式，包含真实轨迹和预测轨迹的坐标点。频率为`20Hz`。



### Usage:
可以对单个运动序列测试，也可以批处理多个运动序列
*	进入```source```文件夹，输入相应参数来运行```ronin_resnet.py```。具体命令参数可以在```ronin_resnet.py```文件中查看。这里列出几个常用参数：

    --mode test/train	#测试模式或者训练模式，默认测试模式
    --test_path		#测试单个序列时该序列路径
    --test_list	        #批处理多个序列时，这些序列的列表文件所在的路径，列表文件一般为.txt文件
    --root_dir	        #批处理时，那些序列所在文件夹的路径
    --out_dir		#最后轨迹图像保存的路径。不填则不保存。
    --model_path		#预训练模型的路径
    --show_plot:		#打印图像
    --no_gt:		#无真值轨迹时输入此参数
*   单个序列测试：
在pycharm的Terminal中输入以下命令：```python ronin_resnet.py --mode test --test_path <path-to-test-data> --out_dir <path-to-output-folder> --model_path <path-to-model-checkpoint>  --show_plot```
*   批处理多个序列:
在pycharm的Terminal中输入以下命令：```python ronin_resnet.py --mode test --test_list <path-to-train-list> --root_dir <path-to-dataset-folder> --out_dir <path-to-output-folder> --model_path <path-to-model-checkpoint>.```

