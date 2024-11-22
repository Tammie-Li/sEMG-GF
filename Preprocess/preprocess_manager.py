from Preprocess.filter import DateFilter
import numpy as np


class PreProcessManager:
    # 采样率为500Hz
    def __init__(self):
        self.sample_rate = 500
        self.filter = DateFilter(sample_rate=500)

    def data_filter(self, x):
        x = self.filter.band_pass_filter(x, freq_low=20, freq_high=150)
        return x
    
    def data_normalize(self, x):
        # z-method 数据归一化
        return x

    
    def data_slice(self, x, y, window_size, window_mov_t, speed):
        # 根据窗口大学和长度执行数据分段
        # 只检查，初始点和结束点的标签，标签一致即为样本
        s_x, s_y, s_speed = [], [], []
        idx_start = 0
        while idx_start < x.shape[0]:
            idx_end = idx_start + int(self.sample_rate * window_size)
            if idx_end >= x.shape[0]: break
            if y[idx_start] == y[idx_end]:
                s_x.append(x[idx_start: idx_end, :])
                s_y.append(y[idx_start])
                s_speed.append(speed[idx_start])
            idx_start = idx_start + int(self.sample_rate * window_mov_t)
        s_x, s_y, s_speed = np.array(s_x), np.array(s_y), np.array(s_speed)


        return s_x, s_y, s_speed
    
    def data_preprocess_all(self, x, y, window_size, window_mov_t, speed):
        # x = self.data_filter(x)
        x, y, speed = self.data_slice(x, y, window_size, window_mov_t, speed=speed)
        x = x.transpose(0, 2, 1)
        return x, y, speed

    def divide_train_test_data(self, x, y, speed, classes):
        num = x.shape[0]
        critical_value_tmp = int(num * 2 / 3)
        bias = 50
        for idx in range(critical_value_tmp - bias, critical_value_tmp + bias):
            if y[idx] == (classes-1) and y[idx+1] == 0: 
                critical_value = idx
        
        x_train, y_train = x[: critical_value, ...], y[: critical_value, ...]
        x_test, y_test = x[critical_value: , ...], y[critical_value: , ...]
        speed_test = speed[critical_value: , ...]

        return x_train, y_train, x_test, y_test, speed_test


                