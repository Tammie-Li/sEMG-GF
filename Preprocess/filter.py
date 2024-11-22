
from scipy import signal


class DateFilter:
    # 数据滤波
    def __init__(self, sample_rate=500):
        self.sample_rate = sample_rate

    def band_pass_filter(self, data, freq_low=20, freq_high=150):
        # 带通滤波
        wn = [freq_low * 2 / self.sample_rate, freq_high * 2 / self.sample_rate]
        b, a = signal.butter(6, wn, "bandpass")
        data = signal.filtfilt(b, a, data, axis=0)
        return data
    
    def high_pass_filter(self, data, freq=20):
        # 高通滤波
        b, a = signal.butter(10, freq / (self.sample_rate / 2), 'high', analog=False, output='ba')
        data = signal.filtfilt(b, a, data, axis=0)
        return data

    
    def notch_filter_50Hz(self, data):
        f0 = 50.0  # 陷波中心频率为50Hz
        Q = 30.0  # 品质因数为30

        # 计算归一化角频率
        w0 = f0 / (self.sample_rate / 2)

        # 创建陷波滤波器系数
        b, a = signal.butter(2, [w0 - 1/(2*Q), w0 + 1/(2*Q)], btype='bandstop')

        # 进行零相位滤波
        data_filtered = signal.filtfilt(b, a, data, axis=0)

        return data_filtered
    
