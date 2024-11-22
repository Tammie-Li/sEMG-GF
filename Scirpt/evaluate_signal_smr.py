# 评估信号质量
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os, struct
from scipy.signal import welch

def plot1D_line(x):
    # 绘制一条线
    # 去除异常值
    plt.figure()
    plt.plot(x, color = 'b')
    plt.show()


def plot1D_line_group(x, row_n=2, col_n =4):
    # 每次绘制一条线, 输入[T, C], 共C个子图
    plt.figure(figsize=(16, 8))  

    # 绘制row_n  x  col_n个子图
    for i in range(x.shape[1]):
        plt.subplot(row_n, col_n, i + 1)  
        plt.plot(x[:, i], color='b')  
        plt.title(f'The raw signal of {i + 1}th channel sEMG')
        plt.xlabel('Sample point')
        plt.ylabel('Amplitude')

    plt.tight_layout()  # 调整子图间的间距和边缘
    plt.show()

def band_pass_filter(data, freq_low=20, freq_high=150):
    # 带通滤波
    wn = [freq_low * 2 / 500, freq_high * 2 / 500]
    b, a = signal.butter(6, wn, "bandpass")
    data = signal.filtfilt(b, a, data, axis=0)
    return data


def notch_filter_50Hz(data):
    f0 = 50.0  # 陷波中心频率为50Hz
    Q = 30.0  # 品质因数为30

    # 计算归一化角频率
    w0 = f0 / (500 / 2)

    # 创建陷波滤波器系数
    b, a = signal.butter(2, [w0 - 1/(2*Q), w0 + 1/(2*Q)], btype='bandstop')

    # 进行零相位滤波
    data_filtered = signal.filtfilt(b, a, data, axis=0)

    return data_filtered


def get_gesture_data(raw_data, gesture_id):
    # 筛选出选择的手势数据
    x, y = [], []
    for idx in range(raw_data.shape[0]):
        if int(raw_data[idx][12]) == gesture_id:
            x.append(raw_data[idx][: 8])
            y.append(raw_data[idx][13])
    x, y = np.array(x), np.array(y)
    return x, y


def get_rest_data(raw_data):
    x = []
    for idx in range(raw_data.shape[0]):
        if int(raw_data[idx][12]) == 1:
            x.append(raw_data[idx][: 8])
    x = np.array(x)
    return x


def divide_data_to_repetitions(x, y):
    # 将数据按照repetitions切分，共有12个block
    s_x, s_y = [], []
    idx_start, idx_end = 0, 0
    while idx_start < x.shape[0]:
        idx_end = idx_end + 1
        if idx_end == x.shape[0]: 
            s_x.append(x[idx_start:, :])
            break
        if y[idx_start] == y[idx_end]:
            continue
        else:
            s_x.append(x[idx_start: idx_end, :])
            idx_start = idx_end
    return s_x


def calculate_psd(signal, fs):
    freqs, psd = welch(signal, fs=fs, nperseg=500)
    return freqs, psd

def calculate_smr(filtered_signal, raw_signal, fs):
    # Calculate PSD for filtered and raw signals
    freqs_filtered, psd_filtered = calculate_psd(filtered_signal, fs)
    freqs_raw, psd_raw = calculate_psd(raw_signal, fs)

    # Linear fit from 0 Hz to the frequency of the highest mean power in raw signal
    max_idx = np.argmax(psd_raw)
    max_freq = freqs_raw[max_idx]
    slope = psd_raw[max_idx] / max_freq

    # Calculate motion artifact power
    motion_artifact_power = 0
    for i in range(len(freqs_raw)):
        if freqs_raw[i] <= 20:
            line_value = slope * freqs_raw[i]
            if psd_raw[i] > line_value:
                motion_artifact_power += psd_raw[i] - line_value

    # Calculate total power in filtered signal below 50 Hz
    total_filtered_power = np.sum(psd_filtered[freqs_filtered <= 50])

    # Calculate SMR
    smr = 10 * np.log10(total_filtered_power / motion_artifact_power)
    return smr


def calculate_signal_to_noise_ratio(raw_x):
    # 计算信噪比SNR
    # input (N, T, C) 
    r_num = 12
    channel = 8
    sum = 0
    for r in range(r_num):
        x = raw_x[r]
        for c in range(channel):
            signal = x[:, c]
            activated_signal = band_pass_filter(notch_filter_50Hz(signal))
            smr = calculate_smr(activated_signal, signal, 500)
            sum = sum + smr
    mean_smr = sum / 12 / 8
    print(mean_smr)
    return mean_smr

def calculate_signal_to_noise_ratio_by_gesture(data, gesture_id):
    x, y = get_gesture_data(data, gesture_id)
    x = divide_data_to_repetitions(x, y)
    smr = calculate_signal_to_noise_ratio(x)
    return smr


if __name__ == "__main__":
    path_dir = os.path.join(os.getcwd(), "Data")
    subject_smr = []
    for sub_id in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        day_smr = []
        for day_id in range(1, 3):
            path = os.path.join(path_dir, f"Subject_{sub_id:>02d}", f"Day_{day_id:>02d}", "raw_data.dat")
            with open(path, 'rb') as file:
                b_data = file.read()
                # 计算文件中64位浮点数的数量
                num_floats = len(b_data) // 8

                # 使用struct模块将二进制数据转化为浮点数
                f_data = struct.unpack('{}d'.format(num_floats), b_data)

            # 去除文件头信息
            data_npy = np.array(f_data[3:])

            # 每个样本长度为15
            data_len = len(data_npy) // 15

            # 数据格式 T * C，T表示采样点，C表示数据通道
            data = np.reshape(data_npy[: data_len*15], (data_len, 15))

            geture_smr = []
            for gesture_id in [2, 3, 4, 5, 6]:
                smr = calculate_signal_to_noise_ratio_by_gesture(data, gesture_id)
                geture_smr.append(smr)
            day_smr.append(geture_smr)
        subject_smr.append(day_smr)
    subject_smr = np.array(subject_smr)
    subject_smr = np.mean(subject_smr, axis=1)
    print(subject_smr.shape)

    np.save("snr.npy", subject_smr)

    subject_smr_mean = np.mean(subject_smr, axis=0)
    subject_smr_std = np.std(subject_smr, axis=0)

    print(subject_smr_mean)
    print(subject_smr_std)




