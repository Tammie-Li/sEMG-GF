# 评估信号质量
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os, struct

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


def calculate_rms(signal):
    return np.sqrt(np.mean(np.square(signal)))


def calculate_signal_to_noise_ratio(raw_x, raw_rest_x):
    # 计算信噪比SNR
    # input (N, T, C) 
    r_num = 12
    channel = 8
    sum = 0
    for r in range(r_num):
        rest_x = raw_rest_x[r]
        x = raw_x[r]
        for c in range(channel):
            signal = x[:, c]
            activated_signal = band_pass_filter(notch_filter_50Hz(signal))
            rms_activated = calculate_rms(activated_signal)
            rms_resting = calculate_rms(band_pass_filter(notch_filter_50Hz(rest_x[:, c])))
            snr = 20 * np.log10(rms_activated / rms_resting)
            sum = sum + snr
    mean_snr = sum / 12 / 8
    print(mean_snr)
    return mean_snr

def calculate_signal_to_noise_ratio_by_gesture(data, gesture_id):
    x, y = get_gesture_data(data, gesture_id)
    rest_x, rest_y = get_gesture_data(data, 1)
    x = divide_data_to_repetitions(x, y)
    rest_x = divide_data_to_repetitions(rest_x, rest_y)
    snr = calculate_signal_to_noise_ratio(x, rest_x)
    return snr


if __name__ == "__main__":
    path_dir = os.path.join(os.getcwd(), "Data")
    subject_snr = []
    for sub_id in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        day_snr = []
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

            geture_snr = []
            for gesture_id in [2, 3, 4, 5, 6]:
                snr = calculate_signal_to_noise_ratio_by_gesture(data, gesture_id)
                geture_snr.append(snr)
            day_snr.append(geture_snr)
        subject_snr.append(day_snr)
    subject_snr = np.array(subject_snr)
    subject_snr = np.mean(subject_snr, axis=1)
    print(subject_snr.shape)

    np.save("snr.npy", subject_snr)

    subject_snr_mean = np.mean(subject_snr, axis=0)
    subject_snr_std = np.std(subject_snr, axis=0)

    print(subject_snr_mean)
    print(subject_snr_std)




