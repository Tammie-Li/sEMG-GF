import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import confusion_matrix, accuracy_score
import os, struct


def plot(confusion_matrix, classes):
    proportion = []
    length = len(confusion_matrix)
    print(length)
    for i in confusion_matrix:
        for j in i:
            temp = j / (np.sum(i))*100
            proportion.append(temp)
    pshow = []
    for i in proportion:
        pt = "%.2f" % (i)
        pshow.append(pt)
    proportion = np.array(proportion).reshape(length, length)  # reshape(列的长度，行的长度)
    pshow = np.array(pshow).reshape(length, length)
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
    }
    rcParams.update(config)
    plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Oranges)  # 按照像素显示出矩阵
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12, rotation=45)
    plt.yticks(tick_marks, classes, fontsize=12)
    thresh = confusion_matrix.max() / 2.
    iters = np.reshape([[[i, j] for j in range(length)] for i in range(length)], (confusion_matrix.size, 2))
    for i, j in iters:
        if (i == j):
            plt.text(j, i, pshow[i, j], va='center', ha='center', fontsize=10, color='white')
        else:
            plt.text(j, i, pshow[i, j], va='center', ha='center', fontsize=10)

    plt.tight_layout()
    plt.show()
    plt.savefig("A.png", dpi=500)


def get_data(data_path):
    with open(data_path, 'rb') as file:
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

    return data


def calculate_acc_with_speed(y_pred, y, s, speed=0):
    idx = np.where(s == speed)[0]
    y_pred, y = y_pred[idx], y[idx]

    acc = accuracy_score(y, y_pred)
    return acc


if __name__ == "__main__":
    y_pred, y, s = [], [], []
    classes = 6

    result0, result4, result6, result8 = [], [], [], []
    for sub_id in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for day_id in [1, 2]:
            data_day = np.load(os.path.join(os.getcwd(), "Result", "SingleDayExp", f"Sub{sub_id:>02d}", f"Day{day_id}", "Window_size_500ms", f"NaiveBayes_best_result_{classes}.npy"))

            speed_path = os.path.join(os.getcwd(), "speed", f"speed_{sub_id:>02d}_{day_id:>02d}.npy")

            speed = np.load(speed_path)

            ys_day, preds_day = data_day[:, 0], data_day[:, 1]
        

            for speed_value in [0, 4, 6 ,8]:
                acc = calculate_acc_with_speed(preds_day, ys_day, speed, speed_value)
                if speed_value == 0:
                    result0.append(acc)
                elif speed_value == 4:
                    result4.append(acc)
                elif speed_value == 6:
                    result6.append(acc)
                elif speed_value == 8:
                    result8.append(acc)
    
    result0, result4, result6, result8 = np.array(result0), np.array(result4), np.array(result6), np.array(result8)

    print(f"speed0: {np.mean(result0)}======={np.std(result0)}")
    print(f"speed4: {np.mean(result4)}======={np.std(result4)}")
    print(f"speed6: {np.mean(result6)}======={np.std(result6)}")
    print(f"speed8: {np.mean(result8)}======={np.std(result8)}")




