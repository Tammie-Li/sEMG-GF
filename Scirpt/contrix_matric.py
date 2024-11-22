import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import confusion_matrix
import os


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


if __name__ == "__main__":
    y_pred, y = [], []
    classes = 6
    for sub_id in range(1, 11):
        data_day1 = np.load(os.path.join(os.getcwd(), "Result", "SingleDayExp", f"Sub{sub_id:>02d}", "Day1", "Window_size_500ms", f"EMGNet_best_result_{classes}.npy"))
        data_day2 = np.load(os.path.join(os.getcwd(), "Result", "SingleDayExp", f"Sub{sub_id:>02d}", "Day2", "Window_size_500ms", f"EMGNet_best_result_{classes}.npy"))

        ys_day1, preds_day1, ys_day2, preds_day2 = data_day1[:, 0], data_day1[:, 1], data_day2[:, 0], data_day2[:, 1]
        
        ys = np.concatenate((ys_day1, ys_day2))
        preds = np.concatenate((preds_day1, preds_day2))

        y.extend(ys)
        y_pred.extend(preds)
    y_pred, y = np.array(y_pred), np.array(y)

    matrix = confusion_matrix(y, y_pred)
    plot(matrix, ["Mode1", "Mode2", "Mode3", "Mode4", "Mode5", "Mode6"])
