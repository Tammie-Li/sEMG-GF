from sklearn import svm
import numpy as np
import os
import joblib
import pywt

from scipy.fft import fft
from scipy.signal import welch

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import accuracy_score

class LDA:
    def __init__(self, classes):
        #instantiate SVM classifier 
        self.grid_matric = [[2], [0.0000001]]
        self.classes = classes

    def get_temp_freq_fearture(self, data, fs=500):
        # 计算频域特征，两个特征，一个是傅里叶变换，一个是功率谱密度
        N, C, T = data.shape

        features = np.zeros((N, C, 4))  # 假设提取2个特征

        for n in range(N):
            for c in range(C):
                signal = data[n, c, :]

                # 时域特征
                mean_value = np.mean(signal)
                rms_value = np.sqrt(np.mean(signal**2))
                
                # 计算傅里叶变换
                spectrum = np.abs(fft(signal))[:T//2+1]
                freq = np.fft.rfftfreq(T, d=1/fs)
                mean_frequency = np.sum(freq * spectrum) / np.sum(spectrum)
                
                # 功率谱密度
                f, Pxx = welch(signal, fs, nperseg=256)
                psd = Pxx.mean() 

                features[n, c, 0] = mean_value
                features[n, c, 1] = rms_value
                features[n, c, 2] = mean_frequency
                features[n, c, 3] = psd
                
        return features

    def train_test(self, model_name, x_train, y_train, x_test, y_test, save_path_dir):

        x_train, x_test = self.get_temp_freq_fearture(x_train), self.get_temp_freq_fearture(x_test)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2]))

        curr_best_acc = 0
        # 网格搜索最优参数
        for c_param in self.grid_matric[0]:
            for gamma_param in self.grid_matric[1]:
                lda_classifier = LinearDiscriminantAnalysis(solver = "svd")    

                # 训练SVM分类器
                lda_classifier.fit(x_train, y_train)

                # 使用SVM分类器预测结果
                tpred = lda_classifier.predict(x_test) 

                # 初步计算分类结果
                tmp_acc = accuracy_score(y_test, tpred)

                if tmp_acc > curr_best_acc:
                    curr_best_acc = tmp_acc

                    result = np.column_stack((y_test, tpred))
                    joblib.dump(lda_classifier, os.path.join(save_path_dir, f"{model_name}_model_{self.classes}.pth"))
                    np.save(os.path.join(save_path_dir, f"{model_name}_result_{self.classes}.npy"), result) 

                    print("当前最优参数：", "C为：", c_param, "gamma为：", gamma_param, "最佳结果为：", curr_best_acc)


