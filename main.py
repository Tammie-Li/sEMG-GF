# 表面肌电信号处理工具箱V1.0，包括数据预处理，数据解码和数据可视化三个方面

# 工具箱设计框架
# main.py ------ preprocess_manager.py ---|--slice.py
#            |                            |--filter.py      
#            |                            |--normalize.py
#            |
#            |-- algorithm_manager.py ----|--1DCNN.py
#            |                            |--2DCNN.py
#            |                            |--LSTM.py
#            |                            |--LDA.py
#            |                            |--.......
#            |
#            |-- plot_manager.py ---------|--tsne.py
#            |                            |--confusion_matrix.py
#            |                            |--waveform.py
#            |                            |--.......
#            |-- data_manager.py ---------|--Subject_01------ Day_01 --- raw_data.dat
#            |                            |--Subject_02   |-- Day_02
#            |                            |--Subject_03
#            |                            |--.......

from Data.data_manager import SingleDataManager, CrossDataManager, CrossSubjectDataManager
from Plot.plot_manager import PlotManager
from Preprocess.preprocess_manager import PreProcessManager
from Decode.algorithm_manager import AlgorithmManager
import os, yaml, warnings, openpyxl, shutil

import numpy as np

from sklearn.metrics import accuracy_score


class Evaluate:
    def __init__(self, path_dir, classes):
        self.result_path_dir = path_dir
        self.classes = classes

    def _get_single_exp_cell_name(self, subject_id, day_id, window_size):
        if self.classes == 6:
            if window_size == 0.25:
                if day_id == 1:
                    cell_name = "B" + f"{subject_id+3}"
                elif day_id == 2:
                    cell_name = "C" + f"{subject_id+3}"
            elif window_size == 0.5:
                if day_id == 1:
                    cell_name = "D" + f"{subject_id+3}"
                elif day_id == 2:
                    cell_name = "E" + f"{subject_id+3}"
            elif window_size == 0.75:
                if day_id == 1:
                    cell_name = "F" + f"{subject_id+3}"
                elif day_id == 2:
                    cell_name = "G" + f"{subject_id+3}"
        elif self.classes == 12:
            if window_size == 0.25:
                if day_id == 1:
                    cell_name = "H" + f"{subject_id+3}"
                elif day_id == 2:
                    cell_name = "I" + f"{subject_id+3}"
            elif window_size == 0.5:
                if day_id == 1:
                    cell_name = "J" + f"{subject_id+3}"
                elif day_id == 2:
                    cell_name = "K" + f"{subject_id+3}"
            elif window_size == 0.75:
                if day_id == 1:
                    cell_name = "L" + f"{subject_id+3}"
                elif day_id == 2:
                    cell_name = "M" + f"{subject_id+3}"
        return cell_name
    
    def _get_cross_exp_cell_name(self, subject_id, window_size):
        if self.classes == 6:
            if window_size == 0.25:
                cell_name = "B" + f"{subject_id+2}"
            elif window_size == 0.5:
                cell_name = "C" + f"{subject_id+2}"
            elif window_size == 0.75:
                cell_name = "D" + f"{subject_id+2}"
        elif self.classes == 12:
            if window_size == 0.25:
                cell_name = "E" + f"{subject_id+2}"
            elif window_size == 0.5:
                cell_name = "F" + f"{subject_id+2}"
            elif window_size == 0.75:
                cell_name = "G" + f"{subject_id+2}"
        return cell_name

    def single_day_exp_evaluate(self, algorithm_list, subject_id, day_id, window_size, speed):
        for alg_name in algorithm_list:
            model_path = os.path.join(self.result_path_dir, "SingleDayExp", f"Sub{subject_id:>02d}", f"Day{day_id}", f"Window_size_{int(window_size*1000)}ms", f"speed_{speed}", f"{alg_name}_model_{self.classes}.pth")
            result_path = os.path.join(self.result_path_dir, "SingleDayExp", f"Sub{subject_id:>02d}", f"Day{day_id}", f"Window_size_{int(window_size*1000)}ms", f"speed_{speed}", f"{alg_name}_result_{self.classes}.npy")
            
            cell_name = self._get_single_exp_cell_name(subject_id, day_id, window_size)
            # excel 统计表路径
            statics_result_path = os.path.join(self.result_path_dir, "SingleDayExp", "speed.xlsx")

            # 从excel中读取当前最优结果
            statics_result_excel = openpyxl.load_workbook(statics_result_path)
            statics_result_sheet = statics_result_excel[f"speed{speed}"]

            curr_best_result = statics_result_sheet[self._get_single_exp_cell_name(subject_id, day_id, window_size)].value

            
            # 计算此次计算的结果
            result = np.load(result_path)
            labels = result[:, 0]
            pred_values = result[:, 1]
            tmp_result = accuracy_score(labels, pred_values)
            print(tmp_result)

            print(self._get_single_exp_cell_name(subject_id, day_id, window_size))
            # 比较结果，如为最优，更新结果和模型
            if tmp_result > curr_best_result:
                statics_result_sheet[self._get_single_exp_cell_name(subject_id, day_id, window_size)] = tmp_result
                # 保存文件
                statics_result_excel.save(os.path.join(self.result_path_dir, "SingleDayExp", "speed.xlsx"))
                best_model_path = os.path.join(self.result_path_dir, "SingleDayExp", f"Sub{subject_id:>02d}", f"Day{day_id}", f"Window_size_{int(window_size*1000)}ms", f"speed_{speed}", f"{alg_name}_best_model_{self.classes}.pth")
                best_result_path = os.path.join(self.result_path_dir, "SingleDayExp", f"Sub{subject_id:>02d}", f"Day{day_id}", f"Window_size_{int(window_size*1000)}ms", f"speed_{speed}", f"{alg_name}_best_result_{self.classes}.npy")
                shutil.copy(model_path, best_model_path)
                shutil.copy(result_path, best_result_path)


    def cross_day_exp_evaluate(self, algorithm_list, subject_id, window_size):
        for alg_name in algorithm_list:
            model_path = os.path.join(self.result_path_dir, "CrossDayExp", f"Sub{subject_id:>02d}", f"Window_size_{int(window_size*1000)}ms", f"{alg_name}_model_{self.classes}.pth")
            result_path = os.path.join(self.result_path_dir, "CrossDayExp", f"Sub{subject_id:>02d}", f"Window_size_{int(window_size*1000)}ms", f"{alg_name}_result_{self.classes}.npy")
            
            cell_name = self._get_cross_exp_cell_name(subject_id, window_size)
            # excel 统计表路径
            statics_result_path = os.path.join(self.result_path_dir, "CrossDayExp", "result.xlsx")

            # 从excel中读取当前最优结果
            statics_result_excel = openpyxl.load_workbook(statics_result_path)
            statics_result_sheet = statics_result_excel[f"{alg_name}"]

            curr_best_result = statics_result_sheet[self._get_cross_exp_cell_name(subject_id, window_size)].value

            
            # 计算此次计算的结果
            result = np.load(result_path)
            labels = result[:, 0]
            pred_values = result[:, 1]
            tmp_result = accuracy_score(labels, pred_values)

            # 比较结果，如为最优，更新结果和模型
            if tmp_result > curr_best_result:
                statics_result_sheet[self._get_cross_exp_cell_name(subject_id, window_size)] = tmp_result
                # 保存文件
                statics_result_excel.save(os.path.join(self.result_path_dir, "CrossDayExp", "result.xlsx"))
                best_model_path = os.path.join(self.result_path_dir, "CrossDayExp", f"Sub{subject_id:>02d}", f"Window_size_{int(window_size*1000)}ms", f"{alg_name}_best_model_{self.classes}.pth")
                best_result_path = os.path.join(self.result_path_dir, "CrossDayExp", f"Sub{subject_id:>02d}", f"Window_size_{int(window_size*1000)}ms", f"{alg_name}_best_result_{self.classes}.npy")
                shutil.copy(model_path, best_model_path)
                shutil.copy(result_path, best_result_path)

    def cross_subject_exp_evaluate(self, algorithm_list, subject_id, window_size):
        for alg_name in algorithm_list:
            model_path = os.path.join(self.result_path_dir, "CrossSubjectExp", f"Sub{subject_id:>02d}", f"Window_size_{int(window_size*1000)}ms", f"{alg_name}_model_{self.classes}.pth")
            result_path = os.path.join(self.result_path_dir, "CrossSubjectExp", f"Sub{subject_id:>02d}", f"Window_size_{int(window_size*1000)}ms", f"{alg_name}_result_{self.classes}.npy")
            
            cell_name = self._get_cross_exp_cell_name(subject_id, window_size)
            # excel 统计表路径
            statics_result_path = os.path.join(self.result_path_dir, "CrossSubjectExp", "result.xlsx")

            # 从excel中读取当前最优结果
            statics_result_excel = openpyxl.load_workbook(statics_result_path)
            statics_result_sheet = statics_result_excel[f"{alg_name}"]

            curr_best_result = statics_result_sheet[self._get_cross_exp_cell_name(subject_id, window_size)].value

            
            # 计算此次计算的结果
            result = np.load(result_path)
            labels = result[:, 0]
            pred_values = result[:, 1]
            tmp_result = accuracy_score(labels, pred_values)

            # 比较结果，如为最优，更新结果和模型
            if tmp_result > curr_best_result:
                statics_result_sheet[self._get_cross_exp_cell_name(subject_id, window_size)] = tmp_result
                # 保存文件
                statics_result_excel.save(os.path.join(self.result_path_dir, "CrossSubjectExp", "result.xlsx"))
                best_model_path = os.path.join(self.result_path_dir, "CrossSubjectExp", f"Sub{subject_id:>02d}", f"Window_size_{int(window_size*1000)}ms", f"{alg_name}_best_model_{self.classes}.pth")
                best_result_path = os.path.join(self.result_path_dir, "CrossSubjectExp", f"Sub{subject_id:>02d}", f"Window_size_{int(window_size*1000)}ms", f"{alg_name}_best_result_{self.classes}.npy")
                shutil.copy(model_path, best_model_path)
                shutil.copy(result_path, best_result_path)




warnings.filterwarnings("ignore")

class SingleDayExperiment:
    def __init__(self, subject_id, day_id, window_size, algoritm_list, speed):
        with open('config.yaml', 'r', encoding='utf-8') as f:
            self.configs = yaml.load(f.read(), Loader=yaml.FullLoader)

        self.preprocess_manager = PreProcessManager()
        self.data_manager = SingleDataManager(subject_id, day_id)
        # self.algorithm_manager = AlgorithmManager(alg_name_list=['rLDA, SVM, 2DCNN, 1DCNN'])
        # self.algorithm_manager = AlgorithmManager(['EMGNet', 'F_SVM'])
        self.algorithm_manager = AlgorithmManager(algoritm_list, window_size)

        self.subject_id, self.day_id, self.window_size, self.speed = subject_id, day_id, window_size, speed

    def make(self):
        x, y, speed = self.data_manager.get_gesture_data(classes=self.configs["exp"]["classes"], speed=self.speed)

        x, y, s_speed = process_manager.data_preprocess_all(x, y, window_size=self.window_size, window_mov_t=0.25, speed=speed)

        # 划分训练集，测试集
        # 划分标准根据前8个block的数据为训练集，后4个block的数据为测试集
        x_train, y_train, x_test, y_test, speed = process_manager.divide_train_test_data(x, y, speed=s_speed, classes=self.configs["exp"]["classes"])
        
        # print(speed.shape)
        # np.save(f"speed_{subject_id:>02d}_{day_id:>02d}_12", speed)
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

        save_path_dir = os.path.join(os.getcwd(), "Result", "SingleDayExp", f"Sub{self.subject_id:>02d}", f"Day{self.day_id}", f"Window_size_{int(self.window_size*1000)}ms", f"speed_{self.speed}")

        if not os.path.exists(save_path_dir):   os.makedirs(save_path_dir)

        # 调用算法，获取结果
        self.algorithm_manager.train_test(x_train, y_train, x_test, y_test, self.configs["exp"]["epoch"], self.configs["exp"]["batchsize"], save_path_dir)


class CrossDayExperiment:
    def __init__(self, subject_id, window_size, algoritm_list):
        with open('config.yaml', 'r', encoding='utf-8') as f:
            self.configs = yaml.load(f.read(), Loader=yaml.FullLoader)

        self.preprocess_manager = PreProcessManager()

        # 需要修改
        self.data_manager = CrossDataManager(subject_id)
        self.algorithm_manager = AlgorithmManager(algoritm_list, window_size)

        self.subject_id, self.window_size = subject_id, window_size

    def make(self):
        x1, y1, x2, y2 = self.data_manager.get_gesture_data()
        x1, y1 = process_manager.data_preprocess_all(x1, y1, window_size=self.window_size, window_mov_t=0.25)
        x2, y2 = process_manager.data_preprocess_all(x2, y2, window_size=self.window_size, window_mov_t=0.25)


        # 划分训练集，测试集
        # 划分标准根据前8个block的数据为训练集，后4个block的数据为测试集
        # 需要修改
        x_train, y_train, x_test, y_test = x1, y1, x2, y2
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

        save_path_dir = os.path.join(os.getcwd(), "Result", "CrossDayExp", f"Sub{self.subject_id:>02d}", f"Window_size_{int(self.window_size*1000)}ms")

        if not os.path.exists(save_path_dir):   os.makedirs(save_path_dir)

        # 调用算法，获取结果
        self.algorithm_manager.train_test(x_train, y_train, x_test, y_test, self.configs["exp"]["epoch"], self.configs["exp"]["batchsize"], save_path_dir)


class CrossSubjectExperiment:
    def __init__(self, subject_id, window_size, algoritm_list):
        with open('config.yaml', 'r', encoding='utf-8') as f:
            self.configs = yaml.load(f.read(), Loader=yaml.FullLoader)

        self.preprocess_manager = PreProcessManager()

        # 需要修改
        self.data_manager = CrossSubjectDataManager(subject_id)
        self.algorithm_manager = AlgorithmManager(algoritm_list, window_size)

        self.subject_id, self.window_size = subject_id, window_size

    def make(self):
        x1, y1, x2, y2 = self.data_manager.get_gesture_data(classes=self.configs["exp"]["classes"])
        x1, y1 = process_manager.data_preprocess_all(x1, y1, window_size=self.window_size, window_mov_t=0.25)
        x2, y2 = process_manager.data_preprocess_all(x2, y2, window_size=self.window_size, window_mov_t=0.25)

        randtmp = np.random.permutation(x1.shape[0])

        x_train, y_train = x1[randtmp[: 10000]], y1[randtmp[: 10000]]
        x_test, y_test = x2, y2
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

        save_path_dir = os.path.join(os.getcwd(), "Result", "CrossSubjectExp", f"Sub{self.subject_id:>02d}", f"Window_size_{int(self.window_size*1000)}ms")

        if not os.path.exists(save_path_dir):   os.makedirs(save_path_dir)

        # 调用算法，获取结果
        self.algorithm_manager.train_test(x_train, y_train, x_test, y_test, self.configs["exp"]["epoch"], self.configs["exp"]["batchsize"], save_path_dir)


if __name__ =="__main__":
    # single_day: 单天单试次实验；cross-day: 跨天实验；cross-subject: 跨被试; 
    # 讨论实验1：speed: 不同速度下的结果，在单single_day实验中
    experiment_type = ["single_day", "cross-day", "cross-subject", "speed"]
    # algoritm_list = ["EMGNet", "SVM", "RF", "LDA", "KNN", "NaiveBayes"]
    algoritm_list = ["EMGNet"]


    process_manager = PreProcessManager()

    evaluater = Evaluate(os.path.join(os.getcwd(), "Result"), classes=6)


    # # single-day实验
    # for subject_id in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    #     for day_id in [1, 2]:
    #         for window_size_l in [0.5]:
    #             for speed in [0, 4, 6, 8]:
    #                 print(f"======SingleDayExperiments=======Class: 6=======Subject_ID: {subject_id:>02d}======Day_ID: {day_id:>02d}=========WindowSize: {int(window_size_l*1000)}ms=====")
    #                 experiment = SingleDayExperiment(subject_id=subject_id, day_id=day_id, window_size=window_size_l, algoritm_list=algoritm_list)
    #                 experiment.make()
    #                 evaluater.single_day_exp_evaluate(algoritm_list, subject_id=subject_id, day_id=day_id, window_size=window_size_l)

    # single-day实验
    for subject_id in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for day_id in [1, 2]:
            for window_size_l in [0.5]:
                for speed in [4, 6, 10]:
                    print(f"======SingleDayExperiments=======Class: 6=======Subject_ID: {subject_id:>02d}======Day_ID: {day_id:>02d}=========WindowSize: {int(window_size_l*1000)}ms=====")
                    experiment = SingleDayExperiment(subject_id=subject_id, day_id=day_id, window_size=window_size_l, algoritm_list=algoritm_list, speed=speed)
                    experiment.make()
                    evaluater.single_day_exp_evaluate(algoritm_list, subject_id=subject_id, day_id=day_id, window_size=window_size_l, speed=speed)

    # # cross-day实验
    # for subject_id in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    #     for window_size_l in [0.25, 0.5, 0.75]:
    #         print(f"======CrossDayExperiments=======Class: 12=======Subject_ID: {subject_id:>02d}=========WindowSize: {int(window_size_l*1000)}ms=====")
    #         experiment = CrossDayExperiment(subject_id=subject_id, window_size=window_size_l, algoritm_list=algoritm_list)
    #         experiment.make()
    #         # evaluater.single_day_exp_evaluate(algoritm_list, subject_id=subject_id, day_id=day_id, window_size=window_size_l, classes=12)
    #         evaluater.cross_day_exp_evaluate(algoritm_list, subject_id=subject_id, window_size=window_size_l, classes=12)


    # # cross-subject实验---留一法
    # for subject_id in [4]:
    #     for window_size_l in [0.75]:
    #         print(f"======CrossSubjectExperiments=======Class: 12=======Subject_ID: {subject_id:>02d}=========WindowSize: {int(window_size_l*1000)}ms=====")
    #         experiment = CrossSubjectExperiment(subject_id=subject_id, window_size=window_size_l, algoritm_list=algoritm_list)
    #         experiment.make()
    #         # evaluater.single_day_exp_evaluate(algoritm_list, subject_id=subject_id, day_id=day_id, window_size=window_size_l, classes=12)
    #         evaluater.cross_subject_exp_evaluate(algoritm_list, subject_id=subject_id, window_size=window_size_l)








