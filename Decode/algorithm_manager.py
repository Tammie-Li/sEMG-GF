from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import yaml
import numpy as np
from tqdm import tqdm
import torch, os
import torch.nn as nn
import torch.optim as optim


from Decode.sEMGNet import EMGNet
from Decode.SVM import SVM
from Decode.LDA import LDA
from Decode.RF import RF
from Decode.KNN import KNN
from Decode.GaussianNB import NaiveBayes
from Decode.OneDCNN import OneDCNN
from Decode.TwoDCNN import TwoDCNN
from Decode.LSTM import LSTM
from Decode.CNN_LSTM import CNN_LSTM

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class DeepLearningDataLoader(Dataset):
    """
    @ Description: 装载数据
    """
    def __init__(self, x, y):
        super(DeepLearningDataLoader, self).__init__()
        self.x = x
        self.y = y
        
    def __getitem__(self, id):
        x = self.x[id, ...]
        y = self.y[id, ...]
        return x, y
    
    def __len__(self):
        return len(self.x)


class AlgorithmManager:
    def __init__(self, model_name_list, window_size):
        with open('config.yaml', 'r', encoding='utf-8') as f:
            self.configs = yaml.load(f.read(), Loader=yaml.FullLoader)
        
        num_classes = self.configs["exp"]["classes"]
        channel = self.configs["exp"]["channels"]

        drop_out = self.configs["model"]["EMGNet"]["drop_out"]
        time_point = self.configs["model"]["EMGNet"]["time_point"]
        N_t = self.configs["model"]["EMGNet"]["N_t"]
        N_s = self.configs["model"]["EMGNet"]["N_s"]

        self.num_classes = num_classes

        # 字典存储所有可以选择的算法
        self.all_model = {} 
        self.all_model["EMGNet"] = EMGNet(num_classes, drop_out, time_point, channel, N_t, N_s, window_size)
        self.all_model["SVM"] = SVM(num_classes)
        self.all_model["LDA"] = LDA(num_classes)
        self.all_model["RF"] = RF(num_classes)
        self.all_model["KNN"] = KNN(num_classes)
        self.all_model["NaiveBayes"] = NaiveBayes(num_classes)
        self.all_model["OneDCNN"] = OneDCNN(num_classes, window_size)
        self.all_model["TwoDCNN"] = TwoDCNN(num_classes, window_size)
        self.all_model["LSTM"] = LSTM(num_classes, window_size)
        self.all_model["CNN_LSTM"] = CNN_LSTM(num_classes, window_size)
        
        # 所有需要计算的模型列表
        self.model_name_list = model_name_list


    def train_test(self, x_train, y_train, x_test, y_test, epoch, batchsize, save_path_dir):
        for model_name in self.model_name_list:
            print(f"============ Model Name========={model_name}========")
            if self.configs["model"][model_name]['is_model_type_deep_learning'] == True:
                self.deep_learning_train(model_name, x_train, y_train, epoch, batchsize, save_path_dir)
                self.deep_learning_test(model_name, x_test, y_test, save_path_dir)
            else:
                self.all_model[model_name].train_test(model_name, x_train, y_train, x_test, y_test, save_path_dir)


    def deep_learning_train(self, model_name, x_train, y_train, epochs, batchsize, save_path_dir):
        # save_path_dir: 保存的是模型参数
        train_data_loader = self._construct_dataloader(x_train, y_train, batchsize)
        # 定义损失函数和优化方法
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.all_model[model_name].parameters())
        # 构造trainloader
        with tqdm(total=epochs, desc='Epoch', leave=True, ncols=100, unit_scale=True) as pbar:
            for epoch in range(epochs):
                self.all_model[model_name].to(device)
                self.all_model[model_name].train()
                running_loss = 0.0
                correct_num = 0
                batch_size = None
                sum_num = 0
                for index, data in enumerate(train_data_loader):
                    x, y = data
                    batch_size = x.shape[0] if index == 0 else batch_size
                    x = torch.tensor(x).to(torch.float32)
                    y = torch.tensor(y).to(torch.long)  
                    x, y = x.to(device), y.to(device)           
                    y_pred = self.all_model[model_name](x)
                    loss = self.criterion(y_pred, y)
                    _, pred = torch.max(y_pred, 1)
                    correct_num += np.sum(pred.cpu().numpy() == y.cpu().numpy())
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    running_loss += float(loss.item())
                    sum_num += x.shape[0]
                batch_num = sum_num // batch_size
                _loss = running_loss / (batch_num + 1)
                acc = correct_num / sum_num * 100
                pbar.update(1)
                pbar.set_description(f'Epoch[{epoch}/{epochs}]')
                pbar.set_postfix(loss = _loss, acc = acc)
        torch.save(self.all_model[model_name].state_dict(), os.path.join(save_path_dir, f"{model_name}_model_{self.num_classes}.pth")) 
  

    def deep_learning_test(self, model_name, x_test, y_test, save_path_dir):
        # save_path_dir: 保存的是预测结果，或者中间特征
        test_data_loader = self._construct_dataloader(x_test, y_test, x_test.shape[0])

        self.all_model[model_name].load_state_dict(torch.load(os.path.join(save_path_dir, f"{model_name}_model_{self.num_classes}.pth")))
        self.all_model[model_name].to(device)

        running_loss = 0.0
        correct_num = 0
        self.all_model[model_name].eval()
        preds = []
        ys = []
        pred_score = []
        test_num = 0
        for index, data in enumerate(test_data_loader):
            x, y = data
            x = torch.tensor(x).to(torch.float32)
            y = torch.tensor(y).to(torch.long)  
            x, y = x.to(device), y.to(device)
            y_pred = self.all_model[model_name](x)
            _, pred = torch.max(y_pred, 1)

            correct_num += np.sum(pred.cpu().numpy() == y.cpu().numpy())
            preds.extend(pred.cpu().numpy().tolist())
            ys.extend(y.cpu().tolist())
            pred_score.extend(y_pred.cpu().detach().numpy())  
            test_num += x.shape[0]
        acc = correct_num / test_num * 100
        print(f'Test acc: {acc:.2f}%') 

        ys, preds, pred_score = np.array(ys).reshape(len(ys), 1), np.array(preds).reshape(len(preds), 1), np.array(pred_score)

        # 打印输出结果
        # for i in range(len(ys)):    print(ys[i], preds[i])

        result = np.concatenate((ys, preds, pred_score), axis=1)

        np.save(os.path.join(save_path_dir, f"{model_name}_result_{self.num_classes}.npy"), result)

        return acc

    def _construct_dataloader(self, x, y, batchsize):
        data = DeepLearningDataLoader(x, y)
        data_loader = DataLoader(data, batch_size=batchsize, shuffle=True)
        return data_loader