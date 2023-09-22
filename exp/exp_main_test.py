from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, Reformer, Autoformer_extend
# from models.etsformer import ETSformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim

# torch.cuda.current_device()
# torch.cuda._initialized = True

import os
import time
import warnings

# 后加的import内容
import copy
import math
from data_provider.data_factory import data_provider_at_test_time


warnings.filterwarnings('ignore')


class Exp_Main_Test(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main_Test, self).__init__(args)

        # 这个可以作为超参数来设置
        self.test_train_num = self.args.test_train_num

        # 判断哪些channels是有周期性的
        data_path = self.args.data_path
        if "ETTh1" in data_path: selected_channels = [1,3]  # [1,3, 2,4,5,6]
        # if "ETTh1" in data_path: selected_channels = [7]
        # elif "ETTh2" in data_path: selected_channels = [1,3,7]
        elif "ETTh2" in data_path: selected_channels = [7]
        elif "ETTm1" in data_path: selected_channels = [1,3, 2,4,5]
        elif "ETTm2" in data_path: selected_channels = [1,7, 3]
        elif "illness" in data_path: selected_channels = [1,2, 3,4,5]
        # elif "weather" in data_path: selected_channels = [17,18,19, 5,8,6,13,20]  # [2,3,11]
        elif "weather" in data_path: selected_channels = [17,18,19]
        # elif "weather" in data_path: selected_channels = [5,8,6,13,20]
        # elif "weather" in data_path: selected_channels = [1,4,7,9,10]
        else: selected_channels = list(range(1, self.args.c_out))
        for channel in range(len(selected_channels)):
            selected_channels[channel] -= 1  # 注意这里要读每个item变成item-1，而非item
        
        self.selected_channels = selected_channels

        # 判断各个数据集的周期是多久
        if "ETTh1" in data_path: period = 24
        elif "ETTh2" in data_path: period = 24
        elif "ETTm1" in data_path: period = 96
        elif "ETTm2" in data_path: period = 96
        elif "electricity" in data_path: period = 24
        elif "traffic" in data_path: period = 24
        elif "illness" in data_path: period = 52.142857
        elif "weather" in data_path: period = 144
        elif "Exchange" in data_path: period = 1
        else: period = 1
        self.period = period


    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'Reformer': Reformer,
            'Autoformer_extend': Autoformer_extend,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader


    # 别忘了这里要加一个用data_provider_at_test_time来提供的data
    def _get_data_at_test_time(self, flag):
        data_set, data_loader = data_provider_at_test_time(self.args, flag)
        return data_set, data_loader


    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
            
        total_loss = np.average(total_loss)
        self.model.train()

        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        
        # 这里为了防止异常，需要做一些修改，要在torch.load后加上map_location='cuda:0'
        # self.model.load_state_dict(torch.load(best_model_path))
        self.model.load_state_dict(torch.load(best_model_path, map_location='cuda:0'))

        return self.model

    def test(self, setting, test=0, flag='test'):
        # test_data, test_loader = self._get_data(flag='test')
        test_data, test_loader = self._get_data(flag=flag)
        if test:
            print('loading model from checkpoint !!!')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        preds = []
        trues = []

        criterion = nn.MSELoss()  # 使用MSELoss
        loss_list = []

        test_time_start = time.time()

        # result save
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                # selected_channels = self.selected_channels
                if self.args.adapt_part_channels:
                    outputs = outputs[:, :, self.selected_channels]
                    batch_y = batch_y[:, :, self.selected_channels]

                # 计算MSE loss
                loss = criterion(outputs, batch_y)
                loss_list.append(loss.item())
                # print(loss)
                
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)

                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        
        # file_name = f"batchsize_32_{setting}" if flag == 'test' else f"batchsize_1_{setting}"
        # # 将adaptation前的loss、adaptation中逐样本做adapt后的loss、以及adaptation之后的loss做统计
        # with open(f"./loss_before_and_after_adapt/{file_name}.txt", "w") as f:
        #     for loss in loss_list:
        #         f.write(f"{loss}\n")


        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        print(f"Test - cost time: {time.time() - test_time_start}s")

        # return
        return loss_list


    def calc_acf(self, setting, test=0, lag=1):
        def acf_func(ts, k):
            """Args:
            ts: time series
            k: lag
            Returns: acf value
            """
            # ACF vs. PACF：https://blog.csdn.net/qq_41103204/article/details/105810742
            # ACF计算参考：https://zhuanlan.zhihu.com/p/430503143
            # ACF其实就是自协方差和方差的比值
            m = np.mean(ts)
            s1 = 0
            for i in range(k, len(ts)):
                s1 = s1 + ((ts[i] - m) * (ts[i - k] - m))

            s2 = 0
            for i in range(0, len(ts)):
                s2 = s2 + ((ts[i] - m)**2)

            return float(s1 / s2)

        # 注意跑下面实验时batch_size一定要取成1，否则就一次就取出多个样本了
        # 也即"--batch_size 1"
        # 另外需要将lag加到pred_len上，以防止取出的数据不正确了

        # 我们在这里把pred_len给加上lag，以方便计算lag后的acf的结果
        self.args.pred_len += lag  # 非常非常重要！！！

        assert self.args.batch_size == 1
        
        for flag in ["train_without_shuffle", "val_without_shuffle", "test"]:

            cur_data, cur_loader = self._get_data(flag=flag)

            # 装有目前取出的数据
            datas = []
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(cur_loader):
                # 因为batch_y中包含label_len & pred_len，所以这里只取出最后一部分
                batch_y = batch_y[:, -self.args.pred_len:, :]

                # 将batch维去掉
                batch_x, batch_y = batch_x[0], batch_y[0]

                cur_data = torch.cat((batch_x, batch_y), dim=0)
                cur_data = np.array(cur_data)
                # 存入datas数组
                datas.append(cur_data)

                if i % 100 == 0:
                    print(f"data {i} have been stored")
            
            # 从datas数组中取出相应数据
            cur_acf_list = []
            for i in range(len(datas)):
                cur_data = datas[i]
                cur_acf_per_channel = []
                # 对各个channel进行遍历
                for channel in range(cur_data.shape[1]):
                    tmp = acf_func(cur_data[:, channel], lag)
                    cur_acf_per_channel.append(tmp)
                # cur_acf_per_channel包含各个channel的信息
                cur_acf_per_channel = np.array(cur_acf_per_channel)
                # print(cur_acf_per_channel.shape)
                cur_acf_list.append(cur_acf_per_channel)

                if i % 100 == 0:
                    print(f"data {i} have been calculated")
            
            cur_acf_list = np.array(cur_acf_list)
            print(cur_acf_list.shape)

            # result save
            folder_path = './ACF_results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            if "train" in flag: file_flag = "train"
            elif "val" in flag: file_flag = "val"
            elif "test" in flag: file_flag = "test"
            file_name = f"lag{lag:02d}_{file_flag}.npy"

            # 将实验结果的cur_acf_list存入npy文件中
            np.save(folder_path + file_name, cur_acf_list)

        self.args.pred_len -= lag  # 非常非常重要！！！

        return


    def calc_KLdiv(self, setting, test=0, lag=1):
        print('loading model from checkpoint !!!')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))
        
        assert self.args.batch_size == 1

        train_npy, val_npy, test_npy = [], [], []
        for flag in ["train_without_shuffle", "val_without_shuffle", "test"]:
            cur_data, cur_loader = self._get_data(flag=flag)

            mid_embedding_list = []

            test_time_start = time.time()

            self.model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(cur_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    pred, true, mid_embedding = self._process_one_batch_with_model(self.model, cur_data,
                        batch_x, batch_y, 
                        batch_x_mark, batch_y_mark,
                        return_mid_embedding=True)
                    
                    mid_embedding = mid_embedding.detach().cpu().numpy()
                    # 去掉最前面的batch_size=1的维度
                    mid_embedding = mid_embedding[0]

                    mid_embedding_list.append(mid_embedding)

                    if i % 100 == 0:
                        print(f"data {i} have been calculated, cost time: {time.time() - test_time_start}s")
            

            mid_embedding_list = np.array(mid_embedding_list)
            print(mid_embedding_list.shape)

            # result save
            folder_path = './mid_embedding_results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            if "train" in flag: file_flag = "train"
            elif "val" in flag: file_flag = "val"
            elif "test" in flag: file_flag = "test"
            file_name = f"pl{self.args.pred_len}_{file_flag}.npy"

            if "train" in flag: train_npy = copy.deepcopy(mid_embedding_list)
            elif "val" in flag: val_npy = copy.deepcopy(mid_embedding_list)
            elif "test" in flag: test_npy = copy.deepcopy(mid_embedding_list)

            # 将实验结果的cur_acf_list存入npy文件中
            np.save(folder_path + file_name, mid_embedding_list)
        
        train_res, val_res, test_res = [], [], []
        def deal_data(cur_npy):
            print(cur_npy.shape)

            # total_length = cur_npy.shape[0] + cur_npy.shape[1] - 1
            result_list = []
            for i in range(cur_npy.shape[0] + cur_npy.shape[1] - 1):
                # 先获取当前需要累加的个数
                if i < cur_npy.shape[1]:
                    total_num = i + 1
                elif i >= cur_npy.shape[1] and i < cur_npy.shape[0]:
                    total_num = cur_npy.shape[1]
                else:
                    total_num = cur_npy.shape[0] + cur_npy.shape[1] - 1 - i
                
                result = 0
                if i < cur_npy.shape[0]:
                    for j in range(0, total_num):
                        result += cur_npy[i-j][j]
                else:
                    for j in range(i+1 - cur_npy.shape[0], cur_npy.shape[1]):
                        result += cur_npy[i-j][j]
                
                result /= total_num
                # print(results.shape)
                result_list.append(result)
            
            # print(len(result_list))
            result_list = np.array(result_list)
            print(result_list.shape)
            
            return result_list
        
        # 获得结果
        train_res = deal_data(train_npy)
        val_res = deal_data(val_npy)
        test_res = deal_data(test_npy)

        # 1.最简单做法-求和
        train_res, val_res, test_res = train_res.sum(axis=1), val_res.sum(axis=1), test_res.sum(axis=1)
        print(train_res.shape, val_res.shape, test_res.shape)

        max_value = max(max(train_res), max(val_res), max(test_res))
        min_value = min(min(train_res), min(val_res), min(test_res))
        print(max(train_res), max(val_res), max(test_res), max_value)
        print(min(train_res), min(val_res), min(test_res), min_value)

        step_num = 21
        interval = (max_value - min_value) / step_num
        x = np.arange(min_value, max_value+interval, interval)
        print(x)

        # 统计各个区间内的数据值信息
        train_stat = np.histogram(a=train_res, bins=x)[0]
        val_stat = np.histogram(a=val_res, bins=x)[0]
        test_stat = np.histogram(a=test_res, bins=x)[0]
        print("train_statistics_num: " + str(train_stat))
        print("val_statistics_num: " + str(val_stat))
        print("test_statistics_num: " + str(test_stat))

        print(sum(train_stat), len(train_res))
        # assert sum(train_stat) == len(train_res)

        train_stat = train_stat/len(train_res)
        val_stat = val_stat / len(val_res)
        test_stat = test_stat / len(test_res)
        EPS = 1e-6
        train_stat += EPS
        val_stat += EPS
        test_stat += EPS
        print("train_stat: " + str(train_stat))
        print("val_stat: " + str(val_stat))
        print("test_stat: " + str(test_stat))

        # KL散度参考：https://zhuanlan.zhihu.com/p/100676922
        # 代码实现可参考：https://zhuanlan.zhihu.com/p/143105854
        from scipy.stats import entropy, ks_2samp, wasserstein_distance
        def KL_divergence(p, q):
            return entropy(p, q)

        def KL_divergence2(p, q):
            KL = 0.0
            px, qx = p / np.sum(p), q / np.sum(q)
            for i in range(len(px)):
                KL += px[i] * np.log(px[i] / qx[i])
            return KL

        # 交叉熵
        def cross_entropy(p, q):
            p = np.float_(p)
            q = np.float_(q)
            return -np.sum([p[i] * np.log2(q[i]) for i in range(len(p))])

        # JS散度
        def JS_divergence(p, q):
            M = (p + q) / 2
            return 0.5 * entropy(p, M) + 0.5 * entropy(q, M)

        # KS检测方法
        def KS_test(p, q):
            res = ks_2samp(p, q).pvalue
            return res

        # Wasserstein距离
        def Wasserstein_distance(p, q):
            dist = wasserstein_distance(p, q)
            return dist

        print("--------------------------------")
        print("KL_divergence between train and val: ", KL_divergence(train_stat, val_stat))
        print("KL_divergence between train and test: ", KL_divergence(train_stat, test_stat))
        print("KL_divergence between val and test: ", KL_divergence(val_stat, test_stat))
        print("--------------------------------")
        print("KL_divergence2 between train and val: ", KL_divergence2(train_stat, val_stat))
        print("KL_divergence2 between train and test: ", KL_divergence2(train_stat, test_stat))
        print("KL_divergence2 between val and test: ", KL_divergence2(val_stat, test_stat))
        print("--------------------------------")
        print("cross_entropy between train and val: ", cross_entropy(train_stat, val_stat))
        print("cross_entropy between train and test: ", cross_entropy(train_stat, test_stat))
        print("cross_entropy between val and test: ", cross_entropy(val_stat, test_stat))
        print("--------------------------------")
        print("JS_divergence between train and val: ", JS_divergence(train_stat, val_stat))
        print("JS_divergence between train and test: ", JS_divergence(train_stat, test_stat))
        print("JS_divergence between val and test: ", JS_divergence(val_stat, test_stat))
        print("--------------------------------")
        print("KS_test between train and val: ", KS_test(train_stat, val_stat))
        print("KS_test between train and test: ", KS_test(train_stat, test_stat))
        print("KS_test between val and test: ", KS_test(val_stat, test_stat))
        print("--------------------------------")
        print("Wasserstein_distance between train and val: ", Wasserstein_distance(train_stat, val_stat))
        print("Wasserstein_distance between train and test: ", Wasserstein_distance(train_stat, test_stat))
        print("Wasserstein_distance between val and test: ", Wasserstein_distance(val_stat, test_stat))


    # def get_normalized_data(self, setting, test=0):
    #     assert self.args.batch_size == 1
    #     for flag in ["train_without_shuffle", "val_without_shuffle", "val"]:
    #         cur_data, cur_loader = self._get_data(flag=flag)
            
    #         all_data = []
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(cur_loader):

    #             cur_data_x = []
    #             cur_data_y = []

    #             # 先加入test_x和test_y
    #             # test_x = batch_x[:, -self.args.seq_len:, :].detach().cpu().numpy()
    #             # test_y = batch_y.detach().cpu().numpy()
    #             test_x = batch_x[0, -self.args.seq_len:, :].detach().cpu().numpy()
    #             test_y = batch_y[0].detach().cpu().numpy()
    #             cur_data_x.append(test_x)
    #             cur_data_y.append(test_y)
    #             # print(test_x.shape, test_y.shape)

    #         all_data_x = np.array(all_data_x)
    #         all_data_y = np.array(all_data_y)
    #         print(all_data_x.shape)
    #         print(all_data_y.shape)
            

    #         lookback_data_dir = "./lookback_data" + '/' + setting
    #         if not os.path.exists(lookback_data_dir):
    #             os.makedirs(lookback_data_dir)
            
    #         lookback_data_npz = f"{lookback_data_dir}/data_{flag}_ttn{self.test_train_num}.npz"

    #         # 将first_grad和all_grads保存为npz文件
    #         np.savez(lookback_data_npz, all_data_x, all_data_y)

            
    #         # 再从npz中获取回来
    #         npzfile = np.load(lookback_data_npz)
    #         all_data_x = npzfile["arr_0"]
    #         all_data_y = npzfile["arr_1"]

    #         test_x, test_y = all_data_x[:, 0:1, :], all_data_y[:, 0:1, :]
    #         lookback_x, lookback_y = all_data_x[:, 1:, :], all_data_y[:, 1:, :]

    #         print(test_x.shape)
    #         print(test_y.shape)
    #         print(lookback_x.shape)
    #         print(lookback_y.shape)
    
    
    def get_data_error(self, setting, test=0):
        print('loading model from checkpoint !!!')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))
        
        assert self.args.batch_size == 1

        for flag in ["train_without_shuffle", "val_without_shuffle", "test"]:
            cur_data, cur_loader = self._get_data(flag=flag)

            test_time_start = time.time()
            
            results = []
            residuals = []

            self.model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(cur_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    pred, true = self._process_one_batch_with_model(self.model, cur_data,
                        batch_x, batch_y, 
                        batch_x_mark, batch_y_mark)


                    pred = pred.detach().cpu().numpy()
                    pred = pred.reshape(pred.shape[1], pred.shape[2])
                    true = true.detach().cpu().numpy()
                    true = true.reshape(true.shape[1], true.shape[2])
                    
                    residual = pred - true
                    residuals.append(residual)
                    
                    mae, mse, rmse, mape, mspe = metric(pred, true)
                    # print('mse:{}, mae:{}'.format(mse, mae))
                    
                    error = pred - true
                    # print(error.shape)
                    err_mean = np.mean(error)
                    err_var = np.var(error)
                    err_abs_mean = np.mean(np.abs(error))
                    err_abs_var = np.var(np.abs(error))
                    pos_num, neg_num = 0, 0
                    for ei in range(error.shape[0]):
                        for ej in range(error.shape[1]):
                            if error[ei][ej] >= 0: pos_num += 1
                            else: neg_num += 1
                    assert pos_num + neg_num == error.shape[0] * error.shape[1]
                    
                    tmp_list = [mae, mse, rmse, mape, mspe, err_mean, err_var, err_abs_mean, err_abs_var, pos_num, neg_num]
                    results.append(tmp_list)
                    
                    if i % 100 == 0:
                        print(f"data {i} have been calculated, cost time: {time.time() - test_time_start}s")
                        print('mse:{}, mae:{}'.format(mse, mae))
                        
            

            # result save
            folder_path = './error_results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            if "train" in flag: file_flag = "train"
            elif "val" in flag: file_flag = "val"
            elif "test" in flag: file_flag = "test"
            
            file_name = folder_path + f"pl{self.args.pred_len}_{file_flag}.txt"
            residual_file_name = folder_path + f"residuals_pl{self.args.pred_len}_{file_flag}.npy"
            
            with open(file_name, "w") as f:
                for result in results:
                    for idx in range(len(result)-1):
                        item = result[idx]
                        f.write(f"{item}, ")
                    f.write(f"{result[-1]}")
                    f.write("\n")
            
            residuals = np.array(residuals)
            np.save(residual_file_name, residuals)
        
        return
            
        

    def get_lookback_data(self, setting, test=0):
        for flag in ["test", "val"]:
            test_data, test_loader = self._get_data_at_test_time(flag=flag)
            
            all_data_x = []
            all_data_y = []
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):

                cur_data_x = []
                cur_data_y = []

                # 先加入test_x和test_y
                # test_x = batch_x[:, -self.args.seq_len:, :].detach().cpu().numpy()
                # test_y = batch_y.detach().cpu().numpy()
                test_x = batch_x[0, -self.args.seq_len:, :].detach().cpu().numpy()
                test_y = batch_y[0].detach().cpu().numpy()
                cur_data_x.append(test_x)
                cur_data_y.append(test_y)
                # print(test_x.shape, test_y.shape)

                # lookback_x = []
                # lookback_y = []
                for ii in range(self.test_train_num):

                    seq_len = self.args.seq_len
                    label_len = self.args.label_len
                    pred_len = self.args.pred_len

                    # tmp_x = batch_x[:, ii : ii+seq_len, :].detach().cpu().numpy()
                    # tmp_y = batch_x[:, ii+seq_len-label_len : ii+seq_len+pred_len, :].detach().cpu().numpy()
                    tmp_x = batch_x[0, ii : ii+seq_len, :].detach().cpu().numpy()
                    tmp_y = batch_x[0, ii+seq_len-label_len : ii+seq_len+pred_len, :].detach().cpu().numpy()
                    # lookback_x.append(cur_x)
                    # lookback_y.append(cur_y)
                    # print(tmp_x.shape, tmp_y.shape)

                    cur_data_x.append(tmp_x)
                    cur_data_y.append(tmp_y)
                
                # cur_data_x = np.array(cur_data_x)
                # cur_data_y = np.array(cur_data_y)
                # print(cur_data_x.shape, cur_data_y.shape)
                all_data_x.append(cur_data_x)
                all_data_y.append(cur_data_y)

            all_data_x = np.array(all_data_x)
            all_data_y = np.array(all_data_y)
            print(all_data_x.shape)
            print(all_data_y.shape)
            

            lookback_data_dir = "./lookback_data" + '/' + setting
            if not os.path.exists(lookback_data_dir):
                os.makedirs(lookback_data_dir)
            
            lookback_data_npz = f"{lookback_data_dir}/data_{flag}_ttn{self.test_train_num}.npz"

            # 将first_grad和all_grads保存为npz文件
            np.savez(lookback_data_npz, all_data_x, all_data_y)

            
            # 再从npz中获取回来
            npzfile = np.load(lookback_data_npz)
            all_data_x = npzfile["arr_0"]
            all_data_y = npzfile["arr_1"]

            test_x, test_y = all_data_x[:, 0:1, :], all_data_y[:, 0:1, :]
            lookback_x, lookback_y = all_data_x[:, 1:, :], all_data_y[:, 1:, :]

            print(test_x.shape)
            print(test_y.shape)
            print(lookback_x.shape)
            print(lookback_y.shape)


    def get_answer_grad(self, is_training_part_params, use_adapted_model, lr, test_data, batch_x, batch_y, batch_x_mark, batch_y_mark, setting):
        cur_model = copy.deepcopy(self.model)
        cur_model.train()

        if is_training_part_params:
            params = []
            names = []
            cur_model.requires_grad_(False)
            for n_m, m in cur_model.named_modules():
                # 注意：这里的名字应当根据Autoformer模型而修改为"decoder.projection"
                if "decoder.projection" == n_m:
                    m.requires_grad_(True)
                    for n_p, p in m.named_parameters():
                        if n_p in ['weight', 'bias']:  # weight is scale, bias is shift
                            params.append(p)
                            names.append(f"{n_m}.{n_p}")
                
                # # 再加入倒数第二层layer_norm层
                # if "decoder.norm.layernorm" == n_m:
                #     for n_p, p in m.named_parameters():
                #         if n_p in ['weight', 'bias']:  # weight is scale, bias is shift
                #             params.append(p)
                #             names.append(f"{n_m}.{n_p}")

            # Adam优化器
            # model_optim = optim.Adam(params, lr=self.args.learning_rate*10 / (2**self.test_train_num))  # 使用Adam优化器
            # model_optim = optim.Adam(params, lr=lr)  # 使用Adam优化器
            # model_optim_norm = optim.Adam(params_norm, lr=self.args.learning_rate*1000 / (2**self.test_train_num))  # 使用Adam优化器
            # 普通的SGD优化器？
            new_lr = lr * 5
            model_optim = optim.SGD(params, lr=new_lr)
        else:
            cur_model.requires_grad_(True)
            # model_optim = optim.Adam(cur_model.parameters(), lr=self.args.learning_rate*10 / (2**self.test_train_num))
            # model_optim = optim.Adam(cur_model.parameters(), lr=self.args.learning_rate)
            model_optim = optim.SGD(cur_model.parameters(), lr=new_lr)
        
        if use_adapted_model:
            pred, true = self._process_one_batch_with_model(cur_model, test_data,
                batch_x[:, -self.args.seq_len:, :], batch_y, 
                batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)
        
        criterion = nn.MSELoss()  # 使用MSELoss
        # 计算MSE loss
        loss_ans_before = criterion(pred, true)
        loss_ans_before.backward()

        w_T = params[0].grad.T  # 先对weight参数做转置
        b = params[1].grad.unsqueeze(0)  # 将bias参数扩展一维
        params_answer = torch.cat((w_T, b), 0)  # 将w_T和b参数concat起来
        params_answer = params_answer.ravel()  # 最后再展开成一维的


        # 让原模型使用在该测试样本上生成的梯度来调整原模型
        # 并验证该梯度能否作为一个标准答案来做指导？
        model_optim.step()

        pred, true = self._process_one_batch_with_model(cur_model, test_data,
                batch_x[:, -self.args.seq_len:, :], batch_y, 
                batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)
        
        # # 计算MSE loss
        # loss_ans_after = criterion(pred, true)
        # # 注意，由于我们存储的是before - after的值，所以当loss_diff为正数才表示loss减少了
        # loss_diff = loss_ans_before - loss_ans_after
        # with open(f"./loss_diff/answer_lr_{new_lr}_{setting}", "a") as f:
        #     f.write(f"{loss_diff}" + "\n")
        

        model_optim.zero_grad()

        cur_model.eval()
        del cur_model

        return params_answer



    def select_with_distance(self, setting, test=0, is_training_part_params=True, use_adapted_model=True, test_train_epochs=1, weights_given=None, adapted_degree="small", weights_from="test"):
        test_data, test_loader = self._get_data_at_test_time(flag='test')
        data_len = len(test_data)
        if test:
            print('loading model from checkpoint !!!')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        self.model.eval()

        preds = []
        trues = []

        a1, a2, a3, a4 = [], [], [], []
        all_angels = []
        all_distances = []

        error_per_pred_index = [[] for i in range(self.args.pred_len)]

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()  # 如果使用amp的话，还需要再生成一个scaler？

        criterion = nn.MSELoss()  # 使用MSELoss
        test_time_start = time.time()

        # # result save
        # folder_path = './test_results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)


        # 加载模型参数到self.model里
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        if weights_given:
            print(f"The given weights is {weights_given}")
            print(f"The length of given weights is {len(weights_given)}")
        

        # self.model.eval()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            # false:
            # if data_len - i < self.args.batch_size: break
            
            # true:
            # if data_len - i < data_len % self.args.batch_size: break
            
            # 从self.model拷贝下来cur_model，并设置为train模式
            # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))
            cur_model = copy.deepcopy(self.model)
            # cur_model.train()
            cur_model.eval()

            if is_training_part_params:
                params = []
                names = []
                params_norm = []
                names_norm = []
                cur_model.requires_grad_(False)
                for n_m, m in cur_model.named_modules():
                    # 注意：这里的名字应当根据Autoformer模型而修改为"decoder.projection"
                    
                    # if "decoder.projection" == n_m:
                    if "decoder.projection" in n_m:
                        m.requires_grad_(True)
                        for n_p, p in m.named_parameters():
                            if n_p in ['weight', 'bias']:  # weight is scale, bias is shift
                                params.append(p)
                                names.append(f"{n_m}.{n_p}")

                # Adam优化器
                # model_optim = optim.Adam(params, lr=self.args.learning_rate*10 / (2**self.test_train_num))  # 使用Adam优化器
                lr = self.args.learning_rate * self.args.adapted_lr_times
                # model_optim = optim.Adam(params, lr=lr)  # 使用Adam优化器
                
                # 普通的SGD优化器？
                model_optim = optim.SGD(params, lr=lr)
            else:
                cur_model.requires_grad_(True)
                # model_optim = optim.Adam(cur_model.parameters(), lr=self.args.learning_rate*10 / (2**self.test_train_num))
                # model_optim = optim.Adam(cur_model.parameters(), lr=self.args.learning_rate)
                lr = self.args.learning_rate * self.args.adapted_lr_times
                model_optim = optim.SGD(cur_model.parameters(), lr=lr)
            

            # tmp loss
            # cur_model.eval()
            seq_len = self.args.seq_len
            pred_len = self.args.pred_len
            adapt_start_pos = self.args.adapt_start_pos
            if not self.args.use_nearest_data or self.args.use_further_data:
                pred, true = self._process_one_batch_with_model(cur_model, test_data,
                    batch_x[:, -seq_len:, :], batch_y, 
                    batch_x_mark[:, -seq_len:, :], batch_y_mark)
            else:
                pred, true = self._process_one_batch_with_model(cur_model, test_data,
                    batch_x[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y, 
                    batch_x_mark[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y_mark)
            if self.args.adapt_part_channels:
                pred = pred[:, :, self.selected_channels]
                true = true[:, :, self.selected_channels]
            # 获取adaptation之前的loss
            loss_before_adapt = criterion(pred, true)
            a1.append(loss_before_adapt.item())
            # cur_model.train()
            

            # 先用原模型的预测值和标签值之间的error，做反向传播之后得到的梯度值gradient_0
            # 并将这个gradient_0作为标准答案
            # 然后，对测试样本做了adaptation之后，会得到一个gradient_1
            # 那么对gradient_1和gradient_0之间做对比，
            # 就可以得到二者之间的余弦值是多少（方向是否一致），以及长度上相差的距离有多少等等。
            # params_answer = self.get_answer_grad(is_training_part_params, use_adapted_model,
            #                                         lr, test_data, 
            #                                         batch_x, batch_y, batch_x_mark, batch_y_mark,
            #                                         setting)
            if use_adapted_model:
                seq_len = self.args.seq_len
                pred_len = self.args.pred_len
                adapt_start_pos = self.args.adapt_start_pos
                if not self.args.use_nearest_data or self.args.use_further_data:
                    pred_answer, true_answer = self._process_one_batch_with_model(cur_model, test_data,
                        batch_x[:, -seq_len:, :], batch_y, 
                        batch_x_mark[:, -seq_len:, :], batch_y_mark)
                else:
                    pred_answer, true_answer = self._process_one_batch_with_model(cur_model, test_data,
                        batch_x[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y, 
                        batch_x_mark[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y_mark)
            
            if self.args.adapt_part_channels:
                pred_answer = pred_answer[:, :, self.selected_channels]
                true_answer = true_answer[:, :, self.selected_channels]
            # criterion = nn.MSELoss()  # 使用MSELoss
            # 计算MSE loss
            loss_ans_before = criterion(pred_answer, true_answer)
            loss_ans_before.backward()

            w_T = params[0].grad.T  # 先对weight参数做转置
            b = params[1].grad.unsqueeze(0)  # 将bias参数扩展一维
            params_answer = torch.cat((w_T, b), 0)  # 将w_T和b参数concat起来
            params_answer = params_answer.ravel()  # 最后再展开成一维的，就得到了标准答案对应的梯度方向

            model_optim.zero_grad()  # 清空梯度


            
            # 选择出合适的梯度
            # 注意：这里是减去梯度，而不是加上梯度！！！！！
            # selected_channels = self.selected_channels

            # 再获得未被选取的unselected_channels
            unselected_channels = list(range(self.args.c_out))
            for item in self.selected_channels:
                unselected_channels.remove(item)
            

            # 在这类我们需要先对adaptation样本的x和测试样本的x之间的距离做对比
            import torch.nn.functional as F
            
            if self.args.adapt_part_channels:  
                test_x = batch_x[:, -seq_len:, self.selected_channels].reshape(-1)
            else:
                test_x = batch_x[:, -seq_len:, :].reshape(-1)
            
            distance_pairs = []
            for ii in range(self.args.test_train_num):
                # 只对周期性样本计算x之间的距离
                if self.args.adapt_cycle:
                    # 为了计算当前的样本和测试样本间时间差是否是周期的倍数
                    # 我们先计算时间差与周期相除的余数
                    if 'illness' in self.args.data_path:
                        import math
                        cycle_remainer = math.fmod(self.args.test_train_num-1 + self.args.pred_len - ii, self.period)
                    cycle_remainer = (self.args.test_train_num-1 + self.args.pred_len - ii) % self.period
                    # 定义判定的阈值
                    threshold = self.period // 12
                    # 如果余数在[-threshold, threshold]之间，那么考虑使用其做fine-tune
                    # 否则的话不将其纳入计算距离的数据范围内
                    if cycle_remainer > threshold or cycle_remainer < -threshold:
                        continue
                    
                if self.args.adapt_part_channels:
                    lookback_x = batch_x[:, ii : ii+seq_len, self.selected_channels].reshape(-1)
                else:
                    lookback_x = batch_x[:, ii : ii+seq_len, :].reshape(-1)
                dist = F.pairwise_distance(test_x, lookback_x, p=2).item()
                distance_pairs.append([ii, dist])

            # 先按距离从小到大排序
            cmp = lambda item: item[1]
            distance_pairs.sort(key=cmp)

            # 筛选出其中最小的selected_data_num个样本出来
            selected_distance_pairs = distance_pairs[:self.args.selected_data_num]
            selected_indices = [item[0] for item in selected_distance_pairs]
            selected_distances = [item[1] for item in selected_distance_pairs]
            # print(f"selected_distance_pairs is: {selected_distance_pairs}")

            all_distances.append(selected_distances)

            # params_adapted = torch.zeros((1)).to(self.device)
            cur_grad_list = []

            # 开始训练
            for epoch in range(test_train_epochs):

                gradients = []
                accpted_samples_num = set()

                # num_of_loss_per_update = 1
                mean_loss = 0


                for ii in selected_indices:

                    model_optim.zero_grad()

                    seq_len = self.args.seq_len
                    label_len = self.args.label_len
                    pred_len = self.args.pred_len

                    # batch_x.requires_grad = True
                    # batch_x_mark.requires_grad = True

                    pred, true = self._process_one_batch_with_model(cur_model, test_data,
                        batch_x[:, ii : ii+seq_len, :], batch_x[:, ii+seq_len-label_len : ii+seq_len+pred_len, :], 
                        batch_x_mark[:, ii : ii+seq_len, :], batch_x_mark[:, ii+seq_len-label_len : ii+seq_len+pred_len, :])

                    # 这里当batch_size为1还是32时
                    # pred和true的size可能为[1, 24, 7]或[32, 24, 7]
                    # 但是结果的loss值均只包含1个值
                    # 这是因为criterion为MSELoss，其默认使用mean模式，会对32个loss值取一个平均值

                    if self.args.adapt_part_channels:
                        pred = pred[:, :, self.selected_channels]
                        true = true[:, :, self.selected_channels]
                    
                    # 判断是否使用最近的数据
                    if not self.args.use_nearest_data or self.args.use_further_data:
                        loss = criterion(pred, true)
                    else:
                        data_used_num = (self.test_train_num - (ii+1)) + self.args.adapt_start_pos
                        if data_used_num < pred_len:
                            loss = criterion(pred[:, :data_used_num, :], true[:, :data_used_num, :])
                        else:
                            loss = criterion(pred, true)
                        # loss = criterion(pred, true)

                    # loss = criterion(pred, true)
                    mean_loss += loss

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        # scaler.step(model_optim_norm)
                        scaler.update()
                    else:
                        # # pass
                        # loss.backward()
                        # model_optim.step()

                        loss.backward()
                        w_T = params[0].grad.T
                        b = params[1].grad.unsqueeze(0)
                        params_tmp = torch.cat((w_T, b), 0)
                        original_shape = params_tmp.shape
                        params_tmp = params_tmp.ravel()

                        # 将该梯度存入cur_grad_list中
                        cur_grad_list.append(params_tmp.detach().cpu().numpy())

                        model_optim.zero_grad()

                    # 记录逐样本做了adaptation之后的loss
                    # mean_loss += tmp_loss
                    # mean_loss += loss
            
            
            # 定义一个权重和梯度相乘函数
            def calc_weighted_params(params, weights):
                results = 0
                for i in range(len(params)):
                    results += params[i] * weights[i]
                return results
            
            # 权重分别乘到对应的梯度上
            if weights_given:
                weighted_params = calc_weighted_params(cur_grad_list, weights_given)
            else:
                weights_all_ones = [1 for i in range(self.test_train_num)]
                weighted_params = calc_weighted_params(cur_grad_list, weights_all_ones)
            
            # 将weighted_params从np.array转成tensor
            weighted_params = torch.tensor(weighted_params)
            weighted_params = weighted_params.to(self.device)


            # 计算标准答案的梯度params_answer和adaptation加权后的梯度weighted_params之间的角度
            import math
            product = torch.dot(weighted_params, params_answer)
            product = product / (torch.norm(weighted_params) * torch.norm(params_answer))
            angel = math.degrees(math.acos(product))
            all_angels.append(angel)
            

            # 还原回原来的梯度
            # 也即将weighted_params变回w_grad和b_grad
            weighted_params = weighted_params.reshape(original_shape)
            w_grad_T, b_grad = torch.split(weighted_params, [weighted_params.shape[0]-1, 1])
            w_grad = w_grad_T.T  # (7, 512)
            b_grad = b_grad.squeeze(0)  # (7)


            # 设置新参数为原来参数 + 梯度值
            from torch.nn.parameter import Parameter
            cur_lr = self.args.learning_rate * self.args.adapted_lr_times

            # 将未选择的channels上的梯度置为0
            if self.args.adapt_part_channels:
                w_grad[unselected_channels, :] = 0
                b_grad[unselected_channels] = 0

            # 注意：这里是减去梯度，而不是加上梯度！！！！！
            cur_model.decoder.projection.weight = Parameter(cur_model.decoder.projection.weight - w_grad * cur_lr)
            cur_model.decoder.projection.bias = Parameter(cur_model.decoder.projection.bias - b_grad * cur_lr)



            # mean_loss = mean_loss / self.test_train_num
            mean_loss = mean_loss / self.args.selected_data_num
            a2.append(mean_loss.item())
            
            # mean_loss.backward()
            # model_optim.step()


            # seq_len = self.args.seq_len
            # label_len = self.args.label_len
            # pred_len = self.args.pred_len
            # tmp_loss = 0
            # for ii in selected_indices:
            #     pred, true = self._process_one_batch_with_model(cur_model, test_data,
            #         batch_x[:, ii : ii+seq_len, :], batch_x[:, ii+seq_len-label_len : ii+seq_len+pred_len, :], 
            #         batch_x_mark[:, ii : ii+seq_len, :], batch_x_mark[:, ii+seq_len-label_len : ii+seq_len+pred_len, :])
            #     if self.args.adapt_part_channels:
            #         pred = pred[:, :, self.selected_channels]
            #         true = true[:, :, self.selected_channels]
            #     tmp_loss += criterion(pred, true)
            # tmp_loss = tmp_loss / self.args.selected_data_num
            # a3.append(tmp_loss.item())
            a3.append(0)



            cur_model.eval()

            seq_len = self.args.seq_len
            pred_len = self.args.pred_len
            adapt_start_pos = self.args.adapt_start_pos
            if use_adapted_model:
                if not self.args.use_nearest_data or self.args.use_further_data:
                    pred, true = self._process_one_batch_with_model(cur_model, test_data,
                        batch_x[:, -seq_len:, :], batch_y, 
                        batch_x_mark[:, -seq_len:, :], batch_y_mark)
                else:
                    pred, true = self._process_one_batch_with_model(cur_model, test_data,
                        batch_x[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y, 
                        batch_x_mark[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y_mark)
            # else:
            #     # pred, true = self._process_one_batch_with_model(self.model, test_data,
            #     #     batch_x[:, -self.args.seq_len:, :], batch_y, 
            #     #     batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)
            #     if not self.args.use_nearest_data or self.args.use_further_data:
            #         pred, true = self._process_one_batch_with_model(cur_model, test_data,
            #             batch_x[:, -seq_len:, :], batch_y, 
            #             batch_x_mark[:, -seq_len:, :], batch_y_mark)
            #     else:
            #         pred, true = self._process_one_batch_with_model(cur_model, test_data,
            #             batch_x[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y, 
            #             batch_x_mark[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y_mark)

            # 如果需要筛选部分维度，那么做一次筛选：
            if self.args.adapt_part_channels:
                pred = pred[:, :, self.selected_channels]
                true = true[:, :, self.selected_channels]

            # 获取adaptation之后的loss
            loss_after_adapt = criterion(pred, true)
            a4.append(loss_after_adapt.item())

            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

            # 对预测结果（如长度为24）中的每个位置/index上的结果分别进行统计
            pred_len = self.args.pred_len
            for index in range(pred_len):
                cur_pred = pred.detach().cpu().numpy()[0][index]
                cur_true = true.detach().cpu().numpy()[0][index]
                cur_error = np.mean((cur_pred - cur_true) ** 2)
                error_per_pred_index[index].append(cur_error)


            if (i+1) % 100 == 0 or (data_len - i) < 100 and (i+1) % 10 == 0:
                print("\titers: {0}, cost time: {1}s".format(i + 1, time.time() - test_time_start))
                print(gradients)
                tmp_p = np.array(preds); tmp_p = tmp_p.reshape(-1, tmp_p.shape[-2], tmp_p.shape[-1])
                tmp_t = np.array(trues); tmp_t = tmp_t.reshape(-1, tmp_t.shape[-2], tmp_t.shape[-1])
                tmp_mae, tmp_mse, _, _, _ = metric(tmp_p, tmp_t)
                print('mse:{}, mae:{}'.format(tmp_mse, tmp_mae))
                
                avg_1, avg_2, avg_3, avg_4 = 0, 0, 0, 0
                avg_angel = 0
                num = len(a1)
                for iii in range(num):
                    avg_1 += a1[iii]; avg_2 += a2[iii]; avg_3 += a3[iii]; avg_4 += a4[iii]; avg_angel += all_angels[iii]
                avg_1 /= num; avg_2 /= num; avg_3 /= num; avg_4 /= num; avg_angel /= num
                print("1.before_adapt, 2.adapt_sample, 3.adapt_sample_after, 4.after_adapt, 5.angel_between_answer")
                print("average:", avg_1, avg_2, avg_3, avg_4, avg_angel)
                print("last one:", a1[-1], a2[-1], a3[-1], a4[-1], all_angels[-1])

                printed_selected_channels = [item+1 for item in self.selected_channels]
                print(f"adapt_part_channels: {self.args.adapt_part_channels}, and adapt_cycle: {self.args.adapt_cycle}")
                print(f"first 25th selected_channels: {printed_selected_channels[:25]}")
                print(f"selected_distance_pairs are: {selected_distance_pairs}")


            # if i % 20 == 0:
            #     input = batch_x.detach().cpu().numpy()
            #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
            #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
            #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

            cur_model.eval()
            # cur_model.cpu()
            del cur_model
            torch.cuda.empty_cache()


        # # 将adaptation前的loss、adaptation中逐样本做adapt后的loss、以及adaptation之后的loss做统计
        # with open(f"./loss_before_and_after_adapt/{setting}.txt", "a") as f:
        #     for i in range(len(a1)):
        #         t1, t2, t3 = a1[i], a2[i], a3[i]
        #         f.write(f"{t1}, {t2}, {t3}\n")


        # 对预测结果（如长度为24）中的每个位置/index上的结果分别进行统计的结果输出出来
        mean_error_per_pred_index = [[] for i in range(pred_len)]
        for index in range(pred_len):
            error_i = error_per_pred_index[index]
            total_err = 0
            total_num = 0
            for tmp_err in error_i:
                total_err += tmp_err
                total_num += 1
            mean_error_per_pred_index[index] = total_err / total_num
        # print(mean_error_per_pred_index)


        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        # f = open("result.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        print(f"Test - cost time: {time.time() - test_time_start}s")


        # 保存distance和loss信息
        distance_dir = "./distances_and_loss_diff/" + setting
        if not os.path.exists(distance_dir):
            os.makedirs(distance_dir)

        if weights_given:
            distance_file = f"{distance_dir}/distances_{weights_from}_select{self.args.selected_data_num}_ttn{self.test_train_num}_lr{self.args.adapted_lr_times:.2f}.txt"
        else:
            distance_file = f"{distance_dir}/distances_allones_select{self.args.selected_data_num}_ttn{self.test_train_num}_lr{self.args.adapted_lr_times:.2f}.txt"

        with open(distance_file, "w") as f:
            for i in range(len(a1)):
                for ii in range(len(all_distances[i])):
                    f.write(f"{all_distances[i][ii]}, ")
                f.write(f"{a1[i]}, {a3[i]}" + "\n")

        return a1, a2, a3, a4


    # def learn_mapping_model_during_vali(self, is_training_part_params):
    #     # 这里用'val_with_batchsize_1'来获取数据
    #     # vali_data, vali_loader = self._get_data(flag='val_with_batchsize_1')
    #     vali_data, vali_loader = self._get_data(flag='val')
    #     criterion = nn.MSELoss()  # 使用MSELoss
        
    #     cur_model = copy.deepcopy(self.model)
    #     cur_model.train()

    #     train_X = []
    #     train_Y = []

    #     if is_training_part_params:
    #         params = []
    #         names = []
    #         cur_model.requires_grad_(False)
    #         for n_m, m in cur_model.named_modules():
    #             # 注意：这里的名字应当根据Autoformer模型而修改为"decoder.projection"
    #             if "decoder.projection" == n_m:
    #                 m.requires_grad_(True)
    #                 for n_p, p in m.named_parameters():
    #                     if n_p in ['weight', 'bias']:  # weight is scale, bias is shift
    #                         params.append(p)
    #                         names.append(f"{n_m}.{n_p}")
                
    #             # # 再加入倒数第二层layer_norm层
    #             # if "decoder.norm.layernorm" == n_m:
    #             #     for n_p, p in m.named_parameters():
    #             #         if n_p in ['weight', 'bias']:  # weight is scale, bias is shift
    #             #             params.append(p)
    #             #             names.append(f"{n_m}.{n_p}")
    #         # 普通的SGD优化器？
    #         tmp_optim = optim.SGD(params, lr=self.args.learning_rate)


    #     for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
    #         # 在验证集上跑
    #         pred, true = self._process_one_batch_with_model(cur_model, vali_data,
    #             batch_x, batch_y, 
    #             batch_x_mark, batch_y_mark)

    #         # pred = pred.detach().cpu()
    #         # true = true.detach().cpu()

    #         # 每次计算前都要将之前的梯度先清零、以避免梯度被累加
    #         tmp_optim.zero_grad()
    #         # 计算MSE loss并作backward
    #         loss = criterion(pred, true)
    #         loss.backward()

    #         w_T = params[0].grad.T  # 先对weight参数的梯度做转置
    #         b = params[1].grad.unsqueeze(0)  # 将bias参数的梯度扩展一维
    #         # print(b)
    #         params_answer = torch.cat((w_T, b), 0)  # 将w_T和b参数concat起来
    #         params_answer = params_answer.ravel()  # 最后再展开成一维的

    #         train_X_sample = torch.cat((batch_x, batch_x_mark), 2).ravel()
    #         train_Y_sample = params_answer
    #         train_X.append(train_X_sample)
    #         train_Y.append(train_Y_sample)
        

    #     # 一定记得要加int，否则dim会是浮点型的！！
    #     X_dim = int(train_X[0].shape[0])
    #     Y_dim = int(train_Y[0].shape[0])
    #     print(f"X_dim: {X_dim}, Y_dim: {Y_dim}")
    #     # 这里要用整除
    #     h_dim = int(self.args.d_model // 1)
    #     # h_dim = int(self.args.d_model // 4)
    #     mapping_model = nn.Sequential(
    #         nn.Linear(X_dim, h_dim), 
    #         nn.Linear(h_dim ,Y_dim)
    #         )
    #     mapping_model.to(self.device)
    #     mapping_model.train()

    #     # loss_fn = nn.MSELoss().to(self.device)
    #     lr = self.args.learning_rate * 1000
    #     model_optim = optim.SGD(mapping_model.parameters(), lr=lr)

    #     for epoch in range(self.args.train_epochs):
    #         print(f"training for mapping model, epoch:{epoch+1}")
    #         for i in range(len(train_X)):
    #             train_X_sample = train_X[i]
    #             train_Y_sample = train_Y[i]
    #             train_X_sample = train_X_sample.to(torch.float32).to(self.device)
    #             train_Y_sample = train_Y_sample.to(torch.float32).to(self.device)
    #             output = mapping_model(train_X_sample)
                
    #             product = torch.dot(output, train_Y_sample)
    #             product = product / (torch.norm(output) * torch.norm(train_Y_sample))
    #             loss = (1 - product) * 100
                
    #             model_optim.zero_grad()
    #             loss.backward()
    #             model_optim.step()

    #             if (i+1) % 10 == 0:
    #                 print(f"epoch:{epoch+1}, iteration:{i+1}, product loss:{loss.item()}")

    #     cur_model.eval()
    #     del cur_model

    #     mapping_model.eval()
    #     return mapping_model


    # def learn_mapping_model_during_vali(self, is_training_part_params, setting, flag="test", test_train_epochs=1, use_adapted_model=True):
    #     if flag == "test":
    #         test_data, test_loader = self._get_data_at_test_time(flag='test')
    #     elif flag == "val":
    #         # 这里用需要用'val_with_batchsize_1'来获取数据
    #         test_data, test_loader = self._get_data_at_test_time(flag='val')

    #     self.model.eval()

    #     preds = []
    #     trues = []

    #     a1, a2, a3 = [], [], []

    #     if self.args.use_amp:
    #         scaler = torch.cuda.amp.GradScaler()  # 如果使用amp的话，还需要再生成一个scaler？

    #     criterion = nn.MSELoss()  # 使用MSELoss
    #     test_time_start = time.time()

    #     # 先加载模型参数
    #     self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))


    #     # 从self.model拷贝下来cur_model，并设置为train模式
    #     cur_model = copy.deepcopy(self.model)
    #     cur_model.train()

    #     if is_training_part_params:
    #         params = []
    #         names = []
    #         cur_model.requires_grad_(False)
    #         for n_m, m in cur_model.named_modules():
    #             # 注意：这里的名字应当根据Autoformer模型而修改为"decoder.projection"
                
    #             # if "decoder.projection" == n_m:
    #             if "decoder.projection" in n_m:
    #                 m.requires_grad_(True)
    #                 for n_p, p in m.named_parameters():
    #                     if n_p in ['weight', 'bias']:  # weight is scale, bias is shift
    #                         params.append(p)
    #                         names.append(f"{n_m}.{n_p}")

    #         # lr = self.args.learning_rate * self.args.adapted_lr_times
    #         lr = self.args.learning_rate

    #         model_optim = optim.SGD(params, lr=lr)
    #     else:
    #         cur_model.requires_grad_(True)
    #         # lr = self.args.learning_rate * self.args.adapted_lr_times
    #         lr = self.args.learning_rate
    #         model_optim = optim.SGD(cur_model.parameters(), lr=lr)


    #     # self.model.eval()
    #     all_grads = []
    #     for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):

    #         cur_grad_list = []

    #         # tmp loss
    #         cur_model.train()
    #         pred, true = self._process_one_batch_with_model(cur_model, test_data,
    #             batch_x[:, -self.args.seq_len:, :], batch_y, 
    #             batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)
    #         # 获取adaptation之前的loss
    #         loss_before_adapt = criterion(pred, true)


    #         # 计算在测试样本上的梯度
    #         loss_before_adapt.backward()
    #         w_T = params[0].grad.T
    #         b = params[1].grad.unsqueeze(0)
    #         params_tmp = torch.cat((w_T, b), 0)
    #         params_tmp = params_tmp.ravel()

    #         cur_grad_list.append(params_tmp.detach().cpu().numpy())

    #         model_optim.zero_grad()

    #         a1.append(loss_before_adapt.item())
    #         # print(f"loss_before_adapt: {loss_before_adapt}")
    #         # print(f'mse:{tmp_loss}')
    #         cur_model.train()



    #         params_adapted = torch.zeros((1)).to(self.device)
    #         # 开始训练
    #         for epoch in range(test_train_epochs):

    #             gradients = []
    #             accpted_samples_num = set()

    #             import random
    #             is_random = False
    #             # is_random = True
    #             sample_order_list = list(range(self.test_train_num))
    #             if is_random:
    #                 random.shuffle(sample_order_list)
    #             else:
    #                 pass

    #             # 新加了参数num_of_loss_per_update，表示我们将多个loss平均起来做一次更新
    #             num_of_loss_per_update = self.test_train_num
    #             # num_of_loss_per_update = 1
    #             mean_loss = 0


    #             for ii in sample_order_list:

    #                 model_optim.zero_grad()

    #                 seq_len = self.args.seq_len
    #                 label_len = self.args.label_len
    #                 pred_len = self.args.pred_len

    #                 # batch_x.requires_grad = True
    #                 # batch_x_mark.requires_grad = True

    #                 pred, true = self._process_one_batch_with_model(cur_model, test_data,
    #                     batch_x[:, ii : ii+seq_len, :], batch_x[:, ii+seq_len-label_len : ii+seq_len+pred_len, :], 
    #                     batch_x_mark[:, ii : ii+seq_len, :], batch_x_mark[:, ii+seq_len-label_len : ii+seq_len+pred_len, :])

    #                 # 这里当batch_size为1还是32时
    #                 # pred和true的size可能为[1, 24, 7]或[32, 24, 7]
    #                 # 但是结果的loss值均只包含1个值
    #                 # 这是因为criterion为MSELoss，其默认使用mean模式，会对32个loss值取一个平均值
                    
    #                 loss = criterion(pred, true)

    #                 # loss = criterion(pred, true)
    #                 mean_loss += loss

    #                 if self.args.use_amp:
    #                     scaler.scale(loss).backward()
    #                     scaler.step(model_optim)
    #                     # scaler.step(model_optim_norm)
    #                     scaler.update()
    #                 else:
    #                     # # pass
    #                     # loss.backward()
    #                     # model_optim.step()

    #                     loss.backward()

    #                     w_T = params[0].grad.T
    #                     b = params[1].grad.unsqueeze(0)
    #                     params_tmp = torch.cat((w_T, b), 0)
    #                     params_tmp = params_tmp.ravel()
    #                     # params_adapted = params_adapted + params_tmp

    #                     cur_grad_list.append(params_tmp.detach().cpu().numpy())

    #                     model_optim.zero_grad()

    #                     # # 计算标准答案的梯度params_answer和adaptation中的梯度params_tmp之间的角度
    #                     # import math
    #                     # # print()
    #                     # # product = torch.dot(params_tmp, params_answer)
    #                     # # product = product / (torch.norm(params_tmp) * torch.norm(params_answer))
    #                     # product = torch.dot(params_tmp, mapping_answer)
    #                     # product = product / (torch.norm(params_tmp) * torch.norm(mapping_answer))
    #                     # angel = math.degrees(math.acos(product))
    #                     # if angel < 90:
    #                     #     model_optim.step()
    #                     # else:
    #                     #     model_optim.zero_grad()

    #                 # 记录逐样本做了adaptation之后的loss
    #                 # mean_loss += tmp_loss
    #                 # mean_loss += loss
            

    #         mean_loss = mean_loss / self.test_train_num
    #         a2.append(mean_loss.item())
            
    #         # mean_loss.backward()
    #         # model_optim.step()

    #         # 把cur_grad_list存入all_grads中
    #         all_grads.append(cur_grad_list)
            
    #         cur_model.eval()


    #         # # 计算标准答案的梯度params_answer和adaptation中的梯度params_adapted之间的角度
    #         # import math
    #         # product = torch.dot(params_adapted, params_answer)
    #         # product = product / (torch.norm(params_adapted) * torch.norm(params_answer))
    #         # angel = math.degrees(math.acos(product))

    #         # with open(f"./grad_angels/{setting}", "a") as f:
    #         #     f.write(f"{angel}" + "\n")

    #         cur_model.eval()

    #         if use_adapted_model:
    #             pred, true = self._process_one_batch_with_model(cur_model, test_data,
    #                 batch_x[:, -self.args.seq_len:, :], batch_y, 
    #                 batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)
    #         else:
    #             # pred, true = self._process_one_batch_with_model(self.model, test_data,
    #             #     batch_x[:, -self.args.seq_len:, :], batch_y, 
    #             #     batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)
    #             pred, true = self._process_one_batch_with_model(cur_model, test_data,
    #                 batch_x[:, -self.args.seq_len:, :], batch_y, 
    #                 batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)


    #         # 获取adaptation之后的loss
    #         loss_after_adapt = criterion(pred, true)
    #         a3.append(loss_after_adapt.item())


    #         preds.append(pred.detach().cpu().numpy())
    #         trues.append(true.detach().cpu().numpy())


    #         if (i+1) % 100 == 0:
    #             print("\titers: {0}, cost time: {1}s".format(i + 1, time.time() - test_time_start))
    #             print(gradients)
    #             tmp_p = np.array(preds); tmp_p = tmp_p.reshape(-1, tmp_p.shape[-2], tmp_p.shape[-1])
    #             tmp_t = np.array(trues); tmp_t = tmp_t.reshape(-1, tmp_t.shape[-2], tmp_t.shape[-1])
    #             tmp_mae, tmp_mse, _, _, _ = metric(tmp_p, tmp_t)
    #             print('mse:{}, mae:{}'.format(tmp_mse, tmp_mae))
    #             print(type(all_grads), len(all_grads), type(np.array(all_grads)), np.array(all_grads).shape)
                
    #             # avg_1, avg_2, avg_3 = 0, 0, 0
    #             # num = len(a1)
    #             # for iii in range(num):
    #             #     avg_1 += a1[iii]; avg_2 += a2[iii]; avg_3 += a3[iii]
    #             # avg_1 /= num; avg_2 /= num; avg_3 /= num
    #             # print(avg_1, avg_2, avg_3)

    #         cur_model.eval()
    #         # cur_model.cpu()
    #         del cur_model
    #         torch.cuda.empty_cache()


    #     # # 将adaptation前的loss、adaptation中逐样本做adapt后的loss、以及adaptation之后的loss做统计
    #     # with open(f"./loss_before_and_after_adapt/{setting}.txt", "a") as f:
    #     #     for i in range(len(a1)):
    #     #         t1, t2, t3 = a1[i], a2[i], a3[i]
    #     #         f.write(f"{t1}, {t2}, {t3}\n")


    #     preds = np.array(preds)
    #     trues = np.array(trues)
    #     print('test shape:', preds.shape, trues.shape)

    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    #     trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    #     print('test shape:', preds.shape, trues.shape)

    #     # result save
    #     folder_path = './results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     mae, mse, rmse, mape, mspe = metric(preds, trues)
    #     print('mse:{}, mae:{}'.format(mse, mae))


        
    #     class LR(nn.Module):
    #         def __init__(self, dim1, dim2):
    #             super().__init__()
    #             self.linear = nn.Linear(dim1, dim2, bias=False)
            
    #         def forward(self, x):
    #             # out = self.linear(x.permute(1,0)).permute(1,0)
    #             out = self.linear(x)
    #             return out

    #     # the shape of all_grads is (2785, 11, 512*7)
    #     grad_X = all_grads[:, 1:, :]  # (2785, 10, 512*7)
    #     grad_Y = all_grads[:, 0:1, :]  # (2785, 1, 512*7)
    #     grad_X = grad_X.transpose(0, 2, 1)  # (2785, 512*7, 10)
    #     grad_Y = grad_Y.transpose(0, 2, 1)  # (2785, 512*7, 1)
    #     grad_X = grad_X.reshape(-1, grad_X.shape[-1])
    #     grad_Y = grad_Y.reshape(-1, grad_Y.shape[-1])

    #     grad_X_mean = grad_X.mean(axis=1)
    #     dim1 = 
    #     mapping_model = LR()

        

    #     # 一定记得要加int，否则dim会是浮点型的！！
    #     X_dim = int(train_X[0].shape[0])
    #     Y_dim = int(train_Y[0].shape[0])
    #     print(f"X_dim: {X_dim}, Y_dim: {Y_dim}")
    #     # 这里要用整除
    #     h_dim = int(self.args.d_model // 1)
    #     # h_dim = int(self.args.d_model // 4)
    #     mapping_model = nn.Sequential(
    #         nn.Linear(X_dim, h_dim), 
    #         nn.Linear(h_dim ,Y_dim)
    #         )
    #     mapping_model.to(self.device)
    #     mapping_model.train()

    #     # loss_fn = nn.MSELoss().to(self.device)
    #     lr = self.args.learning_rate * 1000
    #     model_optim = optim.SGD(mapping_model.parameters(), lr=lr)

    #     for epoch in range(self.args.train_epochs):
    #         print(f"training for mapping model, epoch:{epoch+1}")
    #         for i in range(len(train_X)):
    #             train_X_sample = train_X[i]
    #             train_Y_sample = train_Y[i]
    #             train_X_sample = train_X_sample.to(torch.float32).to(self.device)
    #             train_Y_sample = train_Y_sample.to(torch.float32).to(self.device)
    #             output = mapping_model(train_X_sample)
                
    #             product = torch.dot(output, train_Y_sample)
    #             product = product / (torch.norm(output) * torch.norm(train_Y_sample))
    #             loss = (1 - product) * 100
                
    #             model_optim.zero_grad()
    #             loss.backward()
    #             model_optim.step()

    #             if (i+1) % 10 == 0:
    #                 print(f"epoch:{epoch+1}, iteration:{i+1}, product loss:{loss.item()}")

    #     cur_model.eval()
    #     del cur_model

    #     mapping_model.eval()
    #     return mapping_model


    def calc_analytical_solution(self, mid_embedding_list, y_list, weight_list, W_orig):
        # self.args.lambda_reg = 

        dim = mid_embedding_list[0].shape[-1]  # H+1
        eye = torch.eye(dim).to(self.device)  # 对角矩阵I

        # left = self.args.lambda_reg * eye
        # for ii in range(len(mid_embedding_list)):
        #     mid_embedding = mid_embedding_list[ii].unsqueeze(0)  # 1*(H+1)
        #     weight = weight_list[ii]
        #     left += weight * (mid_embedding.T @ mid_embedding)  # (H+1)*(H+1)

        # right = self.args.lambda_reg * W_orig
        # for ii in range(len(mid_embedding_list)):
        #     mid_embedding = mid_embedding_list[ii].unsqueeze(0)  # 1*(H+1)
        #     y_i = y_list[ii].unsqueeze(0)  # 1*F
        #     weight = weight_list[ii]
        #     right += weight * (mid_embedding.T @ y_i)  # (H+1)*F

        left = self.args.lambda_reg * eye
        right = self.args.lambda_reg * W_orig
        
        # 先将X中均乘以weight_list中的权重值
        weighted_mid_embedding_list = []
        for ii in range(len(mid_embedding_list)):
            weighted_mid_embedding_list.append(mid_embedding_list[ii] * weight_list[ii])

        weighted_mid_embedding = torch.stack(weighted_mid_embedding_list)
        mid_embedding = torch.stack(mid_embedding_list)
        y_matrix = torch.stack(y_list)
        
        # left
        left += weighted_mid_embedding.T @ mid_embedding

        # right
        right += weighted_mid_embedding.T @ y_matrix
        
        # return torch.linalg.inv(left) @ right  # (H+1)*F
        return torch.linalg.solve(left, right)  # (H+1)*F
    

    def calc_test(self, setting, test=0, is_training_part_params=True, use_adapted_model=True, test_train_epochs=1, weights_given=None):
        test_data, test_loader = self._get_data_at_test_time(flag='test')
        if test:
            print('loading model from checkpoint !!!')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        self.model.eval()

        preds = []
        trues = []

        a1, a2, a3, a4 = [], [], [], []

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()  # 如果使用amp的话，还需要再生成一个scaler？

        criterion = nn.MSELoss()  # 使用MSELoss
        test_time_start = time.time()


        cur_model = self.model
        if is_training_part_params:
            params = []
            names = []
            cur_model.requires_grad_(False)
            for n_m, m in cur_model.named_modules():
                # 注意：这里的名字应当根据Autoformer模型而修改为"decoder.projection"
                if "decoder.projection" == n_m:
                    # m.requires_grad_(True)
                    for n_p, p in m.named_parameters():
                        if n_p in ['weight', 'bias']:  # weight is scale, bias is shift
                            params.append(p)
                            names.append(f"{n_m}.{n_p}")
        else:
            # cur_model.requires_grad_(True)
            pass
        
        # 将原始的w_original和b_original记录下来
        # w_original = params[0].detach()  # F*H
        # b_original = params[1].detach()
        w_original = copy.deepcopy(params[0])  # F*H
        b_original = copy.deepcopy(params[1])


        if weights_given:
            print(f"The given weights is {weights_given}")
            print(f"The length of given weights is {len(weights_given)}")

        # self.model.eval()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            # # 从self.model拷贝下来cur_model，并设置为train模式
            # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))
            # cur_model = self.model
            # cur_model = copy.deepcopy(self.model)
            # # cur_model.train()

            # 每次load的时候只load了w_original和b_original，无需load整个模型的参数，也无需做磁盘io了
            cur_model = self.model
            cur_model.decoder.projection.weight.set_(w_original)
            cur_model.decoder.projection.bias.set_(b_original)


            # tmp loss
            cur_model.eval()
            pred, true = self._process_one_batch_with_model(cur_model, test_data,
                batch_x[:, -self.args.seq_len:, :], batch_y, 
                batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)
            # 获取adaptation之前的loss
            loss_before_adapt = criterion(pred, true)
            a1.append(loss_before_adapt.item())
            # print(f"loss_before_adapt: {loss_before_adapt}")
            # print(f'mse:{tmp_loss}')

            # cur_model.train()

            gradients = []

            mean_loss = 0
            mid_embedding_list = []
            weight_list = []
            y_list = []

            seq_len = self.args.seq_len
            label_len = self.args.label_len
            pred_len = self.args.pred_len

            x_stacked_list, y_stacked_list, x_mark_stacked_list, y_mark_stacked_list = [], [], [], []

            # prepare params for analytical solution
            for ii in range(self.test_train_num):
                x_stacked_list.append(batch_x[:, ii : ii+seq_len, :].squeeze(0))
                y_stacked_list.append(batch_x[:, ii+seq_len-label_len : ii+seq_len+pred_len, :].squeeze(0))
                x_mark_stacked_list.append(batch_x_mark[:, ii : ii+seq_len, :].squeeze(0))
                y_mark_stacked_list.append(batch_x_mark[:, ii+seq_len-label_len : ii+seq_len+pred_len, :].squeeze(0))

            # x_stacked维度为(ttn, seq_len, F)
            x_stacked, y_stacked = torch.stack(x_stacked_list), torch.stack(y_stacked_list)
            x_mark_stacked, y_mark_stacked = torch.stack(x_mark_stacked_list), torch.stack(y_mark_stacked_list)


            # 获得中间变量mid_embedding等信息
            # 并行化后，pred和true的维度为(ttn, pred_len, F)，mid_embedding的维度为(ttn, pred_len, mid_dim)
            pred, true, all_mid_embedding = self._process_one_batch_with_model(cur_model, test_data,
                x_stacked, y_stacked, 
                x_mark_stacked, y_mark_stacked,
                return_mid_embedding=True)

            # # 获得中间变量mid_embedding等信息
            # pred, true, mid_embedding = self._process_one_batch_with_model(cur_model, test_data,
            #     batch_x[:, ii : ii+seq_len, :], batch_x[:, ii+seq_len-label_len : ii+seq_len+pred_len, :], 
            #     batch_x_mark[:, ii : ii+seq_len, :], batch_x_mark[:, ii+seq_len-label_len : ii+seq_len+pred_len, :],
            #     return_mid_embedding=True)


            # 并行化计算完后再将mid_embedding拆出来
            for ii in range(self.test_train_num):
                # mid_embedding = mid_embedding.squeeze(0)  # L*H
                mid_embedding = all_mid_embedding[ii]  # L*H，注意这里要取出all_mid_embedding的第ii维

                ones = torch.ones(mid_embedding.shape[-2]).unsqueeze(1).to(self.device)  # 在最后面加入一列1进来
                mid_embedding = torch.cat((mid_embedding, ones), 1)  # L*(H+1)

                mid_embedding_list.extend(mid_embedding.unbind())  # 包含L项的list，每个形状为(H+1)维

                length = mid_embedding.shape[0]  # 取出L，这L项数据的weight均为：1 / (N - ii)**alpha
                if weights_given:
                    weights = [weights_given[ii]] * length
                else:
                    weights = [(self.test_train_num - ii) ** (-self.args.alpha)] * length
                weight_list.extend(weights)  # 将权重值放入

                # y_value = true.squeeze(0)  # L*F
                y_value = true[ii]  # L*F，注意这里也要取出true的第ii维
                y_value = y_value.float().to(self.device)
                # y_list.append(true)
                y_list.extend(y_value.unbind())
            

            # # calculation a2
            # for ii in range(self.test_train_num):
            #     seq_len = self.args.seq_len
            #     label_len = self.args.label_len
            #     pred_len = self.args.pred_len

            #     pred, true = self._process_one_batch_with_model(cur_model, test_data,
            #         batch_x[:, ii : ii+seq_len, :], batch_x[:, ii+seq_len-label_len : ii+seq_len+pred_len, :], 
            #         batch_x_mark[:, ii : ii+seq_len, :], batch_x_mark[:, ii+seq_len-label_len : ii+seq_len+pred_len, :])

            #     loss = criterion(pred, true)
            #     mean_loss += loss
            # mean_loss = mean_loss / self.test_train_num
            # a2.append(mean_loss.item())


            # w_T = params[0].T  # H*F
            # b = params[1].unsqueeze(0)

            # 改成从w_original和b_original中加载w和b的参数
            w_T = w_original.T  # H*F
            b = b_original.unsqueeze(0)
            
            W_orig = torch.cat((w_T, b), 0)  # (H+1)*F

            x_0 = torch.stack(mid_embedding_list)  # (N*L), (H+1)
            y_label = torch.stack(y_list)  # (N*L), F
            y_pred = x_0 @ W_orig
            loss_0 = criterion(y_pred, y_label)
            a2.append(loss_0.item())

            # t0 = time.time()
            W_adapted = self.calc_analytical_solution(mid_embedding_list, y_list, weight_list, W_orig)
            # print(f"calc_analytical_solution, time={time.time() - t0}")

            y_pred = x_0 @ W_adapted
            loss_1 = criterion(y_pred, y_label)
            a3.append(loss_1.item())

            # print(f"loss_0: {loss_0}, loss_1:{loss_1}")

            w_T_adapted, b_adapted = W_adapted[:-1, :], W_adapted[-1:, :]
            w_T_adapted = w_T_adapted.T
            b_adapted = b_adapted.squeeze(0)
            
            cur_model.decoder.projection.weight.set_(w_T_adapted)
            cur_model.decoder.projection.bias.set_(b_adapted)
            
            cur_model.eval()

            if use_adapted_model:
                pred, true = self._process_one_batch_with_model(cur_model, test_data,
                    batch_x[:, -self.args.seq_len:, :], batch_y, 
                    batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)
            else:
                # pred, true = self._process_one_batch_with_model(self.model, test_data,
                #     batch_x[:, -self.args.seq_len:, :], batch_y, 
                #     batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)
                pred, true = self._process_one_batch_with_model(cur_model, test_data,
                    batch_x[:, -self.args.seq_len:, :], batch_y, 
                    batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)

            # 获取adaptation之后的loss
            loss_after_adapt = criterion(pred, true)
            a4.append(loss_after_adapt.item())


            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())


            if (i+1) % 100 == 0:
                print("\titers: {0}, cost time: {1}s".format(i + 1, time.time() - test_time_start))
                print(gradients)
                tmp_p = np.array(preds); tmp_p = tmp_p.reshape(-1, tmp_p.shape[-2], tmp_p.shape[-1])
                tmp_t = np.array(trues); tmp_t = tmp_t.reshape(-1, tmp_t.shape[-2], tmp_t.shape[-1])
                tmp_mae, tmp_mse, _, _, _ = metric(tmp_p, tmp_t)
                print('mse:{}, mae:{}'.format(tmp_mse, tmp_mae))

                avg_1, avg_2, avg_3, avg_4 = 0, 0, 0, 0
                num = len(a1)
                for iii in range(num):
                    avg_1 += a1[iii]; avg_2 += a2[iii]; avg_3 += a3[iii]; avg_4 += a4[iii]
                avg_1 /= num; avg_2 /= num; avg_3 /= num; avg_4 /= num
                print(avg_1, avg_2, avg_3, avg_4)

            cur_model.eval()
            # cur_model.cpu()
            # del cur_model
            torch.cuda.empty_cache()

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        print(f"Test - cost time: {time.time() - test_time_start}s")

        return mse, mae


    def get_grads(self, setting, test=0, is_training_part_params=True, use_adapted_model=True, test_train_epochs=1, flag='test', adapted_degree="small"):
        if flag == "test":
            test_data, test_loader = self._get_data_at_test_time(flag='test')
        elif flag == "val":
            # 这里用需要用'val_with_batchsize_1'来获取数据
            test_data, test_loader = self._get_data_at_test_time(flag='val')
        
        if test:
            print('loading model from checkpoint !!!')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        self.model.eval()

        preds = []
        trues = []

        # mapping_model = self.learn_mapping_model_during_vali(is_training_part_params)
        # mapping_model.eval()

        a1, a2, a3 = [], [], []

        error_per_pred_index = [[] for i in range(self.args.pred_len)]

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()  # 如果使用amp的话，还需要再生成一个scaler？

        criterion = nn.MSELoss()  # 使用MSELoss
        test_time_start = time.time()

        # 先加载模型参数
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        # self.model.eval()
        first_grad = []
        all_grads = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            # 从self.model拷贝下来cur_model，并设置为train模式
            cur_model = copy.deepcopy(self.model)
            # cur_model.train()
            cur_model.eval()

            if is_training_part_params:
                params = []
                names = []
                cur_model.requires_grad_(False)
                for n_m, m in cur_model.named_modules():
                    # 注意：这里的名字应当根据Autoformer模型而修改为"decoder.projection"
                    
                    # if "decoder.projection" == n_m:
                    if "decoder.projection" in n_m:
                        m.requires_grad_(True)
                        for n_p, p in m.named_parameters():
                            if n_p in ['weight', 'bias']:  # weight is scale, bias is shift
                                params.append(p)
                                names.append(f"{n_m}.{n_p}")

                # Adam优化器
                # model_optim = optim.Adam(params, lr=self.args.learning_rate*10 / (2**self.test_train_num))  # 使用Adam优化器
                # lr = self.args.learning_rate * 1
                lr = self.args.learning_rate * self.args.adapted_lr_times
                # model_optim = optim.Adam(params, lr=lr)  # 使用Adam优化器
                # model_optim_norm = optim.Adam(params_norm, lr=self.args.learning_rate*1000 / (2**self.test_train_num))  # 使用Adam优化器
                
                # 普通的SGD优化器？
                model_optim = optim.SGD(params, lr=lr)
            else:
                cur_model.requires_grad_(True)
                # model_optim = optim.Adam(cur_model.parameters(), lr=self.args.learning_rate*10 / (2**self.test_train_num))
                # model_optim = optim.Adam(cur_model.parameters(), lr=self.args.learning_rate)
                lr = self.args.learning_rate * self.args.adapted_lr_times
                model_optim = optim.SGD(cur_model.parameters(), lr=lr)
        

            cur_grad_list = []

            # # tmp loss
            # cur_model.eval()
            # pred, true = self._process_one_batch_with_model(cur_model, test_data,
            #     batch_x[:, -self.args.seq_len:, :], batch_y, 
            #     batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)
            # # 获取adaptation之前的loss
            # loss_before_adapt = criterion(pred, true)
            # a1.append(loss_before_adapt.item())
            # # print(f"loss_before_adapt: {loss_before_adapt}")
            # # print(f'mse:{tmp_loss}')
            # cur_model.train()


            # tmp loss
            # cur_model.train()
            pred, true = self._process_one_batch_with_model(cur_model, test_data,
                batch_x[:, -self.args.seq_len:, :], batch_y, 
                batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)
            # 获取adaptation之前的loss
            loss_before_adapt = criterion(pred, true)

            # 计算在测试样本上的梯度
            loss_before_adapt.backward()
            w_T = params[0].grad.T
            b = params[1].grad.unsqueeze(0)
            params_tmp = torch.cat((w_T, b), 0)
            params_tmp = params_tmp.ravel()

            # 将该梯度存入cur_grad_list中
            cur_grad_list.append(params_tmp.detach().cpu().numpy())

            model_optim.zero_grad()

            a1.append(loss_before_adapt.item())
            # print(f"loss_before_adapt: {loss_before_adapt}")
            # print(f'mse:{tmp_loss}')
            # cur_model.train()



            params_adapted = torch.zeros((1)).to(self.device)
            # 开始训练
            # cur_lr = self.args.learning_rate*10 / (2**self.test_train_num)
            for epoch in range(test_train_epochs):
                # cur_lr = self.args.learning_rate*10 / (2**self.test_train_num)

                gradients = []
                accpted_samples_num = set()

                import random
                is_random = False
                # is_random = True
                sample_order_list = list(range(self.test_train_num))
                # print("before random, sample_order_list is: ", sample_order_list)
                if is_random:
                    random.shuffle(sample_order_list)
                    # print("after random, sample_order_list is: ", sample_order_list)
                else:
                    # print("do not use random.")
                    pass

                # 新加了参数num_of_loss_per_update，表示我们将多个loss平均起来做一次更新
                num_of_loss_per_update = self.test_train_num
                # num_of_loss_per_update = 1
                mean_loss = 0

                # 将第一个梯度的情况分开考虑
                if i == 0:
                    for ii in sample_order_list:

                        model_optim.zero_grad()

                        seq_len = self.args.seq_len
                        label_len = self.args.label_len
                        pred_len = self.args.pred_len

                        pred, true = self._process_one_batch_with_model(cur_model, test_data,
                            batch_x[:, ii : ii+seq_len, :], batch_x[:, ii+seq_len-label_len : ii+seq_len+pred_len, :], 
                            batch_x_mark[:, ii : ii+seq_len, :], batch_x_mark[:, ii+seq_len-label_len : ii+seq_len+pred_len, :])

                        # 这里当batch_size为1还是32时
                        # pred和true的size可能为[1, 24, 7]或[32, 24, 7]
                        # 但是结果的loss值均只包含1个值
                        # 这是因为criterion为MSELoss，其默认使用mean模式，会对32个loss值取一个平均值
                        
                        loss = criterion(pred, true)
                        # loss = decay_MAE(pred, true)

                        # loss = criterion(pred, true)
                        mean_loss += loss

                        if self.args.use_amp:
                            scaler.scale(loss).backward()
                            scaler.step(model_optim)
                            # scaler.step(model_optim_norm)
                            scaler.update()
                        else:
                            # # pass
                            # loss.backward()
                            # model_optim.step()

                            loss.backward()

                            w_T = params[0].grad.T
                            b = params[1].grad.unsqueeze(0)
                            params_tmp = torch.cat((w_T, b), 0)
                            params_tmp = params_tmp.ravel()
                            # params_adapted = params_adapted + params_tmp

                            cur_grad_list.append(params_tmp.detach().cpu().numpy())

                            model_optim.zero_grad()

                            # # 计算标准答案的梯度params_answer和adaptation中的梯度params_tmp之间的角度
                            # import math
                            # # print()
                            # # product = torch.dot(params_tmp, params_answer)
                            # # product = product / (torch.norm(params_tmp) * torch.norm(params_answer))
                            # product = torch.dot(params_tmp, mapping_answer)
                            # product = product / (torch.norm(params_tmp) * torch.norm(mapping_answer))
                            # angel = math.degrees(math.acos(product))
                            # if angel < 90:
                            #     model_optim.step()
                            # else:
                            #     model_optim.zero_grad()

                        # 记录逐样本做了adaptation之后的loss
                        # mean_loss += tmp_loss
                        # mean_loss += loss
                    
                    # 将第一个的梯度放到first_grad里
                    first_grad.append(cur_grad_list)

                else:
                    # 只需要对最后一个做了
                    for ii in [self.test_train_num-1]:

                        model_optim.zero_grad()

                        seq_len = self.args.seq_len
                        label_len = self.args.label_len
                        pred_len = self.args.pred_len

                        pred, true = self._process_one_batch_with_model(cur_model, test_data,
                            batch_x[:, ii : ii+seq_len, :], batch_x[:, ii+seq_len-label_len : ii+seq_len+pred_len, :], 
                            batch_x_mark[:, ii : ii+seq_len, :], batch_x_mark[:, ii+seq_len-label_len : ii+seq_len+pred_len, :])
                        
                        loss = criterion(pred, true)

                        mean_loss += loss

                        if self.args.use_amp:
                            scaler.scale(loss).backward()
                            scaler.step(model_optim)
                            # scaler.step(model_optim_norm)
                            scaler.update()
                        else:
                            # # pass
                            # loss.backward()
                            # model_optim.step()

                            loss.backward()

                            w_T = params[0].grad.T
                            b = params[1].grad.unsqueeze(0)
                            params_tmp = torch.cat((w_T, b), 0)
                            params_tmp = params_tmp.ravel()
                            # params_adapted = params_adapted + params_tmp

                            cur_grad_list.append(params_tmp.detach().cpu().numpy())

                            model_optim.zero_grad()
                    
                    # 将剩余梯度放到all_grads中
                    all_grads.append(cur_grad_list)

            mean_loss = mean_loss / self.test_train_num
            a2.append(mean_loss.item())
            
            # mean_loss.backward()
            # model_optim.step()

            # all_grads.append(cur_grad_list)
            
            cur_model.eval()


            # # 计算标准答案的梯度params_answer和adaptation中的梯度params_adapted之间的角度
            # import math
            # product = torch.dot(params_adapted, params_answer)
            # product = product / (torch.norm(params_adapted) * torch.norm(params_answer))
            # angel = math.degrees(math.acos(product))

            # with open(f"./grad_angels/{setting}", "a") as f:
            #     f.write(f"{angel}" + "\n")

            cur_model.eval()

            if use_adapted_model:
                pred, true = self._process_one_batch_with_model(cur_model, test_data,
                    batch_x[:, -self.args.seq_len:, :], batch_y, 
                    batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)
            else:
                # pred, true = self._process_one_batch_with_model(self.model, test_data,
                #     batch_x[:, -self.args.seq_len:, :], batch_y, 
                #     batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)
                pred, true = self._process_one_batch_with_model(cur_model, test_data,
                    batch_x[:, -self.args.seq_len:, :], batch_y, 
                    batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)

            # 获取adaptation之后的loss
            loss_after_adapt = criterion(pred, true)
            a3.append(loss_after_adapt.item())


            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

            # 对预测结果（如长度为24）中的每个位置/index上的结果分别进行统计
            pred_len = self.args.pred_len
            for index in range(pred_len):
                cur_pred = pred.detach().cpu().numpy()[0][index]
                cur_true = true.detach().cpu().numpy()[0][index]
                cur_error = np.mean((cur_pred - cur_true) ** 2)
                error_per_pred_index[index].append(cur_error)


            if (i+1) % 100 == 0:
                print("\titers: {0}, cost time: {1}s".format(i + 1, time.time() - test_time_start))
                print(gradients)
                tmp_p = np.array(preds); tmp_p = tmp_p.reshape(-1, tmp_p.shape[-2], tmp_p.shape[-1])
                tmp_t = np.array(trues); tmp_t = tmp_t.reshape(-1, tmp_t.shape[-2], tmp_t.shape[-1])
                tmp_mae, tmp_mse, _, _, _ = metric(tmp_p, tmp_t)
                print('mse:{}, mae:{}'.format(tmp_mse, tmp_mae))
                print(type(all_grads), len(all_grads), type(np.array(all_grads)), np.array(all_grads).shape)
                
                # avg_1, avg_2, avg_3 = 0, 0, 0
                # num = len(a1)
                # for iii in range(num):
                #     avg_1 += a1[iii]; avg_2 += a2[iii]; avg_3 += a3[iii]
                # avg_1 /= num; avg_2 /= num; avg_3 /= num
                # print(avg_1, avg_2, avg_3)

            cur_model.eval()
            # cur_model.cpu()
            del cur_model
            torch.cuda.empty_cache()


        # # 将adaptation前的loss、adaptation中逐样本做adapt后的loss、以及adaptation之后的loss做统计
        # with open(f"./loss_before_and_after_adapt/{setting}.txt", "a") as f:
        #     for i in range(len(a1)):
        #         t1, t2, t3 = a1[i], a2[i], a3[i]
        #         f.write(f"{t1}, {t2}, {t3}\n")


        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))


        # 保存npy文件和最后的权重文件的路径
        weight_path = "./grads_npy/" + setting + "/"
        if not os.path.exists(weight_path):
            os.makedirs(weight_path)


        # # 在all_grads中随机抽取1%并存入random_sampled_grads中
        # from random import sample
        # # sample_rate = 0.01
        # sample_rate = 0.1
        # # sample_rate = 1
        # sample_num = int(sample_rate * len(all_grads))
        # random_sampled_grads = sample(all_grads, sample_num)

        # random_sampled_grads = np.array(random_sampled_grads)
        # np.save(f"{weight_path}grads_{flag}_{adapted_degree}_ttn{self.test_train_num}_sample{sample_rate:.2f}.npy", random_sampled_grads)

        # print(first_grad[0].shape)
        # print(all_grads[0].shape)

        first_grad = np.array(first_grad)
        all_grads = np.array(all_grads)
        print(first_grad.shape)
        print(all_grads.shape)

        sample_rate = 1
        npz_file_name = f"{weight_path}grads_{flag}_{adapted_degree}_ttn{self.test_train_num}_sample{sample_rate:.2f}.npz"

        # 将first_grad和all_grads保存为npz文件
        np.savez(npz_file_name, first_grad, all_grads)

        
        npzfile = np.load(npz_file_name)
        first_grad = npzfile["arr_0"]
        all_grads = npzfile["arr_1"]

        grad_X, grad_Y = [], []
        # 先加载fisrt_grad
        grad_Y.append(first_grad[0, 0:1, :])  # (1, 3591)
        grad_X.append(first_grad[0, 1:, :])  # (10, 3591)

        # 再加载all_grads
        for i in range(len(all_grads)):
            cur_grad = all_grads[i]
            grad_Y.append(cur_grad[0:1])  # (1, 3591)
            # grad_X[i][1:, :] is (9, 3591), cur_grad[1:] is (1, 3591), 合并后为(10, 3591)
            grad_X.append(np.concatenate((grad_X[i][1:, :], cur_grad[1:]), axis=0))
        
        grad_X = np.array(grad_X)
        grad_Y = np.array(grad_Y)
        print(grad_X.shape)  # (2785, 10, 512*7)
        print(grad_Y.shape)  # (2785, 1, 512*7)


        if adapted_degree == "small":  # 对每个adaptation的梯度共用一个权重
            # grad_X = all_grads[:, 1:, :]  # (2785, 10, 512*7)
            # grad_Y = all_grads[:, 0:1, :]  # (2785, 1, 512*7)
            grad_X = grad_X.transpose(0, 2, 1)
            grad_Y = grad_Y.transpose(0, 2, 1)
            grad_X = grad_X.reshape(-1, grad_X.shape[-1])
            grad_Y = grad_Y.reshape(-1, grad_Y.shape[-1])

            grad_X_mean = grad_X.mean(axis=1)
        
        elif adapted_degree == "large":  # 对每个adaptation梯度，向量中的每个值均有一个权重
            # grad_X_orig = all_grads[:, 1:, :]  # (2785, 10, 512*7)
            # grad_Y_orig = all_grads[:, 0:1, :]  # (2785, 1, 512*7)
            # # grad_X = grad_X.reshape(-1, grad_X.shape[-1])
            # # grad_Y = grad_Y.reshape(-1, grad_Y.shape[-1])
            # grad_X = grad_X_orig.reshape(grad_X_orig.shape[0], -1)  # (2785, 10*512*7)
            # grad_Y = grad_Y_orig.reshape(grad_Y_orig.shape[0], -1)  # (2785, 1*512*7)

            # # 注意是对grad_X_orig做mean
            # grad_X_mean = grad_X_orig.mean(axis=1)  # (2785, 512*7)
            pass

        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error as MSE
        LR_model = LinearRegression(fit_intercept=False)
        LR_model.fit(grad_X, grad_Y)

        yhat = LR_model.predict(grad_X)

        # print(f"score of grad_X and grad_Y is: {LR_model.score(grad_X, grad_Y)}")
        print(f"the coef is: {LR_model.coef_}")
        print(f"the intercept is: {LR_model.intercept_}")

        mse_predict = MSE(yhat, grad_Y)
        mse_mean = MSE(grad_X_mean, grad_Y)
        print(f"mse_predict between yhat and grad_Y is: {mse_predict}")
        print(f"mse_mean between grad_X_mean and grad_Y is: {mse_mean}")
        print(f"grad_Y.mean: {grad_Y.mean()}, yhat.mean: {yhat.mean()}")

        weights = LR_model.coef_[0].tolist()

        
        # if flag == "test":
        #     weight_file = f"{weight_path}weights_{flag}_ttn{self.test_train_num}.txt"
        # elif flag == "val":
        #     weight_file = f"{weight_path}weights_{flag}_ttn{self.test_train_num}.txt"

        weight_file = f"{weight_path}weights_{flag}_{adapted_degree}_ttn{self.test_train_num}.txt"
        
        # with open(weight_file, "w") as f:
        #     for i in range(len(weights)-1):
        #         f.write(f"{weights[i]}, ")
        #     f.write(f"{weights[-1]}")


        print(f"Test - cost time: {time.time() - test_time_start}s")

        # return a1, a2, a3
        return



    def test_with_mapping_model(self, setting, test=0, is_training_part_params=True, use_adapted_model=True, test_train_epochs=1):
        test_data, test_loader = self._get_data_at_test_time(flag='test')
        if test:
            print('loading model from checkpoint !!!')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        self.model.eval()

        preds = []
        trues = []

        mapping_model = self.learn_mapping_model_during_vali(is_training_part_params)
        mapping_model.eval()

        a1, a2, a3 = [], [], []

        error_per_pred_index = [[] for i in range(self.args.pred_len)]

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()  # 如果使用amp的话，还需要再生成一个scaler？

        criterion = nn.MSELoss()  # 使用MSELoss
        test_time_start = time.time()

        def MAE(pred, true):
            return torch.mean(torch.abs(pred - true))
        def MSE(pred, true):
            return torch.mean((pred - true) ** 2)
        def decay_MSE(pred, true):
            dim = pred.shape[1]  # 获取L维
            loss = 0
            for i in range(dim):
                loss += (1 / (i+1)) * torch.mean((pred[:,i,:] - true[:,i,:]) ** 2)
            loss /= dim
            return loss
        def decay_MAE(pred, true):
            dim = pred.shape[1]  # 获取L维
            loss = 0
            for i in range(dim):
                loss += (1 / math.sqrt(i+1)) * torch.mean(torch.abs(pred[:,i,:] - true[:,i,:]))
            loss /= dim
            return loss

        # # result save
        # folder_path = './test_results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        # self.model.eval()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            # 从self.model拷贝下来cur_model，并设置为train模式
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))
            cur_model = copy.deepcopy(self.model)
            cur_model.train()

            if is_training_part_params:
                params = []
                names = []
                params_norm = []
                names_norm = []
                cur_model.requires_grad_(False)
                for n_m, m in cur_model.named_modules():
                    # 注意：这里的名字应当根据Autoformer模型而修改为"decoder.projection"
                    
                    # if "decoder.projection" == n_m:
                    if "decoder.projection" in n_m:
                        m.requires_grad_(True)
                        for n_p, p in m.named_parameters():
                            if n_p in ['weight', 'bias']:  # weight is scale, bias is shift
                                params.append(p)
                                names.append(f"{n_m}.{n_p}")

                # Adam优化器
                # model_optim = optim.Adam(params, lr=self.args.learning_rate*10 / (2**self.test_train_num))  # 使用Adam优化器
                # lr = self.args.learning_rate * 1
                lr = self.args.learning_rate * self.args.adapted_lr_times
                # model_optim = optim.Adam(params, lr=lr)  # 使用Adam优化器
                # model_optim_norm = optim.Adam(params_norm, lr=self.args.learning_rate*1000 / (2**self.test_train_num))  # 使用Adam优化器
                
                # 普通的SGD优化器？
                model_optim = optim.SGD(params, lr=lr)
            else:
                cur_model.requires_grad_(True)
                # model_optim = optim.Adam(cur_model.parameters(), lr=self.args.learning_rate*10 / (2**self.test_train_num))
                # model_optim = optim.Adam(cur_model.parameters(), lr=self.args.learning_rate)
                lr = self.args.learning_rate * self.args.adapted_lr_times
                model_optim = optim.SGD(cur_model.parameters(), lr=lr)
            

            # tmp loss
            cur_model.eval()
            pred, true = self._process_one_batch_with_model(cur_model, test_data,
                batch_x[:, -self.args.seq_len:, :], batch_y, 
                batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)
            # 获取adaptation之前的loss
            loss_before_adapt = criterion(pred, true)
            a1.append(loss_before_adapt.item())
            # print(f"loss_before_adapt: {loss_before_adapt}")
            # print(f'mse:{tmp_loss}')
            cur_model.train()

            

            # tmp_train_X_sample = torch.cat((batch_x[:, -self.args.seq_len:, :], batch_x_mark[:, -self.args.seq_len:, :]), 2).ravel()
            # tmp_train_X_sample = tmp_train_X_sample.to(torch.float32).to(self.device)
            # mapping_answer = mapping_model(tmp_train_X_sample)
            # # mapping_answer = mapping_answer.reshape()
            # # print(mapping_answer)
            

            # 先用原模型的预测值和标签值之间的error，做反向传播之后得到的梯度值gradient_0
            # 并将这个gradient_0作为标准答案
            # 然后，对测试样本做了adaptation之后，会得到一个gradient_1
            # 那么对gradient_1和gradient_0之间做对比，
            # 就可以得到二者之间的余弦值是多少（方向是否一致），以及长度上相差的距离有多少等等。
            # params_answer = self.get_answer_grad(is_training_part_params, use_adapted_model,
            #                                         lr, test_data, 
            #                                         batch_x, batch_y, batch_x_mark, batch_y_mark,
            #                                         setting)


            params_adapted = torch.zeros((1)).to(self.device)
            # 开始训练
            # cur_lr = self.args.learning_rate*10 / (2**self.test_train_num)
            # cur_lr_norm = self.args.learning_rate*1000 / (2**self.test_train_num)
            for epoch in range(test_train_epochs):
                # cur_lr = self.args.learning_rate*10 / (2**self.test_train_num)
                # cur_lr_norm = self.args.learning_rate*1000 / (2**self.test_train_num)

                gradients = []
                accpted_samples_num = set()

                # for pass_num in range(2):
                #     if pass_num == 1:  # the second pass
                #         cmp = lambda lst: torch.norm(lst[1], lst[2])
                #         gradients.sort(key=cmp)
                #         half_num = self.test_train_num // 2
                #         # just get half of samples with smaller gradients
                #         gradients = gradients[:-half_num or None]
                #         for grad in gradients:
                #             accpted_samples_num.add(grad[0])
                

                import random
                is_random = False
                # is_random = True
                sample_order_list = list(range(self.test_train_num))
                # print("before random, sample_order_list is: ", sample_order_list)
                if is_random:
                    random.shuffle(sample_order_list)
                    # print("after random, sample_order_list is: ", sample_order_list)
                else:
                    # print("do not use random.")
                    pass

                # 新加了参数num_of_loss_per_update，表示我们将多个loss平均起来做一次更新
                num_of_loss_per_update = self.test_train_num
                # num_of_loss_per_update = 1
                mean_loss = 0


                # # 验证adaptation结果
                # pred, true = self._process_one_batch_with_model(cur_model, test_data,
                #     batch_x[:, -self.args.seq_len:, :], batch_y, 
                #     batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)
                
                # loss = criterion(pred, true)
                # mean_loss += loss

                # loss.backward()
                # model_optim.step()



                for ii in sample_order_list:
                    # if pass_num == 1 and ii not in accpted_samples_num:
                    #     continue

                    # if not ((self.test_train_num - 1 - ii) + self.args.pred_len) % 96 == 0:
                    #     continue

                    model_optim.zero_grad()

                    seq_len = self.args.seq_len
                    label_len = self.args.label_len
                    pred_len = self.args.pred_len

                    # batch_x.requires_grad = True
                    # batch_x_mark.requires_grad = True

                    pred, true = self._process_one_batch_with_model(cur_model, test_data,
                        batch_x[:, ii : ii+seq_len, :], batch_x[:, ii+seq_len-label_len : ii+seq_len+pred_len, :], 
                        batch_x_mark[:, ii : ii+seq_len, :], batch_x_mark[:, ii+seq_len-label_len : ii+seq_len+pred_len, :])

                    # 这里当batch_size为1还是32时
                    # pred和true的size可能为[1, 24, 7]或[32, 24, 7]
                    # 但是结果的loss值均只包含1个值
                    # 这是因为criterion为MSELoss，其默认使用mean模式，会对32个loss值取一个平均值
                    
                    loss = criterion(pred, true)
                    # loss = decay_MAE(pred, true)

                    # from functorch import vmap
                    # from functorch.experimental import replace_all_batch_norm_modules_
                    # replace_all_batch_norm_modules_(cur_model)
                    # tmp_batch_x = batch_x.unsqueeze(1)
                    # tmp_batch_x_mark = batch_x_mark.unsqueeze(1)
                    # vmap_func = vmap(self._process_one_batch_with_model, 
                    #                  in_dims=(None, None, 0, 0, 0, 0), out_dims=(0, 0), 
                    #                  randomness='different')
                    # pred, true = vmap_func(cur_model, test_data,
                    #     tmp_batch_x[:, :, ii : ii+seq_len, :], tmp_batch_x[:, :, ii+seq_len-label_len : ii+seq_len+pred_len, :], 
                    #     tmp_batch_x_mark[:, :, ii : ii+seq_len, :], tmp_batch_x_mark[:, :, ii+seq_len-label_len : ii+seq_len+pred_len, :])


                    # loss = criterion(pred, true)
                    mean_loss += loss

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        # scaler.step(model_optim_norm)
                        scaler.update()
                    else:
                        # pass
                        loss.backward()
                        model_optim.step()

                        # # 多个损失合并起来做一次更新
                        # if (ii+1) % num_of_loss_per_update == 0:
                        #     mean_loss = mean_loss / num_of_loss_per_update
                        #     mean_loss.backward()
                        #     model_optim.step()
                        #     mean_loss = 0
                        # elif (ii+1) == len(sample_order_list):
                        #     mean_loss = mean_loss / (len(sample_order_list) % num_of_loss_per_update)
                        #     mean_loss.backward()
                        #     model_optim.step()
                        #     mean_loss = 0
                        # else:
                        #     pass

                        # w_T = params[0].grad.T
                        # b = params[1].grad.unsqueeze(0)
                        # params_tmp = torch.cat((w_T, b), 0)
                        # params_tmp = params_tmp.ravel()
                        # params_adapted = params_adapted + params_tmp

                        # # 计算标准答案的梯度params_answer和adaptation中的梯度params_tmp之间的角度
                        # import math
                        # # print()
                        # # product = torch.dot(params_tmp, params_answer)
                        # # product = product / (torch.norm(params_tmp) * torch.norm(params_answer))
                        # product = torch.dot(params_tmp, mapping_answer)
                        # product = product / (torch.norm(params_tmp) * torch.norm(mapping_answer))
                        # angel = math.degrees(math.acos(product))
                        # if angel < 90:
                        #     model_optim.step()
                        # else:
                        #     model_optim.zero_grad()


                        # if pass_num == 0:
                        #     w_grad = torch.norm(params[0].grad)
                        #     b_grad = torch.norm(params[1].grad)
                        #     gradients.append([ii, w_grad, b_grad])

                        #     with open(f"./logs_loss_and_grad/{setting}", "a") as f:
                        #         f.write(f"{loss}, {torch.norm(w_grad, b_grad)}" + "\n")
                            
                        # elif pass_num == 1:

                        # model_optim.step()
                        
                        # model_optim_norm.step()

                        # w_T = params[0].grad.T  # 先对weight参数做转置
                        # b = params[1].grad.unsqueeze(0)  # 将bias参数扩展一维
                        # params_grad = torch.cat((w_T, b), 0)  # 将w_T和b参数concat起来
                        # params_grad = params_grad.ravel()  # 最后再展开成一维的

                        # with open(f"./logs_loss_and_grad/batch{self.args.adapted_batch_size}_{setting}", "a") as f:
                        #     f.write(f"{loss}, {torch.norm(params_grad)}" + "\n")

                    # cur_model.eval()
                    # tmp_pred, tmp_true = self._process_one_batch_with_model(cur_model, test_data,
                    #     batch_x[:, ii : ii+seq_len, :], batch_x[:, ii+seq_len-label_len : ii+seq_len+pred_len, :], 
                    #     batch_x_mark[:, ii : ii+seq_len, :], batch_x_mark[:, ii+seq_len-label_len : ii+seq_len+pred_len, :])
                    # tmp_loss = criterion(tmp_pred, tmp_true)
                    # cur_model.train()

                    # 记录逐样本做了adaptation之后的loss
                    # mean_loss += tmp_loss
                    # mean_loss += loss


                    # cur_lr = cur_lr * 2
                    # for param_group in model_optim.param_groups:
                    #     param_group['lr'] = cur_lr
                    
                    # cur_lr_norm = cur_lr_norm * 2
                    # for param_group in model_optim_norm.param_groups:
                    #     param_group['lr'] = cur_lr_norm
                

            mean_loss = mean_loss / self.test_train_num
            a2.append(mean_loss.item())
            
            # mean_loss.backward()
            # model_optim.step()
            
            cur_model.eval()


            # # 计算标准答案的梯度params_answer和adaptation中的梯度params_adapted之间的角度
            # import math
            # product = torch.dot(params_adapted, params_answer)
            # product = product / (torch.norm(params_adapted) * torch.norm(params_answer))
            # angel = math.degrees(math.acos(product))

            # with open(f"./grad_angels/{setting}", "a") as f:
            #     f.write(f"{angel}" + "\n")

            cur_model.eval()

            if use_adapted_model:
                pred, true = self._process_one_batch_with_model(cur_model, test_data,
                    batch_x[:, -self.args.seq_len:, :], batch_y, 
                    batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)
            else:
                # pred, true = self._process_one_batch_with_model(self.model, test_data,
                #     batch_x[:, -self.args.seq_len:, :], batch_y, 
                #     batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)
                pred, true = self._process_one_batch_with_model(cur_model, test_data,
                    batch_x[:, -self.args.seq_len:, :], batch_y, 
                    batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)

            # 获取adaptation之后的loss
            loss_after_adapt = criterion(pred, true)
            a3.append(loss_after_adapt.item())


            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

            # 对预测结果（如长度为24）中的每个位置/index上的结果分别进行统计
            pred_len = self.args.pred_len
            for index in range(pred_len):
                cur_pred = pred.detach().cpu().numpy()[0][index]
                cur_true = true.detach().cpu().numpy()[0][index]
                cur_error = np.mean((cur_pred - cur_true) ** 2)
                error_per_pred_index[index].append(cur_error)


            if (i+1) % 100 == 0:
                print("\titers: {0}, cost time: {1}s".format(i + 1, time.time() - test_time_start))
                print(gradients)
                tmp_p = np.array(preds); tmp_p = tmp_p.reshape(-1, tmp_p.shape[-2], tmp_p.shape[-1])
                tmp_t = np.array(trues); tmp_t = tmp_t.reshape(-1, tmp_t.shape[-2], tmp_t.shape[-1])
                tmp_mae, tmp_mse, _, _, _ = metric(tmp_p, tmp_t)
                print('mse:{}, mae:{}'.format(tmp_mse, tmp_mae))
                
                avg_1, avg_2, avg_3 = 0, 0, 0
                num = len(a1)
                for iii in range(num):
                    avg_1 += a1[iii]; avg_2 += a2[iii]; avg_3 += a3[iii]
                avg_1 /= num; avg_2 /= num; avg_3 /= num
                print(avg_1, avg_2, avg_3)

            # if i % 20 == 0:
            #     input = batch_x.detach().cpu().numpy()
            #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
            #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
            #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

            cur_model.eval()
            # cur_model.cpu()
            del cur_model
            torch.cuda.empty_cache()


        # # 将adaptation前的loss、adaptation中逐样本做adapt后的loss、以及adaptation之后的loss做统计
        # with open(f"./loss_before_and_after_adapt/{setting}.txt", "a") as f:
        #     for i in range(len(a1)):
        #         t1, t2, t3 = a1[i], a2[i], a3[i]
        #         f.write(f"{t1}, {t2}, {t3}\n")


        # 对预测结果（如长度为24）中的每个位置/index上的结果分别进行统计的结果输出出来
        mean_error_per_pred_index = [[] for i in range(pred_len)]
        for index in range(pred_len):
            error_i = error_per_pred_index[index]
            total_err = 0
            total_num = 0
            for tmp_err in error_i:
                total_err += tmp_err
                total_num += 1
            mean_error_per_pred_index[index] = total_err / total_num
        # print(mean_error_per_pred_index)


        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        # f = open("result.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        print(f"Test - cost time: {time.time() - test_time_start}s")

        return a1, a2, a3


    def my_test(self, setting, test=0, is_training_part_params=True, use_adapted_model=True, test_train_epochs=1, weights_given=None, adapted_degree="small", weights_from="test"):
        test_data, test_loader = self._get_data_at_test_time(flag='test')
        if test:
            print('loading model from checkpoint !!!')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        self.model.eval()

        preds = []
        trues = []

        # mapping_model = self.learn_mapping_model_during_vali(is_training_part_params)
        # mapping_model.eval()

        a1, a2, a3 = [], [], []
        all_angels = []

        error_per_pred_index = [[] for i in range(self.args.pred_len)]

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()  # 如果使用amp的话，还需要再生成一个scaler？

        criterion = nn.MSELoss()  # 使用MSELoss
        test_time_start = time.time()

        def MAE(pred, true):
            return torch.mean(torch.abs(pred - true))
        def MSE(pred, true):
            return torch.mean((pred - true) ** 2)
        def decay_MSE(pred, true):
            dim = pred.shape[1]  # 获取L维
            loss = 0
            for i in range(dim):
                loss += (1 / (i+1)) * torch.mean((pred[:,i,:] - true[:,i,:]) ** 2)
            loss /= dim
            return loss
        def decay_MAE(pred, true):
            dim = pred.shape[1]  # 获取L维
            loss = 0
            for i in range(dim):
                loss += (1 / math.sqrt(i+1)) * torch.mean(torch.abs(pred[:,i,:] - true[:,i,:]))
            loss /= dim
            return loss

        # # result save
        # folder_path = './test_results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)


        # 加载模型参数到self.model里
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        if weights_given:
            print(f"The given weights is {weights_given}")
            print(f"The length of given weights is {len(weights_given)}")


        # self.model.eval()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            # 从self.model拷贝下来cur_model，并设置为train模式
            # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))
            cur_model = copy.deepcopy(self.model)
            cur_model.train()

            if is_training_part_params:
                params = []
                names = []
                params_norm = []
                names_norm = []
                cur_model.requires_grad_(False)
                for n_m, m in cur_model.named_modules():
                    # 注意：这里的名字应当根据Autoformer模型而修改为"decoder.projection"
                    
                    # if "decoder.projection" == n_m:
                    if "decoder.projection" in n_m:
                        m.requires_grad_(True)
                        for n_p, p in m.named_parameters():
                            if n_p in ['weight', 'bias']:  # weight is scale, bias is shift
                                params.append(p)
                                names.append(f"{n_m}.{n_p}")
                    
                    # # 再加入倒数第二层layer_norm层
                    # if "decoder.norm.layernorm" == n_m:
                    #     m.requires_grad_(True)
                    #     for n_p, p in m.named_parameters():
                    #         if n_p in ['weight', 'bias']:  # weight is scale, bias is shift
                    #             params.append(p)
                    #             names.append(f"{n_m}.{n_p}")
                    
                    # if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                    #     m.requires_grad_(True)
                    #     for n_p, p in m.named_parameters():
                    #         if n_p in ['weight', 'bias']:  # weight is scale, bias is shift
                    #             params_norm.append(p)
                    #             names_norm.append(f"{n_m}.{n_p}")

                # Adam优化器
                # model_optim = optim.Adam(params, lr=self.args.learning_rate*10 / (2**self.test_train_num))  # 使用Adam优化器
                # lr = self.args.learning_rate * 1
                lr = self.args.learning_rate * self.args.adapted_lr_times
                # model_optim = optim.Adam(params, lr=lr)  # 使用Adam优化器
                # model_optim_norm = optim.Adam(params_norm, lr=self.args.learning_rate*1000 / (2**self.test_train_num))  # 使用Adam优化器
                
                # 普通的SGD优化器？
                model_optim = optim.SGD(params, lr=lr)
            else:
                cur_model.requires_grad_(True)
                # model_optim = optim.Adam(cur_model.parameters(), lr=self.args.learning_rate*10 / (2**self.test_train_num))
                # model_optim = optim.Adam(cur_model.parameters(), lr=self.args.learning_rate)
                lr = self.args.learning_rate * self.args.adapted_lr_times
                model_optim = optim.SGD(cur_model.parameters(), lr=lr)
            

            # tmp loss
            cur_model.eval()
            seq_len = self.args.seq_len
            pred_len = self.args.pred_len
            adapt_start_pos = self.args.adapt_start_pos
            if not self.args.use_nearest_data or self.args.use_further_data:
                pred, true = self._process_one_batch_with_model(cur_model, test_data,
                    batch_x[:, -seq_len:, :], batch_y, 
                    batch_x_mark[:, -seq_len:, :], batch_y_mark)
            else:
                pred, true = self._process_one_batch_with_model(cur_model, test_data,
                    batch_x[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y, 
                    batch_x_mark[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y_mark)
            # 获取adaptation之前的loss
            loss_before_adapt = criterion(pred, true)
            a1.append(loss_before_adapt.item())
            # print(f"loss_before_adapt: {loss_before_adapt}")
            # print(f'mse:{tmp_loss}')
            cur_model.train()

            

            # tmp_train_X_sample = torch.cat((batch_x[:, -self.args.seq_len:, :], batch_x_mark[:, -self.args.seq_len:, :]), 2).ravel()
            # tmp_train_X_sample = tmp_train_X_sample.to(torch.float32).to(self.device)
            # mapping_answer = mapping_model(tmp_train_X_sample)
            # # mapping_answer = mapping_answer.reshape()
            # # print(mapping_answer)
            

            # 先用原模型的预测值和标签值之间的error，做反向传播之后得到的梯度值gradient_0
            # 并将这个gradient_0作为标准答案
            # 然后，对测试样本做了adaptation之后，会得到一个gradient_1
            # 那么对gradient_1和gradient_0之间做对比，
            # 就可以得到二者之间的余弦值是多少（方向是否一致），以及长度上相差的距离有多少等等。
            # params_answer = self.get_answer_grad(is_training_part_params, use_adapted_model,
            #                                         lr, test_data, 
            #                                         batch_x, batch_y, batch_x_mark, batch_y_mark,
            #                                         setting)
            if use_adapted_model:
                seq_len = self.args.seq_len
                pred_len = self.args.pred_len
                adapt_start_pos = self.args.adapt_start_pos
                if not self.args.use_nearest_data or self.args.use_further_data:
                    pred_answer, true_answer = self._process_one_batch_with_model(cur_model, test_data,
                        batch_x[:, -seq_len:, :], batch_y, 
                        batch_x_mark[:, -seq_len:, :], batch_y_mark)
                else:
                    pred_answer, true_answer = self._process_one_batch_with_model(cur_model, test_data,
                        batch_x[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y, 
                        batch_x_mark[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y_mark)
            
            # criterion = nn.MSELoss()  # 使用MSELoss
            # 计算MSE loss
            loss_ans_before = criterion(pred_answer, true_answer)
            loss_ans_before.backward()

            w_T = params[0].grad.T  # 先对weight参数做转置
            b = params[1].grad.unsqueeze(0)  # 将bias参数扩展一维
            params_answer = torch.cat((w_T, b), 0)  # 将w_T和b参数concat起来
            params_answer = params_answer.ravel()  # 最后再展开成一维的

            model_optim.zero_grad()  # 清空梯度


            # params_adapted = torch.zeros((1)).to(self.device)
            cur_grad_list = []

            # 开始训练
            # cur_lr = self.args.learning_rate*10 / (2**self.test_train_num)
            # cur_lr_norm = self.args.learning_rate*1000 / (2**self.test_train_num)
            for epoch in range(test_train_epochs):
                # cur_lr = self.args.learning_rate*10 / (2**self.test_train_num)
                # cur_lr_norm = self.args.learning_rate*1000 / (2**self.test_train_num)

                gradients = []
                accpted_samples_num = set()

                # for pass_num in range(2):
                #     if pass_num == 1:  # the second pass
                #         cmp = lambda lst: torch.norm(lst[1], lst[2])
                #         gradients.sort(key=cmp)
                #         half_num = self.test_train_num // 2
                #         # just get half of samples with smaller gradients
                #         gradients = gradients[:-half_num or None]
                #         for grad in gradients:
                #             accpted_samples_num.add(grad[0])
                

                import random
                is_random = False
                # is_random = True
                sample_order_list = list(range(self.test_train_num))
                # print("before random, sample_order_list is: ", sample_order_list)
                if is_random:
                    random.shuffle(sample_order_list)
                    # print("after random, sample_order_list is: ", sample_order_list)
                else:
                    # print("do not use random.")
                    pass

                # 新加了参数num_of_loss_per_update，表示我们将多个loss平均起来做一次更新
                num_of_loss_per_update = self.test_train_num
                # num_of_loss_per_update = 1
                mean_loss = 0


                # # 验证adaptation结果
                # pred, true = self._process_one_batch_with_model(cur_model, test_data,
                #     batch_x[:, -self.args.seq_len:, :], batch_y, 
                #     batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)
                
                # loss = criterion(pred, true)
                # mean_loss += loss

                # loss.backward()
                # model_optim.step()



                for ii in sample_order_list:
                    # if pass_num == 1 and ii not in accpted_samples_num:
                    #     continue

                    # if not ((self.test_train_num - 1 - ii) + self.args.pred_len) % 96 == 0:
                    #     continue

                    model_optim.zero_grad()

                    seq_len = self.args.seq_len
                    label_len = self.args.label_len
                    pred_len = self.args.pred_len

                    # batch_x.requires_grad = True
                    # batch_x_mark.requires_grad = True

                    pred, true = self._process_one_batch_with_model(cur_model, test_data,
                        batch_x[:, ii : ii+seq_len, :], batch_x[:, ii+seq_len-label_len : ii+seq_len+pred_len, :], 
                        batch_x_mark[:, ii : ii+seq_len, :], batch_x_mark[:, ii+seq_len-label_len : ii+seq_len+pred_len, :])

                    # 这里当batch_size为1还是32时
                    # pred和true的size可能为[1, 24, 7]或[32, 24, 7]
                    # 但是结果的loss值均只包含1个值
                    # 这是因为criterion为MSELoss，其默认使用mean模式，会对32个loss值取一个平均值
                    
                    # 判断是否使用最近的数据
                    if not self.args.use_nearest_data or self.args.use_further_data:
                        loss = criterion(pred, true)
                    else:
                        data_used_num = (self.test_train_num - (ii+1)) + self.args.adapt_start_pos
                        if data_used_num < pred_len:
                            loss = criterion(pred[:, :data_used_num, :], true[:, :data_used_num, :])
                        else:
                            loss = criterion(pred, true)
                        # loss = criterion(pred, true)
                    # loss = decay_MAE(pred, true)

                    # from functorch import vmap
                    # from functorch.experimental import replace_all_batch_norm_modules_
                    # replace_all_batch_norm_modules_(cur_model)
                    # tmp_batch_x = batch_x.unsqueeze(1)
                    # tmp_batch_x_mark = batch_x_mark.unsqueeze(1)
                    # vmap_func = vmap(self._process_one_batch_with_model, 
                    #                  in_dims=(None, None, 0, 0, 0, 0), out_dims=(0, 0), 
                    #                  randomness='different')
                    # pred, true = vmap_func(cur_model, test_data,
                    #     tmp_batch_x[:, :, ii : ii+seq_len, :], tmp_batch_x[:, :, ii+seq_len-label_len : ii+seq_len+pred_len, :], 
                    #     tmp_batch_x_mark[:, :, ii : ii+seq_len, :], tmp_batch_x_mark[:, :, ii+seq_len-label_len : ii+seq_len+pred_len, :])


                    # loss = criterion(pred, true)
                    mean_loss += loss

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        # scaler.step(model_optim_norm)
                        scaler.update()
                    else:
                        # # pass
                        # loss.backward()
                        # model_optim.step()

                        # # 多个损失合并起来做一次更新
                        # if (ii+1) % num_of_loss_per_update == 0:
                        #     mean_loss = mean_loss / num_of_loss_per_update
                        #     mean_loss.backward()
                        #     model_optim.step()
                        #     mean_loss = 0
                        # elif (ii+1) == len(sample_order_list):
                        #     mean_loss = mean_loss / (len(sample_order_list) % num_of_loss_per_update)
                        #     mean_loss.backward()
                        #     model_optim.step()
                        #     mean_loss = 0
                        # else:
                        #     pass

                        loss.backward()
                        w_T = params[0].grad.T
                        b = params[1].grad.unsqueeze(0)
                        params_tmp = torch.cat((w_T, b), 0)
                        original_shape = params_tmp.shape
                        params_tmp = params_tmp.ravel()

                        # 将该梯度存入cur_grad_list中
                        cur_grad_list.append(params_tmp.detach().cpu().numpy())

                        model_optim.zero_grad()

                        # # 计算标准答案的梯度params_answer和adaptation中的梯度params_tmp之间的角度
                        # import math
                        # # print()
                        # # product = torch.dot(params_tmp, params_answer)
                        # # product = product / (torch.norm(params_tmp) * torch.norm(params_answer))
                        # product = torch.dot(params_tmp, mapping_answer)
                        # product = product / (torch.norm(params_tmp) * torch.norm(mapping_answer))
                        # angel = math.degrees(math.acos(product))
                        # if angel < 90:
                        #     model_optim.step()
                        # else:
                        #     model_optim.zero_grad()


                        # if pass_num == 0:
                        #     w_grad = torch.norm(params[0].grad)
                        #     b_grad = torch.norm(params[1].grad)
                        #     gradients.append([ii, w_grad, b_grad])

                        #     with open(f"./logs_loss_and_grad/{setting}", "a") as f:
                        #         f.write(f"{loss}, {torch.norm(w_grad, b_grad)}" + "\n")
                            
                        # elif pass_num == 1:

                        # model_optim.step()
                        
                        # model_optim_norm.step()

                        # w_T = params[0].grad.T  # 先对weight参数做转置
                        # b = params[1].grad.unsqueeze(0)  # 将bias参数扩展一维
                        # params_grad = torch.cat((w_T, b), 0)  # 将w_T和b参数concat起来
                        # params_grad = params_grad.ravel()  # 最后再展开成一维的

                        # with open(f"./logs_loss_and_grad/batch{self.args.adapted_batch_size}_{setting}", "a") as f:
                        #     f.write(f"{loss}, {torch.norm(params_grad)}" + "\n")

                    # cur_model.eval()
                    # tmp_pred, tmp_true = self._process_one_batch_with_model(cur_model, test_data,
                    #     batch_x[:, ii : ii+seq_len, :], batch_x[:, ii+seq_len-label_len : ii+seq_len+pred_len, :], 
                    #     batch_x_mark[:, ii : ii+seq_len, :], batch_x_mark[:, ii+seq_len-label_len : ii+seq_len+pred_len, :])
                    # tmp_loss = criterion(tmp_pred, tmp_true)
                    # cur_model.train()

                    # 记录逐样本做了adaptation之后的loss
                    # mean_loss += tmp_loss
                    # mean_loss += loss


                    # cur_lr = cur_lr * 2
                    # for param_group in model_optim.param_groups:
                    #     param_group['lr'] = cur_lr
                    
                    # cur_lr_norm = cur_lr_norm * 2
                    # for param_group in model_optim_norm.param_groups:
                    #     param_group['lr'] = cur_lr_norm
                
            # 定义一个权重平均函数
            def calc_weighted_params(params, weights):
                results = 0
                for i in range(len(params)):
                    results += params[i] * weights[i]
                return results
            # 加载权重到对应的梯度上
            if weights_given:
                weighted_params = calc_weighted_params(cur_grad_list, weights_given)
            else:
                weights_all_ones = [1 for i in range(self.test_train_num)]
                weighted_params = calc_weighted_params(cur_grad_list, weights_all_ones)
            # 将weighted_params从np.array转成tensor
            weighted_params = torch.tensor(weighted_params)
            weighted_params = weighted_params.to(self.device)

            # 计算标准答案的梯度params_answer和adaptation加权后的梯度weighted_params之间的角度
            import math
            product = torch.dot(weighted_params, params_answer)
            product = product / (torch.norm(weighted_params) * torch.norm(params_answer))
            angel = math.degrees(math.acos(product))
            all_angels.append(angel)
            
            # 还原回原来的梯度
            weighted_params = weighted_params.reshape(original_shape)
            w_grad_T, b_grad = torch.split(weighted_params, [weighted_params.shape[0]-1, 1])
            w_grad = w_grad_T.T  # (7, 512)
            b_grad = b_grad.squeeze(0)  # (7)

            # 设置新参数为原来参数 + 梯度值
            from torch.nn.parameter import Parameter
            cur_lr = self.args.learning_rate * self.args.adapted_lr_times

            # if angel < 70:
            #     # 注意：这里是减去梯度，而不是加上梯度！！！！！
            #     # cur_model.decoder.projection.weight.set_(params[0] - w_grad * cur_lr)
            #     # cur_model.decoder.projection.bias.set_(params[1] - b_grad * cur_lr)
            #     cur_model.decoder.projection.weight = Parameter(cur_model.decoder.projection.weight - w_grad * cur_lr)
            #     cur_model.decoder.projection.bias = Parameter(cur_model.decoder.projection.bias - b_grad * cur_lr)

            cur_model.decoder.projection.weight = Parameter(cur_model.decoder.projection.weight - w_grad * cur_lr)
            cur_model.decoder.projection.bias = Parameter(cur_model.decoder.projection.bias - b_grad * cur_lr)

            mean_loss = mean_loss / self.test_train_num
            a2.append(mean_loss.item())
            
            # mean_loss.backward()
            # model_optim.step()
            
            cur_model.eval()


            # # 计算标准答案的梯度params_answer和adaptation中的梯度params_adapted之间的角度
            # import math
            # product = torch.dot(params_adapted, params_answer)
            # product = product / (torch.norm(params_adapted) * torch.norm(params_answer))
            # angel = math.degrees(math.acos(product))

            # with open(f"./grad_angels/{setting}", "a") as f:
            #     f.write(f"{angel}" + "\n")

            cur_model.eval()


            seq_len = self.args.seq_len
            pred_len = self.args.pred_len
            adapt_start_pos = self.args.adapt_start_pos
            if use_adapted_model:
                if not self.args.use_nearest_data or self.args.use_further_data:
                    pred, true = self._process_one_batch_with_model(cur_model, test_data,
                        batch_x[:, -seq_len:, :], batch_y, 
                        batch_x_mark[:, -seq_len:, :], batch_y_mark)
                else:
                    pred, true = self._process_one_batch_with_model(cur_model, test_data,
                        batch_x[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y, 
                        batch_x_mark[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y_mark)
            else:
                # pred, true = self._process_one_batch_with_model(self.model, test_data,
                #     batch_x[:, -self.args.seq_len:, :], batch_y, 
                #     batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)
                if not self.args.use_nearest_data or self.args.use_further_data:
                    pred, true = self._process_one_batch_with_model(cur_model, test_data,
                        batch_x[:, -seq_len:, :], batch_y, 
                        batch_x_mark[:, -seq_len:, :], batch_y_mark)
                else:
                    pred, true = self._process_one_batch_with_model(cur_model, test_data,
                        batch_x[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y, 
                        batch_x_mark[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y_mark)


            # 获取adaptation之后的loss
            loss_after_adapt = criterion(pred, true)
            a3.append(loss_after_adapt.item())

            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

            # 对预测结果（如长度为24）中的每个位置/index上的结果分别进行统计
            pred_len = self.args.pred_len
            for index in range(pred_len):
                cur_pred = pred.detach().cpu().numpy()[0][index]
                cur_true = true.detach().cpu().numpy()[0][index]
                cur_error = np.mean((cur_pred - cur_true) ** 2)
                error_per_pred_index[index].append(cur_error)


            if (i+1) % 100 == 0:
                print("\titers: {0}, cost time: {1}s".format(i + 1, time.time() - test_time_start))
                print(gradients)
                tmp_p = np.array(preds); tmp_p = tmp_p.reshape(-1, tmp_p.shape[-2], tmp_p.shape[-1])
                tmp_t = np.array(trues); tmp_t = tmp_t.reshape(-1, tmp_t.shape[-2], tmp_t.shape[-1])
                tmp_mae, tmp_mse, _, _, _ = metric(tmp_p, tmp_t)
                print('mse:{}, mae:{}'.format(tmp_mse, tmp_mae))
                
                avg_1, avg_2, avg_3 = 0, 0, 0
                avg_angel = 0
                num = len(a1)
                for iii in range(num):
                    avg_1 += a1[iii]; avg_2 += a2[iii]; avg_3 += a3[iii]; avg_angel += all_angels[iii]
                avg_1 /= num; avg_2 /= num; avg_3 /= num; avg_angel /= num
                print(avg_1, avg_2, avg_3, avg_angel)

            # if i % 20 == 0:
            #     input = batch_x.detach().cpu().numpy()
            #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
            #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
            #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

            cur_model.eval()
            # cur_model.cpu()
            del cur_model
            torch.cuda.empty_cache()


        # # 将adaptation前的loss、adaptation中逐样本做adapt后的loss、以及adaptation之后的loss做统计
        # with open(f"./loss_before_and_after_adapt/{setting}.txt", "a") as f:
        #     for i in range(len(a1)):
        #         t1, t2, t3 = a1[i], a2[i], a3[i]
        #         f.write(f"{t1}, {t2}, {t3}\n")


        # 对预测结果（如长度为24）中的每个位置/index上的结果分别进行统计的结果输出出来
        mean_error_per_pred_index = [[] for i in range(pred_len)]
        for index in range(pred_len):
            error_i = error_per_pred_index[index]
            total_err = 0
            total_num = 0
            for tmp_err in error_i:
                total_err += tmp_err
                total_num += 1
            mean_error_per_pred_index[index] = total_err / total_num
        # print(mean_error_per_pred_index)


        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        # f = open("result.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        print(f"Test - cost time: {time.time() - test_time_start}s")


        # # 保存angel和loss信息
        # angle_dir = "./angels_and_loss_diff/" + setting
        # if not os.path.exists(angle_dir):
        #     os.makedirs(angle_dir)

        # if weights_given:
        #     angel_file = f"{angle_dir}/angels_{weights_from}_ttn{self.test_train_num}_lr{self.args.adapted_lr_times:.2f}.txt"
        # else:
        #     angel_file = f"{angle_dir}/angels_allones_ttn{self.test_train_num}_lr{self.args.adapted_lr_times:.2f}.txt"

        # with open(angel_file, "w") as f:
        #     for i in range(len(a1)):
        #         f.write(f"{all_angels[i]}, {a1[i]}, {a3[i]}" + "\n")
        

        return a1, a2, a3



    # def subprocess_of_my_test_mp(self, test_data, setting, test, is_training_part_params, use_adapted_model, test_train_epochs, i, batch_x, batch_y, batch_x_mark, batch_y_mark, criterion, scaler, preds, trues, test_time_start):
    #     # 从self.model拷贝下来cur_model，并设置为train模式
    #     cur_model = copy.deepcopy(self.model)
    #     cur_model.train()

    #     if is_training_part_params:
    #         params = []
    #         names = []
    #         cur_model.requires_grad_(False)
    #         for n_m, m in cur_model.named_modules():
    #             # 注意：这里的名字应当根据Autoformer模型而修改为"decoder.projection"
    #             if "decoder.projection" == n_m:
    #                 m.requires_grad_(True)
    #       


    def _process_one_batch_with_model(self, model, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, return_mid_embedding=False):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                if return_mid_embedding:
                    outputs, mid_embedding = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_mid_embedding=return_mid_embedding)  # 这里返回的结果为[B, L, D]，例如[32, 24, 12]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)  # 这里返回的结果为[B, L, D]，例如[32, 24, 12]
        
        # if self.args.inverse:
        #     outputs = dataset_object.inverse_transform(outputs)
        
        f_dim = -1 if self.args.features=='MS' else 0

        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        # outputs为我们预测出的值pred，而batch_y则是对应的真实值true
        if return_mid_embedding:
            return outputs, batch_y, mid_embedding
        else:
            return outputs, batch_y


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return