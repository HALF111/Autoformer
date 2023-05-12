from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, Reformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim

# torch.cuda.current_device()
# torch.cuda._initialized = True

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

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

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'Reformer': Reformer,
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

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        
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

        return


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
            self.model.requires_grad_(True)
            # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate*10 / (2**self.test_train_num))
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        
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
        


    def learn_mapping_model_during_vali(self, is_training_part_params):
        # 这里用'val_with_batchsize_1'来获取数据
        # vali_data, vali_loader = self._get_data(flag='val_with_batchsize_1')
        vali_data, vali_loader = self._get_data(flag='val')
        criterion = nn.MSELoss()  # 使用MSELoss
        
        cur_model = copy.deepcopy(self.model)
        cur_model.train()

        train_X = []
        train_Y = []

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
            # 普通的SGD优化器？
            tmp_optim = optim.SGD(params, lr=self.args.learning_rate)


        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            # 在验证集上跑
            pred, true = self._process_one_batch_with_model(cur_model, vali_data,
                batch_x, batch_y, 
                batch_x_mark, batch_y_mark)

            # pred = pred.detach().cpu()
            # true = true.detach().cpu()

            # 每次计算前都要讲梯度先清零
            tmp_optim.zero_grad()
            # 计算MSE loss并作backward
            loss = criterion(pred, true)
            loss.backward()

            w_T = params[0].grad.T  # 先对weight参数做转置
            b = params[1].grad.unsqueeze(0)  # 将bias参数扩展一维
            # print(b)
            params_answer = torch.cat((w_T, b), 0)  # 将w_T和b参数concat起来
            params_answer = params_answer.ravel()  # 最后再展开成一维的

            train_X_sample = torch.cat((batch_x, batch_x_mark), 2).ravel()
            train_Y_sample = params_answer
            train_X.append(train_X_sample)
            train_Y.append(train_Y_sample)
        

        # 一定记得要加int，否则会是浮点型的！！
        X_dim = int(train_X[0].shape[0])
        Y_dim = int(train_Y[0].shape[0])
        print(f"X_dim: {X_dim}, Y_dim: {Y_dim}")
        # 这里要用整除
        h_dim = int(self.args.d_model // 1)
        # h_dim = int(self.args.d_model // 4)
        mapping_model = nn.Sequential(
            nn.Linear(X_dim, h_dim), 
            nn.Linear(h_dim ,Y_dim)
            )
        mapping_model.to(self.device)
        mapping_model.train()

        # loss_fn = nn.MSELoss().to(self.device)
        lr = self.args.learning_rate * 1000
        model_optim = optim.SGD(mapping_model.parameters(), lr=lr)

        for epoch in range(self.args.train_epochs):
            print(f"training for mapping model, epoch:{epoch+1}")
            for i in range(len(train_X)):
                train_X_sample = train_X[i]
                train_Y_sample = train_Y[i]
                train_X_sample = train_X_sample.to(torch.float32).to(self.device)
                train_Y_sample = train_Y_sample.to(torch.float32).to(self.device)
                output = mapping_model(train_X_sample)
                
                product = torch.dot(output, train_Y_sample)
                product = product / (torch.norm(output) * torch.norm(train_Y_sample))
                loss = (1 - product) * 100
                
                model_optim.zero_grad()
                loss.backward()
                model_optim.step()

                if (i+1) % 10 == 0:
                    print(f"epoch:{epoch+1}, iteration:{i+1}, product loss:{loss.item()}")

        cur_model.eval()
        del cur_model

        mapping_model.eval()
        return mapping_model



    def my_test(self, setting, test=0, is_training_part_params=True, use_adapted_model=True, test_train_epochs=1):
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

        # # result save
        # folder_path = './test_results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        # self.model.eval()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            # 从self.model拷贝下来cur_model，并设置为train模式
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
                    if "decoder.projection" == n_m:
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
                self.model.requires_grad_(True)
                # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate*10 / (2**self.test_train_num))
                model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
            

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

            

            tmp_train_X_sample = torch.cat((batch_x[:, -self.args.seq_len:, :], batch_x_mark[:, -self.args.seq_len:, :]), 2).ravel()
            tmp_train_X_sample = tmp_train_X_sample.to(torch.float32).to(self.device)
            mapping_answer = mapping_model(tmp_train_X_sample)
            # mapping_answer = mapping_answer.reshape()
            # print(mapping_answer)
            

            # 先用原模型的预测值和标签值之间的error，做反向传播之后得到的梯度值gradient_0
            # 并将这个gradient_0作为标准答案
            # 然后，对测试样本做了adaptation之后，会得到一个gradient_1
            # 那么对gradient_1和gradient_0之间做对比，
            # 就可以得到二者之间的余弦值是多少（方向是否一致），以及长度上相差的距离有多少等等。
            params_answer = self.get_answer_grad(is_training_part_params, use_adapted_model,
                                                    lr, test_data, 
                                                    batch_x, batch_y, batch_x_mark, batch_y_mark,
                                                    setting)


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
                sample_order_list = list(range(self.test_train_num))
                # print("before random, sample_order_list is: ", sample_order_list)
                if is_random:
                    random.shuffle(sample_order_list)
                    # print("after random, sample_order_list is: ", sample_order_list)
                else:
                    # print("do not use random.")
                    pass

                mean_loss = 0
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

                    loss = criterion(pred, true)

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
                    # mean_loss += loss

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        # scaler.step(model_optim_norm)
                        scaler.update()
                    else:
                        # pass
                        loss.backward()
                        # model_optim.step()

                        w_T = params[0].grad.T
                        b = params[1].grad.unsqueeze(0)
                        params_tmp = torch.cat((w_T, b), 0)
                        params_tmp = params_tmp.ravel()
                        params_adapted = params_adapted + params_tmp

                        # 计算标准答案的梯度params_answer和adaptation中的梯度params_tmp之间的角度
                        import math
                        # print()
                        # product = torch.dot(params_tmp, params_answer)
                        # product = product / (torch.norm(params_tmp) * torch.norm(params_answer))
                        product = torch.dot(params_tmp, mapping_answer)
                        product = product / (torch.norm(params_tmp) * torch.norm(mapping_answer))
                        angel = math.degrees(math.acos(product))
                        if angel < 90:
                            model_optim.step()
                        else:
                            model_optim.zero_grad()


                        # if pass_num == 0:
                        #     w_grad = torch.norm(params[0].grad)
                        #     b_grad = torch.norm(params[1].grad)
                        #     gradients.append([ii, w_grad, b_grad])

                        #     with open(f"./logs_loss_and_grad/{setting}", "a") as f:
                        #         f.write(f"{loss}, {torch.norm(w_grad, b_grad)}" + "\n")
                            
                        # elif pass_num == 1:

                        # model_optim.step()
                        
                        # model_optim_norm.step()

                    # cur_model.eval()
                    # tmp_pred, tmp_true = self._process_one_batch_with_model(cur_model, test_data,
                    #     batch_x[:, ii : ii+seq_len, :], batch_x[:, ii+seq_len-label_len : ii+seq_len+pred_len, :], 
                    #     batch_x_mark[:, ii : ii+seq_len, :], batch_x_mark[:, ii+seq_len-label_len : ii+seq_len+pred_len, :])
                    # tmp_loss = criterion(tmp_pred, tmp_true)
                    # cur_model.train()

                    # 记录逐样本做了adaptation之后的loss
                    # mean_loss += tmp_loss
                    mean_loss += loss


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
                pred, true = self._process_one_batch_with_model(self.model, test_data,
                    batch_x[:, -self.args.seq_len:, :], batch_y, 
                    batch_x_mark[:, -self.args.seq_len:, :], batch_y_mark)

            # 获取adaptation之后的loss
            loss_after_adapt = criterion(pred, true)
            a3.append(loss_after_adapt.item())


            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

            # 对预测结果（如长度为24）中的每个位置/index上的结果分别进行统计
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
        print(mean_error_per_pred_index)


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

        return



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


    def _process_one_batch_with_model(self, model, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
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
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)  # 这里返回的结果为[B, L, D]，例如[32, 24, 12]
        
        # if self.args.inverse:
        #     outputs = dataset_object.inverse_transform(outputs)
        
        f_dim = -1 if self.args.features=='MS' else 0

        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        # outputs为我们预测出的值pred，而batch_y则是对应的真实值true
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