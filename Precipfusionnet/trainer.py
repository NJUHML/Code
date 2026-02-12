from loss import LossFunctions, weighted_mae
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
from utils import *
from metrics import *
from datautils import *
import os
import numpy as np
import math
from tqdm import tqdm
from unet_model import UNet, ResUnet,AttU_Net,R2AttU_Net
from dataset import load_dataset
from AutomaticWeightedLoss import AutomaticWeightedLoss
from archs import UKAN,AttUKAN
import time



class trainer_unet():
    def __init__(self, opt):
        self.opt = opt
        self.save_dir = opt.log_dir
        self.dataset = opt.dataset
        self.batch_size = opt.batch_size

        self.dtype = torch.cuda.FloatTensor

        self.start_epoch = 0
        self.total_iter = 0
        num_iters = int(opt.total_epoch * opt.epoch_size / 2)
        self.delta = float(1) / num_iters

        self.epoch_size = opt.epoch_size

        self.fft = 0
        self.datatype = opt.datatype
        self.evaltype = "all"

        self.SCALE = opt.scale
        self.MEAN = opt.mean
        self.model_name = opt.model_name
        self.torlerate_limit = opt.torlerate_limit
        self.awl = AutomaticWeightedLoss(3,self.epoch_size)

        print('GFS DONE =============================================================================================================================')
        print('GPM DONE ============================================================================================================================')
        train_data, valid_data = load_dataset(data_type = self.datatype, input_type = 'P',time_type = 'f024')
        self.train_loader = DataLoader(train_data,num_workers=opt.data_threads,batch_size=opt.batch_size,shuffle=True,drop_last=True,pin_memory=True)
        self.val_loader = DataLoader(valid_data,num_workers=opt.data_threads,batch_size=opt.batch_size,shuffle=True,drop_last=False,pin_memory=True)

        # self.model=UNet(1,1).cuda()
        # self.model=AttU_Net(1,1).cuda()
        self.model = AttUKAN(num_classes=1,input_channels=1,img_size=[160,256]).cuda()
        self.optimizer = optim.Adam([{'params': self.model.parameters()},{'params': self.awl.parameters(), 'weight_decay': 0},],lr=opt.lr)

        self.scheduler = get_scheduler(self.optimizer, self.opt, opt.epoch_size)


        if opt.resume_from:
            checkpoint = torch.load(opt.resume_from)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.epoch = checkpoint['epoch']
            self.best_mse = checkpoint['best_mse']
            self.torlerate = checkpoint['torlerate']
            print("Loaded partially trained model from checkpoint: ", opt.resume_from)
        else:
            self.epoch=0
            self.best_mse=math.inf
            self.torlerate = 0

        if opt.resume == 'True':
            print("previous parameter is loaded !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.model.load_state_dict(torch.load(os.path.join(opt.log_dir,opt.model_name+'.pkl')))

        if not os.path.exists(opt.log_dir):
            os.mkdir(opt.log_dir)

        self.lambda_tem = 5
        self.lambda_tem_1 = 1
        self.Loss_func2 = LossFunctions()

    def name(self):
        return 'Trainer UNet'

    def train_forward(self,data1,data2,t,epoch, train_loss, count_train):
        inputs, targets = data1, data2
        inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
        self.optimizer.zero_grad()
        preds = self.model(inputs)
        loss1 = self.Loss_func2.weighted_mae(preds,targets)
        loss2 = self.Loss_func2.multi_diff(preds,targets)
        loss3 = self.Loss_func2.mse(preds,targets)
        loss,_ = self.awl([[loss1,loss2,loss3]],epoch)
        train_loss = train_loss + loss.detach().cpu().numpy()
        count_train = count_train + 1
        t.set_postfix({
            'loss': '{:.6f}'.format(loss),
            'ep': '{:02d}'.format(epoch),
        })
        
        loss.backward()
        self.optimizer.step()
        return train_loss, count_train

    def train(self):
        self.model.train()
        epoch= self.epoch
        best_mse = self.best_mse
        torlerate = self.torlerate

        for epoch in range(self.epoch_size):
            train_loss = 0
            count_train = 0
            t = tqdm(self.train_loader, leave=False, total=len(self.train_loader))
            start_time = time.time()
            for i, (data1, data2)  in enumerate(t):
                train_loss, count_train = self.train_forward(data1,data2,t, epoch, train_loss, count_train)
            end_time = time.time()
            print('time:',end_time-start_time)
            if self.evaltype == 'all':
                val_loss, count_val = self.evalid(epoch)
                print('train loss: ' + str(train_loss / count_train)+'               '+'valid loss: ' + str(val_loss / count_val)+'              '+'epoch:'+ str(epoch)+'              '+'torlerate:'+ str(torlerate))
                self.scheduler.step()
                print('learning rate = %.7f' % self.scheduler.get_last_lr()[0])
                val_mse = val_loss / count_val
                if val_mse < best_mse:
                    best_mse = val_mse
                    checkpoint_dict = {'epoch': epoch, 
                                    'model_state_dict': self.model.state_dict(), 
                                    'optimizer_state_dict': self.optimizer.state_dict(),
                                    'best_mse':best_mse,
                                    'torlerate':torlerate,
                                    'loss_weights': self.awl.params, 
                                            }
                    torch.save(checkpoint_dict, os.path.join(self.save_dir,self.model_name+'.pkl'))
                    torlerate = 0
                else:
                    torlerate = torlerate + 1

                if epoch%5==0 and epoch>0:
                    checkpoint_dict = {'epoch': epoch, 
                    'model_state_dict': self.model.state_dict(), 
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_mse':best_mse,
                    'torlerate':torlerate,
                    'loss_weights': self.awl.params,  
                                        }
                    torch.save(checkpoint_dict, os.path.join(self.save_dir,self.model_name+'_epoch_'+str(epoch)+'.pkl'))
                print(f'weights: {self.awl.params}')


    def eval_forward(self, data1,data2, epoch,val_loss, count_val):
        inputs, targets = data1, data2
        inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
        preds = self.model(inputs)
        loss1 = self.Loss_func2.weighted_mae(preds,targets)
        loss2 = self.Loss_func2.multi_diff(preds,targets)
        loss3 = self.Loss_func2.mse(preds,targets)
        loss,_ = self.awl([[loss1,loss2,loss3]],epoch)
        val_loss = val_loss + loss.detach().cpu().numpy()
        count_val = count_val + 1
        return val_loss, count_val


    def evalid(self,epoch):
        with torch.no_grad():
            self.model.eval()
            val_loss = 0
            count_val = 0
            t = tqdm(self.val_loader, leave=False, total=len(self.val_loader))
            for i, (data1, data2)  in enumerate(t):
                val_loss, count_val = self.eval_forward(data1,data2,epoch, val_loss, count_val)
        return val_loss, count_val

    def test_event(self):
        self.model.eval()
        t = tqdm(self.val_loader, leave=False, total=len(self.val_loader))
        for i, (data1,data2) in enumerate(t):
            if i == self.event:
                self.test_event_forward(data1,data2,0)
            if i == self.event-3:
                self.test_event_forward(data1,data2,1)
            if i == self.event-6:
                self.test_event_forward(data1,data2,2)
            if i == self.event+3:
                self.test_event_forward(data1,data2,3)
            if i == self.event+6:
                self.test_event_forward(data1,data2,4)
                break


    def test_forward(self,data1,data2,epoch,count_test,test_loss,TS74_0,TS74,TS133_0,TS133, SSIM_0,SSIM,TS_0,TS):
        inputs, targets = data1, data2
        inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())
        preds = self.model(inputs)
        inputs = torchcuda_numpy(normalize(inputs[:,0:1], self.SCALE, self.MEAN, reverse=True))
        preds = torchcuda_numpy(normalize(preds, self.SCALE, self.MEAN, reverse=True))
        targets = torchcuda_numpy(normalize(targets, self.SCALE, self.MEAN, reverse=True))  
        a=0.0001

        count_test += 1
        ts74_0 = CSI(inputs, targets,0.1)
        ts74 = CSI(preds, targets,0.1 )
        ts133_0= CSI(inputs, targets, 10)
        ts133 = CSI(preds, targets, 10)
        ts74_0[np.isnan(ts74_0)] = 0
        ts74[np.isnan(ts74)] = 0
        ts133_0[np.isnan(ts133_0)] = 0
        ts133[np.isnan(ts133)] = 0

        ts_0= CSI(inputs, targets, 25)
        ts = CSI(preds, targets, 25)
        ts_0[np.isnan(ts_0)] = 0
        ts[np.isnan(ts)] = 0
        ssim_0 = SSIM_func(inputs / 1000., targets / 1000.)
        ssim = SSIM_func(preds / 1000., targets / 1000.)

        TS74_0 += ts74_0.mean(axis=0)
        TS74 += ts74.mean(axis=0)
        TS133_0 += ts133_0.mean(axis=0)
        TS133 += ts133.mean(axis=0)
        TS_0 += ts_0.mean(axis=0)
        TS += ts.mean(axis=0)
        SSIM_0 += ssim_0.mean(axis=0)
        SSIM += ssim.mean(axis=0)
        return count_test, test_loss, TS74,TS133, SSIM,TS

    def test_metrics(self,epoch):
        with torch.no_grad():
            self.model.eval()
            count_test = 0
            test_loss = 0
            TS74 = np.zeros(self.pre_len)
            TS74_0 = np.zeros(self.pre_len)
            TS133 = np.zeros(self.pre_len)
            TS133_0 = np.zeros(self.pre_len)
            SSIM_0 = np.zeros(self.pre_len)
            SSIM = np.zeros(self.pre_len)

            TS = np.zeros(self.pre_len)
            TS_0 = np.zeros(self.pre_len)
            t = tqdm(self.val_loader, leave=False, total=len(self.val_loader))
            for i, (data1,data2) in enumerate(t):
                count_test, test_loss,  TS74, TS133, SSIM,TS = self.test_forward(data1,data2,epoch, count_test, test_loss,  TS74_0,TS74,TS133_0, TS133, SSIM_0,SSIM,TS_0,TS)
        
            print('test loss:')
            print(test_loss/count_test)
            print('TS0.1_0: ')
            print(TS74_0/count_test)
            print('TS0.1: ')
            print(TS74/count_test)
            print('TS10_0: ')
            print(TS133_0/count_test)
            print('TS10: ')
            print(TS133/count_test)
            print('TS25_0: ')
            print(TS_0/count_test)
            print('TS25: ')
            print(TS/count_test)
            print('SSIM_0: ')
            print(SSIM_0/count_test)
            print('SSIM: ')
            print(SSIM/count_test)
