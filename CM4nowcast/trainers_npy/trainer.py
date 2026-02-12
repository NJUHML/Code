import sys
sys.path.append("/root/code/CM4nowcast")
from dataset.datareader_amplifer_l import get_sevir_loader_spilt_l
from dataset.datareader import get_sevir_loader, get_sevir_loader_spilt
from dataset.datautils import *
from metrics.metrics import *
from utils.Loss_Assemble import BMSELoss, BMAELoss
from utils.utils import *
from utils.ManualWeightLoss import LinearWeightLoss
from utils.DynamicWeightAverageLoss import DynamicWeightAverageLoss
from utils.AutomaticWeightedLoss import AutomaticWeightedLoss
from utils.multi_scale_temporal_loss import MTloss, MTloss_add_linear, WTloss
from utils.wsloss1.wavelet_ssim_loss import WSloss_init_0, WSloss_init_2, WSloss_init_0_11, WSloss, WSloss_linear_add, WSloss_linear_add_adhoc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import datetime
import torch.optim as optim
from torch.autograd import Variable
import os
import numpy as np
import math
from tqdm import tqdm
sys.path.append("/root/code/CM4nowcast/models/UNet")
from UNet_MEFM import UNet_CM
sys.path.append("/root/code/CM4nowcast/models/DGMR-ConvRNN/DGMR")
from DGMRModels_scale import DGMRGenerator
sys.path.append("/root/code/CM4nowcast/models/EarthFormer_scale_run/earth-forecasting-transformer/scripts/cuboid_transformer/sevir")
from form_model import form_model_normal


class trainer():
    def __init__(self, opt):
        self.opt = opt
        self.save_dir = opt.log_dir
        self.dataset = opt.dataset
        self.batch_size = opt.batch_size
        self.patch_size = opt.patch_size

        self.dtype = torch.cuda.FloatTensor

        self.start_epoch = 0
        self.total_iter = 0
        num_iters = int(opt.total_epoch * opt.epoch_size / 2)
        self.delta = float(1) / num_iters

        self.seq_len = opt.seq_len
        self.pre_len = opt.pre_len

        self.epoch_size = opt.epoch_size

        self.row = opt.row
        self.col = opt.col
        self.datatype = opt.datatype
        self.evaltype = opt.evaltype
        self.losstype = opt.losstype
        self.left_resample = 0
        self.lossweighttype = opt.lossweighttype

        self.img_size = (self.row, self.col)
        self.SCALE = opt.scale
        self.MEAN = opt.mean
        self.model_name = opt.model_name
        self.torlerate_limit = opt.torlerate_limit
        self.event = opt.event_show
        self.display_interval = opt.display_interval
        
        if self.datatype == "npy_sevir":
            train_data, valid_data, test_data = load_dataset(opt)
            self.train_loader = DataLoader(train_data,num_workers=opt.data_threads,batch_size=opt.batch_size,shuffle=True,drop_last=True,pin_memory=True)
            self.test_loader = DataLoader(test_data,num_workers=opt.data_threads,batch_size=opt.batch_size,shuffle=False,drop_last=False,pin_memory=True)
            self.val_loader = DataLoader(valid_data,num_workers=opt.data_threads,batch_size=opt.batch_size,shuffle=True,drop_last=False,pin_memory=True)

        if self.datatype == "sevir":
            _, _, test_data = load_dataset(opt)
            self.test_loader = DataLoader(test_data,num_workers=opt.data_threads,batch_size=opt.batch_size,shuffle=False,drop_last=False,pin_memory=True)
            self.train_loader, self.val_loader = get_sevir_loader_spilt(batch_size=opt.batch_size,start_date=datetime.datetime(2017, 1, 1),end_date=datetime.datetime(2019, 6, 30),shuffle = True,num_workers = opt.data_threads,data_spilt = 0.9, random_seed = 18)
            if self.left_resample == 1:
                self.train_loader_l, self.val_loader_l = get_sevir_loader_spilt_l(batch_size=opt.batch_size,start_date=datetime.datetime(2017, 1, 1),end_date=datetime.datetime(2019, 6, 30),shuffle = True,num_workers = opt.data_threads, data_amplifer=4, data_spilt = 0.9, random_seed = 18)
            

        if opt.modeltype == 'UNet':
            self.model = UNet_CM(input_channel = opt.seq_len, n_classes = opt.pre_len,size = opt.row).cuda()
        elif opt.modeltype == 'CRNN':
            self.model = DGMRGenerator(in_step=self.seq_len, out_step=self.pre_len, size = 384).cuda()
        elif opt.modeltype == 'EF':
            self.model = form_model_normal().cuda()

        if self.losstype == 'multi':
            if self.lossweighttype == 'linear':
               self.mlw = LinearWeightLoss(num = 19)
            elif self.lossweighttype == 'DWAL':
               self.mlw = DynamicWeightAverageLoss(num = 19)
            elif self.lossweighttype == 'AWL':
               self.mlw = AutomaticWeightedLoss(num = 19)

        if self.lossweighttype == 'AWL':
            self.optimizer = optim.Adam([{'params': self.model.parameters(), 'lr': opt.lr}, {'params': self.mlw.parameters(), 'weight_decay': 0}])
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=opt.lr)

        self.scheduler = get_scheduler(self.optimizer, self.opt, opt.epoch_size)

        if opt.resume == 'True':
            print("previous parameter is loaded !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.model.load_state_dict(torch.load(os.path.join(opt.log_dir,opt.model_name+'.pkl')))

        if not os.path.exists(opt.log_dir):
            os.mkdir(opt.log_dir)

        if not os.path.exists(os.path.join(opt.log_dir,'npy_batch_'+str(self.batch_size)+'_event_'+str(self.event)+'_'+self.model_name)):
            os.mkdir(os.path.join(opt.log_dir,'npy_batch_'+str(self.batch_size)+'_event_'+str(self.event)+'_'+self.model_name))

        
        if self.losstype == 'multi':
            self.Loss_func3 = WSloss().cuda()
            self.Loss_func4 = MTloss(scaler=1.).cuda()
        else:
            #self.Loss_func3 = WSloss_linear_add().cuda()
            self.Loss_func3 = WSloss_linear_add_adhoc(multi_scaler=[opt.multi_scaler_l, opt.multi_scaler_m, opt.multi_scaler_h]).cuda()
            if opt.w_tem == 1:
                print('wavelet=========================================')
                self.Loss_func4 = WTloss().cuda()
            else:
                print('pooling=========================================')
                self.Loss_func4 = MTloss_add_linear(scaler=0.1, iterate=opt.t_iter).cuda()

        self.Loss_func1 = BMSELoss(scale=self.SCALE,mean=self.MEAN).cuda()
        self.Loss_func2 = BMAELoss(scale=self.SCALE,mean=self.MEAN).cuda()

    def Loss_func(self, preds, targets):
        if self.losstype == 'multi':
            return self.Loss_func1(preds,targets), self.Loss_func2(preds,targets), *self.Loss_func3(preds,targets), *self.Loss_func4(preds,targets)
        elif self.losstype == 'single':
            return self.Loss_func1(preds,targets)+self.Loss_func2(preds,targets)+self.Loss_func4(preds,targets)+self.Loss_func3(preds,targets)

        

    def name(self):
        return 'Trainer'

    def train_forward(self,data1,data2,t,epoch, train_loss, count_train):
        inputs, targets = data1, data2
        inputs, targets = datapreprocess_npy(inputs, targets, scale=self.SCALE, mean=self.MEAN)
        inputs, targets = Variable(inputs[:,(13-self.seq_len):].cuda()), Variable(targets.cuda())
        self.optimizer.zero_grad()
        preds = self.model(inputs)
        if self.losstype == 'single':
            loss = self.Loss_func(preds, targets)
            train_loss.append(loss.detach().cpu().numpy())
        if self.losstype == 'multi':
            loss, loss_item = self.mlw([self.Loss_func(preds, targets)], epoch_idx = self.epoch)
            train_loss.append(loss_item)
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
        best_mse = math.inf
        torlerate = 0
        train_loss_record = []
        val_loss_record = []
        #test_loss_record = []
        limit = self.torlerate_limit
        for epoch in range(self.epoch_size):
            self.epoch = epoch
            print(self.epoch)
            train_loss = []
            count_train = 0
            if self.left_resample == 1:
                t = tqdm(self.train_loader_l, leave=False, total=len(self.train_loader_l))
                for i, (data1, data2)  in enumerate(t):
                    train_loss, count_train = self.train_forward(data1,data2, t, epoch, train_loss, count_train)
            t = tqdm(self.train_loader, leave=False, total=len(self.train_loader))
            for i, (data1, data2)  in enumerate(t):
                train_loss, count_train = self.train_forward(data1,data2, t, epoch, train_loss, count_train)
            #if self.left_resample == 1:
            #    t = tqdm(self.train_loader_l, leave=False, total=len(self.train_loader_l))
            #    for i, (data1, data2)  in enumerate(t):
            #        train_loss, count_train = self.train_forward(data1,data2, t, epoch, train_loss, count_train)

                #if (i+1) % self.display_interval == 0:
                #    print('epoch: ' + str(epoch))
                #    print('training loss: ' + str(loss1)+', '+ str(loss2)+', '+ str(loss3))
            
            train_loss_record.append(np.array(train_loss).mean(0))
            #print('train loss: ' + str(train_loss / count_train))
            if self.evaltype == 'tolerance':
                val_loss, count_val = self.evalid()
                val_loss_record.append(np.array(val_loss).mean(0))
                print('train loss: ' + str(np.array(train_loss).mean(0))+'               '+'valid loss: ' + str(np.array(val_loss).mean(0))+'              '+'epoch:'+ str(epoch)+'              '+'torlerate:'+ str(torlerate))
                self.scheduler.step()
                print('learning rate = %.7f' % self.scheduler.get_last_lr()[0])
                if self.losstype == 'multi':
                    if self.lossweighttype == 'AWL':
                        print('weight: ')
                        print(self.mlw.lambda_weight[:,epoch]/count_train)
                        print('residual: ')
                        print(self.mlw.residual[:,epoch]/count_train)
                    else:
                        print('weight: ')
                        print(self.mlw.lambda_weight[:,epoch])
                if self.losstype == 'single':
                    val_mse = np.array(val_loss).mean(0)
                else:
                    if self.lossweighttype == 'AWL':
                        val_mse = np.array(val_loss).mean(0).mean(0)
                    else:
                        val_mse = np.array(val_loss).mean(0).mean(0)
                if val_mse < best_mse:
                    best_mse = val_mse
                    torch.save(self.model.state_dict(), os.path.join(self.save_dir,self.model_name+'.pkl'))
                    torlerate = 0
                else:
                    torlerate = torlerate + 1

                if torlerate == limit:
                    self.model.load_state_dict(torch.load(os.path.join(self.save_dir,self.model_name+'.pkl')))
                    np.save(os.path.join(self.save_dir, 'npy_batch_'+str(self.batch_size)+'_event_'+str(self.event)+'_'+self.model_name, 'train_loss_record.npy'),np.array(train_loss_record))
                    np.save(os.path.join(self.save_dir, 'npy_batch_'+str(self.batch_size)+'_event_'+str(self.event)+'_'+self.model_name, 'val_loss_record.npy'),np.array(val_loss_record))
                    if self.losstype == 'multi':
                        np.save(os.path.join(self.save_dir, 'npy_batch_'+str(self.batch_size)+'_event_'+str(self.event)+'_'+self.model_name, 'weight.npy'),self.mlw.lambda_weight)
                        if self.lossweighttype == 'AWL':
                            np.save(os.path.join(self.save_dir, 'npy_batch_'+str(self.batch_size)+'_event_'+str(self.event)+'_'+self.model_name, 'weight.npy'),self.mlw.lambda_weight/(count_train+count_val))
                            np.save(os.path.join(self.save_dir, 'npy_batch_'+str(self.batch_size)+'_event_'+str(self.event)+'_'+self.model_name, 'residual.npy'),self.mlw.residual/(count_train+count_val))
                    self.test_metrics()
                    self.test_event()
                    break


    def eval_forward(self, data1,data2, val_loss, count_val):
        inputs, targets = data1, data2
        inputs, targets = datapreprocess_npy(inputs, targets, scale=self.SCALE, mean=self.MEAN)
        inputs, targets = Variable(inputs[:,(13-self.seq_len):].cuda()), Variable(targets.cuda())
        preds = self.model(inputs)
        if self.losstype == 'single':
            loss = self.Loss_func(preds, targets)
            val_loss.append(loss.detach().cpu().numpy())
        if self.losstype == 'multi':
            loss, loss_item = self.mlw([self.Loss_func(preds, targets)],epoch_idx = self.epoch)
            val_loss.append(loss_item)
        count_val = count_val + 1
        return val_loss, count_val


    def evalid(self):
        with torch.no_grad():
            self.model.eval()
            val_loss = []
            count_val = 0
            t = tqdm(self.val_loader, leave=False, total=len(self.val_loader))
            for i, (data1, data2)  in enumerate(t):
                val_loss, count_val = self.eval_forward(data1,data2, val_loss, count_val)
            if self.left_resample == 1:
                t = tqdm(self.val_loader_l, leave=False, total=len(self.val_loader_l))
                for i, (data1, data2)  in enumerate(t):
                    val_loss, count_val = self.eval_forward(data1,data2, val_loss, count_val)
        return val_loss, count_val

    def test_event_forward(self, data1,data2,idx1):
        inputs, targets = data1, data2
        inputs, targets = datapreprocess_npy(inputs, targets, scale=self.SCALE, mean=self.MEAN)
        inputs, targets = Variable(inputs[:,(13-self.seq_len):].cuda()), Variable(targets.cuda())
        preds = self.model(inputs)
        preds = torchcuda_numpy(normalize(preds, self.SCALE, self.MEAN, reverse=True))
        targets = torchcuda_numpy(normalize(targets, self.SCALE, self.MEAN, reverse=True))
        np.save(os.path.join(self.save_dir,'npy_batch_'+str(self.batch_size)+'_event_'+str(self.event)+'_'+self.model_name,'preds'+str(idx1)+'.npy'),preds)
        np.save(os.path.join(self.save_dir, 'npy_batch_'+str(self.batch_size)+'_event_'+str(self.event)+'_'+self.model_name, 'targets' + str(idx1) + '.npy'), targets)

    def test_event(self):
        self.model.eval()
        t = tqdm(self.test_loader, leave=False, total=len(self.test_loader))
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


    def test_forward(self,data1,data2,count_test,test_loss,TS74,TS133,SSIM,t):
        inputs, targets = data1, data2
        inputs, targets = datapreprocess_npy(inputs, targets, scale=self.SCALE, mean=self.MEAN)
        inputs, targets = Variable(inputs[:,(13-self.seq_len):].cuda()), Variable(targets.cuda())
        preds = self.model(inputs)
        if self.losstype == 'single':
            loss = self.Loss_func(preds, targets)
            test_loss.append(loss.detach().cpu().numpy())
        if self.losstype == 'multi':
            loss, loss_item = self.mlw([self.Loss_func(preds, targets)],epoch_idx = self.epoch)
            test_loss.append(loss_item)
        preds = torchcuda_numpy(normalize(preds, self.SCALE, self.MEAN, reverse=True))
        targets = torchcuda_numpy(normalize(targets, self.SCALE, self.MEAN, reverse=True))
        count_test += 1
        ts74 = CSI(preds, targets, 74)
        ts133 = CSI(preds, targets, 133)
        ts74[np.isnan(ts74)] = 0
        ts133[np.isnan(ts133)] = 0
        ssim = SSIM_func(preds / 255., targets / 255.)
        TS74 += ts74.mean(axis=0)
        TS133 += ts133.mean(axis=0)
        SSIM += ssim.mean(axis=0)
        #t.set_postfix({
        #    'TS74': ts74.mean(axis=0),
        #    'TS133': ts133.mean(axis=0),
        #    'SSIM': ssim.mean(axis=0)
        #})
        return count_test, test_loss, TS74, TS133, SSIM

    def test_metrics(self):
        with torch.no_grad():
            self.model.eval()
            count_test = 0
            test_loss = []
            TS74 = np.zeros(self.pre_len)
            TS133 = np.zeros(self.pre_len)
            SSIM = np.zeros(self.pre_len)
            t = tqdm(self.test_loader, leave=False, total=len(self.test_loader))
            for i, (data1,data2) in enumerate(t):
                count_test, test_loss, TS74, TS133, SSIM = self.test_forward(data1,data2, count_test, test_loss, TS74, TS133, SSIM, t)
        
            print('test loss:')
            print(np.array(test_loss).mean(0))
            print('TS74: ')
            print(TS74/count_test)
            print('TS133: ')
            print(TS133/count_test)
            print('SSIM: ')
            print(SSIM/count_test)
            np.save(os.path.join(self.save_dir, 'npy_batch_'+str(self.batch_size)+'_event_'+str(self.event)+'_'+self.model_name, 'TS74.npy'),TS74/count_test)
            np.save(os.path.join(self.save_dir, 'npy_batch_'+str(self.batch_size)+'_event_'+str(self.event)+'_'+self.model_name, 'TS133.npy'),TS133/count_test)
