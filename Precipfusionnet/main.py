
import sys
from trainer import trainer_unet
import torch
import argparse
import random
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
torch.cuda.set_device(4)
parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--scale', default=1000., type=float, help='scale')
parser.add_argument('--mean', default=0., type=float, help='mean')
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--log_dir', default='/scratch/zhangzy/output/UNetGFS2GPM_China/202504', help='base directory to save logs')
parser.add_argument('--name', default='', help='identifier for directory')

parser.add_argument('--model', default='pamconvlstm', help='model type (convlstm | uconvlstm | predrnn)')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--model_name', default='PrecipFusionNet_epoch', help='model name')
parser.add_argument('--datatype',type = str, default='GFS2GPM_f024')
parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')
parser.add_argument('--trainning', type=int, default=1, help='train or test')
parser.add_argument('--total_epoch', type=int, default=151, help='number of epochs to train for')

parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=151, help='epoch size')
parser.add_argument('--load_size', type=int, default=720, help='the size of the image short edge be loaded in dataset')

parser.add_argument('--input_nc', default=1, type=int)
parser.add_argument('--output_nc', default=1, type=int)
parser.add_argument('--data_threads', type=int, default=16, help='number of data loading threads')
parser.add_argument('--torlerate_limit', type=int, default=20, help='torlerate_limit')
parser.add_argument('--lr_policy', type=str, default='cosine', help='lr_policy')

opt = parser.parse_args()

opt.name = 'model=%s-patch_size=%d-batch_size=%d-rnn_size=%d-nlayer=%d-filter_size=%d-lr=%f' % (
    opt.model, opt.patch_size, opt.batch_size, opt.rnn_size, opt.rnn_nlayer, opt.filter_size, opt.lr)

random.seed(opt.seed)
torch.manual_seed(opt.seed)

print(opt)

trainer = trainer_unet(opt)
# --------- training loop ------------------------------------
if __name__ == '__main__':
    if opt.trainning:
        trainer.train()
    else:
        trainer.test_metrics()
        trainer.test_event()

