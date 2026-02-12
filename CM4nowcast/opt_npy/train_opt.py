import sys
sys.path.append('/root/code/CM4nowcast')
from trainers_npy.trainer import trainer
import torch
import argparse
import random
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--scale', default=255., type=float, help='scale')
parser.add_argument('--mean', default=0., type=float, help='mean')
parser.add_argument('--multi_scaler_l', default=1., type=float)
parser.add_argument('--multi_scaler_m', default=1., type=float)
parser.add_argument('--multi_scaler_h', default=1., type=float)
parser.add_argument('--w_tem', default=0, type=int)
parser.add_argument('--t_iter', default=3, type=int)
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--log_dir', default='/root/autodl-tmp/output/UNet', help='base directory to save logs')
parser.add_argument('--name', default='', help='identifier for directory')
parser.add_argument('--data_root', default='D:/video prediction/code/data/Moving_MNIST',
                    help='root directory for data | smnist or mmnist is data, predminist is /datasets/MovingMnist-example')

parser.add_argument('--model', default='pamconvlstm', help='model type (convlstm | uconvlstm | predrnn)')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--model_name', default='UNet', help='model name')
parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')
parser.add_argument('--modeltype', type=str, default='UNet')
parser.add_argument('--datatype', type=str, default='sevir')
parser.add_argument('--evaltype', type=str, default='tolerance')
parser.add_argument('--losstype', type=str, default='multi')
parser.add_argument('--trainning', type=int, default=1, help='train or test')
parser.add_argument('--total_epoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--criterion', type=str, default='MSE', help='loss function: MSE|BCE|L1|MSE&L1')

parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=100, help='epoch size')
parser.add_argument('--load_size', type=int, default=720, help='the size of the image short edge be loaded in dataset')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--image_height', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--patch_size', type=int, default=4,
                    help='the patch size, the input size will be image_width/patch_size')

parser.add_argument('--dataset', default='predmnist',
                    help='dataset to train with | smnist, mmnist, predmnist, kth, bair, ucf, kitti')
parser.add_argument('--input_nc', default=1, type=int)
parser.add_argument('--output_nc', default=1, type=int)
parser.add_argument('--seq_len', type=int, default=13, help='number of prior frames in a sequence')
parser.add_argument('--pre_len', type=int, default=12, help='number of frames be predicted')
parser.add_argument('--row', type=int, default=384, help='row')
parser.add_argument('--col', type=int, default=384, help='col')
parser.add_argument('--eval_len', type=int, default=10, help='number of frames to predict during eval')
parser.add_argument('--rnn_size', type=int, default=64, help='dimensionality of hidden layer')
parser.add_argument('--rnn_nlayer', type=int, default=4, help='number of convrnn layers')
parser.add_argument('--filter_size', type=int, default=5, help='filter size of the convrnn')

parser.add_argument('--data_threads', type=int, default=4, help='number of data loading threads')
parser.add_argument('--num_digits', type=int, default=2, help='number of digits for moving mnist')
parser.add_argument('--torlerate_limit', type=int, default=3, help='torlerate_limit')

parser.add_argument('--lossweighttype', type=str, default='linear', help='lossweighttype: linear, AWL, DWAL')
parser.add_argument('--lr_policy', type=str, default='cosine', help='lr_policy')
parser.add_argument('--niter', type=int, default=5, help='niter')
parser.add_argument('--event_show', type=int, default=33, help='event_show')
parser.add_argument('--display_interval', type=int, default=100, help='display_interval')

opt = parser.parse_args()

opt.name = 'model=%s-patch_size=%d-batch_size=%d-rnn_size=%d-nlayer=%d-filter_size=%d-lr=%f' % (
    opt.model, opt.patch_size, opt.batch_size, opt.rnn_size, opt.rnn_nlayer, opt.filter_size, opt.lr)

random.seed(opt.seed)
torch.manual_seed(opt.seed)

print(opt)

trainer = trainer(opt)

# --------- training loop ------------------------------------
if __name__ == '__main__':
    if opt.trainning:
        trainer.train()
    else:
        trainer.test_metrics()
        trainer.test_event()

