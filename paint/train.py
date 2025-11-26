from utils.segmentation_trainer import SegTrainer
import argparse
import os
import torch
args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--name', default='phase1_pearl',
                        help='the name of the experiment')
    parser.add_argument('--data-dir', default='',
                        help='training data directory')
    parser.add_argument('--save-dir', default='./logs',
                        help='directory to save models.')
    parser.add_argument('--val_path', default='./logs',
                        help='validation path')
    parser.add_argument('--test_path', default='./logs',
                        help='test path')
    parser.add_argument('--train_path', default='./logs',
                        help='train path')
    parser.add_argument('--model-type', default='unet',
                        help='model name unet, unet++, deeplabv3+, fcn8s')
    parser.add_argument('--seed', type=int, default=15,
                        help='random seed')
    parser.add_argument('--ce_weight', type=float, default=0.1,
                        help='cross entropy weight')
    parser.add_argument('--binary_weight', type=float, default=0.0,
                        help='binary loss weight')
    parser.add_argument('--prob_full', type=float, default=0.5,
                        help='probability to use full image')

    parser.add_argument('--lr', type=float, default=5e-5,
                        help='the initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='the weight decay')
    parser.add_argument('--resume', default='',
                        help='the path of resume training model')
    parser.add_argument('--max-model-num', type=int, default=2,
                        help='max models num to save ')
    parser.add_argument('--max-epoch', type=int, default=1000,
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=5,
                        help='the epoch start to val')

    parser.add_argument('--batch-size', type=int, default=8,
                        help='train batch size')

    parser.add_argument('--num-workers', type=int, default=8,
                        help='the num of training process')

    parser.add_argument('--crop-size', type=int, default=1024,
                        help='the crop size of the train image')
    parser.add_argument('--val-resize', type=int, default=1024,
                        help='the crop size of the train image')
    parser.add_argument('--downsample-ratio', type=int, default=32,
                        help='downsample ratio')
    parser.add_argument('--num-cls', type=int, default=9,
                        help='number of classes')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    trainer = SegTrainer(args)
    trainer.setup()
    trainer.train()
