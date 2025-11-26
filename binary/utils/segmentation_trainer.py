import torch
from utils.trainer import Trainer
import logging
from datasets.plant_dataset import Plant
from torch.utils.data import DataLoader
import os
from models.unetplus import ResNet34UnetPlus
from models.unet import UnetResnet34
from torch import optim
from utils.helper import Save_Handle, AverageMeter
import numpy as np
import time
import random
from losses.diceloss import DiceLoss
from utils.val_metric import multi_class_dice_coefficient
from utils.calc_f1 import calculate_multiclass_f1
from utils.calc_miou import compute_miou
from models.deeplab_resnet import DeepLabv3_plus
from models.FCN8s import fcn8s
from models.pspnet import PSPNet
from models.icnet import ICNet
from models.bisenet import BiSeNet
from models.danet import DANet
from models.stdc import STDC
from models.stdc import LaplacianConv
import torch.nn.functional as F
import torch.nn as nn


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class SegTrainer(Trainer):
    def setup(self):
        args = self.args

        if args.seed != -1:
            setup_seed(args.seed)
            print('Random seed is set as {}'.format(args.seed))

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            assert self.device_count == 1
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        self.downsample_ratio = args.downsample_ratio
        self.datasets ={x: Plant(args.data_dir,
                                 x, args.crop_size) for x in ['train', 'val']}
        logging.info('Number of Train {}, Number of Val:{}'.format(len(self.datasets['train']), len(self.datasets['val'])))
        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          batch_size=(args.batch_size
                                                      if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          pin_memory=(True if x == 'train' else False)) for x in ['train', 'val']}
        if args.model_type == 'unet++':
            self.model = ResNet34UnetPlus(num_channels=3, num_classes=args.num_cls)
        elif args.model_type == 'unet':
            self.model = UnetResnet34(num_classes=args.num_cls)
        elif args.model_type == 'deeplabv3+':
            self.model = DeepLabv3_plus(nInputChannels=3, n_classes=args.num_cls, os=16, pretrained=True, _print=True)
        elif args.model_type == 'fcn8s':
            self.model = fcn8s(num_classes=args.num_cls)
        elif args.model_type == 'pspnet':
            self.model = PSPNet(num_classes=args.num_cls)
        elif args.model_type == 'icnet':
            self.model = ICNet(nclass=args.num_cls)
        elif args.model_type == 'bisenet':
            self.model = BiSeNet(num_classes=args.num_cls, context_path='resnet101')
        elif args.model_type == 'danet':
            self.model = DANet(n_classes=args.num_cls)
        elif args.model_type == 'stdc':
            self.model = STDC(num_class=args.num_cls, use_detail_head=True)
            self.laplacian_conv = LaplacianConv(self.device)
        self.model.to(self.device)
        self.loss = DiceLoss(args.num_cls).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, map_location=self.device))

        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_val = 0


    def train(self):
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-' * 50 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-' * 50)
            self.epoch = epoch
            self.train_epoch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()

    def train_epoch(self):
        epoch_loss = AverageMeter()
        epoch_start = time.time()
        self.model.train() # Set model to training mode

        # Iterate over data.
        for step, (img, targets) in enumerate(self.dataloaders['train']):
            inputs = img.to(self.device)
            targets = targets.to(self.device)
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                if type(outputs) == tuple:
                    #loss = self.loss(outputs[0], targets, softmax=True)
                    loss = nn.CrossEntropyLoss()(outputs, targets.squeeze(1))
                    if self.args.model_type == 'stdc':
                        details = self.laplacian_conv(targets.float())
                        details = self.model.detail_conv(details)
                        details[details > 0.1] = 1
                        details[details <= 0.1] = 0
                        detail_size = details.size()[2:]
                        preds_detail = F.interpolate(outputs[1].sigmoid(), detail_size, mode='bilinear', align_corners=True)
                        loss += self.loss(preds_detail, details, softmax=False)
                    else:
                        for i in range(len(outputs) - 1):
                            loss += 0.4 * self.loss(outputs[i + 1], targets, softmax=True)
                else:
                    loss = self.loss(outputs, targets, softmax=True)
                    #loss = nn.CrossEntropyLoss()(outputs, targets.squeeze(1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                N = inputs.size(0)
                epoch_loss.update(loss.item(), N)

        logging.info('Epoch {} Train, Loss: {:.2f},  Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(),
                             time.time() - epoch_start))
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)
        self.save_list.append(save_path)  # control the number of saved models

    def val_epoch(self):
        epoch_start = time.time()
        self.model.eval()
        epoch_eval_dice = []
        epoch_eval_miou = []
        epoch_eval_acc = []
        epoch_eval_f1 = []
        for img, labels in self.dataloaders['val']:
            inputs = img.to(self.device)
            labels = labels.squeeze(1).to(self.device)
            assert inputs.size(0) == 1
            with torch.set_grad_enabled(False):
                B, C, H, W = inputs.shape
                inputs = torch.nn.functional.interpolate(inputs, size=(1024, 1024), mode='bilinear')
                outputs = self.model(inputs)
                if type(outputs) == tuple:
                    outputs = outputs[0]
                outputs = torch.nn.functional.interpolate(outputs, size=(H, W), mode='nearest').argmax(dim=1)
                dice_score = multi_class_dice_coefficient(outputs, labels, self.args.num_cls)
                miou_score = compute_miou(outputs, labels, self.args.num_cls)
                f1_score = calculate_multiclass_f1(outputs, labels, self.args.num_cls)
                acc_score = (outputs == labels).sum().item() / (labels.shape[1] * labels.shape[2])
                epoch_eval_dice.append(dice_score)
                epoch_eval_miou.append(miou_score)
                epoch_eval_f1.append(f1_score)
                epoch_eval_acc.append(acc_score)

        epoch_eval_dice = np.array(epoch_eval_dice)
        epoch_dice = np.mean(np.abs(epoch_eval_dice))*100
        epoch_eval_miou = np.array(epoch_eval_miou)
        epoch_miou = np.mean(np.abs(epoch_eval_miou))*100
        epoch_eval_f1 = np.array(epoch_eval_f1)
        epoch_f1 = np.mean(np.abs(epoch_eval_f1))*100
        epoch_eval_acc = np.array(epoch_eval_acc)
        epoch_acc = np.mean(np.abs(epoch_eval_acc))*100
        logging.info('Epoch {} Val, dice_score: {:.2f}, miou: {:.2f}, F1:{:.2f}, Acc:{:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_dice, epoch_miou, epoch_f1, epoch_acc, time.time() - epoch_start))

        model_state_dic = self.model.state_dict()
        if epoch_dice > self.best_val:
            self.best_val = epoch_dice
            logging.info("save best dice {:.2f} model epoch {}".format(self.best_val,
                                                                                 self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))




