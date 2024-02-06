import datetime
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import math
from torch.cuda import amp
import model, utils
from spikingjelly.clock_driven import functional
from spikingjelly.datasets import cifar10_dvs
from timm.models import create_model
from timm.data import Mixup
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import autoaugment
_seed_ = 2021
import random
random.seed(2021)
root_path = os.path.abspath(__file__)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np

import pdb
from torch.utils import data
from torchvision.transforms import transforms
from numpy.random import randint
import glob
from PIL import Image
from video_transform import *
import os.path as osp



np.random.seed(_seed_)
writer = SummaryWriter("./")
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--model', default='SEWResNet', help='model')
    parser.add_argument('--dataset', default='cifar10dvs', help='dataset')
    parser.add_argument('--num-classes', type=int, default=114, metavar='N',
                        help='number of label classes (default: 1000)')

    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')

    parser.add_argument('--data-path', default='/media/data/cifar10dvs/', help='dataset')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=2, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--print-freq', default=256, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./logs', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        # default=True,
        help="Only test the model",
        action="store_true",
    )

    # Mixed precision training parameters
    parser.add_argument('--amp', default=True, action='store_true',
                        help='Use AMP training')


    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--tb', default=True,  action='store_true',
                        help='Use TensorBoard to record logs')
    parser.add_argument('--T', default=16, type=int, help='simulation steps')
    # parser.add_argument('--adam', default=True, action='store_true',
    #                     help='Use Adam')

    # Optimizer Parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar="OPTIMIZER", help='Optimizer (default: "adamw")')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, metavar='BETA', help='Optimizer Betas')
    parser.add_argument('--weight-decay', default=0.06, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='Momentum for SGD. Adam will not use momentum')

    parser.add_argument('--connect_f', default='ADD', type=str, help='element-wise connect function')
    parser.add_argument('--T_train', default=None, type=int)

    #Learning rate scheduler
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                        help='learning rate cycle limit')
    parser.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--epochs', type=int, default=96, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                        help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--decay-epochs', type=float, default=20, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation & regularization parameters
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--mixup', type=float, default=0.5,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix', type=float, default=0.,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=0.5,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                        help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
    args = parser.parse_args()
    return args

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])
class VideoDataset(data.Dataset):
    def __init__(self, list_file, num_segments, duration, mode, transform,image_size):
        self.list_file = list_file
        self.duration = duration
        self.num_segments = num_segments
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        self._parse_list()
        pass

    def _parse_list(self):
        # check the frame number is large >=16:
        # form is [video_id, num_frames, class_idx]
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        # tmp = [item for item in tmp if int(item[1]) >= 16]
        tmp = [item for item in tmp]
        self.video_list = [VideoRecord(item) for item in tmp]
        print(('video number:%d' % (len(self.video_list))))

    def _get_train_indices(self, record):
        # split all frames into seg parts, then select frame in each part randomly
        average_duration = (record.num_frames - self.duration + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.duration + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_test_indices(self, record):
        # split all frames into seg parts, then select frame in the mid of each part
        if record.num_frames > self.num_segments + self.duration - 1:
            tick = (record.num_frames - self.duration + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]
        if self.mode == 'train':
            segment_indices = self._get_train_indices(record)
        elif self.mode == 'test':
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)

    def get(self, record, indices):
        ##################################################################################################################################################
        Event_Tensor_dir = 'HARDVS_point_frames_12/'+record.path
        # video_frames_path.sort()
        images = list()
        fileLIST = os.listdir(Event_Tensor_dir)
        fileLIST.sort()
        total_frames_point= len(fileLIST)

        frames_point_clip_offsets = np.array([random.randint(0,total_frames_point)])
        frames_point_inds = frames_point_clip_offsets[:, None] + np.arange(self.num_segments)[None, :] * 1
        frames_point_inds = np.concatenate(frames_point_inds)
        frames_point_inds = np.mod(frames_point_inds,total_frames_point)

        for i, idx in enumerate(frames_point_inds):
            data_path = osp.join(Event_Tensor_dir, fileLIST[idx])
            load_data = np.load(data_path, allow_pickle=True)
            point_frames = load_data['point_frames']
            point_frames = point_frames[np.newaxis, :]
            images.append(point_frames)
        images = np.concatenate(images,axis=0)
        images = torch.tensor(images)  #torch.Size([16, 346, 260, 12])
        ##################################################################################################################################################
        rgb_dir  = 'HARDVS/rawframes/'+record.path+record.path[10:]+'_aps'
        rgb_images = list()
        fileLIST_rgb = os.listdir(rgb_dir)
        fileLIST_rgb.sort()
        total_frames_point_rgb= len(fileLIST_rgb)

        frames_point_clip_offsets = np.array([random.randint(0,total_frames_point_rgb)])
        frames_point_inds = frames_point_clip_offsets[:, None] + np.arange(self.num_segments)[None, :] * 1
        frames_point_inds = np.concatenate(frames_point_inds)
        frames_point_inds = np.mod(frames_point_inds,total_frames_point_rgb)
        for i, idx in enumerate(frames_point_inds):
            data_path = osp.join(rgb_dir, fileLIST_rgb[idx])
            rgb_imgs = [Image.open(os.path.join(data_path)).convert('RGB')]
            rgb_images.extend(rgb_imgs)
        rgb_images = self.transform(rgb_images)
        rgb_images = torch.reshape(rgb_images, (-1, 3, self.image_size, self.image_size))
        muti_model_data=[]
        muti_model_data.append(images)
        muti_model_data.append(rgb_images)
        ##################################################################################################################################################
        return muti_model_data, record.label

    def __len__(self):
        return len(self.video_list)


def train_data_loader(data_set):
    image_size = 240
    train_transforms = torchvision.transforms.Compose([GroupRandomSizedCrop(image_size),
                                                       GroupRandomHorizontalFlip(),
                                                       Stack(),
                                                       ToTorchFormatTensor()])
    train_data = VideoDataset(list_file="HARDVS/HarDvs_list/train_label.txt",
                              # num_segments=16,
                              num_segments=16,
                              duration=1,
                              mode='train',
                              transform=train_transforms,
                              image_size=image_size)
    return train_data

def test_data_loader(data_set):
    image_size = 240
    test_transforms = torchvision.transforms.Compose([GroupRandomSizedCrop(image_size),
                                                       GroupRandomHorizontalFlip(),
                                                       Stack(),
                                                       ToTorchFormatTensor()])
    # test_transforms = torchvision.transforms.Compose([ToTorchFormatTensor()])
    test_data = VideoDataset(list_file="HARDVS/HarDvs_list/test_label.txt",
                             # num_segments=16,
                             num_segments=16,
                             duration=1,
                             mode='test',
                             transform=test_transforms,
                             image_size=image_size)
    return test_data
    ############################################################################################################





def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, scaler=None, T_train=None, aug=None, trival_aug=None, mixup_fn=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)
    for muti_data, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        event_tensor=muti_data[0] #torch.Size([8, 16, 346, 260, 12])
        rgb_images=muti_data[1]    #torch.Size([8, 16, 3, 224, 224])

        #########################################################################################################
        event_tensor, target = event_tensor.to(device), target.to(device)
        event_tensor = event_tensor.float().permute(0,1,4,2,3)  # [N, T, C, H, W]
        N,T,C,H,W = event_tensor.shape
        if aug != None:
            event_tensor = torch.stack([(aug(event_tensor[i])) for i in range(N)])
        if trival_aug != None:
            event_tensor = torch.stack([(trival_aug(event_tensor[i])) for i in range(N)])
        if mixup_fn is not None:
            event_tensor, target = mixup_fn(event_tensor, target)
            target_for_compu_acc = target.argmax(dim=-1)
        if T_train:
            sec_list = np.random.choice(event_tensor.shape[1], T_train, replace=False)
            sec_list.sort()
            event_tensor = event_tensor[:, sec_list]
        #########################################################################################################
        rgb_images = rgb_images.to(device)
        rgb_images = rgb_images.float() # [N, T, C, H, W]
        N,T,C,H,W = rgb_images.shape
        if aug != None:
            # image = image.flatten(1, 2).contiguous() # 合并T,C
            rgb_images = torch.stack([(aug(rgb_images[i])) for i in range(N)])
            # image = image.reshape(N,T,C,H,W)

        if trival_aug != None:
            # image = image.flatten(0,1).contiguous()
            rgb_images = torch.stack([(trival_aug(rgb_images[i])) for i in range(N)])
            # image = image.reshape(N,T,C,H,W).contiguous()

        if T_train:
            sec_list = np.random.choice(rgb_images.shape[1], T_train, replace=False)
            sec_list.sort()
            rgb_images = rgb_images[:, sec_list]
        #########################################################################################################
        if scaler is not None:
            with amp.autocast():
                output = model(event_tensor,rgb_images)
                loss = criterion(output, target)
        else:
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            optimizer.step()

        functional.reset_net(model)
        if mixup_fn is not None:
            acc1, acc5 = utils.accuracy(output, target_for_compu_acc, topk=(1, 5))
        else:
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = event_tensor.shape[0]
        loss_s = loss.item()
        if math.isnan(loss_s):
            raise ValueError('loss is Nan')
        acc1_s = acc1.item()
        acc5_s = acc5.item()
        metric_logger.update(loss=loss_s, lr=optimizer.param_groups[0]["lr"])

        metric_logger.meters['acc1'].update(acc1_s, n=batch_size)
        metric_logger.meters['acc5'].update(acc5_s, n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg



def evaluate(model, criterion, data_loader, device, print_freq=100, header='Test:'):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    with torch.no_grad():
        for muti_data, target in metric_logger.log_every(data_loader, print_freq, header):

            start_time = time.time()
            event_tensor=muti_data[0] #torch.Size([8, 16, 346, 260, 12])
            rgb_images=muti_data[1]    #torch.Size([8, 16, 3, 224, 224])

            #########################################################################################################
            event_tensor = event_tensor.to(device, non_blocking=True)
            rgb_images = rgb_images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            event_tensor = event_tensor.float().permute(0,1,4,2,3)  #torch.Size([12, 16, 346, 260, 12])
            rgb_images = rgb_images.float()  #torch.Size([12, 16, 346, 260, 12])
            
            output = model(event_tensor,rgb_images)
            loss = criterion(output, target)
            functional.reset_net(model)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = event_tensor.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    loss, acc1, acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    print(f' * Acc@1 = {acc1}, Acc@5 = {acc5}, loss = {loss}')
    return loss, acc1, acc5



def main(args):

    max_test_acc1 = 0.
    test_acc5_at_max_test_acc1 = 0.


    train_tb_writer = None
    te_tb_writer = None

    utils.init_distributed_mode(args)
    print(args)

    output_dir = os.path.join(args.output_dir, f'{args.model}_b{args.batch_size}_T{args.T}')

    if args.T_train:
        output_dir += f'_Ttrain{args.T_train}'

    if args.weight_decay:
        output_dir += f'_wd{args.weight_decay}'


    if args.opt == 'adamw':
        output_dir += '_adamw'
    else:
        output_dir += '_sgd'

    if args.connect_f:
        output_dir += f'_cnf_{args.connect_f}'

    if not os.path.exists(output_dir):
        utils.mkdir(output_dir)

    output_dir = os.path.join(output_dir, f'lr{args.lr}')
    if not os.path.exists(output_dir):
        utils.mkdir(output_dir)

    device = torch.device(args.device)

    data_path = args.data_path
    ###########################################################################################################################################
    # train_transforms = torchvision.transforms.Compose([Stack(),
    #                                                     ToTorchFormatTensor()])
    train_data = train_data_loader(data_set=args.dataset)
    test_data = test_data_loader(data_set=args.dataset)
    data_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)
    data_loader_test = torch.utils.data.DataLoader(test_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    ###########################################################################################################################################

    model = create_model(
        'Spikingformer',
        pretrained=False,
        drop_rate=0.,
        drop_path_rate=0.1,
        drop_block_rate=None,
    )
    print("Creating model")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # criterion_train = LabelSmoothingCrossEntropy()
    criterion_train = SoftTargetCrossEntropy().cuda()
    criterion = nn.CrossEntropyLoss()

    optimizer = create_optimizer(args, model)
    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=128, eta_min=1e-5)
    #op = torch.optim.adamw()
    start_epoch = 0
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        max_test_acc1 = checkpoint['max_test_acc1']
        test_acc5_at_max_test_acc1 = checkpoint['test_acc5_at_max_test_acc1']

    if args.test_only:

        evaluate(model, criterion, data_loader_test, device=device, header='Test:')

        return

    if args.tb and utils.is_main_process():
        purge_step_train = args.start_epoch
        purge_step_te = args.start_epoch
        train_tb_writer = SummaryWriter(output_dir + '_logs/train', purge_step=purge_step_train)
        te_tb_writer = SummaryWriter(output_dir + '_logs/te', purge_step=purge_step_te)
        with open(output_dir + '_logs/args.txt', 'w', encoding='utf-8') as args_txt:
            args_txt.write(str(args))

        print(f'purge_step_train={purge_step_train}, purge_step_te={purge_step_te}')


    train_snn_aug = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5)
                    ])
    train_trivalaug = autoaugment.SNNAugmentWide()
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        mixup_fn = Mixup(**mixup_args)
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, num_epochs):
        save_max = False
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if epoch >= 75:
            mixup_fn.mixup_enabled = False
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model, criterion_train, optimizer, data_loader, device, epoch,
            args.print_freq, scaler, args.T_train,
            train_snn_aug, train_trivalaug, mixup_fn)
        if utils.is_main_process():
            train_tb_writer.add_scalar('train_loss', train_loss, epoch)
            train_tb_writer.add_scalar('train_acc1', train_acc1, epoch)
            train_tb_writer.add_scalar('train_acc5', train_acc5, epoch)
        lr_scheduler.step(epoch + 1)

        test_loss, test_acc1, test_acc5 = evaluate(model, criterion, data_loader_test, device=device, header='Test:')
        if te_tb_writer is not None:
            if utils.is_main_process():

                te_tb_writer.add_scalar('test_loss', test_loss, epoch)
                te_tb_writer.add_scalar('test_acc1', test_acc1, epoch)
                te_tb_writer.add_scalar('test_acc5', test_acc5, epoch)


        if max_test_acc1 < test_acc1:
            max_test_acc1 = test_acc1
            test_acc5_at_max_test_acc1 = test_acc5
            save_max = True


        if output_dir:

            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
                'max_test_acc1': max_test_acc1,
                'test_acc5_at_max_test_acc1': test_acc5_at_max_test_acc1,
            }

            if save_max:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(output_dir, 'checkpoint_max_test_acc1.pth'))
        print(args)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))

        print('Training time {}'.format(total_time_str), 'max_test_acc1', max_test_acc1, 'test_acc5_at_max_test_acc1', test_acc5_at_max_test_acc1)
        print(output_dir)
    if output_dir:
        utils.save_on_master(
            checkpoint,
            os.path.join(output_dir, f'checkpoint_{epoch}.pth'))

    return max_test_acc1





if __name__ == "__main__":
    args = parse_args()
    main(args)
