"""Run training."""

import shutil, random
import time
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torchvision

from dataset import CoviarDataSet
from model import Model
from train_options import parser
from transforms import GroupCenterCrop
from transforms import GroupScale

from torchsummary import summary

from spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter,
                                PickFirstChannels)

is_attention = False
torch.manual_seed(7) #固定參數初始化種子

SAVE_FREQ = 40
PRINT_FREQ = 20
best_prec1 = 0

#Train Record
train_loss = []
train_prec = []
train_lr = []

#Valid Record
valid_loss = []
valid_prec = []


def get_lr(optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lr = float(param_group['lr'])
        lrs.append(lr)

    return max(lrs)


def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()

    random.seed(torch_seed + worker_id)

    if torch_seed >= 2**32:
        torch_seed = torch_seed % 2**32
    np.random.seed(torch_seed + worker_id)


def main():
    global args
    global best_prec1
    args = parser.parse_args()

    print('Training arguments:')
    for k, v in vars(args).items():
        print('\t{}: {}'.format(k, v))

    if args.data_name == 'ucf101':
        num_class = 101
    elif args.data_name == 'hmdb51':
        num_class = 51
    elif args.data_name == 'mine':
        num_class = 2
    else:
        raise ValueError('Unknown dataset '+ args.data_name)

    model = Model(num_class, args.num_segments, args.representation, base_model=args.arch)
    print(model)

    if 'resnet3D' in args.arch:
        train_crop_min_ratio = 0.75
        train_crop_min_scale = 0.25
        mean = [0.4345, 0.4051, 0.3775]
        std = [0.2768, 0.2713, 0.2737]
        value_scale = 1

        train_transform = Compose([
            RandomResizedCrop(model.crop_size, (train_crop_min_scale, 1.0), (train_crop_min_ratio, 1.0 / train_crop_min_ratio)),
            RandomHorizontalFlip(),
            ToTensor(),
            ScaleValue(value_scale),
            Normalize(mean, std)
        ])
        test_trainsform = Compose([
            Resize(model.crop_size),
            CenterCrop(model.crop_size),
            ToTensor(),# range [0, 255] -> [0.0,1.0]
            ScaleValue(1), 
            Normalize(mean, std)
        ])

    train_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            args.data_root,
            args.data_name,
            video_list=args.train_list,
            num_segments=args.num_segments,
            representation=args.representation,
            transform=model.get_augmentation(), #train_transform, 
            is_train=True,
            accumulate=(not args.no_accumulation),
            model_name=args.arch
            ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        worker_init_fn=worker_init_fn)

    val_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            args.data_root,
            args.data_name,
            video_list=args.test_list,
            num_segments=args.num_segments,
            representation=args.representation,
            transform=torchvision.transforms.Compose([GroupScale(int(model.scale_size)),GroupCenterCrop(model.crop_size)]), #test_trainsform, 
            is_train=True,
            accumulate=(not args.no_accumulation),
            model_name=args.arch
            ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        worker_init_fn=worker_init_fn)


    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    cudnn.benchmark = True

    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        decay_mult = 0.0 if 'bias' in key else 1.0

        if ('module.base_model.conv1' in key
                or 'module.base_model.bn1' in key
                or 'data_bn' in key) and args.representation in ['mv', 'residual']:
            lr_mult = 0.1
        elif '.fc.' in key:
            lr_mult = 1.0
        else:
            lr_mult = 0.01

        params += [{'params': value, 'lr': args.lr, 'lr_mult': lr_mult, 'decay_mult': decay_mult}]

    #optimizer = torch.optim.SGD(params, weight_decay=0.001, momentum=0.9, nesterov=False)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    optimizer = torch.optim.Adam(params, weight_decay=args.weight_decay, eps=0.001)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    for epoch in range(args.epochs):
        cur_lr = adjust_learning_rate(optimizer, epoch, args.lr_steps, args.lr_decay)
        #cur_lr = get_lr(optimizer)

        train(train_loader, model, criterion, optimizer, epoch, cur_lr)
        #prec1, prev_val_loss = validate(val_loader, model, criterion)
        #scheduler.step(prev_val_loss)

        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1, _ = validate(val_loader, model, criterion)
            
            # 紀錄訓練歷程
            np.savez("train_history/train_history.npz",
                        loss=np.array(train_loss),
                        top1=np.array(train_prec),
                        lr=np.array(train_lr))
            np.savez("train_history/valid_history.npz",
                        loss=np.array(valid_loss),
                        top1=np.array(valid_prec))

            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            if is_best or epoch % SAVE_FREQ == 0:
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_prec1,
                    },
                    is_best,
                    filename='checkpoint.pth.tar')


def train(train_loader, model, criterion, optimizer, epoch, cur_lr):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()

    model.train()

    end = time.time()
    
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        if not 'resnet3D' in args.arch:
            output = output.view((-1, args.num_segments) + output.size()[1:])
            output = torch.mean(output, dim=1)

        loss = criterion(output, target_var)
        
        prec1 = accuracy(output.data, target)
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        # top5.update(prec5, input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % PRINT_FREQ == 0:
            print('Epoch: [{0}][{1}/{2}], lr: {lr:.7f}\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       epoch, i, len(train_loader),
                       batch_time=batch_time,
                       data_time=data_time,
                       loss=losses,
                       top1=top1,
                       lr=cur_lr))

    # 紀錄訓練歷程
    if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
        train_loss.append(losses.avg)
        train_prec.append(top1.avg)
        train_lr.append(cur_lr)


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()

    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        output = model(input_var)
        if not 'resnet3D' in args.arch:
            output = output.view((-1, args.num_segments) + output.size()[1:])
            output = torch.mean(output, dim=1)
        
        loss = criterion(output, target_var)

        prec1 = accuracy(output.data, target)

        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        # top5.update(prec5, input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % PRINT_FREQ == 0:
            print('Test: [{0}/{1}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       i, len(val_loader),
                       batch_time=batch_time,
                       loss=losses,
                       top1=top1))

    # 紀錄訓練歷程
    valid_loss.append(losses.avg)
    valid_prec.append(top1.avg)

    print('Testing Results: Prec@1 {top1.avg:.3f} Loss {loss.avg:.5f}'.format(top1=top1, loss=losses))

    return top1.avg, losses.avg


def save_checkpoint(state, is_best, filename):
    filename = '_'.join((args.model_prefix, args.representation.lower(), filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.model_prefix, args.representation.lower(), 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps, lr_decay):
    decay = lr_decay ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    wd = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = wd * param_group['decay_mult']
    return lr


def accuracy(output, target):
    batch_size = target.size(0)

    _, pred = output.max(1) # 找出最大值的索引
    pred = pred.view(1,-1) #從1維變2維: (資料數) -> (1, 資料數)
    correct = pred.eq(target.view(1, -1)) # 比對每筆資料是否正確
    res = correct.view(-1).float().sum(0).mul_(100.0 / batch_size)

    return res



if __name__ == '__main__':
    main()