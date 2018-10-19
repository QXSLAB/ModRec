import argparse
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix

from support_mods import class_num, ModulationDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='PyTorch RF Training')
parser.add_argument('--epochs', default=10000, type=int, metavar='N',
                    help='number of total epochs to run (default: 100)')
parser.add_argument('--snr', default=10, type=int, metavar='N',
                    help='signal to noise ratio of signal (default: 10)')
parser.add_argument('--batch_size', default=128, type=int, metavar='N',
                    help='batch size of train data (default: 128)')
parser.add_argument('--print_freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--idle_epochs', default=500, type=int,
                    help='Early Stoping threshold')
parser.add_argument('--confusion_matrix', default='confusion_matrix',
                    action='store_true', help='output confusion matrix')


def main():
    """This is the main function to train and evaluate the model"""
    idle_epochs = 0
    global args, best_prec1
    args = parser.parse_args()
    best_prec1 = 0

    model = Discriminator().cuda()
    print(model)
    optimizer = torch.optim.SGD(model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    train_loader = Data.DataLoader(ModulationDataset(True),
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=4)

    val_loader = Data.DataLoader(ModulationDataset(False),
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=4)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch, args.lr)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)

        # Early Stop
        if is_best:
            idle_epochs = 0
        else:
            idle_epochs = idle_epochs + 1

        if idle_epochs > args.idle_epochs:
            print("Early Stoped at epoch: {}, best_prec1: {}"
                  .format(epoch, best_prec1))
            return


class Discriminator(nn.Module):
    """Define the model"""

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 256, 3, padding=1),  # batch, 256, 1024
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Dropout2d()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 80, 3, padding=1),  # batch, 80, 1024
            nn.BatchNorm1d(80),
            nn.ReLU(),
            # nn.Dropout2d()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(80 * 1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.6)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, class_num),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def train(train_loader, model, criterion, optimizer, epoch):
    """Train the model using training dataset"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for step, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda().long()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec3 = accuracy(output, target, topk=(1, 3))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top3.update(prec3[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                epoch, step, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top3=top3))


def validate(val_loader, model, criterion):
    """Evaluate the performance of the model using vallidation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    outputs = []
    targets = []
    with torch.no_grad():
        end = time.time()
        for step, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda().long()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top3.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            targets.append(target)
            outputs.append(output)

        if args.confusion_matrix:
            cm = confusion_matrix(torch.cat(targets),
                                  torch.cat(outputs).argmax(1))
            print(cm)

        print(' * Best@1 {best:.3f} Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
              .format(top1=top1, top3=top3, best=best_prec1))

    return top1.avg


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


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.9 ** (epoch // 100))
    print('Learning Rate %s' % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Save checkpoint of training process"""
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ == '__main__':
    main()
