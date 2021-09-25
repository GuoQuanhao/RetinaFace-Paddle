from __future__ import print_function
import os
import paddle
import paddle.optimizer as optim
import argparse
from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50, make_dataloader
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.retinaface import RetinaFace

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--training_dataset', default='/home/aistudio/Data/widerface/train/label.txt', help='Training dataset directory')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./test/', help='Location to save checkpoint models')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50

rgb_mean = (104, 117, 123) # bgr order
rgb_std = (57.1,57.4,58.4)
num_classes = 2
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.training_dataset
save_folder = args.save_folder

net = RetinaFace(cfg=cfg)
print("Printing net...")
print(net)

if args.resume_net is not None:
    print('Loading resume network...')
    state_dict = paddle.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.set_state_dict(new_state_dict)

if num_gpu > 1 and gpu_train:
    net = paddle.DataParallel(net)

optimizer = optim.Momentum(parameters=net.parameters(), learning_rate=initial_lr, momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with paddle.no_grad():
    priors = priorbox.forward()

def train():
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    dataset = WiderFaceDetection(training_dataset, preproc(img_dim, rgb_mean, rgb_std))

    epoch_size = math.ceil(len(dataset) / batch_size)

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0
    max_iter = max_epoch * epoch_size - start_iter
    batch_iterator = make_dataloader(dataset, shuffle=True, batchsize=batch_size, distributed=False, num_workers=0, num_iters=max_iter, start_iter=0, collate_fn=detection_collate)
    iteration = start_iter
    for images, labels in batch_iterator:
        if iteration % epoch_size == 0:
            if (epoch % 5 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
                paddle.save(net.state_dict(), save_folder + cfg['name']+ '_epoch_' + str(epoch) + '.pdparams')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

        # forward
        out = net(images)
        # backprop
        loss_l, loss_c, loss_landm = criterion(out, priors, [anno for anno in labels])
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        loss.backward()
        optimizer.step()
        optimizer.clear_gradients()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
              .format(epoch, max_epoch, (iteration % epoch_size) + 1,
              epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))
        iteration += 1

    paddle.save(net.state_dict(), save_folder + cfg['name'] + '_Final.pdparams')

def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    optimizer.set_lr(lr)
    return lr

if __name__ == '__main__':
    train()
