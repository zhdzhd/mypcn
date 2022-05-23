
import argparse
import importlib
import os
import sys
import random
from datetime import datetime

from sqlalchemy import false
from extensions.chamfer_dist import ChamferDistanceL2

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='my_pc_model', help='Model file name [default: my_pc_model]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--dataset_path', default='/home/zhang/pcc', help='dataset path [default: None]')
parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--dump_dir', default=None, help='Dump dir to save sample outputs [default: None]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 20000]')
parser.add_argument('--max_epoch', type=int, default=150, help='Epoch to run [default: 180]')
parser.add_argument('--batch_size', type=int, default=6, help='Batch Size during training [default: 8]')
parser.add_argument('--learning_rate', type=float, default=0.00005, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step', type=int, default=4, help='Period of BN decay (in epochs) [default: 20]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='50,80,110', help='When to decay the learning rate (in epochs) [default: 80,120,160]')
parser.add_argument('--lr_decay_rates', default='0.5,0.5,0.5', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
parser.add_argument('--alpha_epoch', default='10, 25, 50', help='When to decay the learning rate (in epochs) [default: 80,120,160]')
parser.add_argument('--alpha_constant', default='0.01, 0.1, 0.5, 1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing log and dump folders.')
parser.add_argument('--manual_seed', type=int, default=7218, help='manual_seed  log6-5830.')
args = parser.parse_args()


# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
batch_size = args.batch_size
input_num_point = args.num_point
dataset_path = args.dataset_path
max_epoch = args.max_epoch
base_learning_rate = args.learning_rate
bn_decay_step = args.bn_decay_step
bn_decay_rate = args.bn_decay_rate
lr_decay_steps = [int(x) for x in args.lr_decay_steps.split(',')]
lr_decay_rates = [float(x) for x in args.lr_decay_rates.split(',')]
alpha_epoch = [int(x) for x in args.alpha_epoch.split(',')]
alpha_constant = [float(x) for x in args.alpha_constant.split(',')]
assert(len(lr_decay_steps)==len(lr_decay_rates))
log_dir = args.log_dir
default_dump_dir = os.path.join(BASE_DIR, os.path.basename(log_dir))
dump_dir = args.dump_dir if args.dump_dir is not None else default_dump_dir
default_checkpoint_path = os.path.join(log_dir, 'checkpoint.tar')
checkpoint_path = args.checkpoint_path if args.checkpoint_path is not None \
    else default_checkpoint_path
args.dump_dir = dump_dir

# Prepare LOG_DIR and DUMP_DIR
if os.path.exists(log_dir) and args.overwrite:
    print('Log folder %s already exists. Are you sure to overwrite? (Y/N)'%(log_dir))
    c = input()
    if c == 'n' or c == 'N':
        print('Exiting..')
        exit()
    elif c == 'y' or c == 'Y':
        print('Overwrite the files in the log and dump folers...')
        os.system('rm -r %s %s'%(log_dir, dump_dir))

if not os.path.exists(log_dir):
    os.mkdir(log_dir)

# 设置log
log_fout = open(os.path.join(log_dir, 'log_train.txt'), 'a')
log_fout.write(str(args)+'\n')
def log_string(out_str):
    log_fout.write(out_str+'\n')
    log_fout.flush()
    print(out_str)
if not os.path.exists(dump_dir): os.mkdir(dump_dir)

# 创建数据集
from my_data import my_pc_dataset
TRAIN_DATASET = my_pc_dataset(root = dataset_path, npoints = input_num_point ,train = True)
TEST_DATASET = my_pc_dataset(root = dataset_path, npoints = input_num_point ,train = False)
print(len(TRAIN_DATASET), len(TEST_DATASET))
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=batch_size,shuffle=True, drop_last=False)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=batch_size, shuffle=True,drop_last=False)
print(len(TRAIN_DATALOADER), len(TEST_DATALOADER))

# 设置随机数种子
if not args.manual_seed:
    seed = random.randint(1, 10000)
else:
    seed = int(args.manual_seed)
log_string('Random Seed: %d' % seed)
random.seed(seed)
torch.manual_seed(seed)

# 载入模型
MODEL = importlib.import_module(args.model) # import network module
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = MODEL.Model(args)
net.to(device)
# if torch.cuda.device_count() > 1:
#   log_string("Let's use %d GPUs!" % (torch.cuda.device_count()))
#   # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#   net = nn.DataParallel(net)

# 载入损失函数和优化函数
optimizer = optim.Adam(net.parameters(), lr=base_learning_rate, weight_decay=args.weight_decay)

# Load checkpoint if there is any
it = -1 # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0

if checkpoint_path is not None and os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #checkpoint['epoch'] = 0
    start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)"%(checkpoint_path, start_epoch))

# Decay Batchnorm momentum from 0.5 to 0.999
# note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum


# lr_lbmd = lambda e: max(args.lr_decay ** (e / args.decay_step), args.lowest_decay)
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lbmd)

def get_current_lr(epoch):
    lr = base_learning_rate
    for i,lr_decay_epoch in enumerate(lr_decay_steps):
        if epoch >= lr_decay_epoch:
            lr *= lr_decay_rates[i]
    return lr

def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_current_alpha(epoch):

    for ind, ep in enumerate(alpha_epoch):
        if epoch < ep:
            alpha = alpha_constant[ind]
            break
        elif ind == len(alpha_epoch)-1 and epoch >= ep:
            alpha = alpha_constant[ind+1]
            break
    return alpha

pc_loss = ChamferDistanceL2()

def get_loss(fine,coarse,gt_pc,center,gt_cen,m,gt_m,alpha = 1):
    # print(fine.shape)
    b,_,_ = fine.shape
    
    
    loss_fine = pc_loss(fine,gt_pc)
    loss_coarse = pc_loss(coarse,gt_pc)
    loss_pc = alpha*loss_fine + loss_coarse
# 
    loss_cen = torch.sum((center-gt_cen)**2)/b
    loss_m = torch.sum((m-gt_m)**2)/b

    loss_cenm = loss_cen+0.5*loss_m
    # loss2 = torch.from_numpy(np.array(0).astype(np.float32))
    loss = loss_pc + 0.5*loss_cenm
    return loss , loss_pc,loss_fine, loss_coarse, loss_cen,loss_m

def rotate_point_cloud(points, rotation_matrix=None):
    """ Input: (n,3), Output: (n,3) """
    # Rotate in-place around Z axis.
    if rotation_matrix is None:
        rotation_angle = np.random.uniform() * 2 * np.pi
        sinval, cosval = np.sin(rotation_angle), np.cos(rotation_angle)     
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
    ctr = points.mean(1)
    ctr = ctr.unsqueeze(1)
    rotated_data = torch.matmul(points-ctr,rotation_matrix) + ctr
    return rotated_data, rotation_matrix
# ------------------------------------------------------------------------- GLOBAL CONFIG END

def train_one_epoch():
    stat_dict = {} # collect statistics
    stat_dict['coarse_loss'] = 0
    stat_dict['fine_loss'] = 0
    stat_dict['m_loss'] = 0
    stat_dict['cen_loss'] = 0
    stat_dict['pc_loss'] = 0
    stat_dict['loss'] = 0
    
    adjust_learning_rate(optimizer, EPOCH_CNT)
    alpha = get_current_alpha(EPOCH_CNT)
    # bnm_scheduler.step() # decay BN momentum
    net.train() # set model to training mode
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        par_pc,gt_pc,matrix,gt_cen,gt_m,par_m = batch_data_label
        gt_pc,m = rotate_point_cloud(gt_pc,matrix)

        par_pc = par_pc.to(device)    
        gt_pc = gt_pc.to(device)
        matrix = matrix.to(device)
        gt_cen = gt_cen.to(device)
        gt_m = gt_m.to(device)
        par_m = par_m.to(device)
        # print(par_pc.shape)
        # Forward pass
        optimizer.zero_grad()
        center ,m, coarse , fine = net(par_pc,gt_cen,gt_m,par_m,train = True)
        # print(center)
        # print(gt_cen)
        # Compute loss and gradients, update parameters.
        loss , loss_pc,loss_fine, loss_coarse, loss_cen,loss_m = get_loss(fine,coarse,gt_pc,center,gt_cen,m,gt_m/par_m,alpha)
        # print(loss1)
        loss.backward()
        optimizer.step()


        # Accumulate statistics and print out
        stat_dict['coarse_loss'] += loss_coarse
        stat_dict['fine_loss'] += loss_fine
        stat_dict['cen_loss'] += loss_cen
        stat_dict['m_loss'] += loss_m
        stat_dict['pc_loss'] += loss_pc
        stat_dict['loss'] += loss
        #end_points = post_pred_grasp_pose(end_points,'/home/aemc/grasp_scene_raw/')
        batch_interval = 20
        if (batch_idx+1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            # TRAIN_VISUALIZER.log_scalars({key:stat_dict[key]/batch_interval for key in stat_dict},
            #     (EPOCH_CNT*len(TRAIN_DATALOADER)+batch_idx)*BATCH_SIZE)
            for key in sorted(stat_dict.keys()):
                log_string('mean %s: %f'%(key, stat_dict[key]/(batch_interval)))
                stat_dict[key] = 0


def evaluate_one_epoch():
    stat_dict = {} # collect statistics
    stat_dict['coarse_loss'] = 0
    stat_dict['fine_loss'] = 0
    stat_dict['m_loss'] = 0
    stat_dict['cen_loss'] = 0
    stat_dict['pc_loss'] = 0
    stat_dict['loss'] = 0
    net.eval() # set model to eval mode (for bn and dp)
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        if batch_idx % 5 == 0:
            print('Eval batch: %d'%(batch_idx))

        par_pc,gt_pc,matrix,gt_cen,gt_m,par_m = batch_data_label
        gt_pc,m = rotate_point_cloud(gt_pc,matrix)

        par_pc = par_pc.to(device)    
        gt_pc = gt_pc.to(device)
        matrix = matrix.to(device)
        gt_cen = gt_cen.to(device)
        gt_m = gt_m.to(device)
        par_m = par_m.to(device)
 

        with torch.no_grad():
            center , m,coarse , fine = net(par_pc)

        # Compute loss
        loss , loss_pc,loss_fine, loss_coarse, loss_cen,loss_m = get_loss(fine,coarse,gt_pc,center,gt_cen,m,gt_m/par_m)

        # Accumulate statistics and print out
        stat_dict['coarse_loss'] += loss_coarse
        stat_dict['fine_loss'] += loss_fine
        stat_dict['cen_loss'] += loss_cen
        stat_dict['m_loss'] += loss_m
        stat_dict['pc_loss'] += loss_pc
        stat_dict['loss'] += loss


        # batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT)
        # batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
        # ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)



        # Log statistics
        # TEST_VISUALIZER.log_scalars({key:stat_dict[key]/float(batch_idx+1) for key in stat_dict},
        #     (EPOCH_CNT+1)*len(TRAIN_DATALOADER)*BATCH_SIZE)
        for key in sorted(stat_dict.keys()):
            log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))

        # Evaluate average precision
        # metrics_dict = ap_calculator.compute_metrics()
        # for key in metrics_dict:
        #     log_string('eval %s: %f'%(key, metrics_dict[key]))

        mean_loss = stat_dict['loss']/(float(batch_idx+1))
        return mean_loss

def train(start_epoch):
    global EPOCH_CNT 
    min_loss = 1e10
    loss = 0
    for epoch in range(start_epoch, max_epoch):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % (epoch))
        log_string('Current learning rate: %f'%(get_current_lr(epoch)))
        log_string('Current alpha: %f'%(get_current_alpha(epoch)))
        # log_string('Current BN decay momentum: %f'%(bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
        log_string(str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        train_one_epoch()
        if EPOCH_CNT == 0 or EPOCH_CNT % 10 == 9: # Eval every 10 epochs
            loss = evaluate_one_epoch()
        # Save checkpoint
        save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }
        try: # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = net.module.state_dict()
        except:
            save_dict['model_state_dict'] = net.state_dict()
        torch.save(save_dict, os.path.join(log_dir, 'checkpoint.tar'))

if __name__=='__main__':
    train(start_epoch)
