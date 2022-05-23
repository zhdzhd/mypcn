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
import open3d

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
parser.add_argument('--max_epoch', type=int, default=2, help='Epoch to run [default: 180]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step', type=int, default=4, help='Period of BN decay (in epochs) [default: 20]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='16,24,32', help='When to decay the learning rate (in epochs) [default: 80,120,160]')
parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing log and dump folders.')
parser.add_argument('--manual_seed', type=int, default=10, help='manual_seed.')
args = parser.parse_args()


# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
pc_loss = ChamferDistanceL2()
MODEL = importlib.import_module(args.model) # import network module
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = MODEL.Model(args)
net.to(device)
net.eval()
CHECKPOINT_PATH = os.path.join( os.path.dirname(os.path.abspath(__file__)),'log/checkpoint.tar' )
print(CHECKPOINT_PATH)
# CHECKPOINT_PATH = '/home/zhang/pcc/src/pc_dateset/scripts/log/checkpoint.tar'
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    print('load')

# def rotate_point_cloud(points, rotation_matrix=None):
#     """ Input: (n,3), Output: (n,3) """
#     # Rotate in-place around Z axis.
#     if rotation_matrix is None:
#         rotation_angle = np.random.uniform() * 2 * np.pi
#         sinval, cosval = np.sin(rotation_angle), np.cos(rotation_angle)     
#         rotation_matrix = np.array([[cosval, sinval, 0],
#                                     [-sinval, cosval, 0],
#                                     [0, 0, 1]])
#     ctr = points.mean(axis=0)
#     rotated_data = np.dot(points-ctr, rotation_matrix) + ctr
#     return rotated_data, rotation_matrix
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

def read_pcd(pc_path):
    pc = open3d.io.read_point_cloud(pc_path)
    ptcloud = np.array(pc.points)
    return ptcloud

def read_matrix(matrix_path):

    with open(matrix_path, 'r') as f:
        data = f.readlines()[0]
        data = data.strip()
        data = data.split(' ')
        data = [float(x) for x in data]


        return np.array([[data[0],  data[1],  data[2]],
                [data[3],  data[4],  data[5]],
                [data[6], data[7],  data[8]]])

pc_loss = ChamferDistanceL2()

def get_loss(pc_output,coarse,pc_gt,cen_output,cen_gt):
    b,_, = cen_output.shape
    
    
    loss11 = pc_loss(pc_output,pc_gt)
    loss12 = pc_loss(coarse,pc_gt)
    loss1 = loss11 + loss12

    loss2 = torch.sum((cen_output-cen_gt)**2)/b
    # loss2 = torch.from_numpy(np.array(0).astype(np.float32))
    loss = loss1 + loss2
    return loss , loss1, loss2,loss11,loss12

from my_data import my_pc_dataset
TRAIN_DATASET = my_pc_dataset(npoints = 2048 ,train = True)
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=1,shuffle=True, drop_last=False)
for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        par_pc,sep_pc,gt_pc,matrix,gt_cen = batch_data_label
        # print(par_pc.shape)
        # print(sep_pc.shape)
        # print(gt_pc.shape)
        # print(matrix.shape)
        # print(gt_cen.shape)
        break

gt_pc,m= rotate_point_cloud(gt_pc,matrix)
b,_,_ = sep_pc.shape
empty = torch.zeros([b,4,3])
if torch.equal(empty,sep_pc):
    sep_pc = None
else:
    sep_pc = sep_pc.to(device)


par_pc = par_pc.to(device)
print(sep_pc)

gt_pc = gt_pc.to(device)
matrix = matrix.to(device)
gt_cen = gt_cen.to(device)

with torch.no_grad():
    center , pc_out, coarse , fine = net(par_pc)

# Compute loss
loss, loss1, loss2 ,loss11,loss12= get_loss(pc_out,coarse,gt_pc,center,gt_cen)

print(loss)
print(loss2)
print(loss11)
print(loss12)
out1 = pc_out.cpu().squeeze(0).numpy()
fine = fine.cpu().squeeze(0).numpy()
coarse = coarse.cpu().squeeze(0).numpy()
gt = gt_pc.cpu().squeeze(0).numpy()
# loss2 = torch.sum((center-centroid)**2)
# print(loss2)
print('loss')
# print(centroid)
# print(center)
# pcd3 = open3d.geometry.PointCloud()
# pcd3.points = open3d.utility.Vector3dVector(p1)
# open3d.io.write_point_cloud("/home/zhang/pc_test/input.pcd", pcd3)

pcd = open3d.geometry.PointCloud()
pcd.points = open3d.utility.Vector3dVector(out1)
open3d.io.write_point_cloud("/home/zhang/pc_test/outpredict.pcd", pcd)

pcd1 = open3d.geometry.PointCloud()
pcd1.points = open3d.utility.Vector3dVector(gt)
open3d.io.write_point_cloud("/home/zhang/pc_test/outgt.pcd", pcd1)

pcd2 = open3d.geometry.PointCloud()
pcd2.points = open3d.utility.Vector3dVector(coarse)
open3d.io.write_point_cloud("/home/zhang/pc_test/outcoutse.pcd", pcd2)

# pcd2 = open3d.geometry.PointCloud()
# pcd2.points = open3d.utility.Vector3dVector(fine)
# open3d.io.write_point_cloud("/home/zhang/pc_test/fine.pcd", pcd2)









