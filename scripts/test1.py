import logging
import os
import sys
import importlib
import argparse
import munch
import yaml
import open3d
import numpy as np
import random
import os
import torch


def read_matrix(matrix_path):

    with open(matrix_path, 'r') as f:
        data = f.readlines()[0]
        data = data.strip()
        data = data.split(' ')
        data = [float(x) for x in data]


        return np.array([[data[0],  data[1],  data[2]],
                [data[3],  data[4],  data[5]],
                [data[6], data[7],  data[8]]])

def seprate_point_cloud(xyz, num_points, crop):
    '''
     seprate point cloud: usage : using to generate the incomplete point cloud with a setted number.
    '''
    n,c = xyz.shape

    assert n == num_points
    assert c == 3
    if crop == num_points:
        return xyz, None
        
    
    if isinstance(crop,list):
        num_crop = random.randint(crop[0],crop[1])
    else:
        num_crop = crop

    # points = points.unsqueeze(0)

    # if fixed_points is None:       
    #     center = F.normalize(torch.randn(1,1,3),p=2,dim=-1).cuda()
    # else:
    #     if isinstance(fixed_points,list):
    #         fixed_point = random.sample(fixed_points,1)[0]
    #     else:
    #         fixed_point = fixed_points
    center = xyz[np.random.choice(np.arange(n),1)[0]]

    distance_matrix = np.linalg.norm(center - xyz, ord =2 ,axis = -1)  # 1 1 2048

    idx = np.argsort(distance_matrix,axis=-1) # 2048

    input_data = xyz.copy()[idx[:num_crop]] #  N 3

    other_data =  xyz.copy()[idx[num_crop:]]

    num_sep = int((n-num_crop)/4)

    sep_data_range = xyz.copy()[idx[(num_crop+num_sep):]]
    index = np.random.choice(np.arange(sep_data_range.shape[0]),4,replace=False)
    sep_data = sep_data_range[index]

    return input_data, other_data, sep_data


def rotate_point_cloud(points, rotation_matrix=None):
    """ Input: (n,3), Output: (n,3) """
    # Rotate in-place around Z axis.
    if rotation_matrix is None:
        rotation_angle = np.random.uniform() * 2 * np.pi
        sinval, cosval = np.sin(rotation_angle), np.cos(rotation_angle)     
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
    ctr = points.mean(axis=0)
    rotated_data = np.dot(points-ctr, rotation_matrix) + ctr
    return rotated_data, rotation_matrix

def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                    [0,  1,  0],
                    [-s, 0,  c]])

def rotx(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                    [0,  c,  s],
                    [0, -s,  c]])

def random_rot_matrix():
    x = np.random.uniform() * 2 * np.pi
    y = np.random.uniform() * 2 * np.pi
    z = np.random.uniform() * 2 * np.pi

    rot_matrix_x = rotx(x)
    rot_matrix_y = roty(y)
    rot_matrix_z = rotz(z)

    matrix = np.dot(np.dot(rot_matrix_x,rot_matrix_y),rot_matrix_z)
    return matrix

def read_pcd(pc_path):
    pc = open3d.io.read_point_cloud(pc_path)
    ptcloud = np.array(pc.points)
    return ptcloud

file = "/home/zhang/pcc/data/model"

p1_path = '/home/zhang/pcc/data/dataset/1/part_pc/par_0.pcd'
p2_path = '/home/zhang/pcc/data/dataset/1/gt_pc/pc_1.pcd'
m_path = '/home/zhang/pcc/data/dataset/1/rotate_matrix/0.txt'

p1 = read_pcd(p1_path)
p2 = read_pcd(p2_path)
# p2 = read_pcd(p2_path)
matrix = read_matrix(m_path)

pc,_ = rotate_point_cloud(p2,matrix)

pcd = open3d.geometry.PointCloud()
pcd.points = open3d.utility.Vector3dVector(pc)
pcd1 = open3d.geometry.PointCloud()
pcd1.points = open3d.utility.Vector3dVector(p1)
open3d.io.write_point_cloud("/home/zhang/pc_test/gtt.pcd", pcd)
open3d.io.write_point_cloud("/home/zhang/pc_test/parr.pcd", pcd1)
# print(p1.shape)
# print(p2.shape)

# cen = np.mean(p1, axis=0)
# print(cen)

# combine_pc = np.concatenate((p1,p2),axis=0)
# print(combine_pc.shape)

# a = []
# a.append([None])
# b=np.array(a).astype(np.float32)
# print(b)
# c= torch.from_numpy(b)
# print(c)
        
# for i in range(88):
#     path = os.path.join(file,'pc%d.pcd'%i)
#     pc = open3d.io.read_point_cloud(path)
#     ptcloud = np.array(pc.points)
#     print(ptcloud.shape)
#     print(i)
# print(ptcloud.size)
# m = np.max(np.sqrt(np.sum(ptcloud**2, axis=1)))
# print(m)

# size,_ = ptcloud.shape
# print(size)
# center_idx = np.random.choice(np.arange(size),1)[0]
# print(center_idx)
# center = ptcloud[center_idx]
# distance_matrix = np.linalg.norm(center - ptcloud, ord =2 ,axis = -1)
# idx = np.argsort(distance_matrix,axis=-1)
# print(idx)
# size,_ = ptcloud.shape

# matrix = random_rot_matrix()
# def write_matrix(path,matrix):
#     # os.mkdir(path)
#     print(matrix)
#     with open(path,"w") as f:
#         for row in matrix:
#             for col in row:
#                 f.write('%f '%col)

# write_matrix('/home/zhang/pc_test/aaa.txt',matrix)
# pcc,_ = rotate_point_cloud(ptcloud,matrix)

# pardata,otherdata,sepdata = seprate_point_cloud(ptcloud,size,[int(size/4),int(size/2)])
# # print(sepdata.size)
# pcd = open3d.geometry.PointCloud()
# pcd.points = open3d.utility.Vector3dVector(pcc)
# # pcd1 = open3d.geometry.PointCloud()
# # pcd1.points = open3d.utility.Vector3dVector(sepdata)
# open3d.io.write_point_cloud("/home/zhang/pc_test/pcc.pcd", pcd)
# open3d.io.write_point_cloud("/home/zhang/pc_test/septest.pcd", pcd1)

# center = np.random.choice(np.arange(size),1)
# print(center[0])
# print(ptcloud[center[0]])
# cen = np.mean(ptcloud, axis=0)
# print(cen)
# print(ptcloud.shape)