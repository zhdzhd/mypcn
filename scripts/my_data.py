import os
import json
import warnings
import numpy as np
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import open3d
from extensions.chamfer_dist import ChamferDistanceL2

class my_pc_dataset(Dataset):
    def __init__(self,root = '/home/zhang/mypcn', npoints = 2048 ,train = True,get_one=False,get_dir=['0']):
        print("Start init dataset!")
        self.npoints = npoints
        self.gt_npoints = npoints * 4
        self.pc_input = []
        self.pc_gt = []
        self.pc_matrix =[]
        self.pc_cen_gt = []
        self.pc_m_gt = []
        self.par_m = []
        # self.m_gt = []
        
        if train:
            root = os.path.join(root,'data')
        else:
            root = os.path.join(root,'data_test')
        
        file_path = os.path.join(root,'dataset')

        if get_one:
            file_dir = get_dir
        else:
            file_dir = os.listdir(file_path)
        max_m = 0
        min_m = 10
        max_parm = 0
        min_parm = 10
        
        for file in file_dir:
            
            matrix_path = os.path.join(file_path,file,'rotate_matrix')
            part_pc_path = os.path.join(file_path,file,'part_pc')
            gt_pc_path = os.path.join(file_path,file,'gt_pc')

            
            matrix_dir = os.listdir(matrix_path)
            part_pc_dir = os.listdir(part_pc_path)
            gt_pc_dir = os.listdir(gt_pc_path)
            
            for par_pc_file in part_pc_dir:

                matrix_file = par_pc_file.replace('par_','')
                matrix_file = matrix_file.replace('.pcd','.txt')

                par_pc = self.read_pcd(os.path.join(part_pc_path,par_pc_file))
                gt_pc = self.read_pcd(os.path.join(gt_pc_path,gt_pc_dir[0]))
                matrix = self.read_matrix(os.path.join(matrix_path,matrix_file))

                # choice = np.random.choice(len(par_pc[:,0]), self.npoints, replace=True)
                # par_pc = par_pc[choice, :]

                # choice = np.random.choice(len(gt_pc[:,0]), self.gt_npoints, replace=True)
                # gt_pc = gt_pc[choice, :]


                cen = np.mean(par_pc, axis=0)
                par_pc_input = par_pc-cen
                gt_cen = -cen
                par_m = np.max(np.sqrt(np.sum(par_pc_input**2, axis=1)))
                par_pc_input = par_pc_input/par_m
                gt_cen = gt_cen/par_m

                if par_m < min_parm:
                    min_parm=par_m
                if par_m>max_parm:
                    max_parm=par_m

                m = np.max(np.sqrt(np.sum(gt_pc**2, axis=1)))
                if m < min_m:
                    min_m=m
                if m>max_m:
                    max_m=m
                # gt_pc = gt_pc/par_m


                self.pc_input.append(par_pc_input)
                self.pc_gt.append(gt_pc)
                self.pc_matrix.append(matrix)
                self.pc_cen_gt.append(gt_cen)
                # self.pc_m_gt.append(np.array(m))
                self.par_m.append(np.array(par_m))
                

        print('Finish load dataset!')
        print('max m = %f'%max_m)
        print('min_m = %f'%min_m)
        print('parmax m = %f'%max_parm)
        print('parmin_m = %f'%min_parm)
                
    def read_pcd(self,pc_path):
        pc = open3d.io.read_point_cloud(pc_path)
        ptcloud = np.array(pc.points)
        return ptcloud

    def read_matrix(self,matrix_path):

        with open(matrix_path, 'r') as f:
            data = f.readlines()[0]
            data = data.strip()
            data = data.split(' ')
            data = [float(x) for x in data]


            return np.array([[data[0],  data[1],  data[2]],
                    [data[3],  data[4],  data[5]],
                    [data[6], data[7],  data[8]]])

    def __getitem__(self, index):
        par_pc = torch.from_numpy(self.pc_input[index].astype(np.float32))
        gt_pc = torch.from_numpy(self.pc_gt[index].astype(np.float32))
        matrix = torch.from_numpy(self.pc_matrix[index].astype(np.float32))
        gt_cen = torch.from_numpy(self.pc_cen_gt[index].astype(np.float32))
        # m_gt = torch.from_numpy(self.pc_m_gt[index].astype(np.float32))
        par_m = torch.from_numpy(self.par_m[index].astype(np.float32))
        return par_pc,gt_pc,matrix,gt_cen,par_m

    def __len__(self):
        return len(self.pc_input)



class my_pc_dataset_get_one(Dataset):
    def __init__(self,root = '/home/zhang/mypcn', npoints = 2048 ,train = True):
        print("Start init dataset!")
        self.npoints = npoints
        self.gt_npoints = npoints * 4
        self.pc_input = []
        self.pc_gt = []
        self.pc_matrix =[]
        self.pc_cen_gt = []
        self.pc_m_gt = []
        self.par_m = []
        
        if train:
            root = os.path.join(root,'data')
        else:
            root = os.path.join(root,'data_test')
        
        file_path = os.path.join(root,'dataset')
        file_dir = ['20']
        
        
        for file in file_dir:
            
            matrix_path = os.path.join(file_path,file,'rotate_matrix')
            part_pc_path = os.path.join(file_path,file,'part_pc')
            gt_pc_path = os.path.join(file_path,file,'gt_pc')

            
            matrix_dir = os.listdir(matrix_path)
            part_pc_dir = os.listdir(part_pc_path)
            gt_pc_dir = os.listdir(gt_pc_path)
            
            for par_pc_file in part_pc_dir:

                matrix_file = par_pc_file.replace('par_','')
                matrix_file = matrix_file.replace('.pcd','.txt')

                par_pc = self.read_pcd(os.path.join(part_pc_path,par_pc_file))
                gt_pc = self.read_pcd(os.path.join(gt_pc_path,gt_pc_dir[0]))
                matrix = self.read_matrix(os.path.join(matrix_path,matrix_file))

                # choice = np.random.choice(len(par_pc[:,0]), self.npoints, replace=True)
                # par_pc = par_pc[choice, :]

                # choice = np.random.choice(len(gt_pc[:,0]), self.gt_npoints, replace=True)
                # gt_pc = gt_pc[choice, :]


                cen = np.mean(par_pc, axis=0)
                par_pc_input = par_pc-cen
                gt_cen = -cen
                par_m = np.max(np.sqrt(np.sum(par_pc_input**2, axis=1)))
                par_pc_input = par_pc_input/par_m
                gt_cen = gt_cen/par_m

                # m = np.max(np.sqrt(np.sum(gt_pc**2, axis=1)))
                


                self.pc_input.append(par_pc_input)
                self.pc_gt.append(gt_pc)
                self.pc_matrix.append(matrix)
                self.pc_cen_gt.append(gt_cen)
                # self.pc_m_gt.append(np.array(m))
                self.par_m.append(np.array(par_m))

                

                

                

        print('Finish load dataset!')
                
    def read_pcd(self,pc_path):
        pc = open3d.io.read_point_cloud(pc_path)
        ptcloud = np.array(pc.points)
        return ptcloud

    def read_matrix(self,matrix_path):

        with open(matrix_path, 'r') as f:
            data = f.readlines()[0]
            data = data.strip()
            data = data.split(' ')
            data = [float(x) for x in data]


            return np.array([[data[0],  data[1],  data[2]],
                    [data[3],  data[4],  data[5]],
                    [data[6], data[7],  data[8]]])

    def __getitem__(self, index):
        par_pc = torch.from_numpy(self.pc_input[index].astype(np.float32))
        gt_pc = torch.from_numpy(self.pc_gt[index].astype(np.float32))
        matrix = torch.from_numpy(self.pc_matrix[index].astype(np.float32))
        gt_cen = torch.from_numpy(self.pc_cen_gt[index].astype(np.float32))
        # m_gt = torch.from_numpy(self.pc_m_gt[index].astype(np.float32))
        par_m = torch.from_numpy(self.par_m[index].astype(np.float32))
        return par_pc,gt_pc,matrix,gt_cen,par_m

    def __len__(self):
        return len(self.pc_input)


if __name__ == "__main__":
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
    pc_loss = ChamferDistanceL2()
    data = my_pc_dataset(train=True)
    train_data = DataLoader(data, batch_size=1,shuffle=True,drop_last=False)
    for batch_idx, batch_data_label in enumerate(train_data):
        par_pc,gt_pc,matrix,gt_cen,gt_m,par_m = batch_data_label
        print(par_pc.shape)
        print(gt_pc.shape)
        print(matrix.shape)
        print(gt_cen.shape)
        break

    
    gt_cen = gt_cen.unsqueeze(1)
    gt_m = gt_m.unsqueeze(1)
    par_m = par_m.unsqueeze(1)
    par_pc = par_pc - gt_cen
    par_pc = par_pc*par_m
    par_pc = par_pc/gt_m
    gt,_ = rotate_point_cloud(gt_pc,matrix)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    par_pc = par_pc.to(device)
    gt = gt.to(device)
    print(par_pc.size())
    print(gt.size())
    loss = pc_loss(par_pc,gt)
    print(loss)
    out1 = par_pc.cpu().squeeze(0).numpy()
    out2 = gt.cpu().squeeze(0).numpy()
    print(out1.shape)

    m = np.max(np.sqrt(np.sum(out2**2, axis=1)))
    print(m)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(out1)
    open3d.io.write_point_cloud("/home/zhang/pc_test/par.pcd", pcd)

    pcd1 = open3d.geometry.PointCloud()
    pcd1.points = open3d.utility.Vector3dVector(out2)
    open3d.io.write_point_cloud("/home/zhang/pc_test/gt.pcd", pcd1)
            
            