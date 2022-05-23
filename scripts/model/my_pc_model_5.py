from pyexpat import features
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
from pointnet2_ops import pointnet2_modules
import open3d

def read_pcd(pc_path):
    pc = open3d.io.read_point_cloud(pc_path)
    ptcloud = np.array(pc.points)
    return ptcloud

class Fold(nn.Module):
    def __init__(self, in_channel , step , hidden_dim = 512):
        super().__init__()

        self.in_channel = in_channel
        self.step = step

        a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).cuda()

        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

    def forward(self, x):
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)

        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)

        return fd2


class Pointnet2(nn.Module):
    def __init__(self, input_dim=0, size_z=128, global_feature_size=1024):
        super().__init__()

        self.sa1=pointnet2_modules.PointnetSAModule(
                mlp=[input_dim, 32, 32, 64],
                npoint=512,
                radius=0.02,
                nsample=32,
                use_xyz=True,
                bn=True,)

        self.sa2=pointnet2_modules.PointnetSAModule(
                mlp=[64, 128, 128, 256],
                npoint=256,
                radius=0.04,
                nsample=16,
                use_xyz=True,
                bn=True,)

        self.sa3=pointnet2_modules.PointnetSAModule(
                mlp=[256, 256, 512, 512],
                npoint=128,
                radius=0.08,
                nsample=16,
                use_xyz=True,
                bn=True,)

        self.sa4=pointnet2_modules.PointnetSAModule(
                mlp=[512, 512, 1024, 1024],
                use_xyz=True,
                bn=True,)

        self.sa_sep=pointnet2_modules.PointnetSAModule(
                mlp=[input_dim, 256, 512, 1024],
                use_xyz=True,
                bn=True,)


        self.conv1 = torch.nn.Conv1d(2048, 1024, 1)
        self.conv2 = torch.nn.Conv1d(2048, 2048, 1)
        self.conv3 = torch.nn.Conv1d(2048, 2048, 1)
        self.bn1 = torch.nn.BatchNorm1d(1024)
        self.bn2 = torch.nn.BatchNorm1d(2048)
    
    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def fps(self,pc, num):
        fps_idx = pointnet2_utils.furthest_point_sample(pc, num) 
        sub_pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
        return sub_pc

    def forward(self, pointcloud,sep_pc = None):
        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)


        xyz, features= self.sa1(xyz, features)
        xyz, features= self.sa2(xyz, features)
        xyz, features= self.sa3(xyz, features)
        _,features= self.sa4(xyz, features)  #[1,128,3]  [1,1024,1]

        if sep_pc is not None:
            sep_xyz, sep_features = self._break_up_pc(sep_pc)
        else:
            sep_pc = self.fps(pointcloud,4)
            sep_xyz, sep_features = self._break_up_pc(sep_pc)

        _,sep_features = self.sa_sep(sep_xyz, sep_features)
        features = torch.cat((features,sep_features), dim=1)
        out = F.relu(self.bn1(self.conv1(features)))
        out = torch.cat((out,sep_features), dim=1)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.conv3(out)



        return out
        

        


class Model(nn.Module):
    def __init__(self, args, num_pred=6144, grid_size=4, global_feature_size=1024):
        super(Model, self).__init__()
        self.number_fine = num_pred
        grid_size = 4 # set default
        self.grid_size = grid_size
        assert self.number_fine % grid_size**2 == 0
        self.number_coarse = self.number_fine // (grid_size ** 2 )
        self.fps_num = 128
        

        
        self.backbone_modules = Pointnet2()
        
        self.center_map= nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3)
        )

        self.offset_map = nn.Linear(1024 + 3, 1024)

        self.mlp = nn.Sequential(
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,3*self.number_coarse)
        )

        self.reduce_map = nn.Linear(1024 + 3, 1024)

        self.foldingnet = Fold(1024, step = self.grid_size, hidden_dim = 512)  # rebuild a cluster point

        
        
        
 
    def fps(self,pc, num):
        fps_idx = pointnet2_utils.furthest_point_sample(pc, num) 
        sub_pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
        return sub_pc


    def forward(self,pc_input,sep_pc = None):

        bs , n , _ = pc_input.shape

        features = self.backbone_modules(pc_input,sep_pc)  #B 2048 1
        

        features = features.squeeze(-1)  
        center_features = features[:,0:1024]
        global_features = features[:,1024:]          # B 1024
        center = self.center_map(center_features)   # B 3

        global_features = self.offset_map(torch.cat([global_features,center],dim=1))  # B 1024




        coarse = self.mlp(global_features).reshape(-1,self.number_coarse,3) # B M 3
        
        rebuild_feature = torch.cat([
            global_features.unsqueeze(-2).expand(-1, self.number_coarse, -1),
            coarse], dim=-1)  # B M 1024 + 3


        rebuild_feature = self.reduce_map(rebuild_feature.reshape(bs*self.number_coarse, -1)) # BM 1024


        relative_xyz = self.foldingnet(rebuild_feature).reshape(bs, self.number_coarse, 3, -1)    # B M 3 S

        fine = (relative_xyz + coarse.unsqueeze(-1)).transpose(2,3).reshape(bs, -1, 3)  # B N 3

        pc_input = pc_input - center.unsqueeze(-2)
        pc_out = torch.cat([pc_input,fine], dim=1)

        inp_sparse = self.fps(pc_input, self.fps_num)
        coarse = torch.cat([coarse,inp_sparse],dim=1)

        return (center.contiguous(), pc_out.contiguous(), coarse.contiguous(), fine.contiguous())








if __name__ == "__main__":
    p1_path = '/home/zhang/pcc/data/dataset/0/part_pc/par_0.pcd'

    p1 = read_pcd(p1_path)
    
    choice = np.random.choice(len(p1[:,0]), 2048, replace=True)
    input_pc = p1[choice, :]


    args = 0
    test = Model(args)
    a = torch.from_numpy(input_pc.astype(np.float32))
    a = a.unsqueeze(dim=0)
    a = torch.cat((a,a), dim=0)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test.to(device)
    a = a.to(device)

    center , pc_out, coarse , fine = test(a)
    print(center.shape)
    print(pc_out.shape)
    print(fine.shape)
