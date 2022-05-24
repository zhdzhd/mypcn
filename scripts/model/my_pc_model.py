from pyexpat import features
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
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

        a = torch.linspace(-0.1, 0.1, steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
        b = torch.linspace(-0.1, 0.1, steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
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


class Pointnet2_center(nn.Module):
    def __init__(self, input_dim=0,radius=[0.04,0.08,0.16],number_point=[512,128,32]):
        super().__init__()


        self.sa1=pointnet2_modules.PointnetSAModule(
                mlp=[input_dim, 32, 32, 64],
                npoint=number_point[0],
                radius=radius[0],
                nsample=64,
                use_xyz=True,
                bn=True,)

        self.sa2=pointnet2_modules.PointnetSAModule(
                mlp=[64, 128, 128, 256],
                npoint=number_point[1],
                radius=radius[1],
                nsample=32,
                use_xyz=True,
                bn=True,)

        self.sa3=pointnet2_modules.PointnetSAModule(
                mlp=[256, 256, 512, 512],
                npoint=number_point[2],
                radius=radius[2],
                nsample=16,
                use_xyz=True,
                bn=True,)

        self.sa4=pointnet2_modules.PointnetSAModule(
                mlp=[512, 512, 1024, 1024],
                use_xyz=True,
                bn=True,)


    
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

    def forward(self, pointcloud):
        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)


        xyz, features = self.sa1(xyz, features)
        xyz2, features2 = self.sa2(xyz, features)
        xyz3, features3 = self.sa3(xyz2, features2)
        _, features4 = self.sa4(xyz3, features3)



        return features4.squeeze(-1)
        
class Pointnet2_coarse(nn.Module):
    def __init__(self, input_dim=0,radius=[0.04,0.08,0.16,0.32],number_point=[1024,384,256,128]):
        super().__init__()

        self.sa1=pointnet2_modules.PointnetSAModule(
                mlp=[input_dim, 32, 64, 128],
                npoint=number_point[0],
                radius=radius[0],
                nsample=64,
                use_xyz=True,
                bn=True,)

        self.sa2=pointnet2_modules.PointnetSAModule(
                mlp=[128, 256, 512, 1024],
                npoint=number_point[1],
                radius=radius[1],
                nsample=32,
                use_xyz=True,
                bn=True,)

        self.sa3=pointnet2_modules.PointnetSAModule(
                mlp=[1024, 512, 512, 1024],
                npoint=number_point[2],
                radius=radius[2],
                nsample=16,
                use_xyz=True,
                bn=True,)

        self.sa4=pointnet2_modules.PointnetSAModule(
                mlp=[1024, 512, 512, 1024],
                npoint=number_point[3],
                radius=radius[3],
                nsample=8,
                use_xyz=True,
                bn=True,)

        self.saglobal=pointnet2_modules.PointnetSAModule(
                mlp=[1024, 512, 512, 1024],
                use_xyz=True,
                bn=True,)

        self.fp1=pointnet2_modules.PointnetFPModule(mlp=[1024+1024,1024])
        self.fp2=pointnet2_modules.PointnetFPModule(mlp=[1024+1024,1024])

    
    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features


    def forward(self, pointcloud):
        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)

        xyz, features = self.sa1(xyz, features)
        xyz2, features2 = self.sa2(xyz, features)
        xyz3, features3 = self.sa3(xyz2, features2)
        xyz4, features4 = self.sa4(xyz3, features3)
        _,global_features = self.saglobal(xyz4, features4)  #[1,1024,1]

        features = self.fp1(xyz3,xyz4,features3,features4)
        coarse_features = self.fp1(xyz2,xyz3,features2,features)  #B 1024 384

        return global_features,coarse_features


class Model(nn.Module):
    def __init__(self, args, num_pred=6144, grid_size=4, global_feature_size=1024):
        super(Model, self).__init__()
        self.number_fine = num_pred
        grid_size = 4 # set default
        self.grid_size = grid_size
        assert self.number_fine % grid_size**2 == 0
        self.number_coarse = self.number_fine // (grid_size ** 2 )
        self.fps_num = 128
        

        
        self.backbone_modules1 = Pointnet2_center(radius=[0.04,0.08,0.16],number_point=[512,128,32])
        self.backbone_modules2 = Pointnet2_coarse(radius=[0.004,0.08,0.016,0.032],number_point=[self.number_coarse*2,self.number_coarse,self.number_coarse//2,self.number_coarse//4])
        
        self.center_map= nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3)
        )


        self.mlp = nn.Sequential(
            nn.Conv1d(2048, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 3, 1)
        )

        self.reduce_map = nn.Linear(1024 + 3 + 1024, 1024)

        self.foldingnet = Fold(1024, step = self.grid_size, hidden_dim = 512)  # rebuild a cluster point

        
        
        
 
    def fps(self,pc, num):
        fps_idx = pointnet2_utils.furthest_point_sample(pc, num) 
        sub_pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
        return sub_pc


    def forward(self,pc_input,par_m,cen_gt = None,train = False):

        bs , n , _ = pc_input.shape

        if train:
            center_features = self.backbone_modules1(pc_input)  #B 1024
            center = self.center_map(center_features)   # B 4

            pc_input = pc_input - cen_gt.unsqueeze(-2)
            pc_input = pc_input*par_m.unsqueeze(-1).unsqueeze(-2)

        else:
            if cen_gt is None:
                center_features = self.backbone_modules1(pc_input)  #B 1024
                center = self.center_map(center_features)   # B 3

            else:
                center = cen_gt

            pc_input = pc_input - center.unsqueeze(-2)
            pc_input = pc_input*par_m.unsqueeze(-2)
   



                

        global_features,coarse_features = self.backbone_modules2(pc_input)  #B 1024 1  B 1024 M

        features = torch.cat([coarse_features,global_features.expand(-1,-1,self.number_coarse)],dim=1) #B 2048 M

        coarse = self.mlp(features).transpose(1,2).contiguous() # B M 3

        rebuild_feature = torch.cat([
            global_features.transpose(1,2).contiguous().expand(-1, self.number_coarse, -1),
            coarse_features.transpose(1,2).contiguous(),
            coarse], dim=-1)  # B M 1024 + 3 + 1024
        

        rebuild_feature = self.reduce_map(rebuild_feature.reshape(bs*self.number_coarse, -1)) # BM 1024


        relative_xyz = self.foldingnet(rebuild_feature).reshape(bs, self.number_coarse, 3, -1)    # B M 3 S

        fine = (relative_xyz + coarse.unsqueeze(-1)).transpose(2,3).reshape(bs, -1, 3)  # B N 3
        fine = torch.cat([pc_input,fine], dim=1)

        inp_sparse = self.fps(pc_input, self.fps_num)
        coarse = torch.cat([coarse,inp_sparse],dim=1)

        return (center.contiguous(), coarse.contiguous(), fine.contiguous())








if __name__ == "__main__":
    p1_path = '/home/zhang/mypcn/data/dataset/0/part_pc/par_0.pcd'

    p1 = read_pcd(p1_path)
    
    choice = np.random.choice(len(p1[:,0]), 2048, replace=True)
    input_pc = p1[choice, :]
    m = np.max(np.sqrt(np.sum(input_pc**2, axis=1)))

    args = 0
    test = Model(args)
    a = torch.from_numpy(input_pc.astype(np.float32))
    a = a.unsqueeze(dim=0)
    a = torch.cat((a,a), dim=0)
    m = torch.from_numpy(np.array([m]).astype(np.float32))
    m = m.unsqueeze(dim=0)
    m = torch.cat((m,m), dim=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test.to(device)
    a = a.to(device)
    m = m.to(device)

    center , coarse , fine = test(a,m)
    print(center.shape)
    print(coarse.shape)
    print(fine.shape)
