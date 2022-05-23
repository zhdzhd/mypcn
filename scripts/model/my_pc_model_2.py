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

class FoldingNet(nn.Module):
    def __init__(self, num_pred ,encoder_channel):
        super().__init__()
        self.num_pred = num_pred
        self.encoder_channel = encoder_channel
        self.grid_size = int(pow(self.num_pred,0.5) + 0.5)

        self.first_conv = nn.Sequential(
            nn.Conv1d(3,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128,256,1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512,512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512,self.encoder_channel,1)
        )

        self.folding1 = nn.Sequential(
            nn.Conv1d(self.encoder_channel + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(self.encoder_channel + 3, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1),
        )

        a = torch.linspace(-0.5, 0.5, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.5, 0.5, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda() # 1 2 N

    def get_loss(self, ret, gt):
        loss_coarse = self.loss_func(ret[0], gt)
        loss_fine = self.loss_func(ret[1], gt)
        return loss_coarse, loss_fine

    def forward(self, xyz):
        bs , n , _ = xyz.shape
        # encoder
        feature = self.first_conv(xyz.transpose(2,1))  # B 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # B 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# B 512 n
        feature = self.second_conv(feature) # B 1024 n
        feature_global = torch.max(feature,dim=2,keepdim=False)[0] # B 1024
        # folding decoder
        fd1, fd2 = self.decoder(feature_global) # B N 3
        return (fd2, fd2) # FoldingNet producing final result directly
        
    def decoder(self,x):
        num_sample = self.grid_size * self.grid_size
        bs = x.size(0)
        features = x.view(bs, self.encoder_channel, 1).expand(bs, self.encoder_channel, num_sample)
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)

        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)

        return fd1.transpose(2,1).contiguous() , fd2.transpose(2,1).contiguous()


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
        self.conv3 = torch.nn.Conv1d(2048, 1024, 1)
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
        
        self.offset_conv= nn.Sequential(
            nn.Conv1d(1024,1024,1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024,1024,1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024,3 + 1024,1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,3*self.number_coarse)
        )

        

        self.final_conv = nn.Sequential(
            nn.Conv1d(1024+3+2,512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512,512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512,3,1)
        )
        
        a = torch.linspace(-0.01, 0.01, steps=grid_size, dtype=torch.float).view(1, grid_size).expand(grid_size, grid_size).reshape(1, -1)
        b = torch.linspace(-0.01, 0.01, steps=grid_size, dtype=torch.float).view(grid_size, 1).expand(grid_size, grid_size).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, grid_size ** 2).cuda() # 1 2 S
        # self.folding = FoldingNet()
 
    def fps(self,pc, num):
        fps_idx = pointnet2_utils.furthest_point_sample(pc, num) 
        sub_pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
        return sub_pc


    def forward(self,pc_input,sep_pc = None):

        bs , n , _ = pc_input.shape

        global_features = self.backbone_modules(pc_input,sep_pc)
        # out = self.offset_conv(global_features)
        # center = out[:,0:3,:].transpose(1,2)
        # feature_offset = out[:,3:,:]
        # global_features = global_features - feature_offset
        # pc_input = pc_input - center
        global_features = global_features.squeeze(-1)    
        center = torch.zeros([bs,3])
        coarse = self.mlp(global_features).reshape(-1,self.number_coarse,3) # B M 3
        point_feat = coarse.unsqueeze(2).expand(-1,-1,self.grid_size**2,-1) # B M S 3
        point_feat = point_feat.reshape(-1,self.number_fine,3).transpose(2,1) # B 3 N

        seed = self.folding_seed.unsqueeze(2).expand(bs,-1,self.number_coarse, -1) # B 2 M S
        seed = seed.reshape(bs,-1,self.number_fine)  # B 2 N

        feature_global = global_features.unsqueeze(2).expand(-1,-1,self.number_fine) # B 1024 N
        feat = torch.cat([feature_global, seed, point_feat], dim=1) # B C N
    
        fine = self.final_conv(feat) + point_feat   # B 3 N


        # global_features = self.backbone_modules(pc_input,sep_pc)
        # coarse = self.coarse_conv(global_features).reshape(-1,self.number_coarse,3) # B M 3

        # point_feat = coarse.unsqueeze(2).expand(-1,-1,self.grid_size**2,-1) # B M S 3

        # point_feat = point_feat.reshape(-1,self.number_fine,3).transpose(2,1) # B 3 N

        # seed = self.folding_seed.unsqueeze(2).expand(bs,-1,self.number_coarse, -1) # B 2 M S

        # seed = seed.reshape(bs,-1,self.number_fine)  # B 2 N

        # feature_global = global_features.expand(-1,-1,self.number_fine) # B 1024 N
        # feat = torch.cat([feature_global, seed, point_feat], dim=1) # B C N
    
        # fine = self.final_conv(feat) + point_feat   # B 3 N
        fine = fine.transpose(1,2)
        pc_out = torch.cat([pc_input,fine], dim=1)

        # center = center.squeeze(1)

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
    # print(center.shape)
    # print(pc_out.shape)
    # print(fine.shape)
