import os
import sys
import random

import sys
sys.path.append("/home/zhang/pcc/src/pc_dateset/scripts")


from extensions.chamfer_dist import ChamferDistanceL2

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def get_loss(pc_output,pc_gt,cen_output,cen_gt):
    b,_ = cen_output.shaep
    
    pc_loss = ChamferDistanceL2()
    loss1 = pc_loss(pc_output,pc_gt)

    loss2 = torch.sum((cen_output-cen_gt)**2)/b

    loss = loss1 + 0.5*loss2
    return loss





if __name__ == "__main__":
    a=np.array([[1,2,3],[2,3,4]]).astype(np.float32)
    b=np.array([[2,3,4],[3,4,5]]).astype(np.float32)

    cen_output = torch.from_numpy(a)
    print(cen_output)
    cen_output = cen_output.unsqueeze(-1).repeat(1,1,3)
    print(cen_output)
    # cen_gt = torch.from_numpy(b)
    # print(cen_output-cen_gt)

    # loss2 = torch.sum((cen_output-cen_gt)**2)

    # print(loss2)