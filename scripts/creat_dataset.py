import open3d
import numpy as np
import random
import os 




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

def write_matrix(path,matrix):
    # os.mkdir(path)
    with open(path,"w") as f:
        for row in matrix:
            for col in row:
                f.write('%f '%col)

base_path = "/home/zhang/pcc/data"
modelpc_path = "/home/zhang/pcc/data/model"
dataset_path = "/home/zhang/pcc/data/dataset"
# base_path = "/home/zhang/pcc/data_test"
# modelpc_path = "/home/zhang/pcc/data_test/model"
# dataset_path = "/home/zhang/pcc/data_test/dataset"
num_data = 50

for i in range(70):
    pc_path = os.path.join(modelpc_path , ("pc%d.pcd" % (i)))


    dataset_pc_path = os.path.join(dataset_path,('%d')%i)


    par_path = os.path.join(dataset_pc_path,'part_pc')
    sep_path = os.path.join(dataset_pc_path,'separate_pc')
    matrix_path = os.path.join(dataset_pc_path,'rotate_matrix')
    gt_path = os.path.join(dataset_pc_path,'gt_pc')
    if not os.path.exists(par_path):
        os.makedirs(par_path) 
    if not os.path.exists(sep_path):
        os.makedirs(sep_path) 
    if not os.path.exists(matrix_path):
        os.makedirs(matrix_path) 
    if not os.path.exists(gt_path):
        os.makedirs(gt_path) 

    gt_save_path = os.path.join(gt_path,('pc_%d.pcd')%(i))

    pc = open3d.io.read_point_cloud(pc_path)
    ptcloud = np.array(pc.points)
    choice = np.random.choice(len(ptcloud[:,0]), 8192, replace=True)
    ptcloud = ptcloud[choice, :]
    pc.points = open3d.utility.Vector3dVector(ptcloud)
    open3d.io.write_point_cloud(gt_save_path, pc)
    num_points,_ = ptcloud.shape
    for rotate_num in range(10):
        matrix = random_rot_matrix()
        pc_after_rotate,_ = rotate_point_cloud(ptcloud,matrix)

        

        for creat_num in range(10):
            par_save_path = os.path.join(par_path,('par_%d.pcd')%(rotate_num*10+creat_num))
            sep_save_path = os.path.join(sep_path,('sep_%d.pcd')%(rotate_num*10+creat_num))

            pardata,otherdata,sepdata = seprate_point_cloud(pc_after_rotate,num_points,int(num_points/2)) 

            choice = np.random.choice(len(pardata[:,0]), 2048, replace=True)
            pardata = pardata[choice, :]

            pardata_pc = open3d.geometry.PointCloud()
            pardata_pc.points = open3d.utility.Vector3dVector(pardata)
            open3d.io.write_point_cloud(par_save_path, pardata_pc)

            sepdata_pc = open3d.geometry.PointCloud()
            sepdata_pc.points = open3d.utility.Vector3dVector(sepdata)
            open3d.io.write_point_cloud(sep_save_path, sepdata_pc)

            matrix_save_path = os.path.join(matrix_path,('%d.txt')%(rotate_num*10+creat_num))
            write_matrix(matrix_save_path,matrix)


                    








