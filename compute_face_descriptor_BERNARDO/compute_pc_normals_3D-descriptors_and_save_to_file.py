import sys
import os
import numpy as np
import argparse
from pathlib import Path
from glob import glob
from tqdm import tqdm
import torch

import pcl

from pointnet2.train.train_triplet import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for point cloud normal computing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "-num_points", type=int, default=20000, help="Number of points to train with"
    )
    parser.add_argument(
        "-weight_decay", type=float, default=1e-5, help="L2 regularization coeff"
    )
    parser.add_argument("-lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument(
        "-lr_decay", type=float, default=0.7, help="Learning rate decay gamma"
    )
    parser.add_argument(
        "-decay_step", type=float, default=5e3, help="Learning rate decay step"
    )
    parser.add_argument(
        "-bn_momentum", type=float, default=0.5, help="Initial batch norm momentum"
    )
    parser.add_argument(
        "-bnm_decay", type=float, default=0.5, help="Batch norm momentum decay gamma"
    )
    parser.add_argument(
        "-model_checkpoint", type=str, default=None, help="Checkpoint to start from"
    )
    parser.add_argument(
        "-cls_checkpoint", type=str, default=None, help="Checkpoint to start from"
    )
    parser.add_argument(
        "-epochs", type=int, default=30, help="Number of epochs to train for"
    )
    parser.add_argument(
        "-run_name",
        type=str,
        default="cls_run_1",
        help="Name for run in tensorboard_logger",
    )
    # loss Classifier
    parser.add_argument('--margin', type=float, default=0.4, metavar='MARGIN',
                        help='the margin value for the triplet loss function (default: 1.0')
    parser.add_argument('--num_triplet', type=int, default=10000, metavar='num_triplet',
                        help='the margin value for the triplet loss function (default: 1e4')

    parser.add_argument('--num_class', type=int, default=500,
                        help='number of people(class)')
    parser.add_argument('--classifier_type', type=str, default='AL',
                        help='Which classifier for train. (MCP, AL, L)')

    # BERNARDO
    parser.add_argument(
        "-dataset_path", type=str, default='',
        help="Path of dataset root folder containing 3D face reconstructions (OBJ or PLY format)"
    )
    return parser.parse_args()


# BERNARDO
class Tree:
    def walk(self, dir_path: Path):
        contents = list(dir_path.iterdir())
        for path in contents:
            if path.is_dir():  # extend the prefix and recurse:
                yield str(path)
                yield from self.walk(path)

    def get_all_sub_folders(self, dir_path: str):
        folders = [dir_path]
        for folder in Tree().walk(Path(os.getcwd()) / dir_path):
            # print(folder)
            folders.append(folder)
        return folders


def get_normals(cloud):
    """
    FROM: https://pcl.gitbook.io/tutorial/part-2/part02-chapter03/part02-chapter03-normal-pcl-python
    The actual *compute* call from the NormalEstimation class does nothing internally but:
    for each point p in cloud P
        1. get the nearest neighbors of p
        2. compute the surface normal n of p
        3. check if n is consistently oriented towards the viewpoint and flip otherwise

    # normals: pcl._pcl.PointCloud_Normal,size: 26475
    # cloud: pcl._pcl.PointCloud
    """
    # cloud = pcl.load(cloud_path)
    feature = cloud.make_NormalEstimation()
    # feature.set_RadiusSearch(0.1) #Use all neighbors in a sphere of radius 3cm
    feature.set_KSearch(3)
    normals = feature.compute()

    # return normals            # original
    return normals.to_array()   # BERNARDO


def get_pointcloud_with_normals(cloud):
    # print('cloud:', cloud)
    cloud_with_normals = cloud.to_array()
    # print('cloud_with_normals:', cloud_with_normals)
    normals = get_normals(cloud)
    cloud_with_normals = np.hstack((cloud_with_normals, normals))
    # print('cloud_with_normals.shape:', cloud_with_normals.shape)
    # print('cloud_with_normals:', cloud_with_normals)
    return cloud_with_normals


def preprocess_pointcloud_with_normals(pc_with_normals):
    point_set = pc_with_normals
    # normlize
    point_set[:, 0:3] = (point_set[:, 0:3]) / 100
    point_set = torch.from_numpy(point_set)
    point_set[:, 6] = torch.pow(point_set[:, 6], 0.1)
    # print('point_set.shape:', point_set.shape)

    input = point_set
    input = input.unsqueeze(0).contiguous()
    input = input.to("cuda", non_blocking=True)
    # print('input.shape:', input.shape)
    return input



def load_pc_and_compute_normals(model, folder):
    # image_paths = sorted(glob(folder + '/*.obj')) + sorted(glob(folder + '/*.ply'))
    image_paths = sorted(glob(folder + '/*.obj'))
    for image_path_OBJ in tqdm(image_paths):
        image_path_PLY = image_path_OBJ.replace('.obj', '.ply')
        # print('image_path:', image_path)
        name = Path(image_path_OBJ).stem
        # print('name:', name)
        print('Loading point cloud:', image_path_OBJ)
        cloud_from_OBJ = pcl.load(image_path_OBJ)
        cloud_from_PLY = pcl.load(image_path_PLY)
        # print('cloud_from_OBJ.to_array().shape:', cloud_from_OBJ.to_array().shape)
        # print('cloud_from_PLY.to_array().shape:', cloud_from_PLY.to_array().shape)

        pc_with_normals_from_OBJ = get_pointcloud_with_normals(cloud_from_OBJ)
        pc_with_normals_from_PLY = get_pointcloud_with_normals(cloud_from_PLY)
        # print('pc_with_normals_from_OBJ.shape:', pc_with_normals_from_OBJ.shape)
        # print('pc_with_normals_from_PLY.shape:', pc_with_normals_from_PLY.shape)

        pc_with_normals_from_OBJ = preprocess_pointcloud_with_normals(pc_with_normals_from_OBJ)
        pc_with_normals_from_PLY = preprocess_pointcloud_with_normals(pc_with_normals_from_PLY)

        path_pc_with_normals_from_OBJ = image_path_OBJ.replace('.obj', '_OBJ_with_normals.pt')
        path_pc_with_normals_from_PLY = image_path_PLY.replace('.ply', '_PLY_with_normals.pt')
        print('Saving mesh with normals:', path_pc_with_normals_from_OBJ, end=' ... ')
        torch.save(pc_with_normals_from_OBJ, path_pc_with_normals_from_OBJ)
        torch.save(pc_with_normals_from_PLY, path_pc_with_normals_from_PLY)
        print('Saved!')


        print('Computing 3D face descriptor ...')
        p_feature_from_OBJ = torch.zeros((1, 1, 512))  # BERNARDO
        p_feature_from_PLY = torch.zeros((1, 1, 512))  # BERNARDO
        feat_from_OBJ = model.forward(pc_with_normals_from_OBJ)  # 1x512
        feat_from_PLY = model.forward(pc_with_normals_from_PLY)  # 1x512
        p_feature_from_OBJ[:, :, :] = feat_from_OBJ.cpu()  # 1x1x512
        p_feature_from_PLY[:, :, :] = feat_from_PLY.cpu()  # 1x1x512
        p_feature_norm_from_OBJ = torch.norm(p_feature_from_OBJ, p=2, dim=2)
        p_feature_norm_from_PLY = torch.norm(p_feature_from_PLY, p=2, dim=2)
        path_feat_norm_from_OBJ = image_path_OBJ.replace(name+'.obj', '3D_face_descriptor_from_OBJ.pt')
        path_feat_norm_from_PLY = image_path_PLY.replace(name+'.ply', '3D_face_descriptor_from_PLY.pt')
        print('Saving 3D face descriptor:', path_feat_norm_from_OBJ, end=' ... ')
        torch.save(feat_from_OBJ, path_feat_norm_from_OBJ)
        torch.save(feat_from_PLY, path_feat_norm_from_PLY)
        print('Saved!')

        # loaded_feat_from_OBJ = torch.load(path_feat_norm_from_OBJ)
        # loaded_feat_from_PLY = torch.load(path_feat_norm_from_PLY)
        # print(loaded_feat_from_OBJ.shape, 'loaded_feat_from_OBJ[:,0:10]:', loaded_feat_from_OBJ[:,0:10])
        # print(loaded_feat_from_PLY.shape, 'loaded_feat_from_PLY[:,0:10]:', loaded_feat_from_PLY[:,0:10])



def build_Pointnet_model(args):
    model = Pointnet(input_channels=3, use_xyz=True)
    model.cuda()
    # 512 is dimension of feature
    classifier = {
        'MCP': layer.MarginCosineProduct(512, args.num_class).cuda(),
        'AL': layer.AngleLinear(512, args.num_class).cuda(),
        'L': torch.nn.Linear(512, args.num_class, bias=False).cuda()
    }[args.classifier_type]

    # criterion = nn.TripletMarginLoss(margin=0.5, p=2)
    optimizer = optim.Adam(
        [{'params': model.parameters()}, {'params': classifier.parameters()}],
        lr=lr, weight_decay=args.weight_decay
    )

    print('compute_face_descriptor_BERNARDO.py: main(): Loading trained model...')
    if args.model_checkpoint is not None:
        checkpoint_status = pt_utils.load_checkpoint(
            model, optimizer, filename=args.model_checkpoint.split(".")[0]
        )
        if checkpoint_status is not None:
            it, start_epoch, best_loss = checkpoint_status

    model.eval()
    optimizer.zero_grad()
    return model



def main(args):
    # load dataset (LFW and TALFW)
    print('face_recognition_3d_descriptor.py: main(): Loading sub-folders of dataset', args.dataset_path, '...')
    sub_folders = Tree().get_all_sub_folders(args.dataset_path)
    print('len(sub_folders):', len(sub_folders))

    model = build_Pointnet_model(args)

    # compute and save point cloud normals to disk
    # for i in range(10):  # range(len(sub_folders))
    for i in range(len(sub_folders)):
        sub_folder = sub_folders[i]
        # print('sub_folder:', sub_folder)
        load_pc_and_compute_normals(model, sub_folder)



if __name__ == '__main__':
    # sys.argv += ['-epochs', '100']
    # print('__main__(): sys.argv=', sys.argv)

    sys.argv += ['-dataset_path', '/home/bjgbiesseck/GitHub/MICA/demo/output/lfw']
    # sys.argv += ['-dataset_path', '/home/bjgbiesseck/GitHub/MICA/demo/output/TALFW']

    args = parse_args()
    # print('__main__(): args=', args)

    main(args)
