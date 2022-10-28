import sys
import os
import numpy as np
import argparse
from pathlib import Path
from glob import glob

os.environ["CUDA_VISIBLE_DEVICES"]='-1'   # cpu
# os.environ["CUDA_VISIBLE_DEVICES"]='0'  # gpu
# os.environ["CUDA_VISIBLE_DEVICES"]='1'  # gpu

import pcl
import pcl.pcl_visualization
# from mtcnn import MTCNN
import cv2



def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for cls training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-dataset_path", type=str, default='/home/bjgbiesseck/GitHub/MICA/demo/output/lfw', help='')
    parser.add_argument('-input_pc_ext', type=str, default='.ply', help='')
    parser.add_argument('-input_kp_ext', type=str, default='kpt68.npy', help='')
    parser.add_argument('-output_pc_ext', type=str, default='_centralized_nosetip.ply', help='')
    parser.add_argument('-start_from', type=str, default='', help='')
    parser.add_argument("-filter_radius", type=float, default=100.0,  help="Radius of sphere to filter points")
    
    return parser.parse_args()



# BERNARDO
class TreeLFW_3DReconstructedMICA:
    
    def get_all_sub_folders(self, dir_path=''):
        return sorted(glob(dir_path + '/*/*/'))
                
    def get_all_pointclouds_paths(self, dir_path, pc_ext='.ply', kp_ext='kpt68.npy'):
        all_sub_folders = self.get_all_sub_folders(dir_path)
        all_pc_paths = []
        all_kp_paths = []
        for sub_folder_pointcloud in all_sub_folders:
            pc_paths = sorted(glob(sub_folder_pointcloud + '/*' + pc_ext))
            kp_paths = sorted(glob(sub_folder_pointcloud + '/*' + kp_ext))
            assert len(pc_paths) > 0 and len(kp_paths) > 0 and len(pc_paths) == len(kp_paths)
            # print('pc_paths:', pc_paths)
            # print('kp_paths:', kp_paths)
            # print('----------------------')
            all_pc_paths += pc_paths
            all_kp_paths += kp_paths
        
        assert len(all_pc_paths) == len(all_kp_paths)
        return all_pc_paths, all_kp_paths





def load_point_cloud(path_point_cloud):
    cloud, normals = pcl.load(path_point_cloud)
    cloud = cloud.to_array()
    # print('load_point_cloud(): cloud =', cloud[0])
    # cloud -= np.array([0., 0., -100.], dtype=np.float32)
    # cloud /= 100
    # cloud = cloud - np.mean(cloud, 0)
    cloud = pcl.PointCloud(cloud)

    # if not normals is None:
    #     # normals = np.asarray(normals)
    #     normals = normals.to_array()
    #     normals = pcl.PointCloud_Normal(normals)
    
    # ptcloud_centred = pcl.PointCloud_PointXYZRGB()
    return cloud, normals


def load_keypoints(kp_path):
    kp_array = np.load(kp_path)
    if len(kp_array.shape) == 3 and kp_array.shape[0] == 1:
        kp_array = kp_array[0]
    return kp_array


def centralize_pointcloud(cloud, nose_point):
    cloud = cloud.to_array()
    cloud -= nose_point
    return pcl.PointCloud(cloud)


def init_pcl_viewer():
    viewer = pcl.pcl_visualization.PCLVisualizering()
    viewer.SetBackgroundColor(0, 0, 0)
    
    viewer.AddCoordinateSystem(100)
    
    # if args.sphere_radius > 0:
    #     sphere_cloud = generate_random_sphere_point_cloud(n_points=2000, radius=args.sphere_radius)
    #     pccolor_sphere = pcl.pcl_visualization.PointCloudColorHandleringCustom(sphere_cloud, 0, 0, 255)
    #     viewer.AddPointCloud_ColorHandler(sphere_cloud, pccolor_sphere, b'cloud_sphere', 0)
    #     viewer.SetPointCloudRenderingProperties(pcl.pcl_visualization.PCLVISUALIZER_POINT_SIZE, 3, b'cloud_sphere')

    return viewer


def show_point_cloud(cloud):
    viewer = init_pcl_viewer()

    # add point cloud
    pccolor1 = pcl.pcl_visualization.PointCloudColorHandleringCustom(cloud, 255, 0, 0)
    viewer.AddPointCloud_ColorHandler(cloud, pccolor1, b'cloud', 0)
    viewer.SetPointCloudRenderingProperties(pcl.pcl_visualization.PCLVISUALIZER_POINT_SIZE, 3, b'cloud')

    viewer.Spin()



def find_index_of_file_name(paths_list, file_name_search):
    for i in range(len(paths_list)):
        if file_name_search in paths_list[i]:
            return i
    return 0


def get_normals_and_curvatures(cloud, k_neighbours=0, radius=0):
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
    cloud_array = cloud.to_array()
    cloud_array += np.array([0., 0., -100])
    cloud = pcl.PointCloud(cloud_array)
    
    feature = cloud.make_NormalEstimation()
    if k_neighbours > 0:
        feature.set_KSearch(k_neighbours)
    else:
        feature.set_RadiusSearch(radius)
    normals = feature.compute()
    normals = normals.to_array()
    
    # normals[:,0:3] = normals[:,0:3] * -1    # invert only normal vectors
    # normals[:,3] = normals[:,3] * -1        # invert only curvature
    # normals *= -1                           # invert all components (normals and curvature)

    normals = pcl.PointCloud_Normal(normals)
    return normals                # original
    # return normals.to_array()   # BERNARDO


def filter_pointcloud_by_radius_from_origin(cloud, keypoint_ref, sphere_radius=90):
    keypoint_ref = np.expand_dims(np.asarray(keypoint_ref, dtype=np.float32), axis=0)
    # print('keypoint_ref:', keypoint_ref)
    searchPoint = pcl.PointCloud(keypoint_ref)
    # print('searchPoint:', searchPoint[0])
    kdtree = pcl.KdTreeFLANN(cloud)
    [ind, sqdist] = kdtree.radius_search_for_cloud(searchPoint, sphere_radius, cloud.size)
    # [ind, sqdist] = kdtree.nearest_k_search_for_cloud(searchPoint, 5)
    ind = ind[0]
    ind = ind[ind != 0]
    cloud = pcl.PointCloud(cloud.to_array()[ind])
    return cloud


def main_centralize_nosetip_with_normals(args):
    print('Searching point cloud files.....')
    pc_paths, kp_paths = TreeLFW_3DReconstructedMICA().get_all_pointclouds_paths(args.dataset_path, args.input_pc_ext, args.input_kp_ext)

    start_index = find_index_of_file_name(pc_paths, args.start_from)
    # start_index = find_index_of_file_name(kp_paths, args.start_from)

    for i in range(start_index, len(pc_paths)):
        pc_path = pc_paths[i]
        kp_path = kp_paths[i]

        print(str(i) + '/' + str(len(pc_paths)) + ' - pc_path:', pc_path)
        print(str(i) + '/' + str(len(kp_paths)) + ' - kp_path:', kp_path)

        ptcloud, _ = load_point_cloud(pc_path)
        kpt68 = load_keypoints(kp_path)
        print('ptcloud.to_array().shape (original):', ptcloud.to_array().shape)

        ptcloud_centralized = centralize_pointcloud(ptcloud, kpt68[30])
        ptcloud_centralized = filter_pointcloud_by_radius_from_origin(ptcloud_centralized, [0., 0., 0.], sphere_radius=args.filter_radius)
        print('ptcloud_centralized.to_array().shape:', ptcloud_centralized.to_array().shape)
        # show_point_cloud(ptcloud_centralized)
        # sys.exit(0)

        normals_and_curvatures = get_normals_and_curvatures(ptcloud_centralized, k_neighbours=30)
        ptcloud_centralized_with_normals_array = np.hstack((ptcloud_centralized.to_array(), normals_and_curvatures.to_array()))
        # print('ptcloud_centralized_with_normals:', ptcloud_centralized_with_normals.shape)
        # print('ptcloud_centralized_with_normals[1000:1100]:', ptcloud_centralized_with_normals[1000:1100])

        path_centralized_ptcloud_bin = '/'.join(pc_path.split('/')[:-1]) + '/' + pc_path.split('/')[-1].split('.')[0] + args.output_pc_ext
        print('Saving centralized point cloud with normals (binary):', path_centralized_ptcloud_bin)
        np.save(path_centralized_ptcloud_bin, ptcloud_centralized_with_normals_array)
        # pcl.save(ptcloud_centralized, path_centralized_ptcloud, format=args.output_pc_ext.split('.')[1], binary=False)

        path_centralized_ptcloud_txt = '/'.join(pc_path.split('/')[:-1]) + '/' + pc_path.split('/')[-1].split('.')[0] + args.output_pc_ext.replace('.npy', '.xyz')
        print('Saving centralized point cloud with normals (text):', path_centralized_ptcloud_txt)
        np.savetxt(path_centralized_ptcloud_txt, ptcloud_centralized_with_normals_array, newline="\r\n")

        # print('ptcloud:', ptcloud.to_array())
        # print('kpt68:', kpt68)
        # print('kpt68.shape:', kpt68.shape)
        # sys.exit(0)
        print('---------------------------')




if __name__ == '__main__':
    
    if not '-dataset_path' in sys.argv:
        sys.argv += ['-dataset_path', '/home/bjgbiesseck/GitHub/MICA/demo/output/lfw']

    # sys.argv += ['-filter_radius', '90']
    sys.argv += ['-filter_radius', '100']

    # sys.argv += ['-input_pc_ext', '.obj']
    # sys.argv += ['-input_pc_ext', '.ply']
    sys.argv += ['-input_pc_ext', '_upsample_MetaPU.xyz']

    sys.argv += ['-input_kp_ext', 'kpt68.npy']

    # sys.argv += ['-output_pc_ext', '_centralized_nosetip.ply']
    # sys.argv += ['-output_pc_ext', '_centralized-nosetip_with-normals_filter-radius=100.npy']
    sys.argv += ['-output_pc_ext', '_upsample_MetaPU_centralized-nosetip_with-normals_filter-radius=100.npy']

    args = parse_args()
    # print('__main__(): args=', args)

    
    main_centralize_nosetip_with_normals(args)
