import sys
import os
import numpy as np
import argparse
from pathlib import Path
import glob

os.environ["CUDA_VISIBLE_DEVICES"]='-1'   # cpu
# os.environ["CUDA_VISIBLE_DEVICES"]='0'  # gpu
# os.environ["CUDA_VISIBLE_DEVICES"]='1'  # gpu

import pcl
import pcl.pcl_visualization
from mtcnn import MTCNN
import cv2



def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for cls training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-dataset_path", type=str, default='/home/bjgbiesseck/datasets/FRGCv2.0/FRGC-2.0-dist/nd1', help='')
    parser.add_argument('-input_img_ext', type=str, default='.ppm', help='')
    parser.add_argument('-input_pc_ext', type=str, default='.abs.gz', help='')
    parser.add_argument('-output_pc_ext', type=str, default='_centralized_nosetip.ply', help='')
    parser.add_argument('-start_from', type=str, default='', help='')
    parser.add_argument("-filter_radius", type=float, default=90.0,  help="Radius of sphere to filter points")
    
    return parser.parse_args()



# BERNARDO
class TreeFRGCv2:
    def __walk(self, dir_path: Path):
        contents = list(dir_path.iterdir())
        for path in contents:
            if path.is_dir():  # extend the prefix and recurse:
                yield str(path)
                yield from self.__walk(path)

    def get_all_sub_folders(self, dir_path: str):
        # dir_path = dir_path.replace('//', '/')
        folders = [dir_path]
        for folder in self.__walk(Path(dir_path)):
            # print(folder)
            folders.append(folder)
        return sorted(folders)

    def get_sub_folders_one_level(self, dir_path: str):
        # sub_folders = [f.path for f in os.scandir(dir_path) if f.is_dir()]
        sub_folders = [f.name for f in os.scandir(dir_path) if f.is_dir()]
        return sorted(sub_folders)
    
    def get_images_pointclouds_paths(self, dir_path, img_ext, pc_ext):
        all_sub_folders = self.get_all_sub_folders(dir_path)
        # print('all_sub_folders:', all_sub_folders)
        valid_sub_folders = []
        for sub_folder in all_sub_folders:
            files = glob.glob(sub_folder + '/*' + pc_ext)
            if len(files) > 0:
                valid_sub_folders.append(sub_folder)
                # print('sub_folder:', sub_folder)
                # print('files:', files)
        all_img_paths = []
        all_pc_paths = []
        for valid_sub_folder in valid_sub_folders:
            img_paths = sorted(glob.glob(valid_sub_folder + '/*' + img_ext))
            pc_paths =  sorted(glob.glob(valid_sub_folder + '/*' + pc_ext))
            
            all_img_paths += img_paths
            all_pc_paths += pc_paths
        return all_pc_paths, all_img_paths
                



def load_imgRGB(img_path=''):
    # img = cv2.imread(img_path)
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    return img


def detect_nose_landmarks(detector, imgRGB):
    try:
        landmarks = detector.detect_faces(imgRGB)[0]
        nose_xy = landmarks['keypoints']['nose']
        nose_index = nose_xy[0] + nose_xy[1]*imgRGB.shape[1]
        # print('nose_xy:', nose_xy)
        # imgRGB_with_landmark = cv2.circle(imgRGB.copy(), nose_xy, radius=3, color=(0, 0, 255), thickness=-1)
        # cv2.imshow('imgRGB_with_landmark', imgRGB_with_landmark)
        # cv2.imshow('imgRGB', imgRGB)
        # cv2.waitKey(0)
        # sys.exit(0)
        return nose_xy, nose_index

    except IndexError:
        raise IndexError('No face detected in image')


'''
def detect_nose_landmarks(detector, imgRGB):
    landmarks = detector.detect_faces(imgRGB)[0]
    nose_xy = landmarks['keypoints']['nose']
    nose_index = nose_xy[0] + nose_xy[1]*imgRGB.shape[1]
    # print('nose_xy:', nose_xy)
    # imgRGB_with_landmark = cv2.circle(imgRGB.copy(), nose_xy, radius=3, color=(0, 0, 255), thickness=-1)
    # cv2.imshow('imgRGB_with_landmark', imgRGB_with_landmark)
    # cv2.imshow('imgRGB', imgRGB)
    # cv2.waitKey(0)
    # sys.exit(0)
    return nose_xy, nose_index
'''


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


def centralize_and_filter_pointcloud(cloud, nose_point_index, value_invalid_coord=-999999.0):
    cloud = cloud.to_array()
    number_valid_points = sum(cloud[:,0] != value_invalid_coord)
    cloud_filtered = np.zeros((number_valid_points, 3), dtype=np.float32)
    print('nose_point_index:', nose_point_index)
    print('cloud[nose_point_index]:', cloud[nose_point_index])
    j = 0
    for i in range(cloud.shape[0]):
        if cloud[i][0] != value_invalid_coord and cloud[i][1] != value_invalid_coord and cloud[i][2] != value_invalid_coord:
            cloud_filtered[j] = cloud[i]
            j += 1
    cloud_filtered -= cloud[nose_point_index]
    return pcl.PointCloud(cloud_filtered)



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



def main_centralize_nosetip(args):
    print('Searching point cloud files...')
    pc_paths, img_paths = TreeFRGCv2().get_images_pointclouds_paths(args.dataset_path, args.input_img_ext, args.input_pc_ext)
    assert len(img_paths) == len(pc_paths)

    print('Initializing MTCNN face detector...')
    detector = MTCNN()

    # start_index = find_index_of_file_name(pc_paths, args.start_from)
    start_index = find_index_of_file_name(img_paths, args.start_from)

    for i in range(start_index, len(img_paths)):
        img_path = img_paths[i]
        pc_path = pc_paths[i]
        print(str(i) + '/' + str(len(img_paths)) + ' - pc_path:', pc_path)
        print(str(i) + '/' + str(len(img_paths)) + ' - img_path:', img_path)

        imgRGB = load_imgRGB(img_path)
        print('Detecting face landmarks...')

        try:
            nose_landmark_xy, nose_landmark_index = detect_nose_landmarks(detector, imgRGB)

            ptcloud, _ = load_point_cloud(pc_path)
            ptcloud_centralized = centralize_and_filter_pointcloud(ptcloud, nose_landmark_index)
            # show_point_cloud(ptcloud_centralized)

            path_centralized_ptcloud = '/'.join(pc_path.split('/')[:-1]) + '/' + pc_path.split('/')[-1].split('.')[0] + args.output_pc_ext
            print('Saving centralized point cloud:', path_centralized_ptcloud)
            pcl.save(ptcloud_centralized, path_centralized_ptcloud, format=args.output_pc_ext.split('.')[1], binary=False)

            path_imgRGB_with_nose_landmark = img_path.replace(args.input_img_ext, args.output_pc_ext.replace('.ply', '.jpg'))
            imgRGB_with_landmark = cv2.circle(cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR).copy(), nose_landmark_xy, radius=3, color=(0, 0, 255), thickness=-1)
            print('Saving img RGB with nose landmark:', path_imgRGB_with_nose_landmark)
            cv2.imwrite(path_imgRGB_with_nose_landmark, imgRGB_with_landmark)

        except:
            print('No face detected in ' + img_path)

        print('-----------------------------------------\n')
        # sys.exit(0)


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
    print('Searching point cloud files...')
    pc_paths, img_paths = TreeFRGCv2().get_images_pointclouds_paths(args.dataset_path, args.input_img_ext, args.input_pc_ext)
    assert len(img_paths) == len(pc_paths)

    print('Initializing MTCNN face detector...')
    detector = MTCNN()

    # start_index = find_index_of_file_name(pc_paths, args.start_from)
    start_index = find_index_of_file_name(img_paths, args.start_from)

    for i in range(start_index, len(img_paths)):
        img_path = img_paths[i]
        pc_path = pc_paths[i]
        print(str(i) + '/' + str(len(img_paths)) + ' - pc_path:', pc_path)
        print(str(i) + '/' + str(len(img_paths)) + ' - img_path:', img_path)

        imgRGB = load_imgRGB(img_path)
        print('Detecting face landmarks...')

        try:
            nose_landmark_xy, nose_landmark_index = detect_nose_landmarks(detector, imgRGB)

            ptcloud, _ = load_point_cloud(pc_path)
            ptcloud_centralized = centralize_and_filter_pointcloud(ptcloud, nose_landmark_index)
            ptcloud_centralized = filter_pointcloud_by_radius_from_origin(ptcloud_centralized, [0., 0., 0.], sphere_radius=args.filter_radius)
            # show_point_cloud(ptcloud_centralized)

            normals_and_curvatures = get_normals_and_curvatures(ptcloud_centralized, k_neighbours=30)
            ptcloud_centralized_with_normals_array = np.hstack((ptcloud_centralized.to_array(), normals_and_curvatures.to_array()))
            # print('ptcloud_centralized_with_normals:', ptcloud_centralized_with_normals.shape)
            # print('ptcloud_centralized_with_normals[1000:1100]:', ptcloud_centralized_with_normals[1000:1100])
            
            path_centralized_ptcloud = '/'.join(pc_path.split('/')[:-1]) + '/' + pc_path.split('/')[-1].split('.')[0] + args.output_pc_ext
            print('Saving centralized point cloud (text):', path_centralized_ptcloud)
            np.savetxt(path_centralized_ptcloud, ptcloud_centralized_with_normals_array, newline="\r\n")
            
            path_centralized_ptcloud_bin = '/'.join(pc_path.split('/')[:-1]) + '/' + pc_path.split('/')[-1].split('.')[0] + args.output_pc_ext.replace('.xyz', '.npy')
            print('Saving centralized point cloud (binary):', path_centralized_ptcloud_bin)
            np.save(path_centralized_ptcloud_bin, ptcloud_centralized_with_normals_array)
            # pcl.save(ptcloud_centralized, path_centralized_ptcloud, format=args.output_pc_ext.split('.')[1], binary=False)

            path_imgRGB_with_nose_landmark = img_path.replace(args.input_img_ext, args.input_img_ext.replace(args.input_img_ext, '_centralized_nosetip.jpg'))
            imgRGB_with_landmark = cv2.circle(cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR).copy(), nose_landmark_xy, radius=3, color=(0, 0, 255), thickness=-1)
            print('Saving img RGB with nose landmark:', path_imgRGB_with_nose_landmark)
            cv2.imwrite(path_imgRGB_with_nose_landmark, imgRGB_with_landmark)

            # sys.exit(0)

        except IndexError:
            print('No face detected in ' + img_path)

        print('-----------------------------------------\n')
        # sys.exit(0)




if __name__ == '__main__':
    
    if not '-dataset_path' in sys.argv:
        sys.argv += ['-dataset_path', '/home/bjgbiesseck/datasets/FRGCv2.0/FRGC-2.0-dist/nd1']

    sys.argv += ['-input_img_ext', '.ppm']
    sys.argv += ['-input_pc_ext', '.abs.gz']

    # sys.argv += ['-output_pc_ext', '_centralized_nosetip.ply']
    sys.argv += ['-output_pc_ext', '_centralized-nosetip_with-normals_filter-radius=90.xyz']

    # sys.argv += ['-start_from', 'Fall2003range/04729d25.ppm']


    args = parse_args()
    # print('__main__(): args=', args)

    
    # main_centralize_nosetip(args)
    main_centralize_nosetip_with_normals(args)
