from math import sqrt
import numpy as np
import pcl
import pcl.pcl_visualization
import argparse
import sys



def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(
        description="Arguments for visualizing point cloud with normals",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-normals_size", type=float, default=0,  help="Normals vectors size")
    parser.add_argument("-input_path", type=str, default='/home/bjgbiesseck/GitHub/MICA/demo/output/lfw/Aaron_Eckhart/Aaron_Eckhart_0001/mesh.obj',  help="Input file path")
    parser.add_argument("-points_size", type=int, default=3,  help="Size of points to show")
    parser.add_argument("-sphere_radius", type=float, default=0,  help="Radius of a sphere for comparison")
    parser.add_argument("-coord_system_size", type=float, default=100,  help="Size of X, Y and Z axis")
    parser.add_argument("-filter_radius", type=float, default=0.0,  help="Radius of sphere to filter points")
    parser.add_argument("-filter_index", type=int, default=0,  help="Indexes of points to crop out")
    parser.add_argument("-only_valid_points", type=str2bool, default=True,  help="True or False")
    parser.add_argument("-centralize", type=str2bool, default=False,  help="True of False")
    parser.add_argument("-normalize", type=str2bool, default=False,  help="True of False")
    parser.add_argument("-scale", type=int, default=100,  help="Original scale of point cloud to divide each point")
    
    return parser.parse_args()


def generate_random_sphere_point_cloud(n_points=1000, radius=1.0):
    xyz_values = np.random.uniform(low=-10.0, high=10.0, size=(n_points, 3)).astype(dtype=np.float32)
    for i in range(n_points):
        xyz_values[i] /= sqrt(xyz_values[i,0]**2 + xyz_values[i,1]**2 + xyz_values[i,2]**2)
        xyz_values[i] *= radius
    sphere_cloud = pcl.PointCloud(xyz_values)
    return sphere_cloud


def pc_normalize(pc, div=1):
    # Bernardo
    pc /= div
    pc = (pc - pc.min()) / (pc.max() - pc.min())
    # l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def load_point_cloud(path_point_cloud, only_valid_points, centralize, filter_index, normalize, scale):
    cloud, normals = pcl.load(path_point_cloud, only_valid_points=only_valid_points)
    cloud = cloud.to_array()
    print('load_point_cloud(): cloud =', cloud)
    print('original cloud.shape:', cloud.shape)
    # cloud -= np.array([0., 0., -100.], dtype=np.float32)
    # cloud /= 100
    if centralize:
        cloud = cloud - np.mean(cloud, 0)

    if normalize:
        cloud = pc_normalize(cloud, div=scale)

    if not normals is None:
        # normals = np.asarray(normals)
        normals = normals.to_array()
        if filter_index > 0 and filter_index < normals.shape[0]:
            normals = normals[0:filter_index,:]

        normals = pcl.PointCloud_Normal(normals)
    
    if filter_index > 0 and filter_index < cloud.shape[0]:
        cloud = cloud[0:filter_index,:]
        print('filtered index cloud.shape:', cloud.shape)
    cloud = pcl.PointCloud(cloud)

    # ptcloud_centred = pcl.PointCloud_PointXYZRGB()
    return cloud, normals


def get_normals_via_integral_image(cloud, radius=30):
    # cloud = pcl.PointCloud(cloud.to_array() / 100)
    ne = cloud.make_IntegralImageNormalEstimation()

    ne.set_NormalEstimation_Method_AVERAGE_3D_GRADIENT()
    # ne.set_NormalEstimation_Method_COVARIANCE_MATRIX()
    # ne.set_NormalEstimation_Method_AVERAGE_DEPTH_CHANGE()
    # ne.set_NormalEstimation_Method_SIMPLE_3D_GRADIENT()

    ne.set_MaxDepthChange_Factor(0.02)
    ne.set_NormalSmoothingSize(1.0)
    normals = ne.compute()
    return normals


def get_normals(cloud, k_neighbours=30, radius=30):
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

    print('normals.to_array():', normals.shape)
    print('normals (x, y, z) + curvature:', normals[0])
    
    # normals[:,0:3] = normals[:,0:3] * -1    # invert only normal vectors
    # normals[:,3] = normals[:,3] * -1        # invert only curvature
    # normals *= -1                           # invert all components (normals and curvature)

    normals = pcl.PointCloud_Normal(normals)
    return normals                # original
    # return normals.to_array()   # BERNARDO


def filter_points_by_radius(cloud, keypoint_ref, radius=90.0):
    keypoint_ref = np.expand_dims(np.asarray(keypoint_ref, dtype=np.float32), axis=0)
    # print('keypoint_ref:', keypoint_ref)
    searchPoint = pcl.PointCloud(keypoint_ref)
    # print('searchPoint:', searchPoint[0])
    kdtree = pcl.KdTreeFLANN(cloud)
    [ind, sqdist] = kdtree.radius_search_for_cloud(searchPoint, radius, cloud.size)
    # [ind, sqdist] = kdtree.nearest_k_search_for_cloud(searchPoint, 5)
    ind = ind[0]
    ind = ind[ind != 0]
    cloud = pcl.PointCloud(cloud.to_array()[ind])
    return cloud


def init_pcl_viewer(args):
    viewer = pcl.pcl_visualization.PCLVisualizering()
    # viewer.SetBackgroundColor(0, 0, 0)
    viewer.SetBackgroundColor(255, 255, 255)
    
    if args.coord_system_size > 0:
        viewer.AddCoordinateSystem(args.coord_system_size)
        # viewer.InitCameraParameters()
    
    if args.sphere_radius > 0:
        sphere_cloud = generate_random_sphere_point_cloud(n_points=2000, radius=args.sphere_radius)
        pccolor_sphere = pcl.pcl_visualization.PointCloudColorHandleringCustom(sphere_cloud, 0, 0, 255)
        viewer.AddPointCloud_ColorHandler(sphere_cloud, pccolor_sphere, b'cloud_sphere', 0)
        viewer.SetPointCloudRenderingProperties(pcl.pcl_visualization.PCLVISUALIZER_POINT_SIZE, 3, b'cloud_sphere')

    return viewer


def show_point_cloud(args, cloud):
    viewer = init_pcl_viewer(args)

    # add point cloud
    # pccolor1 = pcl.pcl_visualization.PointCloudColorHandleringCustom(cloud, 255, 0, 0)
    pccolor1 = pcl.pcl_visualization.PointCloudColorHandleringCustom(cloud, 50, 50, 50)
    viewer.AddPointCloud_ColorHandler(cloud, pccolor1, b'cloud', 0)
    viewer.SetPointCloudRenderingProperties(pcl.pcl_visualization.PCLVISUALIZER_POINT_SIZE, args.points_size, b'cloud')

    viewer.Spin()



def show_point_cloud_with_normals(args, cloud, normals):
    viewer = init_pcl_viewer(args)
    
    # add point cloud
    # pccolor1 = pcl.pcl_visualization.PointCloudColorHandleringCustom(cloud, 255, 0, 0)
    pccolor1 = pcl.pcl_visualization.PointCloudColorHandleringCustom(cloud, 50, 50, 50)
    viewer.AddPointCloud_ColorHandler(cloud, pccolor1, b'cloud', 0)
    viewer.SetPointCloudRenderingProperties(pcl.pcl_visualization.PCLVISUALIZER_POINT_SIZE, 3, b'cloud')

    # add normals
    # viewer.AddPointCloudNormals(cloud, normals, 1, 10, b'normals')
    viewer.AddPointCloudNormals(cloud, normals, 1, args.normals_size, b'normals')

    viewer.Spin()



def main(args):
    print('Loading point cloud:', args.input_path, '...')
    ptcloud, normals = load_point_cloud(args.input_path, args.only_valid_points, args.centralize, args.filter_index, args.normalize, args.scale)

    if args.filter_radius > 0:
        ptcloud = filter_points_by_radius(ptcloud, [0., 0., 0.], radius=args.filter_radius)
        print('filtered radius ptcloud.size:', str((ptcloud.size,3)))
    
    if args.normals_size > 0:
        if normals is None:
            # k_neighbours = 1
            # k_neighbours = 2
            # k_neighbours = 2
            # k_neighbours = 3
            # k_neighbours = 5
            # k_neighbours = 10
            # k_neighbours = 20
            k_neighbours = 30
            # k_neighbours = 50
            # k_neighbours = 100
            # k_neighbours = 200
            print('Computing normals   k_neighbours:', k_neighbours, '...')
            normals = get_normals(ptcloud, k_neighbours)
            # normals = get_normals_via_integral_image(ptcloud, radius=30)

        print('Showing point cloud with normals...')
        show_point_cloud_with_normals(args, ptcloud, normals)

    else:
        print('Showing point cloud...')
        show_point_cloud(args, ptcloud)



if __name__ == '__main__':

    if not '-input_path' in sys.argv:
        # sys.argv += ['-input_path', '/home/bjgbiesseck/GitHub/MICA/demo/output/lfw/Aaron_Eckhart/Aaron_Eckhart_0001/mesh.obj']
        # sys.argv += ['-input_path', '/home/bjgbiesseck/GitHub/MICA/demo/output/lfw/Aaron_Eckhart/Aaron_Eckhart_0001/mesh.ply']
        # sys.argv += ['-input_path', '/home/bjgbiesseck/GitHub/Meta-PU_biesseck/model/new/result/output_TESTEcarell/mesh.xyz']

        # sys.argv += ['-input_path', '/home/bjgbiesseck/datasets/FRGCv2.0/FRGC-2.0-dist/nd1/Fall2003range/02463d550.abs']
        # sys.argv += ['-input_path', '/home/bjgbiesseck/datasets/FRGCv2.0/FRGC-2.0-dist/nd1/Fall2003range/02463d558.abs']

        # sys.argv += ['-input_path', '/home/bjgbiesseck/datasets/FRGCv2.0/FRGC-2.0-dist/nd1/Fall2003range/02463d562.abs.gz']
        # sys.argv += ['-input_path', '/home/bjgbiesseck/datasets/FRGCv2.0/FRGC-2.0-dist/nd1/Fall2003range/04226d357.abs.gz']

        # sys.argv += ['-input_path', '/home/bjgbiesseck/GitHub/3DFacePointCloudNet/Data/TrainData/400000000/000.bc']
        # sys.argv += ['-input_path', '/home/bjgbiesseck/GitHub/3DFacePointCloudNet/Data/TrainData/400000005/000.bc']

        # sys.argv += ['-input_path', '/home/bjgbiesseck/datasets/modelnet40_normal_resampled/airplane/airplane_0001.txt']
        # sys.argv += ['-input_path', '/home/bjgbiesseck/datasets/modelnet40_normal_resampled/airplane/airplane_0015.txt']
        # sys.argv += ['-input_path', '/home/bjgbiesseck/datasets/modelnet40_normal_resampled/person/person_0008.txt']

        # sys.argv += ['-input_path', '/home/bjgbiesseck/datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal/02691156/1a04e3eab45ca15dd86060f189eb133.txt']
        # sys.argv += ['-input_path', '/home/bjgbiesseck/datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal/03467517/1ae3b398cea3823b49c212147ab9c105.txt']

        sys.argv += ['-input_path', '/home/bjgbiesseck/datasets/FRGCv2.0/FRGC-2.0-dist/nd1/Fall2003range/02463d562_centralized-nosetip_with-normals_filter-radius=90.npy']


    args = parse_args()
    
    main(args)
