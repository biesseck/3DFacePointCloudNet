from math import sqrt
import numpy as np
import pcl
import pcl.pcl_visualization
import argparse



def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for visualizing point cloud with normals",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-show_normals", type=str, default='Y', help="Whether or not to compute and show normals with point cloud")
    parser.add_argument("-normals_size", type=int, default=10,  help="Normals vectors size")
    parser.add_argument("-input_path", type=str, default='/home/bjgbiesseck_home_duo/GitHub/MICA/demo/output/lfw/Aaron_Eckhart/Aaron_Eckhart_0001/mesh.obj',  help="Input file path")
    parser.add_argument("-points_size", type=int, default=3,  help="Size of points to show")
    parser.add_argument("-sphere_radius", type=int, default=100,  help="Radius of a sphere for comparison")
    return parser.parse_args()


def generate_random_sphere_point_cloud(n_points=1000, radius=1.0):
    xyz_values = np.random.uniform(low=-10.0, high=10.0, size=(n_points, 3)).astype(dtype=np.float32)
    for i in range(n_points):
        xyz_values[i] /= sqrt(xyz_values[i,0]**2 + xyz_values[i,1]**2 + xyz_values[i,2]**2)
        xyz_values[i] *= radius
    sphere_cloud = pcl.PointCloud(xyz_values)
    return sphere_cloud


def load_point_cloud(path_point_cloud):
    cloud = pcl.load(path_point_cloud)
    cloud = np.asarray(cloud)
    
    # cloud
    # cloud = cloud - np.mean(cloud, 0)
    cloud = pcl.PointCloud(cloud)
    # ptcloud_centred = pcl.PointCloud_PointXYZRGB()
    return cloud


def get_normals(cloud, radius=30):
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
    feature = cloud.make_NormalEstimation()
    feature.set_KSearch(radius)    # Use all neighbors in a sphere of radius 5 cm
    normals = feature.compute()
    normals = normals.to_array()

    print('normals (x, y, z) + curvature:', normals[0])
    print('normals.to_array():', normals.shape)

    # normals[:,0:3] = normals[:,0:3] * -1    # invert only normal vectors
    # normals[:,3] = normals[:,3] * -1        # invert only curvature
    # normals *= -1                           # invert all components (normals and curvature)

    normals = pcl.PointCloud_Normal(normals)
    return normals                # original
    # return normals.to_array()   # BERNARDO


def init_pcl_viewer(args):
    viewer = pcl.pcl_visualization.PCLVisualizering()
    viewer.SetBackgroundColor(0, 0, 0)
    viewer.AddCoordinateSystem(100.0)
    viewer.InitCameraParameters()
    # viewer.addSphere((0, 0, 0), 1, 0.5, 0.5, 0.0, b'sphere')
    # viewer.AddSphere()

    if args.sphere_radius > 0:
        sphere_cloud = generate_random_sphere_point_cloud(n_points=2000, radius=args.sphere_radius)
        pccolor_sphere = pcl.pcl_visualization.PointCloudColorHandleringCustom(sphere_cloud, 0, 0, 255)
        viewer.AddPointCloud_ColorHandler(sphere_cloud, pccolor_sphere, b'cloud_sphere', 0)
        viewer.SetPointCloudRenderingProperties(pcl.pcl_visualization.PCLVISUALIZER_POINT_SIZE, 3, b'cloud_sphere')

    return viewer


def show_point_cloud(args, cloud):
    viewer = init_pcl_viewer(args)

    # add point cloud
    pccolor1 = pcl.pcl_visualization.PointCloudColorHandleringCustom(cloud, 255, 0, 0)
    viewer.AddPointCloud_ColorHandler(cloud, pccolor1, b'cloud', 0)
    viewer.SetPointCloudRenderingProperties(pcl.pcl_visualization.PCLVISUALIZER_POINT_SIZE, 3, b'cloud')

    viewer.Spin()



def show_point_cloud_with_normals(args, cloud, normals):
    viewer = init_pcl_viewer(args)
    
    # add point cloud
    pccolor1 = pcl.pcl_visualization.PointCloudColorHandleringCustom(cloud, 255, 0, 0)
    viewer.AddPointCloud_ColorHandler(cloud, pccolor1, b'cloud', 0)
    viewer.SetPointCloudRenderingProperties(pcl.pcl_visualization.PCLVISUALIZER_POINT_SIZE, 3, b'cloud')

    # add normals
    viewer.AddPointCloudNormals(cloud, normals, 1, 10, b'normals')

    viewer.Spin()



def main(args, path_point_cloud: str):
    print('Loading point cloud:', path_point_cloud, '...')
    ptcloud = load_point_cloud(path_point_cloud)

    if args.show_normals.upper() == 'Y':
        # radius_search = 1   # 1 mm
        # radius_search = 2   # 2 mm
        # radius_search = 2.5   # 2.5 mm
        # radius_search = 3   # 3 mm
        # radius_search = 5   # 5 mm
        # radius_search = 10  # 1 cm
        # radius_search = 20  # 2 cm
        # radius_search = 30  # 3 cm
        # radius_search = 50  # 5 cm
        # radius_search = 100  # 10 cm
        radius_search = 200  # 20 cm
        print('Computing normals   radius_search:', radius_search, 'mm...')
        normals = get_normals(ptcloud, radius_search)

        print('Showing point cloud with normals...')
        show_point_cloud_with_normals(args, ptcloud, normals)

    else:
        print('Showing point cloud...')
        show_point_cloud(args, ptcloud)



if __name__ == '__main__':
    # path_point_cloud = '/home/bjgbiesseck_home_duo/GitHub/MICA/demo/output/lfw/Aaron_Eckhart/Aaron_Eckhart_0001/mesh.obj'
    # path_point_cloud = '/home/bjgbiesseck_home_duo/GitHub/MICA/demo/output/lfw/Aaron_Eckhart/Aaron_Eckhart_0001/mesh.ply'
    # path_point_cloud = '/home/bjgbiesseck_home_duo/GitHub/Meta-PU_biesseck/model/new/result/output_TESTEcarell/mesh.xyz'
    # path_point_cloud = '/home/bjgbiesseck_home_duo/datasets/FRGCv2.0/FRGC-2.0-dist/nd1/Fall2003range/02463d550.abs'
    # path_point_cloud = '/home/bjgbiesseck_home_duo/datasets/FRGCv2.0/FRGC-2.0-dist/nd1/Fall2003range/02463d558.abs'
    # path_point_cloud = '/home/bjgbiesseck_home_duo/datasets/FRGCv2.0/FRGC-2.0-dist/nd1/Fall2003range/02463d562.abs.gz'
    # path_point_cloud = '/home/bjgbiesseck_home_duo/datasets/FRGCv2.0/FRGC-2.0-dist/nd1/Fall2003range/04226d357.abs.gz'
    # path_point_cloud = '/home/bjgbiesseck_home_duo/GitHub/3DFacePointCloudNet/Data/TrainData/400000000/000.bc'
    path_point_cloud = '/home/bjgbiesseck_duo/GitHub/3DFacePointCloudNet/Data/TrainData/400000000/000.bc'


    args = parse_args()

    path_point_cloud = args.input_path

    main(args, path_point_cloud)
