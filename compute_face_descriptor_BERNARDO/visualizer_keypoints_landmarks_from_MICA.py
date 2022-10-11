import numpy as np
import pcl
import pcl.pcl_visualization

from pcl import NormalEstimation

def load_point_cloud(path_point_cloud):
    cloud = pcl.load(path_point_cloud)
    cloud = np.asarray(cloud)
    # cloud = cloud - np.mean(cloud, 0)
    ptcloud_centred = pcl.PointCloud()
    # ptcloud_centred = pcl.PointCloud_PointXYZRGB()
    ptcloud_centred.from_array(cloud)
    return ptcloud_centred


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

    return normals            # original
    # return normals.to_array()   # BERNARDO


def show_point_cloud_with_keypoints(ptcloud_centred, key_points):
    # visualize normals
    viewer = pcl.pcl_visualization.PCLVisualizering()
    viewer.SetBackgroundColor(0, 0, 0)
    # viewer.SetBackgroundColor(255, 255, 255)

    # viewer.AddPointCloud(ptcloud_centred, b'cloud')
    # viewer.AddPointCloud(key_points, b'cloud')

    pccolor1 = pcl.pcl_visualization.PointCloudColorHandleringCustom(ptcloud_centred, 255, 0, 0)
    viewer.AddPointCloud_ColorHandler(ptcloud_centred, pccolor1, b'cloud', 0)
    viewer.SetPointCloudRenderingProperties(pcl.pcl_visualization.PCLVISUALIZER_POINT_SIZE, 2, b'cloud')

    pccolor2 = pcl.pcl_visualization.PointCloudColorHandleringCustom(ptcloud_centred, 0, 255, 0)
    viewer.AddPointCloud_ColorHandler(key_points, pccolor2, b'keypoints', 0)
    viewer.SetPointCloudRenderingProperties(pcl.pcl_visualization.PCLVISUALIZER_POINT_SIZE, 3, b'keypoints')

    viewer.Spin()


def show_point_cloud_with_keypoints_and_normals(ptcloud_centred, key_points, normals):
    # visualize normals
    viewer = pcl.pcl_visualization.PCLVisualizering()
    viewer.SetBackgroundColor(0, 0, 0)
    # viewer.SetBackgroundColor(255, 255, 255)

    # viewer.AddPointCloud(ptcloud_centred, b'cloud')
    # viewer.AddPointCloud(key_points, b'cloud')

    pccolor1 = pcl.pcl_visualization.PointCloudColorHandleringCustom(ptcloud_centred, 255, 0, 0)
    viewer.AddPointCloud_ColorHandler(ptcloud_centred, pccolor1, b'cloud', 0)
    viewer.SetPointCloudRenderingProperties(pcl.pcl_visualization.PCLVISUALIZER_POINT_SIZE, 2, b'cloud')

    pccolor2 = pcl.pcl_visualization.PointCloudColorHandleringCustom(ptcloud_centred, 0, 255, 0)
    viewer.AddPointCloud_ColorHandler(key_points, pccolor2, b'keypoints', 0)
    viewer.SetPointCloudRenderingProperties(pcl.pcl_visualization.PCLVISUALIZER_POINT_SIZE, 3, b'keypoints')

    viewer.AddPointCloudNormals(ptcloud_centred, normals, 1, 10, b'normals')

    viewer.Spin()



def show_point_cloud_with_normals(cloud, normals):

    # visualize normals
    viewer = pcl.pcl_visualization.PCLVisualizering()
    viewer.SetBackgroundColor(0, 0, 0)
    viewer.AddPointCloud(cloud, b'cloud')

    # viewer.AddPointCloudNormals(cloud, normals, 10, 0.05, b'normals')
    viewer.AddPointCloudNormals(cloud, normals, 1, 10, b'normals')
    viewer.Spin()



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



def main(path_point_cloud: str, path_key_points: str):
    print('Loading point cloud:', path_point_cloud, '...')
    ptcloud_centred = load_point_cloud(path_point_cloud)

    print('Loading keypoints:', path_key_points, '...')
    key_points = load_point_cloud(path_key_points)

    # ptcloud_centred = filter_points_by_radius(ptcloud_centred, key_points[30], radius=90.0)  # 90 mm from nose tip
    ptcloud_centred = filter_points_by_radius(ptcloud_centred, key_points[30], radius=120.0)  # 120 mm from nose tip
    # ptcloud_centred = filter_points_by_radius(ptcloud_centred, key_points[30], radius=150.0)  # 150 mm from nose tip

    print('Computing normals...')
    # radius_search = 3   # 3 mm
    # radius_search = 5   # 5 mm
    # radius_search = 10  # 1 cm
    radius_search = 20  # 2 cm
    # radius_search = 50  # 5 cm
    # radius_search = 100  # 10 cm
    normals = get_normals(ptcloud_centred, radius_search)

    print('Showing point cloud...')
    # show_point_cloud_with_keypoints(ptcloud_centred, key_points)
    show_point_cloud_with_keypoints_and_normals(ptcloud_centred, key_points, normals)




if __name__ == '__main__':
    path_point_cloud = None
    path_point_cloud = '/home/bjgbiesseck_home_duo/GitHub/MICA/demo/output/lfw/Aaron_Eckhart/Aaron_Eckhart_0001/mesh.obj'
    # path_point_cloud = '/home/bjgbiesseck_home_duo/GitHub/MICA/demo/output/lfw/Aaron_Eckhart/Aaron_Eckhart_0001/mesh.ply'
    # path_point_cloud = '/home/bjgbiesseck_home_duo/GitHub/MICA/demo/output/lfw/Adam_Sandler/Adam_Sandler_0001/mesh.obj'
    # path_point_cloud = '/home/bjgbiesseck_home_duo/GitHub/MICA/demo/output/lfw/Aaron_Eckhart/Aaron_Eckhart_0001/mesh_upsample_MetaPU.xyz'

    path_key_points = None
    path_key_points = '/home/bjgbiesseck_home_duo/GitHub/MICA/demo/output/lfw/Aaron_Eckhart/Aaron_Eckhart_0001/kpt68.npy'
    # path_key_points = '/home/bjgbiesseck_home_duo/GitHub/MICA/demo/output/lfw/Aaron_Eckhart/Aaron_Eckhart_0001/kpt7.npy'
    # path_key_points = '/home/bjgbiesseck_home_duo/GitHub/MICA/demo/output/lfw/Adam_Sandler/Adam_Sandler_0001/kpt68.npy'

    main(path_point_cloud, path_key_points)
