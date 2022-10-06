import numpy as np
import pcl
import pcl.pcl_visualization

def load_point_cloud(path_point_cloud):
    cloud = pcl.load(path_point_cloud)
    # cloud = np.asarray(cloud)
    cloud = cloud - np.mean(cloud, 0)
    ptcloud_centred = pcl.PointCloud()
    # ptcloud_centred = pcl.PointCloud_PointXYZRGB()
    ptcloud_centred.from_array(cloud)
    return ptcloud_centred

'''
def load_point_cloud(path_point_cloud):
    cloud = pcl.load(path_point_cloud)
    cloud = cloud - np.mean(cloud, 0)

    color = np.zeros((cloud.shape[0], 1), dtype='float32')
    cloud = np.hstack((cloud, color))

    # ptcloud_centred = pcl.PointCloud()
    ptcloud_centred = pcl.PointCloud_PointXYZRGB()
    ptcloud_centred.from_array(cloud)
    return ptcloud_centred
'''


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


def show_point_cloud(ptcloud_centred):
    visual = pcl.pcl_visualization.CloudViewing()

    # PointXYZ
    visual.ShowMonochromeCloud(ptcloud_centred, b'cloud')
    # visual.ShowGrayCloud(ptcloud_centred, b'cloud')
    # visual.ShowColorCloud(ptcloud_centred, b'cloud')
    # visual.ShowColorACloud(ptcloud_centred, b'cloud')

    v = True
    while v:
        v = not (visual.WasStopped())


def show_point_cloud_with_normals(cloud, normals):

    # visualize normals
    viewer = pcl.pcl_visualization.PCLVisualizering()
    viewer.SetBackgroundColor(0, 0, 0)
    viewer.AddPointCloud(cloud, b'cloud')

    # viewer.AddPointCloudNormals(cloud, normals, 10, 0.05, b'normals')
    viewer.AddPointCloudNormals(cloud, normals, 1, 10, b'normals')
    viewer.Spin()



def main(path_point_cloud: str):
    print('Loading point cloud:', path_point_cloud, '...')
    ptcloud_centred = load_point_cloud(path_point_cloud)

    print('Computing normals...')
    # radius_search = 3   # 3 mm
    # radius_search = 5   # 5 mm
    # radius_search = 10  # 1 cm
    # radius_search = 20  # 2 cm
    radius_search = 50  # 5 cm
    # radius_search = 100  # 10 cm
    normals = get_normals(ptcloud_centred, radius_search)

    print('Showing point cloud...')
    # show_point_cloud(ptcloud_centred)
    show_point_cloud_with_normals(ptcloud_centred, normals)


if __name__ == '__main__':

    # path_point_cloud = '/home/bjgbiesseck_home_duo/GitHub/MICA/demo/output/lfw/Aaron_Eckhart/Aaron_Eckhart_0001/mesh.obj'
    # path_point_cloud = '/home/bjgbiesseck_home_duo/GitHub/MICA/demo/output/lfw/Aaron_Eckhart/Aaron_Eckhart_0001/mesh.obj'
    path_point_cloud = '/home/bjgbiesseck_home_duo/GitHub/Meta-PU_biesseck/model/new/result/output_TESTEcarell/mesh.xyz'

    main(path_point_cloud)
