import sys
import numpy as np

from pointnet2.train.train_triplet import *
import pcl


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

    # BERNARDO
    # print('normals.to_array()[0:5]:', normals.to_array()[0:5])
    # print('normals[0]:', normals[0])

    # return normals            # original
    return normals.to_array()   # BERNARDO


# BERNARDO
def get_pointcloud_with_normals(cloud):
    # print('cloud:', cloud)
    cloud_with_normals = cloud.to_array()
    # print('cloud_with_normals:', cloud_with_normals)
    normals = get_normals(cloud)
    cloud_with_normals = np.hstack((cloud_with_normals, normals))
    # print('cloud_with_normals.shape:', cloud_with_normals.shape)
    # print('cloud_with_normals:', cloud_with_normals)
    return cloud_with_normals


def main():
    cloud_path = '/home/bjgbiesseck/GitHub/MICA/demo/output_TESTE/carell/mesh.obj'
    # cloud_path = '/home/bjgbiesseck/GitHub/MICA/demo/output_TESTE/carell/mesh.ply'

    print('compute_face_descriptor_BERNARDO.py: main(): Loading face cloud...')
    cloud = pcl.load(cloud_path)
    print('compute_face_descriptor_BERNARDO.py: main(): Computing face cloud normals...')
    # normals = get_normals(cloud)
    pointcloud_with_normals = get_pointcloud_with_normals(cloud)
    data_raw = pointcloud_with_normals

    print('compute_face_descriptor_BERNARDO.py: main(): Pre-processing normals...')
    # n = data_raw.shape[0]
    # data = np.asarray(data_raw, dtype=np.float32).reshape((4 + 3), n // (4 + 3))
    data = data_raw
    # point_set = data.T
    point_set = data
    # normlize
    point_set[:, 0:3] = (point_set[:, 0:3]) / 100
    point_set = torch.from_numpy(point_set)
    point_set[:, 6] = torch.pow(point_set[:, 6], 0.1)



    # BASED ON FUNCTION train_triplet.py: validate()
    print('compute_face_descriptor_BERNARDO.py: main(): Loading Pointnet model descriptor...')
    model = Pointnet(input_channels=3, use_xyz=True)
    model.cuda()
    # 512 is dimension of feature
    classifier = {
        'MCP': layer.MarginCosineProduct(512, args.num_class).cuda(),
        'AL': layer.AngleLinear(512, args.num_class).cuda(),
        'L': torch.nn.Linear(512, args.num_class, bias=False).cuda()
    }[args.classifier_type]

    criterion = nn.TripletMarginLoss(margin=0.5, p=2)
    optimizer = optim.Adam(
        [{'params': model.parameters()}, {'params': classifier.parameters()}],
        lr=lr, weight_decay=args.weight_decay
    )

    model.eval()
    optimizer.zero_grad()

    # g_feature = torch.zeros((len(gfile_list), 1, 512))  # original
    # p_feature = torch.zeros((len(pfile_list), 1, 512))  # original
    g_feature = torch.zeros((1, 1, 512))                  # BERNARDO
    p_feature = torch.zeros((1, 1, 512))                  # BERNARDO

    input = point_set   # BERNARDO
    input = input.unsqueeze(0).contiguous()
    input = input.to("cuda", non_blocking=True)
    # feat = model(input)  # 1x512
    feat = model.forward(input)  # 1x512
    g_feature[:, :, :] = feat.cpu()  # 105x1x512

    '''
    print('Loading file:', fname)
    input = loadBCNFile(fname)
    input = input.unsqueeze(0).contiguous()
    input = input.to("cuda", non_blocking=True)
    feat = model(input)  # 1x512
    g_feature[i, :, :] = feat.cpu()  # 105x1x512
    '''

if __name__ == '__main__':
    # sys.argv += ['-epochs', '100']
    # print('__main__(): sys.argv=', sys.argv)

    args = parse_args()
    # print('__main__(): args=', args)

    main()
