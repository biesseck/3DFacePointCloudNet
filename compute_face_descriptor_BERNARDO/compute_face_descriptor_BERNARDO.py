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


# BERNARDO
def preprocess_pointcloud_with_normals(pc_with_normals):
    point_set = pc_with_normals
    # normlize
    point_set[:, 0:3] = (point_set[:, 0:3]) / 100
    point_set = torch.from_numpy(point_set)
    point_set[:, 6] = torch.pow(point_set[:, 6], 0.1)

    input = point_set
    input = input.unsqueeze(0).contiguous()
    input = input.to("cuda", non_blocking=True)
    return input


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
    print('compute_face_descriptor_BERNARDO.py: main(): Loading face cloud', args.cloud_path, '...')
    cloud = pcl.load(args.cloud_path)
    print('cloud.to_array()[0:3]:', cloud.to_array()[0:3])
    print('compute_face_descriptor_BERNARDO.py: main(): Computing face cloud normals...')
    # normals = get_normals(cloud)
    pointcloud_with_normals = get_pointcloud_with_normals(cloud)
    # data_raw = pointcloud_with_normals

    print('compute_face_descriptor_BERNARDO.py: main(): Pre-processing normals...')
    input = preprocess_pointcloud_with_normals(pointcloud_with_normals)


    # BASED ON FUNCTION train_triplet.py: validate()
    print('compute_face_descriptor_BERNARDO.py: main(): Making Pointnet model descriptor...')
    model = build_Pointnet_model(args)



    print('compute_face_descriptor_BERNARDO.py: main(): Computing 3D face descriptor...')
    p_feature = torch.zeros((1, 1, 512))  # BERNARDO

    # feat = model(input)  # 1x512
    feat = model.forward(input)  # 1x512
    print('feat[:,0:3] - total is 512:\n', feat[:,0:21])
    p_feature[:, :, :] = feat.cpu()  # 1x1x512
    # print('p_feature:', p_feature[:, :, 0:100])
    p_feature_norm = torch.norm(p_feature, p=2, dim=2)
    print('p_feature_norm:', p_feature_norm)

    print('\nFinished!\n')



if __name__ == '__main__':
    # sys.argv += ['-epochs', '100']
    # print('__main__(): sys.argv=', sys.argv)

    sys.argv += ['-model_checkpoint', 'checkpoints/20191028_1000cls_model_best.pth.tar']
    # sys.argv += ['-cls_checkpoint', 'checkpoints/20191028_1000cls_model_best.pth.tar']
    
    sys.argv += ['-cloud_path', '/home/bjgbiesseck/GitHub/MICA/demo/output/lfw/Aaron_Eckhart/Aaron_Eckhart_0001/mesh.obj']
    # sys.argv += ['-cloud_path', '/home/bjgbiesseck/GitHub/MICA/demo/output/lfw/Aaron_Eckhart/Aaron_Eckhart_0001/mesh.ply']
    # sys.argv += ['-cloud_path', '/home/bjgbiesseck/GitHub/MICA/demo/output_TESTE/carell/mesh.obj']
    # sys.argv += ['-cloud_path', '/home/bjgbiesseck/GitHub/MICA/demo/output_TESTE/carell/mesh.ply']
    # sys.argv += ['-cloud_path', '/home/bjgbiesseck/GitHub/MICA/demo/output/lfw/Abba_Eban/Abba_Eban_0001/mesh.obj']
    # sys.argv += ['-cloud_path', '/home/bjgbiesseck/GitHub/MICA/demo/output/lfw/Abba_Eban/Abba_Eban_0001/mesh.ply']

    args = parse_args()
    # print('__main__(): args=', args)

    main(args)
