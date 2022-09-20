import sys
import numpy as np

from pointnet2.train.train_triplet import *
import pcl

def load_dataset():
    pass


def main(args):
    # load dataset (LFW and TALFW)

    # compute point cloud normals

    # do matching (1:1)

    pass

if __name__ == '__main__':
    # sys.argv += ['-epochs', '100']
    # print('__main__(): sys.argv=', sys.argv)

    sys.argv += ['-model_checkpoint', 'checkpoints/20191028_1000cls_model_best.pth.tar']
    # sys.argv += ['-cls_checkpoint', 'checkpoints/20191028_1000cls_model_best.pth.tar']

    sys.argv += ['-dataset_path', '/home/bjgbiesseck/GitHub/MICA/demo/output/lfw']


    args = parse_args()
    # print('__main__(): args=', args)

    main(args)
