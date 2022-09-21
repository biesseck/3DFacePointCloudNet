import sys
import numpy as np
import argparse
import os

# from pointnet2.train.train_triplet import *
import pcl

from file_tree import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for cls training",
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
        "-datasets_path", type=str, default='',
        help="Path of dataset root folder containing 3D face descriptors (.pt format)"
    )
    parser.add_argument(
        "-datasets_names", type=list, default='', help="Datasets names"
    )
    parser.add_argument(
        "-desc_file_ext", type=str, default='_OBJ.pt', help="Extension or final part of descriptor file"
    )

    return parser.parse_args()


def filter_dataset_subjects_and_samples(sujects_names: [], samples_per_subjects: [], criterias: []):
    def remove_subjects_by_min_samples(min_samples: int, sujects_names: [], samples_per_subjects: []):
        # num_samples_per_subject = np.zeros((len(samples_per_subjects),), dtype=int)
        # for i in range(len(samples_per_subjects)):
        for i in range(len(samples_per_subjects)-1, -1, -1):
            if len(samples_per_subjects[i]) < min_samples:
                # print(i, '- subject:', sujects_names[i], '   samples:', samples_per_subjects[i])
                sujects_names.pop(i)
                samples_per_subjects.pop(i)

    criteria = 'min_samples'
    if criteria in criterias:
        value = criterias[criterias.index(criteria) + 1]
        remove_subjects_by_min_samples(value, sujects_names, samples_per_subjects)


def load_3D_descriptors_with_labels(args):

    dataset = {}
    for dataset_name in args.datasets_names:
        dataset_sujects, dataset_samples_per_subject \
            = FileTreeLfwDatasets3dReconstructed().get_subjects_and_samples_names(args.datasets_path, dataset_name, 'original')
        # for subject, samples_list_name in zip(dataset_sujects, dataset_samples_per_subject):
        #     print(dataset_name, ':', subject, ':', samples_list_name)
        # input('Paused... press ENTER')
        # print(dataset_name, '  len(dataset_sujects) before:', len(dataset_sujects), '  len(dataset_samples_per_subject) before:', len(dataset_samples_per_subject))

        filter_dataset_subjects_and_samples(dataset_sujects, dataset_samples_per_subject, criterias=['min_samples', 2])
        # for subject, samples_list_name in zip(dataset_sujects, dataset_samples_per_subject):
        #     print(dataset_name, ':', subject, ':', samples_list_name)
        # input('Paused... press ENTER')
        # print(dataset_name, '  len(dataset_sujects) after:', len(dataset_sujects), '  len(dataset_samples_per_subject) after:', len(dataset_samples_per_subject))
        
        # dataset[dataset_name] =



    # TODO: LOAD 3D DESCRIPTORS FROM DISK

    # TODO: ORGANIZE LOADED DESCRIPTORS


# TODO
def do_face_verification():
    pass


def main_verification(args):
    # LOAD DATASETS (LFW and TALFW)
    descriptors_3D, labels_gt = load_3D_descriptors_with_labels(args)

    # LOAD POINT CLOUD DESCRIPTORS

    # DO MATCHING (1:1)

    pass

def main_teste():
    for i in range(10-1, -1, -1):
        print('main_teste: i=', i)

if __name__ == '__main__':
    # sys.argv += ['-epochs', '100']
    # print('__main__(): sys.argv=', sys.argv)

    sys.argv += ['-model_checkpoint', 'checkpoints/20191028_1000cls_model_best.pth.tar']
    # sys.argv += ['-cls_checkpoint', 'checkpoints/20191028_1000cls_model_best.pth.tar']

    sys.argv += ['-datasets_path', '/home/bjgbiesseck/GitHub/MICA/demo/output']

    sys.argv += ['-datasets_names', ['lfw', 'TALFW']]

    sys.argv += ['-desc_file_ext', '_OBJ.pt']


    args = parse_args()
    # print('__main__(): args=', args)

    main_verification(args)
    # main_teste()
