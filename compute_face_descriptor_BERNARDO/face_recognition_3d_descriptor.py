import sys
import numpy as np
import argparse
import os
import glob

# from pointnet2.train.train_triplet import *
import pcl
import torch

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


def load_3D_descriptors_from_disk(dataset_path: str, dataset_name: str, sujects_names: [], samples_per_subject: [], desc_file_ext: str):
    total_samples_dataset = sum([len(samples) for samples in samples_per_subject])
    # print('total_samples_dataset:', total_samples_dataset)
    # descriptors_3D = np.zeros((total_samples_dataset, 512), dtype=float)
    descriptors_3D = torch.zeros((total_samples_dataset, 512), dtype=float)
    labels_gt = [''] * total_samples_dataset

    i = 0
    for j in range(len(sujects_names)):
        for k in range(len(samples_per_subject[j])):
            path_descriptor_to_find = os.path.join(dataset_path, dataset_name, sujects_names[j], samples_per_subject[j][k], desc_file_ext)
            # print('path_descriptor:', path_descriptor)
            path_found_descriptor = glob.glob(path_descriptor_to_find)[0]
            # print('path_descriptor:', path_descriptor)
            # print('Loading 3D face descriptor:', path_found_descriptor, '...   sujects_names[j]:', sujects_names[j])
            one_3D_descriptor = torch.load(path_found_descriptor)
            # print('loaded!   shape:', one_3D_descriptor.shape)
            # input('Paused... press ENTER')
            descriptors_3D[i] = one_3D_descriptor
            labels_gt[i] = sujects_names[j]
            i += 1
    return descriptors_3D, labels_gt


def load_3D_descriptors_with_labels(args):

    datasets = {}
    for dataset_name in args.datasets_names:
        dataset_sujects, dataset_samples_names_per_subject \
            = FileTreeLfwDatasets3dReconstructed().get_subjects_and_samples_names(args.datasets_path, dataset_name, 'original')
        # for subject, samples_list_name in zip(dataset_sujects, dataset_samples_names_per_subject):
        #     print(dataset_name, ':', subject, ':', samples_list_name)
        # input('Paused... press ENTER')
        # print(dataset_name, '  len(dataset_sujects) before:', len(dataset_sujects), '  len(dataset_samples_names_per_subject) before:', len(dataset_samples_names_per_subject))

        filter_dataset_subjects_and_samples(dataset_sujects, dataset_samples_names_per_subject, criterias=['min_samples', 2])
        # for subject, samples_list_name in zip(dataset_sujects, dataset_samples_names_per_subject):
        #     print(dataset_name, ':', subject, ':', samples_list_name)
        # input('Paused... press ENTER')
        # print(dataset_name, '  len(dataset_sujects) after:', len(dataset_sujects), '  len(dataset_samples_names_per_subject) after:', len(dataset_samples_names_per_subject))

        dataset_descriptors_3D, dataset_labels_gt = \
            load_3D_descriptors_from_disk(args.datasets_path, dataset_name, dataset_sujects, dataset_samples_names_per_subject, args.desc_file_ext)

        datasets[(dataset_name, 'sujects')] = dataset_sujects
        datasets[(dataset_name, 'samples_names')] = dataset_samples_names_per_subject
        datasets[(dataset_name, 'descriptors_3D')] = dataset_descriptors_3D
        datasets[(dataset_name, 'labels_gt')] = dataset_labels_gt
    return datasets


# TODO
def do_face_verification_one_dataset(descriptors_3D, labels_gt):
    descriptors_3D = descriptors_3D.unsqueeze(0)
    tp, fp = 0, 0

    # # TESTE
    # n = 20
    # descriptors_3D = descriptors_3D[:, 0:n, :]  # TESTE BERNARDO
    # labels_gt = labels_gt[0:n]            # TESTE BERNARDO
    # print('descriptors_3D.shape:', descriptors_3D.shape, '    len(labels_gt):', len(labels_gt))
    # print('descriptors_3D[:,0,:].shape:', descriptors_3D[:,0,:].shape)
    # for i in range(len(labels_gt)):
    #     print('labels_gt[i]:', labels_gt[i])
    # print('-------------------------------------')
    # # TESTE

    norms_descriptors_3D = torch.norm(descriptors_3D, p=2, dim=2)
    # print('norms_descriptors_3D:', norms_descriptors_3D)

    # # TESTE
    # i = 0
    # dis = torch.sum(torch.mul(descriptors_3D[:,i,:], descriptors_3D.transpose(1, 0)), dim=2) / norms_descriptors_3D[:,i] / norms_descriptors_3D.transpose(1, 0)
    # print('dis:', dis, '    dis.min:', dis.min())
    # # TESTE

    for i in range(descriptors_3D.shape[1]):   # total num samples
        dis = torch.sum(torch.mul(descriptors_3D[:,i,:], descriptors_3D.transpose(1, 0)), dim=2) / norms_descriptors_3D[:,i] / norms_descriptors_3D.transpose(1, 0)
        # top1 = np.equal(torch.argmax(dis, dim=0).numpy(), np.argmax(Label, axis=0)).sum() / len(pfile_list)
        dis[i] = 0      # ignores itself

        # print('dis.transpose(1, 0):', dis.transpose(1, 0))
        # # print()
        # print('torch.argmax(dis, dim=0):', torch.argmax(dis, dim=0)[0])
        # # input('Paused...')

        gt_index = i
        pr_index = torch.argmax(dis, dim=0)[0]
        # print('labels_gt['+str(gt_index)+']:', labels_gt[gt_index], '   labels_gt['+str(pr_index)+']:', labels_gt[pr_index], '   dist:', dis[pr_index].detach().numpy())

        if labels_gt[pr_index] == labels_gt[gt_index]:
            tp += 1
        else:
            fp += 1

    print('tp:', tp, '    fp:', fp)


def main_verification(args):
    # LOAD DATASETS (LFW and TALFW)
    print('Loading datasets (3D face descriptor)...')
    datasets = load_3D_descriptors_with_labels(args)

    if len(args.datasets_names) < 2:
        dataset_sujects = datasets[(args.datasets_names[0], 'sujects')]
        dataset_samples_names_per_subject = datasets[(args.datasets_names[0], 'samples_names')]
        dataset_descriptors_3D = datasets[(args.datasets_names[0], 'descriptors_3D')]
        dataset_labels_gt = datasets[(args.datasets_names[0], 'labels_gt')]
        print('    sujects:', len(dataset_sujects), '   samples:', len(dataset_labels_gt))

        print('Doing face verification...')
        results = do_face_verification_one_dataset(dataset_descriptors_3D, dataset_labels_gt)

    else:
        raise Exception('Face verification for multiple datasets not implemented yet!')


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

    sys.argv += ['-datasets_names', ['lfw']]
    # sys.argv += ['-datasets_names', ['lfw', 'TALFW']]

    sys.argv += ['-desc_file_ext', '*from_OBJ.pt']
    # sys.argv += ['-desc_file_ext', '*from_PLY.pt']


    args = parse_args()
    # print('__main__(): args=', args)

    main_verification(args)
    # main_teste()
