import sys
import numpy as np
import argparse
import os
import glob

# from pointnet2.train.train_triplet import *
import pcl
import torch
import matplotlib as plt
from sklearn.manifold import TSNE

from file_tree import *
from plots_3d_face_descriptors import *


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
    parser.add_argument("-dataset_size", type=str, default='whole', help="whole or subset")

    return parser.parse_args()


def filter_dataset_subjects_and_samples(sujects_names: [], samples_per_subjects: [], criterias: []):
    def remove_subjects_by_min_samples(min_samples: int, sujects_names: [], samples_per_subjects: []):
        # num_samples_per_subject = np.zeros((len(samples_per_subjects),), dtype=int)
        # for i in range(len(samples_per_subjects)):
        for i in range(len(samples_per_subjects) - 1, -1, -1):
            if len(samples_per_subjects[i]) < min_samples:
                # print(i, '- subject:', sujects_names[i], '   samples:', samples_per_subjects[i])
                sujects_names.pop(i)
                samples_per_subjects.pop(i)

    criteria = 'min_samples'
    if criteria in criterias:
        value = criterias[criterias.index(criteria) + 1]
        remove_subjects_by_min_samples(value, sujects_names, samples_per_subjects)


def load_descriptors_from_disk(dataset_path: str, dataset_name: str, sujects_names: [], samples_per_subject: [],
                               desc_file_ext: str):
    total_samples_dataset = sum([len(samples) for samples in samples_per_subject])
    # print('total_samples_dataset:', total_samples_dataset)
    descriptors = np.zeros((total_samples_dataset, 300), dtype=float)
    # descriptors = torch.zeros((total_samples_dataset, 512), dtype=float)
    labels_gt_str = [''] * total_samples_dataset
    labels_gt_int = [0] * total_samples_dataset

    i = 0
    for j in range(len(sujects_names)):
        for k in range(len(samples_per_subject[j])):
            path_descriptor_to_find = os.path.join(dataset_path, dataset_name, sujects_names[j],
                                                   samples_per_subject[j][k], desc_file_ext)
            print('path_descriptor_to_find:', path_descriptor_to_find)
            path_found_descriptor = glob.glob(path_descriptor_to_find)[0]
            # print('path_descriptor:', path_descriptor)
            # print('Loading 3D face descriptor:', path_found_descriptor, '...   sujects_names[j]:', sujects_names[j])
            one_descriptor = np.load(path_found_descriptor)
            # print('loaded!   shape:', one_descriptor.shape)
            # input('Paused... press ENTER')
            descriptors[i] = one_descriptor
            labels_gt_str[i] = sujects_names[j]
            labels_gt_int[i] = j
            i += 1
    return descriptors, labels_gt_str, labels_gt_int


def load_face_descriptors_with_labels(args):
    datasets = {}
    for dataset_name in args.datasets_names:

        if args.dataset_size == 'whole':  # loads whole dataset
            dataset_sujects, dataset_samples_names_per_subject \
                = FileTreeLfwDatasets3dReconstructed().get_subjects_and_samples_names(args.datasets_path, dataset_name,
                                                                                      'original')

        elif args.dataset_size == 'subset':  # loads only a dataset subset for fast tests
            dataset_sujects, dataset_samples_names_per_subject \
                = FileTreeLfwDatasets3dReconstructed().get_subsample_lfw_subjects_and_samples_names(args.datasets_path,
                                                                                                    dataset_name,
                                                                                                    'original')

        filter_dataset_subjects_and_samples(dataset_sujects, dataset_samples_names_per_subject,
                                            criterias=['min_samples', 2])

        dataset_descriptors, dataset_labels_gt_str, dataset_labels_gt_int = \
            load_descriptors_from_disk(args.datasets_path, dataset_name, dataset_sujects,
                                       dataset_samples_names_per_subject, args.desc_file_ext)

        datasets[(dataset_name, 'sujects')] = np.array(dataset_sujects)
        datasets[(dataset_name, 'samples_names')] = dataset_samples_names_per_subject
        datasets[(dataset_name, 'descriptors')] = dataset_descriptors
        datasets[(dataset_name, 'labels_gt_str')] = np.array(dataset_labels_gt_str)
        datasets[(dataset_name, 'labels_gt_int')] = np.array(dataset_labels_gt_int)
    return datasets


def do_face_verification_one_dataset(dataset_name, descriptors_3D, labels_gt_str, labels_gt_int):
    descriptors_3D = descriptors_3D.unsqueeze(0)
    labels_gt = torch.tensor(labels_gt_int)

    # TESTE
    n = 100
    descriptors_3D = descriptors_3D[:, 0:n, :]  # TESTE BERNARDO
    labels_gt = labels_gt[0:n]  # TESTE BERNARDO
    labels_gt_str = labels_gt_str[0:n]
    print('descriptors_3D.shape:', descriptors_3D.shape, '    len(labels_gt):', len(labels_gt))
    print('descriptors_3D[:,0,:].shape:', descriptors_3D[:, 0, :].shape)
    for i in range(len(labels_gt)):
        print('labels_gt[' + str(i) + ']:', labels_gt[i])
    print('-------------------------------------')
    # TESTE

    norms_descriptors_3D = torch.norm(descriptors_3D, p=2, dim=2)
    # print('norms_descriptors_3D:', norms_descriptors_3D)

    # # TESTE
    # i = 0
    # dist = torch.sum(torch.mul(descriptors_3D[:,i,:], descriptors_3D.transpose(1, 0)), dim=2) / norms_descriptors_3D[:,i] / norms_descriptors_3D.transpose(1, 0)
    # print('dist:', dist, '    dist.min:', dist.min())
    # # TESTE

    t_min, t_max, t_interv = 0, 1, 0.01
    # t_min, t_max, t_interv = 0.99, 1, 0.0001
    # thresholds = np.arange(t_min, t_max + t_interv, t_interv)
    thresholds = torch.arange(t_min, t_max + t_interv, t_interv)

    # tp_total = np.zeros((thresholds.shape[0],), dtype=int)
    tp_total = torch.zeros((thresholds.shape[0],), dtype=int)
    fp_total = tp_total.clone()
    tn_total = tp_total.clone()
    fn_total = tp_total.clone()

    for i in range(descriptors_3D.shape[1] - 1):  # the last descriptor has already been compared to all others
        print('sample: ' + str(i) + '/' + str(descriptors_3D.shape[1]), end='\r')
        # dist = torch.sum(torch.mul(descriptors_3D[:,i,:], descriptors_3D.transpose(1, 0)), dim=2) / norms_descriptors_3D[:,i] / norms_descriptors_3D.transpose(1, 0)
        dist = torch.sum(torch.mul(descriptors_3D[:, i, :], descriptors_3D[:, i:, :].transpose(1, 0)),
                         dim=2) / norms_descriptors_3D[:, i] / norms_descriptors_3D[:, i:].transpose(1, 0)
        # print('dist:', dist)

        # print('labels_gt:', labels_gt)
        right_label_indexes = labels_gt[i:] == labels_gt[i]
        # wrong_label_indexes = labels_gt[i:] != labels_gt[i]
        wrong_label_indexes = ~right_label_indexes
        # print('dist.size():', dist.size())
        # print('right_label_indexes.size():', right_label_indexes.size())
        # print('wrong_label_indexes.size():', wrong_label_indexes.size())
        # input('Paused...')

        for t, tresh in enumerate(thresholds):
            tp = torch.sum(dist[right_label_indexes] >= tresh) - 1
            fp = torch.sum(dist[wrong_label_indexes] >= tresh)
            tn = torch.sum(dist[wrong_label_indexes] < tresh)
            fn = torch.sum(dist[right_label_indexes] < tresh)

            tp_total[t] += tp
            fp_total[t] += fp
            tn_total[t] += tn
            fn_total[t] += fn

        print('Saving figure...')
        Plots_3D_Face_Descriptors().plot_distance_one_descriptor_to_all_others(dist.detach().numpy(),
                                                                               labels=labels_gt_str[i:],
                                                                               title=dataset_name + ' - Cosine distance between 3D face descriptors',
                                                                               path_figure=os.path.abspath(
                                                                                   os.getcwd()) + '/cosine_distance.png',
                                                                               save=True)
        sys.exit(0)

    # print('tp_total:', tp_total)
    # print('fp_total:', fp_total)
    # print('tn_total:', tn_total)
    # print('fn_total:', fn_total)

    results = {'tp_total': tp_total, 'fp_total': fp_total, 'tn_total': tn_total, 'fn_total': fn_total}
    return results


def reduce_dimensionality(descriptors, labels_gt_str, labels_gt_int):
    # print('descriptors:', descriptors.shape)
    X_embedded = TSNE(n_components=2, learning_rate='auto', init = 'random', perplexity = 100).fit_transform(descriptors)
    # print('X_embedded:', X_embedded.shape)
    return X_embedded


def main_visualization(args):
    # LOAD DATASETS (LFW and TALFW)
    print('Loading datasets (3D face descriptor):', args.datasets_names)
    datasets = load_face_descriptors_with_labels(args)

    if len(args.datasets_names) < 2:
        dataset_sujects = datasets[(args.datasets_names[0], 'sujects')]
        dataset_samples_names_per_subject = datasets[(args.datasets_names[0], 'samples_names')]
        dataset_descriptors = datasets[(args.datasets_names[0], 'descriptors')]
        dataset_labels_gt_str = datasets[(args.datasets_names[0], 'labels_gt_str')]
        dataset_labels_gt_int = datasets[(args.datasets_names[0], 'labels_gt_int')]
        print('    sujects:', len(dataset_sujects), '   samples:', len(dataset_labels_gt_str))

        # print('Reducing dimensionality with TSNE...')
        # dataset_descriptors_reduced = reduce_dimensionality(dataset_descriptors, dataset_labels_gt_str, dataset_labels_gt_int)
        #
        # Plots_3D_Face_Descriptors().plot_feature_vectors_into_feature_space(vectors=dataset_descriptors_reduced,
        #                                                                   labels=dataset_labels_gt_int,
        #                                                                   title='FLAME face features',
        #                                                                   path_figure=os.path.abspath(os.getcwd()) + '/FLAME_features_2D.png',
        #                                                                   save=True)

        print('Doing face verification (1:1)...')
        results = do_face_verification_one_dataset(args.datasets_names[0], dataset_descriptors,
                                                   dataset_labels_gt_str, dataset_labels_gt_int)

        tp_total = results['tp_total']
        fp_total = results['fp_total']
        tn_total = results['tn_total']
        fn_total = results['fn_total']

    else:
        raise Exception('Face verification for multiple datasets not implemented yet!')

    # LOAD POINT CLOUD DESCRIPTORS

    # DO MATCHING (1:1)

    pass


def main_teste():
    for i in range(10 - 1, -1, -1):
        print('main_teste: i=', i)


if __name__ == '__main__':
    # sys.argv += ['-epochs', '100']
    # print('__main__(): sys.argv=', sys.argv)

    sys.argv += ['-datasets_path', '/home/bjgbiesseck/GitHub/MICA/demo/output']

    sys.argv += ['-datasets_names', ['lfw']]
    # sys.argv += ['-datasets_names', ['TALFW']]
    # sys.argv += ['-datasets_names', ['lfw', 'TALFW']]

    # sys.argv += ['-dataset_size', 'subset']
    sys.argv += ['-dataset_size', 'whole']

    sys.argv += ['-desc_file_ext', 'identity.npy']  # 300 parameter of FLAME face

    args = parse_args()
    # print('__main__(): args=', args)

    main_visualization(args)
    # main_teste()
