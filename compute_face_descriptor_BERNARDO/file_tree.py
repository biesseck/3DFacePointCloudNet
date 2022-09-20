import os.path
from pathlib import Path
import glob


class Tree:
    def __walk(self, dir_path: Path):
        contents = list(dir_path.iterdir())
        for path in contents:
            if path.is_dir():  # extend the prefix and recurse:
                yield str(path)
                yield from self.__walk(path)

    def get_all_sub_folders(self, dir_path: str):
        # dir_path = dir_path.replace('//', '/')
        folders = [dir_path]
        for folder in self.__walk(Path(dir_path)):
            # print(folder)
            folders.append(folder)
        return folders

    def get_sub_folders_one_level(self, dir_path: str):
        # sub_folders = [f.path for f in os.scandir(dir_path) if f.is_dir()]
        sub_folders = [f.name for f in os.scandir(dir_path) if f.is_dir()]
        return sub_folders


class FileTreeLfwDatasets3dReconstructed(Tree):
    def _get_sample_names_per_subject(self, dataset_path, subdirs_list: []):
        # dataset_path = dataset_path.replace('//', '/')
        subjects_sample_paths = [[]] * len(subdirs_list)
        # for subdir in subdirs_list:
        for i, subdir in enumerate(subdirs_list):
            subject_path = os.path.join(dataset_path, subdir)
            # print('FileTreeLfwDatasets: _get_sample_names_from_dirs_list(): subject_path=', subject_path)
            # subjects_sample_paths[i] = glob.glob(subject_path + '/*')
            subjects_sample_paths[i] = self.get_sub_folders_one_level(subject_path)
            # print('    FileTreeLfwDatasets: _get_sample_names_from_dirs_list(): subjects_sample_paths['+str(i)+']=', subjects_sample_paths[i])
        return subjects_sample_paths

    def __get_subjects_names_from_paths(self, sub_folders: []):
        subjects_names = [''] * len(sub_folders)
        for i in range(len(sub_folders)):
            sub_folder = sub_folders[i]
            subjects_names[i] = sub_folder.split('/')[-1]
        return subjects_names

    def get_common_subjects_names(self, dataset_location: str, datasets_names: [], image_type: str):
        subjects_names = {}
        for dataset_name in datasets_names:
            if dataset_name.upper() == 'LFW':
                subjects_names[dataset_name] = self.get_sub_folders_one_level(dataset_location + '/' + dataset_name)
            elif dataset_name.upper() == 'TALFW':
                subjects_names[dataset_name] = self.get_sub_folders_one_level(dataset_location + '/' + dataset_name)
            # TODO
            elif dataset_name.upper() == 'calfw':
                pass
            # TODO
            elif dataset_name.upper() == 'MLFW':
                pass
        common_subjects = sorted(set(subjects_names[datasets_names[0]]).intersection(subjects_names[datasets_names[1]]))
        return common_subjects

    def get_common_samples_names(self, dataset_location: str, datasets_names: [], common_subjects: [], image_type: str):
        # print('datasets_names:', datasets_names)
        samples_lists_names = {}
        for dataset_name in datasets_names:
            samples_lists_names[dataset_name] = self._get_sample_names_per_subject(
                                                                    dataset_path=dataset_location + '/' + dataset_name,
                                                                    subdirs_list=common_subjects)
            # for subject, samples_list_name in zip(common_subjects, samples_lists_names[dataset_name]):
            #     print(subject, ':', samples_list_name)

        # KEEP ONLY COMMON SAMPLES
        common_samples = [[]] * len(common_subjects)
        for i in range(len(samples_lists_names[datasets_names[0]])):
            common_samples[i] = sorted(set(samples_lists_names[datasets_names[0]][i]).intersection(samples_lists_names[datasets_names[1]][i]))
            # print('i:', i, '-', samples_lists_names[datasets_names[0]][i], ':', samples_lists_names[datasets_names[1]][i])
            # print('    common_samples[i]:', common_samples[i])
        return samples_lists_names, common_samples

    def get_common_subjects_and_samples_names(self, dataset_location: str, datasets_names: [], image_type: str):
        common_subjects = self.get_common_subjects_names(dataset_location, datasets_names, image_type)
        samples_lists_names, common_samples_names = self.get_common_samples_names(dataset_location, datasets_names, common_subjects, image_type)
        return common_subjects, samples_lists_names, common_samples_names



if __name__ == '__main__':

    input_path = '/home/bjgbiesseck/GitHub/MICA/demo/output'

    dataset_names = ['lfw', 'TALFW']
    # dataset_names = ['calfw', 'MLFW']

    # exts = ['_OBJ.pt', '_PLY.pt']
    exts = ['_OBJ.pt']
    # exts = ['_PLY.pt']

    common_subjects, samples_lists_names, common_samples_names = FileTreeLfwDatasets3dReconstructed().get_common_subjects_and_samples_names(input_path, dataset_names, 'original')
    # print('len(common_subjects):', len(common_subjects))
    # print('len(common_samples_names):', len(common_samples_names))

    for i, common_subject in enumerate(common_subjects):
        common_sample_names = common_samples_names[i]
        print('common_subject:', common_subject, '   common_sample_names:', common_sample_names)
    print('len(common_subjects):', len(common_subjects))

    # for common_file_name in common_file_names:
    #     print('common_file_name:', common_file_name)
    # print('len(common_file_names):', len(common_file_names))
