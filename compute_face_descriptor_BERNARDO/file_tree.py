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


class FileTreeLfwDatasets(Tree):
    def _get_files_from_dirs_list(self, dataset_path, subdirs_list: [], exts=('jpg', 'png')):
        # dataset_path = dataset_path.replace('//', '/')
        subjects_files_paths = []
        for subdir in subdirs_list:
            subject_path = os.path.join(dataset_path, subdir)
            for ext in exts:
                subjects_files_paths += glob.glob(subject_path + '/*.' + ext)
        subjects_files_names = [''] * len(subjects_files_paths)
        for i in range(len(subjects_files_paths)):
            subjects_files_names[i] = subjects_files_paths[i].split('/')[-1]
        return subjects_files_names

    def __get_subjects_names_from_paths(self, sub_folders: []):
        subjects_names = [''] * len(sub_folders)
        for i in range(len(sub_folders)):
            sub_folder = sub_folders[i]
            subjects_names[i] = sub_folder.split('/')[-1]
        return subjects_names

    def get_common_images_names(self, dataset_location: str, datasets_names: [], exts: [], image_type: str):
        subjects_names = {}
        for dataset_name in datasets_names:
            if dataset_name.upper() == 'LFW':
                sub_folders_lfw = self.get_all_sub_folders(dataset_location + '/' + dataset_name)
                subjects_names[dataset_name] = self.__get_subjects_names_from_paths(sub_folders_lfw)
            if dataset_name.upper() == 'TALFW':
                sub_folders_talfw = self.get_all_sub_folders(dataset_location + '/' + dataset_name)
                subjects_names[dataset_name] = self.__get_subjects_names_from_paths(sub_folders_talfw)

        common_subjects = sorted(set(subjects_names[datasets_names[0]]).intersection(subjects_names[datasets_names[1]]))

        files_names = {}
        for dataset_name in datasets_names:
            files_names[dataset_name] = self._get_files_from_dirs_list(
                                                                dataset_path=dataset_location + '/' + dataset_name,
                                                                subdirs_list=common_subjects,
                                                                exts=exts)

        common_file_names = sorted(set(files_names[datasets_names[0]]).intersection(files_names[datasets_names[1]]))
        return common_subjects, common_file_names

    def get_common_images_names_without_ext(self, dataset_location: str, datasets_names: [], exts: [], image_type: str):
        common_subjects, common_file_names = self.get_common_images_names(dataset_location, datasets_names, exts, image_type)
        for i in range(len(common_file_names)):
            for ext in exts:
                common_file_names[i] = common_file_names[i].replace('.'+ext, '')
        return common_subjects, common_file_names


class FileTreeLfwDatasets3dReconstructed(FileTreeLfwDatasets):

    def get_3d_faces_file_names(self, dataset_location: str, dataset_name: str, subject_names: [], sub_folder_file_names: []):
        reconstruct_faces_file_names = [''] * len(sub_folder_file_names)
        reconstruct_render_file_names = [''] * len(sub_folder_file_names)
        for i in range(len(sub_folder_file_names)):
            sub_folder_file_name = sub_folder_file_names[i]
            subject_name = '_'.join(sub_folder_file_name.split('_')[0:-1])
            path_reconstruct_face = os.path.join(dataset_location, dataset_name, subject_name, sub_folder_file_name)
            reconstruct_faces_file_names[i] = os.path.join(path_reconstruct_face, 'mesh.obj')
            reconstruct_render_file_names[i] = os.path.join(path_reconstruct_face, 'render.jpg')
        return reconstruct_faces_file_names, reconstruct_render_file_names


'''
if __name__ == '__main__':

    input_path = '/mnt/42C8A18221CA5B0F/Local/datasets/imagensRGB/'
    # input_path = '/home/bjgbiesseck/datasets/rgb_images'

    dataset_names = ['lfw', 'TALFW']
    # dataset_names = ['calfw', 'MLFW']

    exts = ['jpg', 'png']

    common_file_names = FileTreeLfwDatasets().get_common_images_names_without_ext(input_path,
                                                                                  dataset_names, exts, 'original')
    pass
'''
