# XXX do a more specific import!
from ._pcl import *
# from .pcl_visualization import *
# from .pcl_grabber import *


import sys
import numpy as np


def load(path, format=None):
    """Load pointcloud from path.
    Currently supports PCD and PLY files.
    Format should be "pcd", "ply", "xyz", or None to infer from the pathname.
    """
    p, n = None, None

    if path.endswith('.xyz'):       # Bernardo
        p = load_XYZ(path)
    elif path.endswith('.npy'):     # Bernardo
        p = load_NPY(path)
    elif path.endswith('.abs'):     # Bernardo
        p = load_ABS(path)
    elif path.endswith('.abs.gz'):  # Bernardo
        p = load_ABS_GZ(path)
    elif path.endswith('.bc'):      # Bernardo
        p = load_BC(path)
    elif path.endswith('.txt'):      # Bernardo
        p, n = load_TXT(path)
    else:
        format = _infer_format(path, format)
        p = PointCloud()
        try:
            loader = getattr(p, "_from_%s_file" % format)
        except AttributeError:
            raise ValueError("unknown file format %s" % format)
        if loader(_encode(path)):
            raise IOError("error while loading pointcloud from %r (format=%r)"
                          % (path, format))
    return p, n


# BERNARDO
def load_TXT(filename):
    def get_format_info(data: str):
        if data.endswith('\n'):
            data = data[:-1]
        if ', ' in data:
            separator = ', '
        elif ',' in data:
            separator = ','
        elif ';' in data:
            separator = ';'
        elif ' ' in data:
            separator = ' '
        number_values = len(data.split(separator))
        return separator, number_values

    with open(filename) as file:
        cloud, normals = None, None
        all_lines = file.readlines()
        separator, number_values = get_format_info(all_lines[0][:-1])
        if number_values == 3:   # XYZ (3)
            nd_array_xyz = np.zeros(shape=(len(all_lines),3), dtype='float32')
            for i, line in enumerate(all_lines):
                if not line.startswith('#'):   # ignore comments
                    xyz_values = line[:-1].split(separator)
                    nd_array_xyz[i,0] = float(xyz_values[0])
                    nd_array_xyz[i,1] = float(xyz_values[1])
                    nd_array_xyz[i,2] = float(xyz_values[2])
            cloud = PointCloud(nd_array_xyz)

        elif number_values == 6:   # XYZ (3) + Normals (3)
            nd_array_xyz = np.zeros(shape=(len(all_lines),3), dtype='float32')
            nd_array_normals = np.zeros(shape=(len(all_lines),4), dtype='float32')
            for i, line in enumerate(all_lines):
                if not line.startswith('#'):   # ignore comments
                    xyz_values = line[:-1].split(separator)
                    nd_array_xyz[i,0] = float(xyz_values[0])
                    nd_array_xyz[i,1] = float(xyz_values[1])
                    nd_array_xyz[i,2] = float(xyz_values[2])
                    nd_array_normals[i,0] = float(xyz_values[3])
                    nd_array_normals[i,1] = float(xyz_values[4])
                    nd_array_normals[i,2] = float(xyz_values[5])
                    nd_array_normals[i,3] = float(0.0)
                cloud = PointCloud(nd_array_xyz)
                normals = PointCloud_Normal(nd_array_normals)

        elif number_values == 7:   # XYZ (3) + Normals (3) + PixelLabel (1)
            nd_array_xyz = np.zeros(shape=(len(all_lines),3), dtype='float32')
            nd_array_normals = np.zeros(shape=(len(all_lines),4), dtype='float32')
            nd_array_pixel_label = np.zeros(shape=(len(all_lines),1), dtype='float32')
            for i, line in enumerate(all_lines):
                if not line.startswith('#'):   # ignore comments
                    xyz_values = line[:-1].split(separator)
                    nd_array_xyz[i,0] = float(xyz_values[0])
                    nd_array_xyz[i,1] = float(xyz_values[1])
                    nd_array_xyz[i,2] = float(xyz_values[2])
                    nd_array_normals[i,0] = float(xyz_values[3])
                    nd_array_normals[i,1] = float(xyz_values[4])
                    nd_array_normals[i,2] = float(xyz_values[5])
                    nd_array_normals[i,3] = float(0.0)
                    nd_array_pixel_label[i] = float(xyz_values[6])
                cloud = PointCloud(nd_array_xyz)
                normals = PointCloud_Normal(nd_array_normals)

        # return cloud, normals, nd_array_pixel_label
        return cloud, normals


# BERNARDO
def load_BC(file):
    import os
    import struct
    npoints = os.path.getsize(file) // 4
    with open(file,'rb') as f:
        raw_data = struct.unpack('f'*npoints,f.read(npoints*4))
        data = np.asarray(raw_data,dtype=np.float32)       
#    data = data.reshape(len(data)//6, 6)
#    data = data.reshape(7, len(data)//7)     # original
    data = data.reshape(3, len(data)//3).T    # Bernardo
    # print('data:', data.shape)
    # translate the nose tip to [0,0,0]
    # data = (data - np.array([0, 0, 100], dtype=np.float32))
    # data = (data - data[8157]) / 100
    # print('data[8157]:', data[8157])
#    data = (data[:,0:2] - data[8157,0:2]) / 100
    # return torch.from_numpy(data.T)
    cloud = PointCloud(data)
    return cloud


# BERNARDO
def load_ABS_GZ(filename):
    import gzip
    with gzip.open(filename, 'rb') as f:
        all_lines = f.read().decode("utf-8").split('\r\n')
        rows = int(all_lines[0].split(' ')[0])
        columns = int(all_lines[1].split(' ')[0])
        # all_lines[2] can be ignored
        
        nd_array_xyz = np.zeros(shape=(rows*columns,3), dtype='float32')
        for i, valid, x, y, z in zip(np.arange(0, nd_array_xyz.shape[0]), all_lines[3].split(' '), all_lines[4].split(' '), all_lines[5].split(' '), all_lines[6].split(' ')):
            # valid = int(valid)
            nd_array_xyz[i,0] = float(x)
            nd_array_xyz[i,1] = float(y)
            nd_array_xyz[i,2] = float(z)
        
        cloud = PointCloud(nd_array_xyz)
        return cloud



# BERNARDO
def load_ABS(filename):
    with open(filename) as file:
        all_lines = file.readlines()
        rows = int(all_lines[0].split(' ')[0])
        columns = int(all_lines[1].split(' ')[0])
        # all_lines[2] can be ignored
        valid_points_indexes = [int(value) for value in all_lines[3].split(' ')[:-1]]
        valid_points_indexes = np.array(valid_points_indexes)
        assert rows*columns == valid_points_indexes.shape[0]
        
        number_of_valid_points = sum(valid_points_indexes == 1)
        print('number_of_valid_points:', number_of_valid_points)
        nd_array_xyz = np.zeros(shape=(number_of_valid_points,3), dtype='float32')
        j = 0
        for i, x, y, z in zip(np.arange(0, valid_points_indexes.shape[0]), all_lines[4].split(' '), all_lines[5].split(' '), all_lines[6].split(' ')):
            if valid_points_indexes[i] == True:
                nd_array_xyz[j,0] = float(x)
                nd_array_xyz[j,1] = float(y)
                nd_array_xyz[j,2] = float(z)
                j += 1
        
        cloud = PointCloud(nd_array_xyz)
        return cloud


# BERNARDO
def load_NPY(filename):
    nd_array_npy = np.load(filename)
    if len(nd_array_npy.shape) == 3:
        nd_array_npy = nd_array_npy[0]
    cloud = PointCloud()
    cloud.from_array(nd_array_npy)
    return cloud


# BERNARDO
def load_XYZ(filename):
    with open(filename) as file:
        all_lines = file.readlines()
        nd_array_xyz = np.zeros(shape=(len(all_lines),3), dtype='float32')
        # print('nd_array_xyz.shape:', nd_array_xyz.shape)
        for i, line in enumerate(all_lines):
            # print('line:', line[:-1])
            if not line.startswith('#'):   # ignore comments
                xyz_values = line[:-1].split(' ')
                nd_array_xyz[i,0] = float(xyz_values[0])
                nd_array_xyz[i,1] = float(xyz_values[1])
                nd_array_xyz[i,2] = float(xyz_values[2])
                # print('i:', i, '  nd_array_xyz[i]:', nd_array_xyz[i])
        cloud = PointCloud()
        cloud.from_array(nd_array_xyz)
        return cloud



def load_XYZI(path, format=None):
    """Load pointcloud from path.

    Currently supports PCD and PLY files.

    Format should be "pcd", "ply", or None to infer from the pathname.
    """
    format = _infer_format(path, format)
    p = PointCloud_PointXYZI()
    try:
        loader = getattr(p, "_from_%s_file" % format)
    except AttributeError:
        raise ValueError("unknown file format %s" % format)
    if loader(_encode(path)):
        raise IOError("error while loading pointcloud from %r (format=%r)"
                      % (path, format))
    return p


def load_XYZRGB(path, format=None):
    """
    Load pointcloud from path.
    Currently supports PCD and PLY files.
    Format should be "pcd", "ply", or None to infer from the pathname.
    """
    format = _infer_format(path, format)
    p = PointCloud_PointXYZRGB()
    try:
        loader = getattr(p, "_from_%s_file" % format)
    except AttributeError:
        raise ValueError("unknown file format %s" % format)
    if loader(_encode(path)):
        raise IOError("error while loading pointcloud from %r (format=%r)"
                      % (path, format))
    return p


def load_XYZRGBA(path, format=None):
    """
    Load pointcloud from path.
    Currently supports PCD and PLY files.
    Format should be "pcd", "ply", or None to infer from the pathname.
    """
    format = _infer_format(path, format)
    p = PointCloud_PointXYZRGBA()
    try:
        loader = getattr(p, "_from_%s_file" % format)
    except AttributeError:
        raise ValueError("unknown file format %s" % format)
    if loader(_encode(path)):
        raise IOError("error while loading pointcloud from %r (format=%r)"
                      % (path, format))
    return p


def load_PointWithViewpoint(path, format=None):
    """
    Load pointcloud from path.
    Currently supports PCD and PLY files.
    Format should be "pcd", "ply", or None to infer from the pathname.
    """
    format = _infer_format(path, format)
    p = PointCloud_PointWithViewpoint()
    try:
        loader = getattr(p, "_from_%s_file" % format)
    except AttributeError:
        raise ValueError("unknown file format %s" % format)
    if loader(_encode(path)):
        raise IOError("error while loading pointcloud from %r (format=%r)"
                      % (path, format))
    return p


def save(cloud, path, format=None, binary=False):
    """Save pointcloud to file.

    Format should be "pcd", "ply", or None to infer from the pathname.
    """
    format = _infer_format(path, format)
    try:
        dumper = getattr(cloud, "_to_%s_file" % format)
    except AttributeError:
        raise ValueError("unknown file format %s" % format)
    if dumper(_encode(path), binary):
        raise IOError("error while saving pointcloud to %r (format=%r)"
                      % (path, format))


def save_XYZRGBA(cloud, path, format=None, binary=False):
    """Save pointcloud to file.

    Format should be "pcd", "ply", or None to infer from the pathname.
    """
    format = _infer_format(path, format)
    try:
        dumper = getattr(cloud, "_to_%s_file" % format)
    except AttributeError:
        raise ValueError("unknown file format %s" % format)
    if dumper(_encode(path), binary):
        raise IOError("error while saving pointcloud to %r (format=%r)"
                      % (path, format))


def save_PointNormal(cloud, path, format=None, binary=False):
    """
    Save pointcloud to file.
    Format should be "pcd", "ply", or None to infer from the pathname.
    """
    format = _infer_format(path, format)
    try:
        dumper = getattr(cloud, "_to_%s_file" % format)
    except AttributeError:
        raise ValueError("unknown file format %s" % format)
    if dumper(_encode(path), binary):
        raise IOError("error while saving pointcloud to %r (format=%r)"
                      % (path, format))


def _encode(path):
    # Encode path for use in C++.
    if isinstance(path, bytes):
        return path
    else:
        return path.encode(sys.getfilesystemencoding())


def _infer_format(path, format):
    if format is not None:
        return format.lower()

    for candidate in ["pcd", "ply", "obj"]:              # original
    # for candidate in ["pcd", "ply", "obj", "xyz"]:     # BERNARDO
        if path.endswith("." + candidate):
            return candidate

    raise ValueError("Could not determine file format from pathname %s" % path)
