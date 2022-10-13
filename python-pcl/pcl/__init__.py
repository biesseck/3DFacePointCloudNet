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

    if path.endswith('.xyz'):
        p = load_XYZ(path)
    elif path.endswith('.npy'):
        p = load_NPY(path)
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
    return p


def load_NPY(filename):
    nd_array_npy = np.load(filename)
    if len(nd_array_npy.shape) == 3:
        nd_array_npy = nd_array_npy[0]
    cloud = PointCloud()
    cloud.from_array(nd_array_npy)
    return cloud


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

    for candidate in ["pcd", "ply", "obj"]:   # original
    # for candidate in ["pcd", "ply", "obj", "xyz"]:     # BERNARDO
        if path.endswith("." + candidate):
            return candidate

    raise ValueError("Could not determine file format from pathname %s" % path)
