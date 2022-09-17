# ORIGINAL (alfredtorres/3DFacePointCloudNet)
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)

__version__ = "2.1.1"

try:
    __POINTNET2_SETUP__
except NameError:
    __POINTNET2_SETUP__ = False

if not __POINTNET2_SETUP__:
    from pointnet2 import utils
    from pointnet2 import data
    from pointnet2 import models



# ADDED BY BERNARDO FROM erikwijmans/Pointnet2_PyTorch
from pointnet2 import data, models, utils
from pointnet2._version import __version__
