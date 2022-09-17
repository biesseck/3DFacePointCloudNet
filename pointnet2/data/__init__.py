# ORIGINAL (alfredtorres/3DFacePointCloudNet)
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
from .GPMM_Normal_Curvature_dataset import GPMMNormalCurvDataset
from .TripletFaceDataset import TripletFaceDataset



# ADDED FROM erikwijmans/Pointnet2_PyTorch
from .Indoor3DSemSegLoader import Indoor3DSemSeg
from .ModelNet40Loader import ModelNet40Cls
