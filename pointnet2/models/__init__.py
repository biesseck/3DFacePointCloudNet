# ORIGINAL (alfredtorres/3DFacePointCloudNet)
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
from .pointnet2_ssg_face import Pointnet2SSG as Pointnet2FaceClsSSG


# ADDED BY BERNARDO FROM erikwijmans/Pointnet2_PyTorch
from pointnet2.models.pointnet2_msg_cls import PointNet2ClassificationMSG
from pointnet2.models.pointnet2_msg_sem import PointNet2SemSegMSG
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG
from pointnet2.models.pointnet2_ssg_sem import PointNet2SemSegSSG
