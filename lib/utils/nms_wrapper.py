# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

# from .nms.cpu_nms import cpu_nms, cpu_soft_nms

# def nms(dets, thresh):
#     """Dispatch to either CPU or GPU NMS implementations."""

#     if dets.shape[0] == 0:
#         return []
#     return cpu_nms(dets, thresh)

# 파이썬용?

from .nms.py_cpu_nms import py_cpu_nms

def nms(dets, thresh):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    return py_cpu_nms(dets, thresh)
