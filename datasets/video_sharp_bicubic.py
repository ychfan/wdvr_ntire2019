import os
import random
from PIL import Image
import numpy as np
import h5py

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import common.modes
import datasets._vsr

LOCAL_DIR = 'data/video/sharp_bicubic/'
TRAIN_LR_DIR = lambda s: LOCAL_DIR + 'train_lr/X{}/'.format(s)
TRAIN_HR_DIR = LOCAL_DIR + 'train_hr/'
EVAL_LR_DIR = lambda s: LOCAL_DIR + 'val_lr/X{}/'.format(s)
EVAL_HR_DIR = LOCAL_DIR + 'val_hr/'
PREDICT_LR_DIR = lambda s: LOCAL_DIR + 'test_lr/X{}/'.format(s)


def update_argparser(parser):
  datasets._vsr.update_argparser(parser)
  parser.set_defaults(
      scale=4,
      num_channels=3,
      train_batch_size=32,
      eval_batch_size=1,
  )


def get_dataset(mode, params):
  if mode == common.modes.TRAIN:
    return VideoSR(mode, params)
  elif mode == common.modes.EVAL:
    return VideoSR(mode, params)
  else:
    raise NotImplementedError


class VideoSR(datasets._vsr.VideoSuperResolutionHDF5Dataset):

  def __init__(self, mode, params):
    lr_cache_file = 'cache/video_sharp_bicubic_{}_lr_x{}.h5'.format(
        mode, params.scale)
    hr_cache_file = 'cache/video_sharp_bicubic_{}_hr.h5'.format(mode)

    lr_dir = {
        common.modes.TRAIN: TRAIN_LR_DIR(params.scale),
        common.modes.EVAL: EVAL_LR_DIR(params.scale),
        common.modes.PREDICT: PREDICT_LR_DIR(params.scale),
    }[mode]
    hr_dir = {
        common.modes.TRAIN: TRAIN_HR_DIR,
        common.modes.EVAL: EVAL_HR_DIR,
        common.modes.PREDICT: '',
    }[mode]

    def list_image_files(lr_dir, hr_dir):
      lr_files = []
      hr_files = []
      for video in sorted(os.listdir(lr_dir)):
        files = sorted(os.listdir(os.path.join(lr_dir, video)))
        files = [f for f in files if f.endswith(".png")]
        lr_files.append((video, [(os.path.join(video, f),
                                  os.path.join(lr_dir, video, f))
                                 for f in files]))
        if mode == common.modes.PREDICT:
          hr_files.append((video,
                           [('{}_{}'.format(video, f), NULL) for f in files]))
        else:
          hr_files.append((video, [(os.path.join(video, f),
                                    os.path.join(hr_dir, video, f))
                                   for f in files]))
      return lr_files, hr_files

    lr_files, hr_files = list_image_files(lr_dir, hr_dir)

    super(VideoSR, self).__init__(
        mode,
        params,
        lr_files,
        hr_files,
        lr_cache_file,
        hr_cache_file,
    )