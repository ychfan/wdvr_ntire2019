import os
import random
from PIL import Image
import numpy as np
import h5py
import tables
import warnings

import torch.utils.data as data
import torchvision.transforms as transforms

import common.modes
import datasets


def update_argparser(parser):
  datasets.update_argparser(parser)
  parser.add_argument(
      '--scale', help='Scale for image super-resolution', default=2, type=int)
  parser.add_argument(
      '--lr_patch_size',
      help='Number of pixels in height or width of LR patches',
      default=48,
      type=int)
  parser.add_argument(
      '--ignored_boundary_size',
      help='Number of ignored boundary pixels of LR patches',
      default=2,
      type=int)
  parser.add_argument(
      '--num_patches',
      help='Number of sampling patches per image for training',
      default=100,
      type=int)
  parser.set_defaults(
      train_batch_size=16,
      eval_batch_size=1,
  )


class ImageSuperResolutionDataset(data.Dataset):

  def __init__(self, mode, params, lr_files, hr_files):
    super(ImageSuperResolutionDataset, self).__init__()
    self.mode = mode
    self.params = params
    self.lr_files = lr_files
    self.hr_files = hr_files

  def __getitem__(self, index):
    if self.mode == common.modes.PREDICT:
      lr_image = np.asarray(Image.open(self.lr_files[index][1]))
      lr_image = transforms.functional.to_tensor(lr_image)
      return lr_image, self.hr_files[index][0]

    if self.mode == common.modes.TRAIN:
      index = index // self.params.num_patches

    lr_image = Image.open(self.lr_files[index][1])
    hr_image = Image.open(self.lr_files[index][1])

    # sample patch while training
    if self.mode == common.modes.TRAIN:
      lr_width, lr_height = lr_image.size
      x = random.randrange(
          self.params.ignored_boundary_size, lr_width -
          self.params.lr_patch_size + 1 - self.params.ignored_boundary_size)
      y = random.randrange(
          self.params.ignored_boundary_size, lr_height -
          self.params.lr_patch_size + 1 - self.params.ignored_boundary_size)
      lr_patch_region = (x, y, x + self.params.lr_patch_size,
                         y + self.params.lr_patch_size)
      lr_image = lr_image.crop(lr_patch_region)
      hr_image = hr_image.crop(
          (pos * self.params.scale for pos in lr_patch_region))
    else:
      lr_width, lr_height = lr_image.size
      hr_image = hr_image.crop((0, 0, lr_width * self.params.scale,
                                lr_height * self.params.scale))
    lr_image = np.asarray(lr_image)
    hr_image = np.asarray(hr_image)

    # augmentation while training
    if self.mode == common.modes.TRAIN:
      if lr_image.ndim == 2:
        lr_image = lr_image[:, :, None]
      if hr_image.ndim == 2:
        hr_image = hr_image[:, :, None]
      if random.random() < 0.5:
        lr_image = lr_image[::-1]
        hr_image = hr_image[::-1]
      if random.random() < 0.5:
        lr_image = lr_image[:, ::-1]
        hr_image = hr_image[:, ::-1]
      if random.random() < 0.5:
        lr_image = lr_image.transpose(1, 0, 2)
        hr_image = hr_image.transpose(1, 0, 2)
      lr_image = np.ascontiguousarray(lr_image)
      hr_image = np.ascontiguousarray(hr_image)

    lr_image = transforms.functional.to_tensor(lr_image)
    hr_image = transforms.functional.to_tensor(hr_image)
    return lr_image, hr_image

  def __len__(self):
    if self.mode == common.modes.TRAIN:
      return len(self.lr_files) * self.params.num_patches
    else:
      return len(self.lr_files)


class ImageSuperResolutionHDF5Dataset(data.Dataset):

  def __init__(
      self,
      mode,
      params,
      lr_files,
      hr_files,
      lr_cache_file,
      hr_cache_file,
  ):
    super(ImageSuperResolutionHDF5Dataset, self).__init__()
    self.mode = mode
    self.params = params
    self.lr_files = lr_files
    self.hr_files = hr_files
    self.lr_cache_file = lr_cache_file
    self.hr_cache_file = hr_cache_file

    cache_dir = os.path.dirname(self.lr_cache_file)
    if not os.path.exists(cache_dir):
      os.makedirs(cache_dir)

    def add_image(image_file, cache_file):
      image = np.asarray(Image.open(image_file[1]))
      cache_file.create_dataset(
          image_file[0],
          data=image,
          maxshape=image.shape,
          compression='lzf',
          shuffle=True,
          track_times=False,
          track_order=False,
      )

    if not os.path.exists(self.lr_cache_file):
      with h5py.File(self.lr_cache_file, 'w', libver='latest') as f:
        for lr_file in self.lr_files:
          add_image(lr_file, f)
    if self.mode != common.modes.PREDICT:
      if not os.path.exists(self.hr_cache_file):
        with h5py.File(self.hr_cache_file, 'w', libver='latest') as f:
          for hr_file in self.hr_files:
            add_image(hr_file, f)

  def __getitem__(self, index):
    if self.mode == common.modes.PREDICT:
      with h5py.File(self.lr_cache_file, 'r', libver='latest') as f:
        lr_image = f[self.lr_files[index][0]]
        lr_image = np.ascontiguousarray(lr_image)
        lr_image = transforms.functional.to_tensor(lr_image)
        return lr_image, self.hr_files[index][0]

    if self.mode == common.modes.TRAIN:
      index = index // self.params.num_patches

    with h5py.File(
        self.lr_cache_file, 'r', libver='latest') as lr_f, h5py.File(
            self.hr_cache_file, 'r', libver='latest') as hr_f:
      lr_image = lr_f[self.lr_files[index][0]]
      hr_image = hr_f[self.hr_files[index][0]]

      if self.mode == common.modes.TRAIN:
        # sample patch while training
        x = random.randrange(
            self.params.ignored_boundary_size, lr_image.shape[0] -
            self.params.lr_patch_size + 1 - self.params.ignored_boundary_size)
        y = random.randrange(
            self.params.ignored_boundary_size, lr_image.shape[1] -
            self.params.lr_patch_size + 1 - self.params.ignored_boundary_size)
        lr_image = lr_image[x:x + self.params.lr_patch_size, y:y +
                            self.params.lr_patch_size]
        hr_image = hr_image[x *
                            self.params.scale:(x + self.params.lr_patch_size) *
                            self.params.scale, y * self.params.scale:
                            (y + self.params.lr_patch_size) * self.params.scale]
      else:
        hr_image = hr_image[:lr_image.shape[0] *
                            self.params.scale, :lr_image.shape[1] *
                            self.params.scale]
      lr_image = np.asarray(lr_image)
      hr_image = np.asarray(hr_image)

      if lr_image.ndim == 2:
        lr_image = lr_image[:, :, None]
      if hr_image.ndim == 2:
        hr_image = hr_image[:, :, None]

      if self.mode == common.modes.TRAIN:
        # augmentation while training
        if random.random() < 0.5:
          lr_image = lr_image[::-1]
          hr_image = hr_image[::-1]
        if random.random() < 0.5:
          lr_image = lr_image[:, ::-1]
          hr_image = hr_image[:, ::-1]
        if random.random() < 0.5:
          lr_image = lr_image.transpose(1, 0, 2)
          hr_image = hr_image.transpose(1, 0, 2)

      lr_image = np.ascontiguousarray(lr_image)
      hr_image = np.ascontiguousarray(hr_image)

    lr_image = transforms.functional.to_tensor(lr_image)
    hr_image = transforms.functional.to_tensor(hr_image)

    return lr_image, hr_image

  def __len__(self):
    if self.mode == common.modes.TRAIN:
      return len(self.lr_files) * self.params.num_patches
    else:
      return len(self.lr_files)
