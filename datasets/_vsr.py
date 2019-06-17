import os
import random
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import common.modes
import datasets._isr


def update_argparser(parser):
  datasets._isr.update_argparser(parser)
  parser.add_argument(
      '--train_temporal_size',
      help='Number of frames for training',
      default=5,
      type=int)
  parser.add_argument(
      '--eval_temporal_size',
      help='Number of frames for evaluation',
      default=5,
      type=int)
  parser.add_argument(
      '--train_temporal_padding_size',
      help='Number of frames for training',
      default=3,
      type=int)
  parser.add_argument(
      '--eval_temporal_padding_size',
      help='Number of frames for evaluation',
      default=3,
      type=int)
  parser.set_defaults(
      num_patches=100,
      train_batch_size=16,
      eval_batch_size=1,
  )


class _SingleVideoSuperResolutionDataset(data.Dataset):

  def __init__(self, mode, params, video_name, lr_files, hr_files):
    super(_SingleVideoSuperResolutionDataset, self).__init__()
    self.mode = mode
    self.params = params
    self.video_name = video_name
    self.lr_files = lr_files
    self.hr_files = hr_files
    self.temporal_size = {
        common.modes.TRAIN: params.train_temporal_size,
        common.modes.EVAL: params.eval_temporal_size,
        common.modes.PREDICT: params.eval_temporal_size,
    }[mode]
    self.temporal_padding_size = {
        common.modes.TRAIN: params.train_temporal_padding_size,
        common.modes.EVAL: params.eval_temporal_padding_size,
        common.modes.PREDICT: params.eval_temporal_padding_size,
    }[mode]

  def __getitem__(self, index):
    t = index * self.temporal_size
    lr_files = [
        self.lr_files[min(len(self.lr_files) - 1, max(0, i))]
        for i in range(t - self.temporal_padding_size, t + self.temporal_size +
                       self.temporal_padding_size)
    ]
    hr_files = [self.hr_files[i] for i in range(t, t + self.temporal_size)]
    if self.mode == common.modes.PREDICT:
      # lr_images = [Image.open(lr_file[1]) for lr_file in lr_files]
      # lr_images = [
      #     lr_image.resize(
      #         (int(lr_image.size[0] / 4), int(lr_image.size[1] / 4)),
      #         Image.BICUBIC) for lr_image in lr_images
      # ]
      # lr_images = [
      #     transforms.functional.to_tensor(np.asarray(lr_image))
      #     for lr_image in lr_images
      # ]
      lr_images = [
          transforms.functional.to_tensor(np.asarray(Image.open(lr_file[1])))
          for lr_file in lr_files
      ]
      lr_images = torch.stack(lr_images, dim=1)
      hr_files = [hr_file[0] for hr_file in hr_files]
      return lr_images, hr_files

    lr_images = [Image.open(lr_file[1]) for lr_file in lr_files]
    hr_images = [Image.open(hr_file[1]) for hr_file in hr_files]

    # sample patch while training
    if self.mode == common.modes.TRAIN:
      lr_width, lr_height = lr_images[0].size
      x = random.randrange(
          self.params.ignored_boundary_size, lr_width -
          self.params.lr_patch_size + 1 - self.params.ignored_boundary_size)
      y = random.randrange(
          self.params.ignored_boundary_size, lr_height -
          self.params.lr_patch_size + 1 - self.params.ignored_boundary_size)
      lr_patch_region = (x, y, x + self.params.lr_patch_size,
                         y + self.params.lr_patch_size)
      lr_images = [lr_image.crop(lr_patch_region) for lr_image in lr_images]
      hr_patch_region = (pos * self.params.scale for pos in lr_patch_region)
      hr_images = [hr_image.crop(hr_patch_region) for hr_image in hr_images]
    lr_images = [np.asarray(lr_image) for lr_image in lr_images]
    hr_images = [np.asarray(hr_image) for hr_image in hr_images]

    # augmentation while training
    if self.mode == common.modes.TRAIN:
      if random.random() < 0.5:
        lr_images = [lr_image[::-1] for lr_image in lr_images]
        hr_images = [hr_image[::-1] for hr_image in hr_images]
      if random.random() < 0.5:
        lr_images = [lr_image[:, ::-1] for lr_image in lr_images]
        hr_images = [hr_image[:, ::-1] for hr_image in hr_images]
      if random.random() < 0.5:
        lr_images = [lr_image.transpose(1, 0, 2) for lr_image in lr_images]
        hr_images = [hr_image.transpose(1, 0, 2) for hr_image in hr_images]
      if random.random() < 0.5:
        lr_images = reversed(lr_images)
        hr_images = reversed(hr_images)
      lr_images = [np.ascontiguousarray(lr_image) for lr_image in lr_images]
      hr_images = [np.ascontiguousarray(hr_image) for hr_image in hr_images]

    lr_images = [
        transforms.functional.to_tensor(lr_image) for lr_image in lr_images
    ]
    hr_images = [
        transforms.functional.to_tensor(hr_image) for hr_image in hr_images
    ]
    lr_images = torch.stack(lr_images, dim=1)
    hr_images = torch.stack(hr_images, dim=1)
    return lr_images, hr_images

  def __len__(self):
    if len(self.hr_files) % self.temporal_size:
      raise NotImplementedError
    return len(self.hr_files) // self.temporal_size


class VideoSuperResolutionDataset(data.ConcatDataset):

  def __init__(self, mode, params, lr_files, hr_files):
    video_datasets = []
    for (v, l), (_, h) in zip(lr_files, hr_files):
      video_datasets.append(
          _SingleVideoSuperResolutionDataset(mode, params, v, l, h))
    if mode == common.modes.TRAIN:
      video_datasets = video_datasets * params.num_patches
    super(VideoSuperResolutionDataset, self).__init__(video_datasets)


class _SingleVideoSuperResolutionHDF5Dataset(data.Dataset):
  import h5py

  def __init__(
      self,
      mode,
      params,
      video_name,
      lr_files,
      hr_files,
      lr_cache_file,
      hr_cache_file,
      init_hdf5=False,
  ):
    super(_SingleVideoSuperResolutionHDF5Dataset, self).__init__()
    self.mode = mode
    self.params = params
    self.video_name = video_name
    self.lr_files = lr_files
    self.hr_files = hr_files
    self.lr_cache_file = lr_cache_file
    self.hr_cache_file = hr_cache_file
    self.temporal_size = {
        common.modes.TRAIN: params.train_temporal_size,
        common.modes.EVAL: params.eval_temporal_size,
        common.modes.PREDICT: params.eval_temporal_size,
    }[mode]
    self.temporal_padding_size = {
        common.modes.TRAIN: params.train_temporal_padding_size,
        common.modes.EVAL: params.eval_temporal_padding_size,
        common.modes.PREDICT: params.eval_temporal_padding_size,
    }[mode]

    if init_hdf5:
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
            #compression="gzip",
            shuffle=True,
            track_times=False,
            track_order=False)

      with self.h5py.File(lr_cache_file, 'a', libver='latest') as f:
        for lr_file in self.lr_files:
          add_image(lr_file, f)
      if self.mode != common.modes.PREDICT:
        with self.h5py.File(hr_cache_file, 'a', libver='latest') as f:
          for hr_file in self.hr_files:
            add_image(hr_file, f)

  def __getitem__(self, index):
    t = index * self.temporal_size
    lr_files = [
        self.lr_files[min(len(self.lr_files) - 1, max(0, i))][0]
        for i in range(t - self.temporal_padding_size, t + self.temporal_size +
                       self.temporal_padding_size)
    ]
    hr_files = [self.hr_files[i][0] for i in range(t, t + self.temporal_size)]
    if self.mode == common.modes.PREDICT:
      with self.h5py.File(self.lr_cache_file, 'r', libver='latest') as f:
        lr_images = [
            transforms.functional.to_tensor(np.ascontiguousarray(f[lr_file]))
            for lr_file in lr_files
        ]
        hr_files = [hr_file[0] for hr_file in hr_files]
        return lr_images, hr_files

    with self.h5py.File(
        self.lr_cache_file, 'r', libver='latest') as lr_f, self.h5py.File(
            self.hr_cache_file, 'r', libver='latest') as hr_f:
      lr_images = [lr_f[lr_file] for lr_file in lr_files]
      hr_images = [hr_f[hr_file] for hr_file in hr_files]

      # sample patch while training
      if self.mode == common.modes.TRAIN:
        x = random.randrange(
            self.params.ignored_boundary_size, lr_images[0].shape[0] -
            self.params.lr_patch_size + 1 - self.params.ignored_boundary_size)
        y = random.randrange(
            self.params.ignored_boundary_size, lr_images[0].shape[1] -
            self.params.lr_patch_size + 1 - self.params.ignored_boundary_size)
        lr_images = [
            lr_image[x:x + self.params.lr_patch_size, y:y +
                     self.params.lr_patch_size] for lr_image in lr_images
        ]
        hr_images = [
            hr_image[x * self.params.scale:(x + self.params.lr_patch_size) *
                     self.params.scale, y * self.params.scale:
                     (y + self.params.lr_patch_size) * self.params.scale]
            for hr_image in hr_images
        ]

      # augmentation while training
      if self.mode == common.modes.TRAIN:
        if random.random() < 0.5:
          lr_images = [lr_image[::-1] for lr_image in lr_images]
          hr_images = [hr_image[::-1] for hr_image in hr_images]
        if random.random() < 0.5:
          lr_images = [lr_image[:, ::-1] for lr_image in lr_images]
          hr_images = [hr_image[:, ::-1] for hr_image in hr_images]
        if random.random() < 0.5:
          lr_images = [lr_image.transpose(1, 0, 2) for lr_image in lr_images]
          hr_images = [hr_image.transpose(1, 0, 2) for hr_image in hr_images]
        if random.random() < 0.5:
          lr_images = reversed(lr_images)
          hr_images = reversed(hr_images)

      lr_images = [np.ascontiguousarray(lr_image) for lr_image in lr_images]
      hr_images = [np.ascontiguousarray(hr_image) for hr_image in hr_images]

    lr_images = [
        transforms.functional.to_tensor(lr_image) for lr_image in lr_images
    ]
    hr_images = [
        transforms.functional.to_tensor(hr_image) for hr_image in hr_images
    ]
    lr_images = torch.stack(lr_images, dim=1)
    hr_images = torch.stack(hr_images, dim=1)

    return lr_images, hr_images

  def __len__(self):
    if len(self.hr_files) % self.temporal_size:
      raise NotImplementedError
    return len(self.hr_files) // self.temporal_size


class VideoSuperResolutionHDF5Dataset(data.ConcatDataset):

  def __init__(
      self,
      mode,
      params,
      lr_files,
      hr_files,
      lr_cache_file,
      hr_cache_file,
  ):
    video_datasets = []
    init_hdf5 = not os.path.exists(lr_cache_file)
    for (v, l), (_, h) in zip(lr_files, hr_files):
      video_datasets.append(
          _SingleVideoSuperResolutionHDF5Dataset(
              mode, params, v, l, h, lr_cache_file, hr_cache_file, init_hdf5))
    if mode == common.modes.TRAIN:
      video_datasets = video_datasets * params.num_patches
    super(VideoSuperResolutionHDF5Dataset, self).__init__(video_datasets)
