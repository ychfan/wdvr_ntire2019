from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import functools

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

import models


def update_argparser(parser):
  models.update_argparser(parser)
  args, _ = parser.parse_known_args()
  if args.dataset.startswith('video'):
    parser.add_argument(
        '--num-blocks',
        help='Number of residual blocks in networks',
        default=16,
        type=int)
    parser.add_argument(
        '--num-residual-units',
        help='Number of residual units in networks',
        default=32,
        type=int)
    parser.add_argument(
        '--width_multiplier',
        help='Width multiplier inside residual blocks',
        default=23.0 / 6,
        type=float)
    parser.set_defaults(
        train_epochs=20,
        learning_rate_milestones=(15, 18),
        save_checkpoints_epochs=1,
        lr_patch_size=48,
        train_temporal_size=10,
        eval_temporal_size=20,
        train_temporal_padding_size=0,
        eval_temporal_padding_size=10,
        train_batch_size=4,
    )
  else:
    raise NotImplementedError('Needs to tune hyper parameters for new dataset.')


def get_model_spec(params):
  model = MODEL(params)
  optimizer = optim.Adam(model.parameters())
  lr_scheduler = optim.lr_scheduler.MultiStepLR(
      optimizer, params.learning_rate_milestones, 0.1)
  loss_fn = PaddingL1Loss()
  metrics = {
      "loss": loss_fn,
      "PSNR": functools.partial(psnr, shave=params.scale + 6)
  }
  return model, loss_fn, optimizer, lr_scheduler, metrics


class PaddingL1Loss(nn.L1Loss):

  def forward(self, input, target):
    temporal_padding_size = (input.shape[2] - target.shape[2]) // 2
    if temporal_padding_size:
      input = input[:, :, temporal_padding_size:-temporal_padding_size, :, :]
    return nn.functional.l1_loss(input, target, reduction=self.reduction)


def psnr(sr, hr, shave):
  temporal_padding_size = (sr.shape[2] - hr.shape[2]) // 2
  if temporal_padding_size:
    sr = sr[:, :, temporal_padding_size:-temporal_padding_size, :, :]
  sr = sr.to(hr.dtype)
  sr = (sr * 255).round().clamp(0, 255) / 255
  diff = sr - hr
  valid = diff[..., shave:-shave, shave:-shave]
  mse = valid.pow(2).mean(-1).mean(-1).mean(-2)
  psnr = -10 * mse.log10()
  return psnr.mean()


class MODEL(nn.Module):

  def __init__(self, params):
    super(MODEL, self).__init__()
    # hyper-params
    # wn = lambda x: x
    wn = lambda x: torch.nn.utils.weight_norm(x)
    out_feats = params.num_channels * (params.scale**2)

    body = []
    body.append(
        wn(
            nn.Conv3d(
                params.num_channels,
                params.num_residual_units,
                (1, 3, 3),
                padding=(0, 3 // 2, 3 // 2),
            )))
    for i in range(params.num_blocks):
      body.append(
          Block(
              params.num_residual_units,
              3,
              params.width_multiplier,
              wn=wn,
              res_scale=1 / math.sqrt(params.num_blocks),
          ))
    body.append(
        wn(
            nn.Conv3d(
                params.num_residual_units,
                out_feats,
                (1, 3, 3),
                padding=(0, 3 // 2, 3 // 2),
            )))

    skip = []
    skip.append(nn.ReplicationPad3d((0, 0, 0, 0, 2, 2)))
    skip.append(
        wn(
            nn.Conv3d(
                params.num_channels,
                out_feats,
                (5, 5, 5),
                padding=(0, 5 // 2, 5 // 2),
            )))

    shuf = []
    shuf.append(PixelShuffleVideo(params.scale))

    # make object members
    self.body = nn.Sequential(*body)
    self.skip = nn.Sequential(*skip)
    self.shuf = nn.Sequential(*shuf)

  def forward(self, x):
    x -= 0.5
    x = self.body(x) + self.skip(x)
    x = self.shuf(x)
    x += 0.5
    return x


class Block(nn.Module):

  def __init__(self,
               n_feats,
               kernel_size,
               width_multiplier,
               wn,
               res_scale=1,
               act=nn.ReLU(True)):
    super(Block, self).__init__()
    self.res_scale = res_scale
    body = []
    body.append(nn.ReplicationPad3d((0, 0, 0, 0, 1, 1)))
    body.append(wn(nn.Conv3d(n_feats, n_feats, (3, 1, 1))))
    body.append(
        wn(
            nn.Conv3d(
                n_feats,
                int(n_feats * width_multiplier),
                (1, kernel_size, kernel_size),
                padding=(0, kernel_size // 2, kernel_size // 2),
            )))
    body.append(act)
    body.append(
        wn(
            nn.Conv3d(
                int(n_feats * width_multiplier),
                n_feats,
                (1, kernel_size, kernel_size),
                padding=(0, kernel_size // 2, kernel_size // 2),
            )))

    self.body = nn.Sequential(*body)

  def forward(self, x):
    res = self.body(x) * self.res_scale
    res += x
    return res


class PixelShuffleVideo(nn.Module):
  __constants__ = ['upscale_factor']

  def __init__(self, upscale_factor):
    super(PixelShuffleVideo, self).__init__()
    self.upscale_factor = upscale_factor

  def forward(self, input):

    def _pixel_shuffle_video(input, upscale_factor):
      batch_size, channels, depth, in_height, in_width = input.size()
      channels //= upscale_factor**2
      out_height = in_height * upscale_factor
      out_width = in_width * upscale_factor

      input_view = input.contiguous().view(batch_size, channels, upscale_factor,
                                           upscale_factor, depth, in_height,
                                           in_width)

      shuffle_out = input_view.permute(0, 1, 4, 5, 2, 6, 3).contiguous()
      return shuffle_out.view(batch_size, channels, depth, out_height,
                              out_width)

    return _pixel_shuffle_video(input, self.upscale_factor)

  def extra_repr(self):
    return 'upscale_factor={}'.format(self.upscale_factor)
