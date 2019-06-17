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
        default=4,
        type=int)
    parser.set_defaults(
        train_epochs=20,
        learning_rate_milestones=(15, 18),
        save_checkpoints_epochs=1,
        lr_patch_size=64,
        train_temporal_size=1,
        eval_temporal_size=1,
    )
  else:
    raise NotImplementedError('Needs to tune hyper parameters for new dataset.')


def get_model_spec(params):
  model = MODEL(params)
  print("# of parameters: ", sum([p.numel() for p in model.parameters()]))
  optimizer = optim.Adam(model.parameters())
  lr_scheduler = optim.lr_scheduler.MultiStepLR(
      optimizer, params.learning_rate_milestones, 0.1)
  loss_fn = torch.nn.L1Loss()
  metrics = {
      "loss": loss_fn,
      "PSNR": functools.partial(psnr, shave=params.scale + 6)
  }
  return model, loss_fn, optimizer, lr_scheduler, metrics


def psnr(sr, hr, shave):
  sr = sr.to(hr.dtype)
  sr = (sr * 255).round().clamp(0, 255) / 255
  diff = sr - hr
  valid = diff[..., shave:-shave, shave:-shave]
  mse = valid.pow(2).mean(-1).mean(-1).mean(-1)
  psnr = -10 * mse.log10()
  return psnr.mean()


class MODEL(nn.Module):

  def __init__(self, params):
    super(MODEL, self).__init__()
    # hyper-params
    scale = params.scale
    n_resblocks = params.num_blocks
    n_feats = params.num_residual_units
    kernel_size = 3
    act = nn.ReLU(True)
    # wn = lambda x: x
    wn = lambda x: torch.nn.utils.weight_norm(x)

    # define head module
    head = []
    head.append(
        wn(
            nn.Conv2d(
                params.num_channels *
                (params.train_temporal_padding_size * 2 + 1),
                n_feats,
                3,
                padding=3 // 2)))

    # define body module
    body = []
    for i in range(n_resblocks):
      body.append(
          Block(
              n_feats,
              kernel_size,
              params.width_multiplier,
              wn=wn,
              res_scale=1 / math.sqrt(params.num_blocks),
              act=act))

    # define tail module
    tail = []
    out_feats = scale * scale * params.num_channels
    tail.append(wn(nn.Conv2d(n_feats, out_feats, 3, padding=3 // 2)))

    skip = []
    skip.append(
        wn(
            nn.Conv2d(
                params.num_channels *
                (params.train_temporal_padding_size * 2 + 1),
                out_feats,
                5,
                padding=5 // 2)))

    shuf = []
    shuf.append(nn.PixelShuffle(scale))

    # make object members
    self.head = nn.Sequential(*head)
    self.body = nn.Sequential(*body)
    self.tail = nn.Sequential(*tail)
    self.skip = nn.Sequential(*skip)
    self.shuf = nn.Sequential(*shuf)

  def forward(self, x):
    x = x.view([x.shape[0], -1, x.shape[3], x.shape[4]])
    x -= 0.5
    s = self.skip(x)
    x = self.head(x)
    x = self.body(x)
    x = self.tail(x)
    x += s
    x = self.shuf(x)
    x += 0.5
    x = x.view([x.shape[0], -1, 1, x.shape[2], x.shape[3]])
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
    body.append(
        wn(
            nn.Conv2d(
                n_feats,
                n_feats * width_multiplier,
                kernel_size,
                padding=kernel_size // 2)))
    body.append(act)
    body.append(
        wn(
            nn.Conv2d(
                n_feats * width_multiplier,
                n_feats,
                kernel_size,
                padding=kernel_size // 2)))

    self.body = nn.Sequential(*body)

  def forward(self, x):
    res = self.body(x) * self.res_scale
    res += x
    return res
