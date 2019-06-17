# WDVR: wide-activated 3D convolutional network for video restoration

## Training
### 2D
```
python trainer.py --dataset video_sharp_bicubic --model wdsr_burst --job_dir ./logs/wdsr_burst --train_temporal_padding_size 2 --eval_temporal_padding_size 2
```

### 3D
```
python trainer.py --dataset video_sharp_bicubic --model wdvr --job_dir ./logs/wdvr
```