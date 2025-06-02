'''
Analyze the model's layer names and data sizes.
The information is important to split the model (pipeline parallelism)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
from easydict import EasyDict

from video_to_video.video_to_video_model import VideoToVideo_sr, pad_to_fit

from torchsummary import summary

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--width', type=int, default=480)
    parser.add_argument('--n_channels', type=int, default=3)
    parser.add_argument('--n_frames', type=int, default=180)
    parser.add_argument('--upscale', type=float, default=4.0)

    parser.add_argument('--model_path', type=str, default='/workspace/shared-dir/zzhu/pretrained/STAR/I2VGen-XL-based/light_deg.pt')
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--encoder_chunk_size', type=int, default=1)

    args = parser.parse_args()
    return args


def main():
    print("Analyzing the model's layer names and data sizes...")

    ## Get args
    args = parse_args()
    height = args.height
    width = args.width
    n_channels = args.n_channels
    n_frames = args.n_frames
    device = args.device
    model_path = args.model_path

    encoder_chunk_size = args.encoder_chunk_size

    ## Prepare dummy input data
    target_h = int(height * args.upscale)
    target_w = int(width * args.upscale)

    dummy_input = torch.randn(n_frames, n_channels, target_h, target_w)
    padding = pad_to_fit(height, width)
    video_data = F.pad(dummy_input, padding, 'constant', 1)
    # video_data = video_data.unsqueeze(0)

    video_data = video_data.to(device)
    x = video_data

    x = x[:encoder_chunk_size, :, :, :]

    ## Initialize the model
    model_cfg = EasyDict(__name__='model_cfg')
    model_cfg.model_path = model_path
    model = VideoToVideo_sr(model_cfg)

    print("-----VAE-----")
    summary(model.vae, x)


if __name__ == "__main__":
    main()
