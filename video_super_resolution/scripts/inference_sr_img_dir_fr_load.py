import os
import torch
from argparse import ArgumentParser, Namespace
import json
from typing import Any, Dict, List, Mapping, Tuple
from easydict import EasyDict
from math import ceil

import sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(base_path)
from video_to_video.video_to_video_model import VideoToVideo_sr, Vid2VidFr, pad_to_fit
from video_to_video.utils.seed import setup_seed
from video_to_video.utils.logger import get_logger
from video_super_resolution.color_fix import adain_color_fix, wavelet_color_fix
import torch.cuda.amp as amp
from video_to_video.modules.autoencoder_kl_temporal_decoder_feature_resetting import AutoencoderKLTemporalDecoderFeatureResetting

from inference_utils import *

logger = get_logger()

def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument("--input_path", required=True, type=str, help="input video path")
    parser.add_argument("--latent_path", required=True, type=str, help="pre-computed latent path")
    parser.add_argument("--save_dir", type=str, default='results', help="save directory")
    parser.add_argument("--file_name", type=str, help="file name")
    parser.add_argument("--model_path", type=str, default='./pretrained_weight/model.pt', help="model path")
    parser.add_argument("--prompt", type=str, default='a good video', help="prompt")
    parser.add_argument("--upscale", type=float, default=4, help='up-scale')
    parser.add_argument("--max_chunk_len", type=int, default=32, help='max_chunk_len')

    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--solver_mode", type=str, default='fast', help='fast | normal')
    parser.add_argument("--steps", type=int, default=15)


    parser.add_argument("--in_win_size", type=int, default=12, help="Window size of encoder & DM")
    parser.add_argument("--in_win_step", type=int, default=6, help="Window step of encoder & DM")
    parser.add_argument("--out_win_step", type=int, default=1, help="Window step of decoder")
    parser.add_argument("--out_win_overlap", type=int, default=1, help="Window overlap of decoder")
    parser.add_argument("--color_cor_method", type=str, default="wavelet")

    parser.add_argument("--device", type=str, default='cuda:0') # Add support for CPU because RAM sufficient

    parser.add_argument("--decoder_tile_size", type=int, default=64, help="decoder tile size")

    return parser.parse_args()

def vae_decode_fr(vae, z, z_prev, feature_map_prev, is_first_batch, out_win_step, out_win_overlap, decoder_tile_size=64):
    z = rearrange(z, "b c f h w -> (b f) c h w")
    num_f = z.shape[0]
    num_steps = int(ceil(num_f/out_win_step))
    video = []
    for i in range(num_steps):
        # Only if both input & output 1st batch, is_first_batch is true
        if i == 0 and is_first_batch:
            is_first_batch = True
            z_prev = z[0,:,:,:].repeat(out_win_overlap,1,1,1)
        else:
            is_first_batch = False
        z_chunk = z[i*out_win_step:(i+1)*out_win_step,:,:,:]
        z_chunk = torch.cat((z_prev, z_chunk), dim=0)
        v, feature_map_prev = vae.decode(z_chunk / vae.config.scaling_factor,
                                        feature_map_prev=feature_map_prev,
                                        num_frames=z_chunk.shape[0],
                                        is_first_batch=is_first_batch,
                                        frame_overlap_num=out_win_overlap,
                                        decoder_tile_size=decoder_tile_size)
        v = v.sample
        video.append(v[out_win_overlap:,:,:,:]) # handle corner case.
        z_prev = z_chunk[-out_win_overlap:,:,:,:] #always ensure out_win_overlap win size

    video = torch.cat(video)

    return video, feature_map_prev, z_prev

def enhance_a_video_fr(vae,
                       input_frames,
                       input_latents,
                       feature_map_prev,
                       z_prev,
                       is_first_batch,
                       out_win_step,
                       out_win_overlap,
                       prompt,
                       max_chunk_len,
                       color_cor_method,
                       device=torch.device(f'cuda:0'),
                       decoder_tile_size=64):
    video_data = preprocess(input_frames)
    bs = 1
    frames_num, _, h, w = video_data.shape
    target_h, target_w = int(h * 4), int(w * 4)  # adjust_resolution(h, w, up_scale=4)
    padding = pad_to_fit(target_h, target_w)
    with torch.no_grad():
        with amp.autocast(enabled=True):
            input_latents = np.stack(input_latents)
            input_latents = torch.from_numpy(input_latents).to(device).float()
            input_latents = torch.permute(input_latents, (1, 0, 2, 3))
            gen_vid = input_latents[None, :]
            vid_tensor_gen, feature_map_prev, z_prev = vae_decode_fr(vae=vae,
                                                                     z=gen_vid,
                                                                     z_prev=z_prev,
                                                                     feature_map_prev=feature_map_prev,
                                                                     is_first_batch=is_first_batch,
                                                                     out_win_step=out_win_step,
                                                                     out_win_overlap=out_win_overlap,
                                                                     decoder_tile_size=decoder_tile_size)
        w1, w2, h1, h2 = padding
        vid_tensor_gen = vid_tensor_gen[:, :, h1:target_h + h1, w1:target_w + w1]

        gen_video = rearrange(
            vid_tensor_gen, '(b f) c h w -> b c f h w', b=bs)

        torch.cuda.empty_cache()

        output = gen_video.type(torch.float32).cpu()

    output = tensor2vid(output)


    # Using color fix
    if color_cor_method == "adain":
        output = adain_color_fix(output, video_data)
    elif color_cor_method == "wavelet":
        from torch.nn import functional as F
        video_data = F.interpolate(video_data, [target_h, target_w], mode='bilinear')
        output = wavelet_color_fix(output, video_data)
    else:
        raise NotImplementedError("Color correction method not implemented.")

    # save_video(output, self.result_dir, self.file_name, fps=input_fps)
    return output, feature_map_prev, z_prev


def main():
    
    args = parse_args()

    input_path = args.input_path
    latent_path = args.latent_path
    prompt = args.prompt
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    in_win_size = args.in_win_size
    in_win_step = args.in_win_step
    out_win_step = args.out_win_step
    out_win_overlap = args.out_win_overlap
    color_cor_method = args.color_cor_method

    solver_mode = args.solver_mode


    max_chunk_len = args.max_chunk_len
    device = torch.device(args.device)

    decoder_tile_size = args.decoder_tile_size

    assert solver_mode in ('fast', 'normal')
    assert in_win_size >= in_win_step

    assert out_win_step >= out_win_overlap, "window step should be no smaller than window overlap."

    print("Loading VAE Model...")
    vae = AutoencoderKLTemporalDecoderFeatureResetting.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid", subfolder="vae", variant="fp16"
    )
    vae.eval()
    vae.requires_grad_(False)
    vae.to(device)

    logger.info('Build Temporal VAE with Feature Resetting Decoder.')

    torch.cuda.empty_cache()

    print(f"Loading latents from: {latent_path}")
    latents_name_list = os.listdir(latent_path)
    latents_name_list.sort()
    n_frames = len(latents_name_list)

    img_name_list = os.listdir(input_path)
    img_name_list.sort()

    n_steps = int(ceil((n_frames - in_win_size) / in_win_step)) + 1
    for w_i in range(n_steps):

        w_start_idx = w_i * in_win_size if w_i != n_steps - 1 else n_frames - in_win_size
        w_end_idx = w_start_idx + in_win_size if w_i != n_steps - 1 else n_frames
        w_latent_name_list = latents_name_list[w_start_idx:w_end_idx]
        w_img_name_list = img_name_list[w_start_idx//4:w_start_idx//4 + len(w_latent_name_list)//4]

        w_img_list_ = []
        for j in range(len(w_img_name_list)):
            for k in range(4):
                w_img_list_.append(w_img_name_list[j])
        w_img_name_list = w_img_list_

        print(f"Step {w_i} processing {len(w_img_name_list)} images")

        if w_i == 0:
            is_first_batch = True
            latents = load_latents(latent_path, w_latent_name_list)
            frames = load_frames(input_path, w_img_name_list)
            feature_map_prev = torch.zeros(0)
            z_prev = None
        else:
            is_first_batch = False
            latents = load_latents(latent_path, w_latent_name_list)
            frames = load_frames(input_path, w_img_name_list)

        video_sr, feature_map_prev, z_prev = enhance_a_video_fr(vae=vae,
                                                                input_frames=frames,
                                                                input_latents=latents,
                                                                feature_map_prev=feature_map_prev,
                                                                z_prev=z_prev,
                                                                is_first_batch=is_first_batch,
                                                                out_win_step=out_win_step,
                                                                out_win_overlap=out_win_overlap,
                                                                prompt=prompt,
                                                                max_chunk_len=max_chunk_len,
                                                                color_cor_method=color_cor_method,
                                                                device=device,
                                                                decoder_tile_size=decoder_tile_size)

        image_sr_list = [(img.numpy()).astype('uint8')[:, :, ::-1] for img in video_sr]
        for i in range(len(w_img_name_list)):
            img_name = os.path.splitext(w_latent_name_list[i])[0]+".png"
            out_path = os.path.join(save_dir, img_name)
            cv2.imwrite(out_path, image_sr_list[i])
            print(f"Saving {out_path}")


if __name__ == '__main__':
    main()
