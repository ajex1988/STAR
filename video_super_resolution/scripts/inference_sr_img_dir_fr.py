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
from video_to_video.video_to_video_model import VideoToVideo_sr, Vid2VidFr
from video_to_video.utils.seed import setup_seed
from video_to_video.utils.logger import get_logger
from video_super_resolution.color_fix import adain_color_fix, wavelet_color_fix

from inference_utils import *

logger = get_logger()


class STAR():
    def __init__(self, 
                 result_dir='./results/',
                 model_path='',
                 solver_mode='fast',
                 steps=15,
                 guide_scale=7.5,
                 upscale=4,
                 max_chunk_len=32,
                 vae_decoder_chunk_size=3,
                 device=torch.device(f'cuda:0')
                 ):
        self.model_path=model_path
        logger.info('checkpoint_path: {}'.format(self.model_path))

        self.result_dir = result_dir
        os.makedirs(self.result_dir, exist_ok=True)

        model_cfg = EasyDict(__name__='model_cfg')
        model_cfg.model_path = self.model_path
        self.model = VideoToVideo_sr(model_cfg, device=device)

        if device == torch.device('cpu'):
            # Use fp32 in the cpu mode
            self.model.generator = self.model.generator.float()
            self.model.text_encoder = self.model.text_encoder.float()
            self.model.vae = self.model.vae.float()

        steps = 15 if solver_mode == 'fast' else steps
        self.solver_mode=solver_mode
        self.steps=steps
        self.guide_scale=guide_scale
        self.upscale = upscale
        self.max_chunk_len=max_chunk_len
        self.vae_decoder_chunk_size=vae_decoder_chunk_size

    def enhance_dir(self, input_frames_dir, input_fps, prompt, win_size, win_step, idx_i, idx_j):
        """
        Enhance the images inside a directory.
        Assume that the dir only contains images of the same format, and the name indicates the frame order.
        """
        img_name_list = os.listdir(input_frames_dir)
        img_name_list.sort()

        n_frames = len(img_name_list)
        assert n_frames >= win_size, "Input frame size should be larger than window size."
        n_steps = ceil((n_frames-win_size)/win_step) + 1

        for w_i in range(n_steps):
            w_start_idx = win_step*w_i
            w_end_idx = w_start_idx + win_size if w_i != n_steps-1 else n_frames
            w_img_name_list = img_name_list[w_start_idx:w_end_idx]
            frames = load_frames(input_frames_dir, w_img_name_list)
            video_sr = self.enhance_a_video(input_frames=frames, input_fps=input_fps, prompt=prompt)
            image_sr_list = [(img.numpy()).astype('uint8')[:,:,::-1] for img in video_sr]
            if w_i == 0:
                save_frame_list = image_sr_list[:idx_j]
                save_name_list = w_img_name_list[:idx_j]
            elif w_i==n_steps-1:
                save_frame_list = image_sr_list[idx_i:]
                save_name_list = w_img_name_list[idx_i:]
            else:
                ## Only save the frames between idx_i and idx_j
                save_frame_list = image_sr_list[idx_i:idx_j]
                save_name_list = w_img_name_list[idx_i:idx_j]
            for i in range(len(save_name_list)):
                out_path = os.path.join(self.result_dir, save_name_list[i])
                cv2.imwrite(out_path, save_frame_list[i])


    def enhance_a_video(self, input_frames, input_fps, prompt):
        text = prompt
        logger.info('text: {}'.format(text))
        caption = text + self.model.positive_prompt

        logger.info('input fps: {}'.format(input_fps))

        video_data = preprocess(input_frames)
        _, _, h, w = video_data.shape
        logger.info('input resolution: {}'.format((h, w)))
        target_h, target_w = int(h * self.upscale), int(w * self.upscale)   # adjust_resolution(h, w, up_scale=4)
        logger.info('target resolution: {}'.format((target_h, target_w)))

        pre_data = {'video_data': video_data, 'y': caption}
        pre_data['target_res'] = (target_h, target_w)

        total_noise_levels = 900
        setup_seed(666)

        with torch.no_grad():
            data_tensor = collate_fn(pre_data, 'cuda:0')
            output = self.model.test(data_tensor, total_noise_levels, steps=self.steps, \
                                solver_mode=self.solver_mode, guide_scale=self.guide_scale, \
                                max_chunk_len=self.max_chunk_len,
                                vae_decoder_chunk_size=self.vae_decoder_chunk_size
                                )

        output = tensor2vid(output)

        # Using color fix
        output = adain_color_fix(output, video_data)

        # save_video(output, self.result_dir, self.file_name, fps=input_fps)
        return output

class StarFR(STAR):
    """
    Feature Resetting trick played in STAR approach
    """
    def __init__(self,
                 result_dir='./results/',
                 model_path='',
                 solver_mode='fast',
                 steps=15,
                 guide_scale=7.5,
                 upscale=4,
                 device=torch.device(f'cuda:0')
                 ):
        super(StarFR, self).__init__(result_dir=result_dir,
                                     model_path=model_path,
                                     solver_mode=solver_mode,
                                     steps=steps,
                                     guide_scale=guide_scale,
                                     upscale=upscale,
                                     max_chunk_len=32,
                                     vae_decoder_chunk_size=3,
                                     device=device
                                     )
        print("STAR with Feature Resetting trick")
        model_cfg = EasyDict(__name__='model_cfg')
        model_cfg.model_path = self.model_path
        self.model = Vid2VidFr(model_cfg, device=device)
        if device == torch.device('cpu'):
            # Use fp32 in the cpu mode
            self.model.generator = self.model.generator.float()
            self.model.text_encoder = self.model.text_encoder.float()
            self.model.vae = self.model.vae.float()
        print("Setting the model to Video2Video with Feature Resetting")


    def enhance_dir_recur(self, input_frames_dir, prompt, in_win_size, in_win_step, out_win_step, out_win_overlap,
                          max_chunk_len=32, color_cor_method="wavelet", device=torch.device(f'cuda:0'), decoder_tile_size=64):
        """
        Enhance the images inside a directory, using an approach in a 'recursive' way.
        For the first and last window, use the 'same' padding strategy.
        For example, win_size = 5 and win_step = 3.
        Step1, pad two frames ahead. [f1, f1,] f1, f2, f3. Step2, f2, f3, f4, f5, [f5]
        """
        assert out_win_step >= out_win_overlap, "window step should be no smaller than window overlap."

        img_name_list = os.listdir(input_frames_dir)
        img_name_list.sort()

        n_frames = len(img_name_list)
        n_steps = int(ceil((n_frames-in_win_size) / in_win_step)) + 1
        for w_i in range(n_steps):
            w_start_idx = w_i * in_win_size if w_i != n_steps - 1 else n_frames - in_win_size
            w_end_idx = w_start_idx + in_win_size if w_i != n_steps - 1 else n_frames
            w_img_name_list = img_name_list[w_start_idx:w_end_idx]

            if w_i == 0:
                is_first_batch = True
                frames = load_frames(input_frames_dir, w_img_name_list)
                feature_map_prev = torch.zeros(0)
                z_prev = None
            else:
                is_first_batch = False
                frames = load_frames(input_frames_dir, w_img_name_list)

            video_sr, feature_map_prev, z_prev = self.enhance_a_video_fr(input_frames=frames,
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
                out_path = os.path.join(self.result_dir, w_img_name_list[i])
                cv2.imwrite(out_path, image_sr_list[i])

    def enhance_a_video_fr(self,
                           input_frames,
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
        """
        Enhance a video volume conditioned on previous volume's output feature maps.
        This trick aims at aligning the sr videos based on previous processed frames.
        """
        text = prompt
        logger.info('text: {}'.format(text))
        caption = text + self.model.positive_prompt

        video_data = preprocess(input_frames)
        _, _, h, w = video_data.shape
        logger.info('input resolution: {}'.format((h, w)))
        target_h, target_w = int(h * self.upscale), int(w * self.upscale)  # adjust_resolution(h, w, up_scale=4)
        logger.info('target resolution: {}'.format((target_h, target_w)))

        pre_data = {'video_data': video_data, 'y': caption}
        pre_data['target_res'] = (target_h, target_w)

        total_noise_levels = 900
        setup_seed(666)

        with torch.no_grad():
            data_tensor = collate_fn(pre_data, device=device)
            output, feature_map_prev, z_prev = self.model.infer(input=data_tensor,
                                                        feature_map_prev=feature_map_prev,
                                                        z_prev=z_prev,
                                                        is_first_batch=is_first_batch,
                                                        out_win_step=out_win_step,
                                                        out_win_overlap=out_win_overlap,
                                                        total_noise_levels=total_noise_levels,
                                                        steps=self.steps,
                                                        solver_mode=self.solver_mode,
                                                        guide_scale=self.guide_scale,
                                                        max_chunk_len=max_chunk_len,
                                                        decoder_tile_size=decoder_tile_size
                                                        )

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

def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument("--input_path", required=True, type=str, help="input video path")
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


def main():
    
    args = parse_args()

    input_path = args.input_path
    prompt = args.prompt
    model_path = args.model_path
    save_dir = args.save_dir
    upscale = args.upscale

    in_win_size = args.in_win_size
    in_win_step = args.in_win_step
    out_win_step = args.out_win_step
    out_win_overlap = args.out_win_overlap
    color_cor_method = args.color_cor_method

    steps = args.steps
    solver_mode = args.solver_mode
    guide_scale = args.cfg

    max_chunk_len = args.max_chunk_len
    device = torch.device(args.device)

    decoder_tile_size = args.decoder_tile_size

    assert solver_mode in ('fast', 'normal')
    assert in_win_size >= in_win_step

    starfr = StarFR(
        result_dir=save_dir,
        model_path=model_path,
        solver_mode=solver_mode,
        steps=steps,
        guide_scale=guide_scale,
        upscale=upscale,
        device=device)

    starfr.enhance_dir_recur(input_frames_dir=input_path,
                             prompt=prompt,
                             in_win_size=in_win_size,
                             in_win_step=in_win_step,
                             out_win_step=out_win_step,
                             out_win_overlap=out_win_overlap,
                             max_chunk_len=max_chunk_len,
                             color_cor_method=color_cor_method,
                             device=device,
                             decoder_tile_size=decoder_tile_size)


if __name__ == '__main__':
    main()
