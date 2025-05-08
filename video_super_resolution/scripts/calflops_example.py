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
from video_to_video.video_to_video_model import VideoToVideo_sr
from video_to_video.utils.seed import setup_seed
from video_to_video.utils.logger import get_logger
from video_super_resolution.color_fix import adain_color_fix

from inference_utils import *

from torch.profiler import profile, record_function, ProfilerActivity

import copy

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
                 vae_decoder_chunk_size=3
                 ):
        self.model_path=model_path
        logger.info('checkpoint_path: {}'.format(self.model_path))

        self.result_dir = result_dir
        os.makedirs(self.result_dir, exist_ok=True)

        model_cfg = EasyDict(__name__='model_cfg')
        model_cfg.model_path = self.model_path
        self.model = VideoToVideo_sr(model_cfg)

        steps = 15 if solver_mode == 'fast' else steps
        self.solver_mode=solver_mode
        self.steps=steps
        self.guide_scale=guide_scale
        self.upscale = upscale
        self.max_chunk_len=max_chunk_len
        self.vae_decoder_chunk_size=vae_decoder_chunk_size

    def enhance_dir(self, input_frames_dir, input_fps, prompt, in_win_size, profiling_log_dir='./'):
        """
        Enhance the images inside a directory.
        Assume that the dir only contains images of the same format, and the name indicates the frame order.
        """
        img_name_list = os.listdir(input_frames_dir)
        img_name_list.sort()

        n_frames = len(img_name_list)
        assert n_frames >= in_win_size, "Input frame size should be larger than window size."

        w_img_name_list = img_name_list[:in_win_size]
        frames = load_frames(input_frames_dir, w_img_name_list)
        video_sr = self.enhance_a_video(input_frames=frames, input_fps=input_fps, prompt=prompt, profiling_log_dir=profiling_log_dir)
        image_sr_list = [(img.numpy()).astype('uint8')[:, :, ::-1] for img in video_sr]
        save_name_list = w_img_name_list
        save_frame_list = image_sr_list
        for i in range(len(save_name_list)):
            out_path = os.path.join(self.result_dir, save_name_list[i])
            cv2.imwrite(out_path, save_frame_list[i])

    def cal_flops(self):
        from calflops import calculate_flops
        flops_vae, macs_vae, params_vae = calculate_flops(model=self.model.vae,
                                              input_shape=(1, 3, 1936, 2944))
        print(self.model.device)
        inputs = {}
        inputs['x'] = torch.zeros(1, 4, 1, 32, 32).to(self.model.device)
        inputs['t'] = torch.zeros(1, ).to(self.model.device)
        inputs['y'] = torch.zeros(1, 77, 1024).to(self.model.device)
        inputs['hint'] = torch.zeros(1, 4, 1, 32, 32).to(self.model.device)
        # inputs['x'] = torch.zeros(1, 4, 1, 242, 368).to(self.model.device)
        # inputs['t'] = torch.zeros(1,).to(self.model.device)
        # inputs['y'] = torch.zeros(1, 77, 1024).to(self.model.device)
        # inputs['hint'] = torch.zeros(1, 4, 1, 242, 368).to(self.model.device)


        flops_dm, macs_dm, params_dm = calculate_flops(model=self.model.generator,
                                                       kwargs=inputs)

        print(f"VAE FLOPs {flops_vae}")
        print(f"DM FLOPSs: {flops_dm}")
        print(f"Total FLOPS: {flops_dm+flops_vae}")

    def cal_flops2(self):
        from torchtnt.utils.flops import FlopTensorDispatchMode
        dm = self.model.generator.float()
        dm_input = {}
        dm_input['x'] = torch.zeros(1, 4, 1, 242, 368).to(self.model.device)
        dm_input['t'] = torch.zeros(1,).to(self.model.device)
        dm_input['y'] = torch.zeros(1, 77, 1024).to(self.model.device)
        dm_input['hint'] = torch.zeros(1, 4, 1, 242, 368).to(self.model.device)
        sum = 0
        with FlopTensorDispatchMode(dm) as ftdm:
            result = dm(**dm_input)
            flops_dm = copy.deepcopy(ftdm.flop_counts)
            for t in flops_dm:
                for tt in flops_dm[t]:
                    sum += flops_dm[t][tt]
            print(f"DM FLOPS:{sum}")



    def enhance_a_video(self, input_frames, input_fps, prompt, profiling_log_dir='./'):
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

        # Profiling parameters
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        with profile(activities=activities,
                     profile_memory=True,
                     record_shapes=True,
                     on_trace_ready=torch.profiler.tensorboard_trace_handler(profiling_log_dir)) as profiling:
            with torch.no_grad():
                flop_counter_vae = FlopCounterMode(self.model.vae, display=False)
                flop_counter_dm = FlopCounterMode(self.model.diffusion, display=False)
                flop_counter_generator = FlopCounterMode(self.model.generator, display=False)
                flop_counter_text_encoder = FlopCounterMode(self.model.text_encoder, display=False)
                with flop_counter_vae, flop_counter_dm, flop_counter_generator, flop_counter_text_encoder:

                    data_tensor = collate_fn(pre_data, 'cuda:0')
                    output = self.model.test(data_tensor, total_noise_levels, steps=self.steps, \
                                            solver_mode=self.solver_mode, guide_scale=self.guide_scale, \
                                            max_chunk_len=self.max_chunk_len,
                                            vae_decoder_chunk_size=self.vae_decoder_chunk_size
                                            )
                    flops_vae = flop_counter_vae.get_total_flops()
                    flops_dm = flop_counter_dm.get_total_flops()
                    flops_generator = flop_counter_generator.get_total_flops()
                    flops_text_encoder = flop_counter_text_encoder.get_total_flops()

                    total_flops = flops_vae + flops_dm + flops_generator + flops_text_encoder

                    print(f"FLOPS VAE:{flops_vae}; FLOPS DM:{flops_dm}; FLOPS GENERATOR:{flops_generator}; FLOPS TEXT ENCODER:{flops_text_encoder}; Total FLOPs: {total_flops}")

        print(profiling.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=50))



        output = tensor2vid(output)

        # Using color fix
        output = adain_color_fix(output, video_data)

        # Release cached gpu memory
        torch.cuda.empty_cache()
        # save_video(output, self.result_dir, self.file_name, fps=input_fps)
        return output
    

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

    parser.add_argument("--input_fps", type=int, default=30)
    parser.add_argument("--in_win_size", type=int, default=12, help="sliding window size")

    parser.add_argument("--vae_decoder_chunk_size",type=int,default=2,help="vae_decoder_chunk_size")

    parser.add_argument("--positive_prompt", type=str, default=None, help="prompt")
    parser.add_argument("--negative_prompt", type=str, default=None, help="prompt")

    parser.add_argument("--profiling_log_dir", type=str, default='./', help="profiling_log_dir")
    return parser.parse_args()


def main():
    
    args = parse_args()

    input_path = args.input_path
    prompt = args.prompt
    model_path = args.model_path
    save_dir = args.save_dir
    upscale = args.upscale
    max_chunk_len = args.max_chunk_len

    in_win_size = args.in_win_size

    vae_decoder_chunk_size = args.vae_decoder_chunk_size

    steps = args.steps
    solver_mode = args.solver_mode
    guide_scale = args.cfg

    profiling_log_dir = args.profiling_log_dir

    assert solver_mode in ('fast', 'normal')

    star = STAR(
                result_dir=save_dir,
                model_path=model_path,
                solver_mode=solver_mode,
                steps=steps,
                guide_scale=guide_scale,
                upscale=upscale,
                max_chunk_len=max_chunk_len,
                vae_decoder_chunk_size=vae_decoder_chunk_size,
                )

    star.cal_flops2()


if __name__ == '__main__':
    main()
