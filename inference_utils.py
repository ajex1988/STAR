import os
import subprocess
import tempfile
import cv2
import torch
from PIL import Image
from typing import Mapping
from einops import rearrange
import numpy as np
import torchvision.transforms.functional as transforms_F
from video_to_video.utils.logger import get_logger

logger = get_logger()


def tensor2vid(video, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    mean = torch.tensor(mean, device=video.device).reshape(1, -1, 1, 1, 1)
    std = torch.tensor(std, device=video.device).reshape(1, -1, 1, 1, 1)
    video = video.mul_(std).add_(mean)
    video.clamp_(0, 1)
    video = video * 255.0
    images = rearrange(video, 'b c f h w -> b f h w c')[0]
    return images


def preprocess(input_frames):
    out_frame_list = []
    for pointer in range(len(input_frames)):
        frame = input_frames[pointer]
        frame = frame[:, :, ::-1]
        frame = Image.fromarray(frame.astype('uint8')).convert('RGB')
        frame = transforms_F.to_tensor(frame)
        out_frame_list.append(frame)
    out_frames = torch.stack(out_frame_list, dim=0)
    out_frames.clamp_(0, 1)
    mean = out_frames.new_tensor([0.5, 0.5, 0.5]).view(-1)
    std = out_frames.new_tensor([0.5, 0.5, 0.5]).view(-1)
    out_frames.sub_(mean.view(1, -1, 1, 1)).div_(std.view(1, -1, 1, 1))
    return out_frames


def adjust_resolution(h, w, up_scale):
    if h*up_scale < 720:
        up_s = 720/h
        target_h = int(up_s*h//2*2)
        target_w = int(up_s*w//2*2)
    elif h*w*up_scale*up_scale > 1280*2048:
        up_s = np.sqrt(1280*2048/(h*w))
        target_h = int(up_s*h//2*2)
        target_w = int(up_s*w//2*2)
    else:
        target_h = int(up_scale*h//2*2)
        target_w = int(up_scale*w//2*2)
    return (target_h, target_w)


def make_mask_cond(in_f_num, interp_f_num):
    mask_cond = []
    interp_cond = [-1 for _ in range(interp_f_num)]
    for i in range(in_f_num):
        mask_cond.append(i)
        if i != in_f_num - 1:
            mask_cond += interp_cond
    return mask_cond


def load_video(vid_path):
    capture = cv2.VideoCapture(vid_path)
    _fps = capture.get(cv2.CAP_PROP_FPS)
    _total_frame_num = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    pointer = 0
    frame_list = []
    stride = 1
    while len(frame_list) < _total_frame_num:
        ret, frame = capture.read()
        pointer += 1
        if (not ret) or (frame is None):
            break
        if pointer >= _total_frame_num + 1:
            break
        if pointer % stride == 0:
            frame_list.append(frame)
    capture.release()
    return frame_list, _fps


def save_video(video, save_dir, file_name, fps=16.0):
    output_path = os.path.join(save_dir, file_name)
    images = [(img.numpy()).astype('uint8') for img in video]
    temp_dir = tempfile.mkdtemp()
    
    for fid, frame in enumerate(images):
        tpth = os.path.join(temp_dir, '%06d.png' % (fid + 1))
        cv2.imwrite(tpth, frame[:, :, ::-1])
    
    tmp_path = os.path.join(save_dir, 'tmp.mp4')
    cmd = f'ffmpeg -y -f image2 -framerate {fps} -i {temp_dir}/%06d.png \
     -vcodec libx264 -preset ultrafast -crf 0 -pix_fmt yuv420p {tmp_path}'
    
    status, output = subprocess.getstatusoutput(cmd)
    if status != 0:
        logger.error('Save Video Error with {}'.format(output))
    
    os.system(f'rm -rf {temp_dir}')
    os.rename(tmp_path, output_path)

def load_frames(frame_dir, frame_name_list):
    frame_list = []
    for frame_name in frame_name_list:
        frame_path = os.path.join(frame_dir, frame_name)
        frame = cv2.imread(frame_path)
        frame_list.append(frame)
    return frame_list

def save_frames(frame_dir, frame_name_list, frame_list):
    assert len(frame_name_list) == len(frame_list), "Number of frame name list and frame list must match"
    for i in range(len(frame_name_list)):
        frame = frame_list[i]
        frame = frame[:,:,::-1]
        frame_name = frame_name_list[i]
        frame_path = os.path.join(frame_dir, frame_name)
        cv2.imwrite(frame_path, frame)

def load_latents(latents_dir, latents_name_list):
    latent_list = []
    for latents_name in latents_name_list:
        latents_path = os.path.join(latents_dir, latents_name)
        latent = np.load(latents_path)
        latent_list.append(latent)
    return latent_list

def collate_fn(data, device):
    """Prepare the input just before the forward function.
    This method will move the tensors to the right device.
    Usually this method does not need to be overridden.

    Args:
        data: The data out of the dataloader.
        device: The device to move data to.

    Returns: The processed data.

    """
    from torch.utils.data.dataloader import default_collate

    def get_class_name(obj):
        return obj.__class__.__name__

    if isinstance(data, dict) or isinstance(data, Mapping):
        return type(data)({
            k: collate_fn(v, device) if k != 'img_metas' else v
            for k, v in data.items()
        })
    elif isinstance(data, (tuple, list)):
        if 0 == len(data):
            return torch.Tensor([])
        if isinstance(data[0], (int, float)):
            return default_collate(data).to(device)
        else:
            return type(data)(collate_fn(v, device) for v in data)
    elif isinstance(data, np.ndarray):
        if data.dtype.type is np.str_:
            return data
        else:
            return collate_fn(torch.from_numpy(data), device)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (bytes, str, int, float, bool, type(None))):
        return data
    else:
        raise ValueError(f'Unsupported data type {type(data)}')