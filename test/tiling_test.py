import os
import argparse
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import math
import copy
from video_to_video.modules.task_queue import build_up_blocks_task_queue
from video_to_video.modules.autoencoder_kl_temporal_decoder_feature_resetting import TileHook, GroupNormParam

from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import TemporalDecoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--num_frames", type=int, default=6)
    parser.add_argument("--num_channels", type=int, default=512)
    parser.add_argument("--num_residual_blocks", type=int, default=3)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--overlap", type=int, default=2)

    parser.add_argument("--out_dir", type=str, default="./output")

    args = parser.parse_args()
    return args


def get_input_feature_map(seed:int=10086, h:int=128, w:int=128, nc:int=512, n_frames:int=6, device:torch.device=torch.device("cuda:0"), requires_grad=False ):
    torch.manual_seed(seed)
    input_shape = (n_frames,nc,h,w)
    input_tensor = torch.rand(input_shape,requires_grad=requires_grad)
    input_tensor = input_tensor.to(device)
    return input_tensor


def save_feature_map(feature_map:torch.Tensor, path:str, frm_idx:int=0, n_channels:int=3):
    feature_map = feature_map.detach().cpu().numpy()
    feature_map = np.clip(feature_map * 255, 0, 255).astype(np.uint8)
    feature_map_img = feature_map[frm_idx,:n_channels, :, :]
    feature_map_img = np.transpose(feature_map_img, (1,2,0))
    feature_map_img = Image.fromarray(feature_map_img)
    feature_map_img.save(path)


class UpModule(nn.Module):
    def __init__(self, up_blocks):
        super(UpModule, self).__init__()
        self.up_blocks = up_blocks

    def forward(self, x, path_prefix):
        image_only_indicator = torch.zeros(1, x.shape[0], dtype=x.dtype, device=x.device)
        for i, block in enumerate(self.up_blocks):
            feat_map_path = path_prefix + f"_{i}.png"
            save_feature_map(x, feat_map_path)
            x = block(x, image_only_indicator=image_only_indicator)
        feat_map_path = path_prefix + f"_out.png"
        save_feature_map(x, feat_map_path)
        return x


class UpModuleTile(nn.Module):
    def __init__(self, up_blocks, tile_size=64, upscale=8, pad=11):
        super(UpModuleTile, self).__init__()
        self.up_blocks = up_blocks
        self.tile_size = tile_size
        self.upscale = upscale
        self.pad = pad

    def forward(self, x, path_prefix):
        image_only_indicator = torch.zeros(1, x.shape[0], dtype=x.dtype, device=x.device)
        tile_hook = TileHook(model=None,tile_size=self.tile_size,scale=self.upscale,pad=self.pad)
        nf, nc, feat_h, feat_w = x.shape
        in_bboxes, out_bboxes = tile_hook._split_tiles(h=feat_h, w=feat_w,scale=8)
        result = torch.zeros((nf, nc, feat_h*self.upscale, feat_w*self.upscale), device=x.device)
        n_tiles = len(in_bboxes)

        for i in range(n_tiles):
            in_bbx = in_bboxes[i]
            out_bbx = out_bboxes[i]
            tile = x[:,:,in_bbx[1]:in_bbx[3],in_bbx[0]:in_bbx[2]]
            for j, block in enumerate(self.up_blocks):
                feat_map_path = path_prefix + f"_tile_{i}_layer_{j}.png"
                save_feature_map(tile, feat_map_path)
                tile = block(tile, image_only_indicator=image_only_indicator)
            result[out_bbx[1]:out_bbx[3],out_bbx[0]:out_bbx[2]] = tile_hook._crop_valid_region(tile=tile,
                                                                                               in_bbox=in_bbx,
                                                                                               t_bbx=out_bbx,
                                                                                               scale=self.upscale)
            feat_map_path = path_prefix + f"_tile_{i}_out.png"
            save_feature_map(tile, feat_map_path)
        feat_map_path = path_prefix + f"_out.png"
        save_feature_map(result, feat_map_path)
        return x


class UpModuleTileTaskQueue(nn.Module):
    def __init__(self, up_blocks, tile_size=64, upscale=8, pad=11):
        super(UpModuleTileTaskQueue, self).__init__()
        self.up_blocks = up_blocks
        self.tile_size = tile_size
        self.upscale = upscale
        self.pad = pad

    def forward(self, x, path_prefix):
        x = self.forward_task_queue(x)
        feat_map_path = path_prefix + "_tile_task_queue.png"
        save_feature_map(x, feat_map_path)
        return x


    def forward_task_queue(self, x):
        device = next(self.up_blocks.parameters()).device
        dtype = next(self.up_blocks.parameters()).dtype

        tile_helper = TileHook(model=None,tile_size=self.tile_size,scale=self.upscale,pad=self.pad)

        nf, nc, feat_h, feat_w = x.shape
        in_bboxes, out_bboxes = tile_helper._split_tiles(h=feat_h, w=feat_w,scale=8)
        result = torch.zeros((nf, nc//4, feat_h * self.upscale, feat_w * self.upscale), device=x.device)
        n_tiles = len(in_bboxes)
        tile_completed = 0

        tiles = []
        for in_bbox in in_bboxes:
            tile = x[:, :, in_bbox[1]:in_bbox[3], in_bbox[0]:in_bbox[2]].cpu()
            tiles.append(tile)

        task_queue = build_up_blocks_task_queue(block=self.up_blocks)
        task_queues = [copy.deepcopy(task_queue) for _ in range(n_tiles)]

        while True:
            group_norm_helper = GroupNormParam()
            for i in range(n_tiles):
                in_bbx = in_bboxes[i]
                out_bbx = out_bboxes[i]
                tile = tiles[i].to(device=device)

                task_queue = task_queues[i]


                while len(task_queue) > 0:
                    task = task_queue.pop(0)

                    if task[0] == "pre_norm":
                        group_norm_helper.add_tile(tile=tile, layer=task[1])
                        break
                    elif task[0] == "store_res":
                        task_id = 0
                        res = task[1](tile)
                        while task_queue[task_id][0] != "add_res":
                            task_id += 1
                        task_queue[task_id][1] = res
                    elif task[0] == "store_x_spatial":
                        task_id = 0
                        x_spatial = task[1](tile)
                        while task_queue[task_id][0] != "alpha_blend":
                            task_id += 1
                        task_queue[task_id] = (task_queue[task_id][0], task_queue[task_id][1], x_spatial)
                    elif task[0] == "add_res":
                        tile += task[1]
                    elif task[0] == "alpha_blend":
                        tile = task[1](task[2], tile)
                    else:
                        # print(f"{task[0]}")
                        tile = task[1](tile)

                if len(task_queue) == 0:
                    # finished current tile
                    tile_completed += 1
                    result[:,:,out_bbx[1]:out_bbx[3], out_bbx[0]:out_bbx[2]] = tile_helper._crop_valid_region(tile=tile,
                                                                                                          in_bbox=in_bbx,
                                                                                                          t_bbx=out_bbx,
                                                                                                          scale=self.upscale)
                    del tile
                else:
                    # Prepare to apply group norm
                    tiles[i] = tile.cpu()
                    del tile

            if tile_completed == n_tiles:
                break

            # insert the group norm task to the head of each task queue
            group_norm_func = group_norm_helper.summary()
            if group_norm_func is not None:
                for i in range(n_tiles):
                    task_queues[i].insert(0, ("apply_norm", group_norm_func))

        return result.to(dtype=dtype)


def test_tiling():
    """
    Tiling without recurrent mechanism
    """
    args = parse_args()
    out_dir = args.out_dir
    height = args.height
    width = args.width
    num_frames = args.num_frames
    num_channels = args.num_channels

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    temporal_decoder = TemporalDecoder()
    up_blocks = temporal_decoder.up_blocks
    up_blocks = up_blocks.to(device=device)

    input_tensor = get_input_feature_map(h=height, w=width, n_frames=num_frames, nc=num_channels, device=device)
    print(f"Input shape: {input_tensor.shape}")
    up_module = UpModuleTile(up_blocks)

    output_tensor = up_module(input_tensor, out_dir)

    print(f"Output shape: {output_tensor.shape}")


def test_no_tiling():
    """
    No tiling, no recurrent mechanism
    """
    print("Testing no tiling and no recurrent mechanism")
    args = parse_args()
    out_dir = args.out_dir
    height = args.height
    width = args.width
    num_frames = args.num_frames
    num_channels = args.num_channels

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    temporal_decoder = TemporalDecoder()
    up_blocks = temporal_decoder.up_blocks
    up_blocks = up_blocks.to(device=device)

    input_tensor = get_input_feature_map(h=height,w=width,n_frames=num_frames, nc=num_channels, device=device)
    print(f"Input shape: {input_tensor.shape}")
    up_module = UpModule(up_blocks)

    output_tensor = up_module(input_tensor, out_dir)

    print(f"Output shape: {output_tensor.shape}")


def test_task_queue_tiling():
    print("Testing task queue tiling and no recurrent mechanism")
    args = parse_args()
    out_dir = args.out_dir
    height = args.height
    width = args.width
    num_frames = args.num_frames
    num_channels = args.num_channels

    with torch.no_grad():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        input_tensor = get_input_feature_map(h=height, w=width, n_frames=num_frames, nc=num_channels, device=device)
        print(f"Input shape: {input_tensor.shape}")

        temporal_decoder = TemporalDecoder().to(device=device)

        up_module = UpModuleTileTaskQueue(up_blocks=temporal_decoder.up_blocks)
        output_tensor = up_module(input_tensor, out_dir)
        print(f"Output shape: {output_tensor.shape}")



def test_tiling_with_recur():
    args = parse_args()

def test_no_tiling_recur():
    args = parse_args()


def main():
    test_no_tiling()
    # test_tiling()
    test_task_queue_tiling()


if __name__ == "__main__":
    main()