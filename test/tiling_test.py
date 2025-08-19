import os
import argparse
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import math
from video_to_video.modules.task_queue import build_up_blocks_task_queue

from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import TemporalDecoder

class TileHook():
    """
    Class for tiling algorithm test
    """
    def __init__(self, model, tile_size: int = 256, scale: int = 8, pad: int = 11):
        self.model = model
        self.tile_size = tile_size
        self.scale = scale
        self.pad = pad

    def __call__(self, x):
        return self._tiled_forward(x)

    def _crop_valid_region(self, tile, in_bbox, t_bbx, is_decoder=True, scale=8):
        """
        Handle cases where the height and width can not be divided by scale
        """
        padded_bbox = [i*scale if is_decoder else i//8 for i in in_bbox]
        margin = [t_bbx[i] - padded_bbox[i] for i in range(4)]
        return tile[:, :, margin[1]:tile.size(2)+margin[3], margin[0]:tile.size(3)+margin[2]] # TODO verify

    def _tiled_forward(self, x):
        x = x.detach() # avoid back propagation
        print(f"Tile input size : {x.shape}, tile size:{self.tile_size}, scale:{self.scale}, pad:{self.pad}")

        batch_size = x.shape[0]
        target_channels = self.model.out_channels
        n_frames = x.shape[2]
        h, w = x.shape[-2:]

        in_tiles, out_tiles = self._split_tiles(h=h, w=w, scale=self.scale)
        n_tiles = len(in_tiles)

        result = torch.zeros((batch_size, target_channels, n_frames, h * self.scale , w * self.scale), device=x.get_device(), requires_grad=False)
        for i in range(n_tiles):
            in_bbox = in_tiles[i]
            tile = x[:, :, :, in_bbox[1]:in_bbox[3], in_bbox[0]:in_bbox[2]]
            tile = self.model(tile)
            result[:, :, :, out_tiles[i][1]:out_tiles[i][3], out_tiles[i][0]:out_tiles[i][2]] = self._crop_valid_region(tile, in_bbox=in_tiles[i], t_bbx=out_tiles[i], scale=self.scale)
            del tile

        return result


    def _get_best_tile_size(self, lowerbound, upperbound):
        """
        Get the best tile size for GPU memory
        """
        divider = 32
        while divider >= 2:
            remainer = lowerbound % divider
            if remainer == 0:
                return lowerbound
            candidate = lowerbound - remainer + divider
            if candidate <= upperbound:
                return candidate
            divider //= 2
        return lowerbound

    def _split_tiles(self, h, w, scale=8):
        """
        Split the feature map into tiles.
        @param h, w: height and width of the feature map
        @param scale: scaling factor for h and w.
        @return: tile_in_bboxes, tile_out_bboxes
        """
        tile_size = self.tile_size
        pad = self.pad
        num_height_tiles = math.ceil((h - 2 * pad) / tile_size)
        num_width_tiles = math.ceil((w - 2 * pad) / tile_size)
        # If any of the numbers are 0, we let it be 1
        # This is to deal with long and thin images
        num_height_tiles = max(num_height_tiles, 1)
        num_width_tiles = max(num_width_tiles, 1)

        # Suggestions from https://github.com/Kahsolt: auto shrink the tile size
        real_tile_height = math.ceil((h - 2 * pad) / num_height_tiles)
        real_tile_width = math.ceil((w - 2 * pad) / num_width_tiles)
        real_tile_height = self._get_best_tile_size(real_tile_height, tile_size)
        real_tile_width = self._get_best_tile_size(real_tile_width, tile_size)

        print(
            f'[Tiled VAE]: split to {num_height_tiles}x{num_width_tiles} = {num_height_tiles * num_width_tiles} tiles. ' +
            f'Optimal tile size {real_tile_width}x{real_tile_height}, original tile size {tile_size}x{tile_size}')

        tile_input_bboxes, tile_output_bboxes = [], []
        for i in range(num_height_tiles):
            for j in range(num_width_tiles):
                # bbox: [x0, y0, x1, y1]
                input_bbox = [
                    pad + j * real_tile_width,
                    pad + i * real_tile_height,
                    min(pad + (j + 1) * real_tile_width, w),
                    min(pad + (i + 1) * real_tile_height, h),
                ]

                output_bbox = [
                    input_bbox[0] if input_bbox[0] > pad else 0,
                    input_bbox[1] if input_bbox[1] > pad else 0,
                    input_bbox[2] if input_bbox[2] < w - pad else w,
                    input_bbox[3] if input_bbox[3] < h - pad else h,
                ]

                # scale to get the fianl output bbox
                output_bbox = [i*scale for i in output_bbox]
                tile_output_bboxes.append(output_bbox)

                tile_input_bboxes.append([
                    max(0, input_bbox[0] - pad),
                    max(0, input_bbox[1] - pad),
                    min(w, input_bbox[2] + pad),
                    min(h, input_bbox[3] + pad),
                ])

        return tile_input_bboxes, tile_output_bboxes

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


def get_input_feature_map(seed:int=10086, h:int=128, w:int=128, nc:int=512, n_frames:int=6, device:torch.device=torch.device("cuda:0") ):
    torch.manual_seed(seed)
    input_shape = (n_frames,nc,h,w)
    input_tensor = torch.rand(input_shape)
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


    def forward_task_queue(self, x):
        pass

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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_tensor = get_input_feature_map(h=height, w=width, n_frames=num_frames, nc=num_channels, device=device)
    print(f"Input shape: {input_tensor.shape}")

    temporal_decoder = TemporalDecoder()

    task_queue = []
    build_up_blocks_task_queue(queue=task_queue,
                                            block=temporal_decoder.up_blocks)
    print(task_queue)


def test_tiling_with_recur():
    args = parse_args()

def test_no_tiling_recur():
    args = parse_args()


def main():
    # test_no_tiling()
    # test_tiling()
    test_task_queue_tiling()


if __name__ == "__main__":
    main()