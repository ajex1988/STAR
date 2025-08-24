import torch
import math
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch.nn.functional as F

from diffusers import AutoencoderKLTemporalDecoder
from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import TemporalDecoder
from diffusers.models.autoencoders.vae import DecoderOutput

from diffusers.utils.import_utils import is_torch_version
from diffusers.configuration_utils import register_to_config
from diffusers.utils.accelerate_utils import apply_forward_hook


class TemporalDecoderFeatureResetting(TemporalDecoder):
    """
    Temporal decoder with feature resetting
    """
    def __init__(
                self,
                in_channels: int = 4,
                out_channels: int = 3,
                block_out_channels: Tuple[int] = (128, 256, 512, 512),
                layers_per_block: int = 2,):
        super(TemporalDecoderFeatureResetting, self).__init__(in_channels=in_channels,
                                                              out_channels=out_channels,
                                                              block_out_channels=block_out_channels,
                                                              layers_per_block=layers_per_block)

    def forward(self,
                sample: torch.Tensor,
                feature_map_prev: Dict,
                image_only_indicator: torch.Tensor,
                frame_overlap_num: int = 1,
                is_first_batch: bool = False,
                num_frames: int = 1):
        """
        feature_map_prev: The feature map of the previous batch. For layer l(conv_in, mid_block, .etc)
                          feature_map_prev[l] indicates the feature map after that layer.
        """

        feature_map_cur = {}
        feature_map_cur["sample"] = sample[-frame_overlap_num:, :, :, :].clone()

        if is_first_batch:
            sample = self.conv_in(sample)
        else:
            sample[:frame_overlap_num, :, :, :] = feature_map_prev["sample"]
            sample = self.conv_in(sample)

        feature_map_cur["conv_in"] = sample[-frame_overlap_num:, :, :, :].clone()

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block),
                    sample,
                    image_only_indicator,
                    use_reentrant=False,
                )
                sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        sample,
                        image_only_indicator,
                        use_reentrant=False,
                    )
            else:
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block),
                    sample,
                    image_only_indicator,
                )
                sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        sample,
                        image_only_indicator,
                    )
        else:
            # middle
            if is_first_batch:
                sample = self.mid_block(sample, image_only_indicator=image_only_indicator)
            else:
                sample[:frame_overlap_num, :, :, :] = feature_map_prev["conv_in"]
                sample = self.mid_block(sample, image_only_indicator=image_only_indicator)
            sample = sample.to(upscale_dtype)

            feature_map_cur["mid_block"] = sample[-frame_overlap_num:, :, :, :].clone()

            # up
            for i, up_block in enumerate(self.up_blocks):
                if i == 0:
                    if is_first_batch:
                        sample = up_block(sample, image_only_indicator=image_only_indicator)
                    else:
                        sample[:frame_overlap_num, :, :, :] = feature_map_prev["mid_block"]
                        sample = up_block(sample, image_only_indicator=image_only_indicator)
                    feature_map_cur["up_block"] = [sample[-frame_overlap_num:, :, :, :].clone()]
                else:
                    if torch.cuda.device_count() == 4:
                        if i == 1:
                            up_block = up_block.to('cuda:0')
                            sample = sample.to('cuda:0')
                        elif i == 2:
                            up_block = up_block.to('cuda:1')
                            sample = sample.to('cuda:1')
                        elif i == 3:
                            up_block = up_block.to('cuda:2')
                            sample = sample.to('cuda:2')
                    if is_first_batch:
                        if sample.device.type == 'cuda:2':
                            sample = sample.to(torch.float32)
                        sample = up_block(sample, image_only_indicator=image_only_indicator)
                        if sample.device.type == 'cuda:2':
                            sample = sample.to(torch.float16)
                    else:
                        sample[:frame_overlap_num, :, :, :] = feature_map_prev["up_block"][i - 1]
                        if sample.device.type == 'cuda:2':
                            sample = sample.to(torch.float32)
                        sample = up_block(sample, image_only_indicator=image_only_indicator)
                        if sample.device.type == 'cuda:2':
                            sample = sample.to(torch.float16)
                    feature_map_cur["up_block"].append(sample[-frame_overlap_num:, :, :, :].clone())
                # if is_first_batch:
                #     sample = up_block(sample, image_only_indicator=image_only_indicator)
                # else:
                #     if i == 0:
                #         sample[:frame_overlap_num, :, :, :] = feature_map_prev["mid_block"][-frame_overlap_num:, :, :, :]
                #         sample = up_block(sample, image_only_indicator=image_only_indicator)
                #         feature_map_cur["up_block"] = [sample.clone()]
                #     else:
                #         sample[:frame_overlap_num, :, :, :] = feature_map_prev["up_block"][i-1][-frame_overlap_num:, :, :, :]
                #         sample = up_block(sample, image_only_indicator=image_only_indicator)
                #         feature_map_cur["up_block"].append(sample.clone())

        # post-process
        if torch.cuda.device_count() == 4:
            sample = sample.to('cuda:2')
            sample = self.conv_norm_out.to('cuda:2')(sample)
            sample = self.conv_act.to('cuda:2')(sample)
        else:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)

        feature_map_cur["conv_act"] = sample[-frame_overlap_num:, :, :, :].clone()
        if is_first_batch:
            if torch.cuda.device_count() == 4:
                sample = self.conv_out.to('cuda:2')(sample)
            else:
                sample = self.conv_out(sample)
        else:
            sample[:frame_overlap_num, :, :, :] = feature_map_prev["conv_act"]
            if torch.cuda.device_count() == 4:
                sample = self.conv_out.to('cuda:2')(sample)
            else:
                sample = self.conv_out(sample)

        batch_frames, channels, height, width = sample.shape
        batch_size = batch_frames // num_frames
        sample = sample[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)

        feature_map_cur["conv_out"] = sample[:, :, -frame_overlap_num:, :,:].clone()
        if is_first_batch:
            if torch.cuda.device_count() == 4:
                sample = self.time_conv_out.to('cuda:2')(sample)
            else:
                sample = self.time_conv_out(sample)
        else:
            sample[:, :, :frame_overlap_num, :, :] = feature_map_prev["conv_out"]
            if torch.cuda.device_count() == 4:
                sample = self.time_conv_out.to('cuda:2')(sample)
            else:
                sample = self.time_conv_out(sample)

        sample = sample.permute(0, 2, 1, 3, 4).reshape(batch_frames, channels, height, width)

        # torch.cuda.empty_cache()

        return sample, feature_map_cur


class TiledTemporalDecoderFeatureResetting(TemporalDecoder):
    """
    Tiled version of TemporalDecoderFeatureResetting
    """

    def __init__(
            self,
            in_channels: int = 4,
            out_channels: int = 3,
            block_out_channels: Tuple[int] = (128, 256, 512, 512),
            layers_per_block: int = 2,
            tile_size: int = 256,
            pad: int = 11):
        super(TiledTemporalDecoderFeatureResetting, self).__init__(in_channels=in_channels,
                                                                   out_channels=out_channels,
                                                                   block_out_channels=block_out_channels,
                                                                   layers_per_block=layers_per_block)
        self.tile_size = tile_size
        self.pad = pad

    def forward(self,
                sample: torch.Tensor,
                feature_map_prev: Dict,
                image_only_indicator: torch.Tensor,
                frame_overlap_num: int = 1,
                is_first_batch: bool = False,
                num_frames: int = 1,
                tile_size: int = 64,
                ) -> torch.Tensor:
        """
        To enable more No. of frames run in the decoder, utilize multiple GPUs:
        1. Store the feature map reloading in GPU No.3
        """
        fm_device = torch.device('cuda:3')
        device_0 = torch.device('cuda:1')
        device_1 = torch.device('cuda:2')
        device_2 = torch.device('cpu')
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus < 4:
                raise RuntimeError(f"The No. of GPUs needed is 4, now only has {num_gpus}")

        # start from the 1st device
        sample = sample.to(device_0)

        feature_map_cur = {}
        feature_map_cur["sample"] = sample[-frame_overlap_num:, :, :, :].clone().to(fm_device)

        if is_first_batch:
            sample = self.conv_in.to(device_0)(sample)
        else:
            sample[:frame_overlap_num, :, :, :] = feature_map_prev["sample"].to(device_0)
            sample = self.conv_in.to(device_0)(sample)

        feature_map_cur["conv_in"] = sample[-frame_overlap_num:, :, :, :].clone().to(fm_device)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block),
                    sample,
                    image_only_indicator,
                    use_reentrant=False,
                )
                sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        sample,
                        image_only_indicator,
                        use_reentrant=False,
                    )
            else:
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block),
                    sample,
                    image_only_indicator,
                )
                sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        sample,
                        image_only_indicator,
                    )
        else:
            # middle
            if is_first_batch:
                sample = self.mid_block.to(device_0)(sample, image_only_indicator=image_only_indicator)
            else:
                sample[:frame_overlap_num, :, :, :] = feature_map_prev["conv_in"].to(device_0)
                sample = self.mid_block.to(device_0)(sample, image_only_indicator=image_only_indicator)
            sample = sample.to(upscale_dtype)

            feature_map_cur["mid_block"] = sample[-frame_overlap_num:, :, :, :].clone().to(fm_device)

            # up
            # Since up layers consumes most of the memory, we use tiling to support large input feature map.
            up_scale = 8
            tile_hook = TileHook(model=None,
                                 tile_size=tile_size,
                                 scale=up_scale,
                                 pad=11)
            feat_h, feat_w = sample.shape[-2:]
            in_bboxes, out_bboxes = tile_hook._split_tiles(h=feat_h, w=feat_w, scale=up_scale)  # 8x
            in_bboxes_2x, out_bboxes_2x = tile_hook._split_tiles(h=feat_h, w=feat_w, scale=2)   # 2x
            in_bboxes_4x, out_bboxes_4x = tile_hook._split_tiles(h=feat_h, w=feat_w, scale=4)   # 4x
            n_tiles = len(in_bboxes)

            bs, nc, h, w = sample.shape
            result = torch.zeros((bs, nc//4, h * up_scale, w * up_scale),
                                 device=sample.get_device(), requires_grad=False)
            for i in range(n_tiles):
                in_bbox = in_bboxes[i]
                tile = sample[:, :, in_bbox[1]:in_bbox[3], in_bbox[0]:in_bbox[2]]
                for j, up_block in enumerate(self.up_blocks):
                    if j == 0:
                        if is_first_batch:
                            tile = up_block.to(device_0)(tile, image_only_indicator=image_only_indicator)
                        else:
                            tile[:frame_overlap_num, :, :, :] = feature_map_prev["mid_block"][:, :, in_bbox[1]:in_bbox[3], in_bbox[0]:in_bbox[2]].to(device_0)
                            tile = up_block.to(device_0)(tile, image_only_indicator=image_only_indicator)
                        if i == 0:
                            feature_map_cur["up_block"] = [torch.zeros((frame_overlap_num, tile.shape[1], feat_h*2, feat_w*2),device=fm_device)]
                        feature_map_cur["up_block"][0][:,:,out_bboxes_2x[i][1]:out_bboxes_2x[i][3], out_bboxes_2x[i][0]:out_bboxes_2x[i][2]] = tile_hook._crop_valid_region(tile=tile[-frame_overlap_num:, :, :, :].clone().to(fm_device),
                                                                                                                                                                             in_bbox=in_bboxes_2x[i],
                                                                                                                                                                             t_bbx=out_bboxes_2x[i],
                                                                                                                                                                             scale=2)
                    elif j == 1:
                        if is_first_batch:
                            tile = up_block.to(device_0)(tile, image_only_indicator=image_only_indicator)
                        else:
                            tile[:frame_overlap_num, :, :, :] = feature_map_prev["up_block"][j - 1][:, :, 2*in_bboxes[i][1]:2*in_bboxes[i][3], 2*in_bboxes[i][0]:2*in_bboxes[i][2]].to(device_0)
                            tile = up_block.to(device_0)(tile, image_only_indicator=image_only_indicator)
                        if i == 0:
                            feature_map_cur["up_block"].append(torch.zeros((frame_overlap_num, tile.shape[1], feat_h*4, feat_w*4)))
                        feature_map_cur["up_block"][1][:,:,out_bboxes_4x[i][1]:out_bboxes_4x[i][3], out_bboxes_4x[i][0]:out_bboxes_4x[i][2]] = tile_hook._crop_valid_region(tile=tile[-frame_overlap_num:, :, :, :].clone().to(fm_device),
                                                                                                                                                                             in_bbox=in_bboxes_4x[i],
                                                                                                                                                                             t_bbx=out_bboxes_4x[i],
                                                                                                                                                                             scale=4)
                    elif j == 2:
                        if is_first_batch:
                            tile = up_block.to(device_0)(tile, image_only_indicator=image_only_indicator)
                        else:
                            tile[:frame_overlap_num, :, :, :] = feature_map_prev["up_block"][j - 1][:, :, 4*in_bboxes[i][1]:4*in_bboxes[i][3], 4*in_bboxes[i][0]:4*in_bboxes[i][2]].to(device_0)
                            tile = up_block.to(device_0)(tile, image_only_indicator=image_only_indicator)
                        if i == 0:
                            feature_map_cur["up_block"].append(torch.zeros((frame_overlap_num, tile.shape[1], feat_h*8, feat_w*8)))
                        feature_map_cur["up_block"][2][:,:,out_bboxes[i][1]:out_bboxes[i][3], out_bboxes[i][0]:out_bboxes[i][2]] = tile_hook._crop_valid_region(tile=tile[-frame_overlap_num:, :, :, :].clone().to(fm_device),
                                                                                                                                                                 in_bbox=in_bbox,
                                                                                                                                                                 t_bbx=out_bboxes[i],
                                                                                                                                                                 scale=8)
                    else:
                        tile = tile.to(device_1)
                        if is_first_batch:
                            tile = up_block.to(device_1)(tile, image_only_indicator=image_only_indicator)
                        else:
                            tile[:frame_overlap_num, :, :, :] = feature_map_prev["up_block"][j - 1][:, :,
                                                                8*in_bbox[1]:8*in_bbox[3], 8*in_bbox[0]:8*in_bbox[2]].to(device_1)
                            tile = up_block.to(device_1)(tile, image_only_indicator=image_only_indicator)
                        if i == 0:
                            feature_map_cur["up_block"].append(torch.zeros((frame_overlap_num, tile.shape[1], feat_h * 8, feat_w * 8)))
                        feature_map_cur["up_block"][3][:, :, out_bboxes[i][1]:out_bboxes[i][3],out_bboxes[i][0]:out_bboxes[i][2]] = tile_hook._crop_valid_region(tile=tile[-frame_overlap_num:, :, :, :].clone().to(fm_device),
                                                                                                                                                                  in_bbox=in_bbox,
                                                                                                                                                                  t_bbx=out_bboxes[i],
                                                                                                                                                                  scale=8)

                result[:, :, out_bboxes[i][1]:out_bboxes[i][3],out_bboxes[i][0]:out_bboxes[i][2]] = tile_hook._crop_valid_region(tile=tile, in_bbox=in_bboxes[i],t_bbx=out_bboxes[i], scale=up_scale)
                del tile
            sample = result

        # post-process
        sample = self.conv_norm_out.to(device_2)(sample.to(device_2))
        sample = self.conv_act.to(device_2)(sample.to(device_2))

        feature_map_cur["conv_act"] = sample[-frame_overlap_num:, :, :, :].clone().to(fm_device)
        if is_first_batch:
            sample = self.conv_out.to(device_2)(sample.to(device_2))
        else:
            sample[:frame_overlap_num, :, :, :] = feature_map_prev["conv_act"].to(fm_device)
            sample = self.conv_out.to(device_2)(sample.to(device_2))

        batch_frames, channels, height, width = sample.shape
        batch_size = batch_frames // num_frames
        sample = sample[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)

        feature_map_cur["conv_out"] = sample[:, :, -frame_overlap_num:, :, :].clone().to(fm_device)
        if is_first_batch:
            sample = self.time_conv_out.to(device_2)(sample.to(device_2))
        else:
            sample[:, :, :frame_overlap_num, :, :] = feature_map_prev["conv_out"].to(device_2)
            sample = self.time_conv_out.to(device_2)(sample.to(device_2))

        sample = sample.permute(0, 2, 1, 3, 4).reshape(batch_frames, channels, height, width)

        # torch.cuda.empty_cache()

        return sample, feature_map_cur


class AutoencoderKLTemporalDecoderFeatureResetting(AutoencoderKLTemporalDecoder):
    @register_to_config
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
            block_out_channels: Tuple[int] = (64,),
            layers_per_block: int = 1,
            latent_channels: int = 4,
            sample_size: int = 32,
            scaling_factor: float = 0.18215,
            force_upcast: float = True,
    ):
        super(AutoencoderKLTemporalDecoderFeatureResetting, self).__init__(in_channels=in_channels,
                                                                           out_channels=out_channels,
                                                                           down_block_types=down_block_types,
                                                                           block_out_channels=block_out_channels,
                                                                           layers_per_block=layers_per_block,
                                                                           latent_channels=latent_channels,
                                                                           sample_size=sample_size,
                                                                           scaling_factor=scaling_factor,
                                                                           force_upcast=force_upcast)
        # self.decoder = TemporalDecoderFeatureResetting(in_channels=latent_channels,
        #                                                out_channels=out_channels,
        #                                                block_out_channels=block_out_channels,
        #                                                layers_per_block=layers_per_block)
        self.decoder = TiledTemporalDecoderFeatureResetting(in_channels=latent_channels,
                                                            out_channels=out_channels,
                                                            block_out_channels=block_out_channels,
                                                            layers_per_block=layers_per_block)

    @apply_forward_hook
    def decode(
        self,
        z: torch.Tensor,
        feature_map_prev: Dict,
        num_frames: int,
        frame_overlap_num: int,
        is_first_batch: bool = False,
        return_dict: bool = True,
        decoder_tile_size=64
    ) -> Union[DecoderOutput, torch.Tensor]:
        batch_size = z.shape[0] // num_frames
        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=z.dtype, device=z.device)

        decoded, feature_map_cur = self.decoder(sample=z,
                                                feature_map_prev=feature_map_prev,
                                                image_only_indicator=image_only_indicator,
                                                frame_overlap_num=frame_overlap_num,
                                                is_first_batch=is_first_batch,
                                                num_frames=num_frames,
                                                tile_size=decoder_tile_size)

        if not return_dict:
            return (decoded, feature_map_cur)

        return DecoderOutput(sample=decoded), feature_map_cur

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

        # print(
        #     f'[Tiled VAE]: split to {num_height_tiles}x{num_width_tiles} = {num_height_tiles * num_width_tiles} tiles. ' +
        #     f'Optimal tile size {real_tile_width}x{real_tile_height}, original tile size {tile_size}x{tile_size}')

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


class GroupNormParam:

    def __init__(self):
        self.var_list = []
        self.mean_list = []
        self.pixel_list = []
        self.weight = None
        self.bias = None

    def add_tile(self, tile, layer):
        var, mean = self._get_var_mean(tile, 32)
        # For giant images, the variance can be larger than max float16
        # In this case we create a copy to float32
        if var.dtype == torch.float16 and var.isinf().any():
            fp32_tile = tile.float()
            var, mean = self._get_var_mean(fp32_tile, 32)
        # ============= DEBUG: test for infinite =============
        # if torch.isinf(var).any():
        #    print('var: ', var)
        # ====================================================
        self.var_list.append(var)
        self.mean_list.append(mean)
        self.pixel_list.append(
            tile.shape[2]*tile.shape[3])
        if hasattr(layer, 'weight'):
            self.weight = layer.weight
            self.bias = layer.bias
        else:
            self.weight = None
            self.bias = None

    def _get_var_mean(self, input, num_groups, eps=1e-6):
        """
        Get mean and var for group norm
        """
        b, c = input.size(0), input.size(1)
        channel_in_group = int(c / num_groups)
        input_reshaped = input.contiguous().view(1, int(b * num_groups), channel_in_group, *input.size()[2:])
        var, mean = torch.var_mean(input_reshaped, dim=[0, 2, 3, 4], unbiased=False)
        return var, mean

    def _custom_group_norm(self, input, num_groups, mean, var, weight=None, bias=None, eps=1e-6):
        """
        Custom group norm with fixed mean and var

        @param input: input tensor
        @param num_groups: number of groups. by default, num_groups = 32
        @param mean: mean, must be pre-calculated by get_var_mean
        @param var: var, must be pre-calculated by get_var_mean
        @param weight: weight, should be fetched from the original group norm
        @param bias: bias, should be fetched from the original group norm
        @param eps: epsilon, by default, eps = 1e-6 to match the original group norm

        @return: normalized tensor
        """
        b, c = input.size(0), input.size(1)
        channel_in_group = int(c / num_groups)
        input_reshaped = input.contiguous().view(
            1, int(b * num_groups), channel_in_group, *input.size()[2:])

        out = F.batch_norm(input_reshaped, mean.to(input), var.to(input), weight=None, bias=None, training=False,
                           momentum=0, eps=eps)
        out = out.view(b, c, *input.size()[2:])

        # post affine transform
        if weight is not None:
            weight.requires_grad = False
            out *= weight.view(1, -1, 1, 1)
        if bias is not None:
            bias.requires_grad = False
            out += bias.view(1, -1, 1, 1)
        return out

    def summary(self):
        """
        summarize the mean and var and return a function
        that apply group norm on each tile
        """
        if len(self.var_list) == 0: return None

        var = torch.vstack(self.var_list)
        mean = torch.vstack(self.mean_list)
        max_value = max(self.pixel_list)
        pixels = torch.tensor(self.pixel_list, dtype=torch.float32, device=var.device) / max_value
        sum_pixels = torch.sum(pixels)
        pixels = pixels.unsqueeze(1) / sum_pixels
        var = torch.sum(var * pixels, dim=0)
        mean = torch.sum(mean * pixels, dim=0)
        return lambda x:  self._custom_group_norm(x, 32, mean, var, self.weight, self.bias)


class VAEDecoderHook():
    pass

def test_tiled_decoder():
    """
    Test the tiled decoder.
    """
    from PIL import Image
    test_3dconv_img_path = "/workspace/shared-dir/zzhu/tmp/20250728/ori.png"
    tiled_3dconv_img_path = "/workspace/shared-dir/zzhu/tmp/20250728/tiled.png"

    torch.manual_seed(10086)
    x = torch.randn(1, 3, 8, 480, 720)


    con3d = torch.nn.Conv3d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
    tile_hook = TileHook(model=con3d, scale=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # move to gpu
    x = x.to(device)
    con3d = con3d.to(device)

    x_ori = con3d(x)
    x_ori = x_ori.cpu().detach().numpy()
    x_ori = np.clip(x_ori, 0, 1)
    x_ori = x_ori*255.0
    x_ori = x_ori.astype(np.uint8)
    x_ori = x_ori[0, :, 0, :,:]
    x_ori = x_ori.transpose(1, 2, 0)
    x_ori_img = Image.fromarray(x_ori)
    x_ori_img.save(test_3dconv_img_path)

    x_processed = tile_hook(x)
    x_processed = x_processed.cpu().detach().numpy()
    x_processed = np.clip(x_processed, 0, 1)
    x_processed = x_processed*255.0
    x_processed = x_processed.astype(np.uint8)
    x_processed = x_processed[0, :, 0, :,:]
    x_processed = x_processed.transpose(1, 2, 0)
    x_processed_img = Image.fromarray(x_processed)
    x_processed_img.save(tiled_3dconv_img_path)


def main():
    test_tiled_decoder()


if __name__ == '__main__':
    main()
