import torch
from typing import Dict, Optional, Tuple, Union

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
        feature_map_cur["sample"] = sample.clone()

        if is_first_batch:
            sample = self.conv_in(sample)
        else:
            sample[:frame_overlap_num, :, :, :] = feature_map_prev["sample"][-frame_overlap_num:, :, :, :]
            sample = self.conv_in(sample)

        feature_map_cur["conv_in"] = sample.clone()

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
                sample[:frame_overlap_num, :, :, :] = feature_map_prev["conv_in"][-frame_overlap_num:, :, :, :]
                sample = self.mid_block(sample, image_only_indicator=image_only_indicator)
            sample = sample.to(upscale_dtype)

            feature_map_cur["mid_block"] = sample.clone()

            # up
            for i, up_block in enumerate(self.up_blocks):
                if is_first_batch:
                    sample = up_block(sample, image_only_indicator=image_only_indicator)
                else:
                    if i == 0:
                        sample[:frame_overlap_num, :, :, :] = feature_map_prev["mid_block"][-frame_overlap_num:, :, :, :]
                        sample = up_block(sample, image_only_indicator=image_only_indicator)
                        feature_map_cur["up_block"] = [sample.clone()]
                    else:
                        sample[:frame_overlap_num, :, :, :] = feature_map_prev["up_block"][i-1][-frame_overlap_num:, :, :, :]
                        sample = up_block(sample, image_only_indicator=image_only_indicator)
                        feature_map_cur["up_block"].append(sample.clone())

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)

        feature_map_cur["conv_act"] = sample.clone()
        if is_first_batch:
            sample = self.conv_out(sample)
        else:
            sample[:frame_overlap_num, :, :, :] = feature_map_prev["conv_act"][-frame_overlap_num:, :, :, :]
            sample = self.conv_out(sample)

        batch_frames, channels, height, width = sample.shape
        batch_size = batch_frames // num_frames
        sample = sample[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)

        feature_map_cur["conv_out"] = sample.clone()
        if is_first_batch:
            sample = self.time_conv_out(sample)
        else:
            sample[:frame_overlap_num, :, :, :, :] = feature_map_prev["conv_out"][-frame_overlap_num:, :, :, :, :]
            sample = self.time_conv_out(sample)

        sample = sample.permute(0, 2, 1, 3, 4).reshape(batch_frames, channels, height, width)

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
        self.decoder = TemporalDecoderFeatureResetting(in_channels=in_channels,
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
    ) -> Union[DecoderOutput, torch.Tensor]:
        batch_size = z.shape[0] // num_frames
        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=z.dtype, device=z.device)

        decoded, feature_map_cur = self.decoder(sample=z,
                                                feature_map_prev=feature_map_prev,
                                                image_only_indicator=image_only_indicator,
                                                frame_overlap_num=frame_overlap_num,
                                                is_first_batch=is_first_batch,
                                                num_frames=num_frames)

        if not return_dict:
            return (decoded, feature_map_cur)

        return DecoderOutput(sample=decoded), feature_map_cur