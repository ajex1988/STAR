import torch
import torch.nn.functional as F


def inplace_nonlinearity(x):
    # Test: fix for Nans
    return F.silu(x, inplace=True)

def build_res_block_task_queue(queue, block):
    """
    Build the spatial/temporal resolution block task queue.
    """

    if block.use_in_shortcut:
        queue.append(('store_res', block.conv_shortcut))
    else:
        queue.append(('store_res', lambda x: x))

    queue.append(('pre_norm', block.norm1))
    queue.append(('silu', inplace_nonlinearity))
    queue.append(('conv1', block.conv1))
    queue.append(('pre_norm', block.norm2))
    queue.append(('silu', inplace_nonlinearity))
    queue.append(('conv2', block.conv2))
    queue.append(['add_res', None])


def vec4d_to_5d(hidden_states):
    num_frames, channels, height, width = hidden_states.shape
    hidden_states = (
        hidden_states[None, :].reshape(1, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
    )
    return hidden_states

def vec5d_to_4d(hidden_states):
    channels, num_frames, height, width = hidden_states.shape
    hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(num_frames, channels, height, width)
    return hidden_states


def build_up_blocks_task_queue(block):
    """
    Build the up-block task queue.
    """
    queue = []
    for i in range(4):
        for j in range(3):
            build_res_block_task_queue(queue, block[i].resnets[j].spatial_res_block)
            queue.append(('vec4d_to_5d', vec4d_to_5d))
            queue.append(('store_x_spatial', lambda x:x))
            build_res_block_task_queue(queue, block[i].resnets[j].temporal_res_block)
            queue.append(('alpha_blend', block[i].resnets[j].time_mixer))
            queue.append(('vec5d_to_4d', vec5d_to_4d))
        if i != 3:
            # Up-sample the latent
            queue.append(('upsample_2d',block[i].upsamplers[0]))

    return queue

def build_decoder_task_queue():
    pass


def print_layer_names(model):
    for name, module in model.named_modules():
        print(f"Layer name: {name}, Layer type: {type(module)}")

def task_print_kl_temporal_decoder():
    from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import TemporalDecoder
    decoder = TemporalDecoder()
    print_layer_names(decoder)


def main():
    task_print_kl_temporal_decoder()


if __name__ == "__main__":
    main()
