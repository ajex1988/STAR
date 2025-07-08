import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dirs', type=str, default='', help='input video dir')
    parser.add_argument('--out_dirs', type=str, default='', help='output video dir')
    parser.add_argument('--split_num', type=int, default=4, help='split num')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    in_vid_names = os.listdir(args.in_dirs)

    for in_vid_name in in_vid_names:
        in_vid_dir = os.path.join(args.in_dirs, in_vid_name)
        in_frame_names = os.listdir(in_vid_dir)
        in_frame_names.sort()
        step = len(in_frame_names) // args.split_num
        in_frame_name_list = []
        for i in range(0, len(in_frame_names), step):
            in_frame_name_list.append(in_frame_names[i:i + step])
        for i in range(args.split_num):
            out_vid_dir = os.path.join(args.out_dirs, in_vid_name+f'_{i}')
            os.makedirs(out_vid_dir, exist_ok=True)
            in_frame_name = in_frame_name_list[i]
            for j in range(len(in_frame_name)):
                in_frame_path = os.path.join(in_vid_dir, in_frame_name[j])
                out_frame_path = os.path.join(out_vid_dir, in_frame_name[j])
                os.symlink(in_frame_path, out_frame_path)




if __name__ == '__main__':
    main()
