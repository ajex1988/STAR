import os
import argparse
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--w', type=int)
    parser.add_argument('--h', type=int)
    parser.add_argument('--x', type=int)
    parser.add_argument('--y', type=int)
    parser.add_argument('--color', type=int, default=0)
    return parser.parse_args()

def main():
    args = parse_args()
    in_dir = args.in_dir
    out_dir = args.out_dir
    h = args.h
    w = args.w
    x = args.x
    y = args.y
    color = args.color

    os.makedirs(out_dir, exist_ok=True)

    img_name_list = os.listdir(in_dir)
    img_name_list.sort()
    for img_name in img_name_list:
        img_path = os.path.join(in_dir, img_name)
        out_path = os.path.join(out_dir, img_name)

        img = Image.open(img_path)

        canvas = Image.new('RGB', (w, h), color=(color, color, color))

        canvas.paste(img, (x, y))
        canvas.save(out_path)


if __name__ == '__main__':
    main()
