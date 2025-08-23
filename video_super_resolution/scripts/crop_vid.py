from PIL import Image
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default='./video/')
    parser.add_argument('--out_dir', type=str, default='./output/')
    parser.add_argument('--x', type=int, default=0)
    parser.add_argument('--y', type=int, default=0)
    parser.add_argument('--w', type=int, default=64)
    parser.add_argument('--h', type=int, default=64)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    in_dir = args.in_dir
    out_dir = args.out_dir
    x = args.x
    y = args.y
    w = args.w
    h = args.h

    os.makedirs(out_dir, exist_ok=True)

    img_name_list = os.listdir(in_dir)
    img_name_list.sort()

    for img_name in img_name_list:
        img_path = os.path.join(in_dir, img_name)
        img = Image.open(img_path)
        img_cropped = img.crop((x, y, x + w, y + h))
        out_path = os.path.join(out_dir, img_name)
        img_cropped.save(out_path)

if __name__ == "__main__":
    main()
