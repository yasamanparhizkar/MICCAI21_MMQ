import torch
import torchvision.transforms.functional as F
import clip
import argparse
import os
import json
from PIL import Image
import _pickle as cPickle


def preprocess(args):
    # load the clip preprocessor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load(args.model, device=device)

    # load CLIP images
    image_names = []
    image_paths = []
    images = []
    for split in ['train', 'val', 'test']:
        image_dir = os.path.join(args.input, split)
        split_image_names = os.listdir(image_dir)
        split_image_paths = [os.path.join(image_dir, image_name) for image_name in split_image_names]
        split_images = [preprocess(Image.open(image_path).convert('RGB')).to('cpu') for image_path in split_image_paths]
        image_names.extend(split_image_names)
        image_paths.extend(split_image_paths)
        images.extend(split_images)

    # create and dump imgid2idx.json
    imgid2idx = {imgid: idx for idx, imgid in enumerate(image_names)}
    output_path_json = os.path.join(args.output, 'imgid2idx.json')
    with open(output_path_json, 'w') as file:
        json.dump(imgid2idx, file)
        print('imgid2idx dumped to %s' % output_path_json)
    
    # dump CLIP images to file
    output_path_pkl = os.path.join(args.output, 'images_clip.pkl')
    with open(output_path_pkl, 'wb') as file:
        cPickle.dump(images, file)
        print('CLIP images dumped to %s' % output_path_pkl)
    
    # load AE images
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert('L')
        image = F.center_crop(image, output_size=[min(image.size)])
        image = F.resize(image, size=[args.aedim])
        image = F.to_tensor(image).to('cpu')
        image = torch.transpose(image, 0, 2)
        images.append(image)
    
    # dump AE images to file
    output_path_pkl = os.path.join(args.output, 'images{}x{}.pkl'.format(args.aedim, args.aedim))
    with open(output_path_pkl, 'wb') as file:
        cPickle.dump(images, file)
        print('AE images dumped to %s' % output_path_pkl)

def load_images(file_path):
    with open(file_path, 'rb') as file:
        images = cPickle.load(file)
        images = torch.stack(images)
    return images

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./',
                        help='path to the folder containing the images.')
    parser.add_argument('--output', type=str, default='./',
                        help='path to where the .pkl file is saved.')
    parser.add_argument('--model', type=str, default='ViT-B/32',
                        help='vision model used in clip, avialable options can be listed by `clip.available_models()`.')
    parser.add_argument('--aedim', type=int, default=128,
                        help='dimension of AE images.')
    args = parser.parse_args()
    preprocess(args)