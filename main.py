from __future__ import print_function

import argparse
import os

import matplotlib.pyplot as plt
import torch
import torchvision.models as models

from data_utils import image_loader, masks_loader, plt_images, tensor_to_image
from neural_style import run_style_transfer
import preprocessing


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Style Transfer')
    # arguments for preprocessing raw data
    parser.add_argument('--prepare_data', action='store_true', help='Set, if images should be preprocessed')
    parser.add_argument('--path_style_raw', default='images/raw', help='Path of style images to be preprocessed')
    parser.add_argument('--path_content_raw', default='../fashion-product-images-dataset/fashion-dataset/images',
                        help='Path of content images to be preprocessed')
    parser.add_argument('--path_preprocessed', default='images/preprocessed', help='Path of images after preprocessing')
    # arguments for preparing segmentation images
    parser.add_argument('--prepare_segmentation', action='store_true', help='Set, if segmentations should be prepared')
    parser.add_argument('--path_segmentation', default='images/segmentation',
                        help='Path of segmentation images after preprocessing')
    # images to apply style transfer on
    parser.add_argument('--style_image_id', default=0, type=int)
    parser.add_argument('--content_image_id', default=0, type=int)
    # paths for results
    parser.add_argument('--path_results', default='results', help='Directory for results')
    parser.add_argument('--file_results', default='result', help='Filename of output images')
    # loss function and number of epochs
    parser.add_argument('--method', default='deepobjstyle', type=str, help='Whether neural_style or deepobjstyle loss')
    parser.add_argument('--epochs', default=500, type=int, help='Number of optimization steps')
    # weights of deepobjstyle loss
    parser.add_argument('--alpha11', default=1e4, type=float, help='Weight of content loss in DPS')
    parser.add_argument('--alpha12', default=5e6, type=float, help='Weight of augmented style loss in DPS')
    parser.add_argument('--alpha13', default=1e-4, type=float, help='Weight of regularization in DPS')
    parser.add_argument('--alpha2', default=1e-2, type=float, help='Weight of content loss in DOS')
    # weight of neural_style loss
    parser.add_argument('--alpha', default=1, type=float, help='Weight of content loss in neural_style')
    parser.add_argument('--beta', default=5e6, type=float, help='Weight of style loss in neural_style')

    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imsize = (512, 512) # if torch.cuda.is_available() else (128, 128)

    if args.prepare_data:
        preprocessing.prepare_images(path_images_c=args.path_content_raw, path_images_s=args.path_style_raw,
                                     path_save=args.path_preprocessed, imsize=imsize)
    if args.prepare_segmentation:
        preprocessing.prepare_segmentation(path_images=args.path_preprocessed, path_save=args.path_segmentation)

    content_img = image_loader('{}/fashion_{}.jpeg'.format(args.path_preprocessed, args.content_image_id),
                               imsize).to(device, torch.float)
    style_img = image_loader('{}/style_{}.jpeg'.format(args.path_preprocessed, args.style_image_id),
                             imsize).to(device, torch.float)
    input_img = content_img.clone()

    vgg = models.vgg19(pretrained=True).features.to(device).eval()

    vgg_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    vgg_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    if args.method == 'deepobjstyle':
        style_masks, content_masks = masks_loader(
            path_style='{}/segmentation_style_{}.jpeg'.format(args.path_segmentation, args.style_image_id),
            path_content='{}/segmentation_fashion_{}.jpeg'.format(args.path_segmentation, args.content_image_id),
            size=imsize
        )

        style_masks = [mask.to(device) for mask in style_masks]
        content_masks = [mask.to(device) for mask in content_masks]

    elif args.method == 'neural_style':
        style_masks = None
        content_masks = None

    else:
        ValueError('Invalid method {}'.format(args.method))

    output = run_style_transfer(
        cnn=vgg,
        normalization_mean=vgg_normalization_mean, normalization_std=vgg_normalization_std,
        style_img=style_img, content_img=content_img, input_img=input_img,
        style_masks=style_masks, content_masks=content_masks,
        device=device,
        reg=True,
        num_steps=args.epochs,
        alpha11=args.alpha11, alpha12=args.alpha12, alpha13=args.alpha13, alpha2=args.alpha2,
        method=args.method,
        alpha=args.alpha, beta=args.beta
    )

    args.path_results = '{}/c{}_s{}'.format(args.path_results, args.content_image_id, args.style_image_id)
    if not os.path.exists(args.path_results):
        os.mkdir(args.path_results)

    plt.imsave(arr=tensor_to_image(input_img), fname='{}/{}.pdf'.format(args.path_results, args.file_results))