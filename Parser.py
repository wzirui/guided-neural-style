import argparse

def parse_args():
    parser = argparse.ArgumentParser()

# ('--weights', nargs='+', type=float, default=[1.0], choices=[1, 2, 3], help='')

    parser.add_argument('--content_img', type=str, 
        help='content image path')

    parser.add_argument('--style_img', type=str,
        help='style image path')

    parser.add_argument('--target_mask', type=str,
        help='target mask path')

    parser.add_argument('--style_mask', type=str,
        help='style mask path')

    # colors = 1: only use white region
    # colors > 1: use all colors
    parser.add_argument('--mask_n_colors', type=int,
        default=1,
        help='Number of colors in the given mask')

    parser.add_argument('--hard_width', type=int,
        help='If set, resize the content, style and mask images to the same width')

    parser.add_argument('--init_noise_ratio', type=float,
        default=0.0,
        help='The ratio between noise and content, ranging from 0. to 1.')

    parser.add_argument('--model_path', type=str,
        default='imagenet-vgg-verydeep-19.mat',
        help='The path of the vgg model')

    parser.add_argument('--feature_pooling_type', type=str,
        default='avg',
        choices=['avg', 'max'],
        help='pooling type of the vgg model')

    parser.add_argument('--mask_downsample_type', type=str,
        default='simple',
        choices=['simple', 'all', 'inside', 'mean'],
        help='How to propagate masks to different layers')

    parser.add_argument('--content_layers', nargs='+', type=str,
        default=['relu4_2'],
        help='VGG19 layers used for the content image')

    parser.add_argument('--content_layers_weights', nargs='+', type=float,
        default=[1.],
        help='weights of each content layer')

    parser.add_argument('--style_layers', nargs='+', type=str,
        default=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'],
        help='VGG19 layers used for the style image')

    parser.add_argument('--style_layers_weights', nargs='+', type=float,
        default=[1., 1., 1., 1., 1.],
        help='weights of each style layer')

    parser.add_argument('--content_loss_normalization', type=int,
        default=1,
        choices=[1, 2],
        help='1 for 1./(N * M); 2 for 1./(2. * N**0.5 * M**0.5)')

    parser.add_argument('--mask_normalization_type', type=str,
        default='square_sum',
        choices=['square_sum', 'sum'],
        help='How to normalize a masked gram matrix')

    parser.add_argument('--content_weight', type=float,
        default=1.,
        help='Content loss weight')

    parser.add_argument('--style_weight', type=float,
        default=0.2,
        help='Style loss weight')

    parser.add_argument('--tv_weight', type=float,
        default=0.,
        help='Total variation loss weight')

    parser.add_argument('--optimizer', type=str,
        default='lbfgs',
        choices=['lbfgs', 'adam'],
        help='choose optimizer')

    parser.add_argument('--learning_rate', type=float,
        default=10.,
        help='learning rate for adam optimizer')

    parser.add_argument('--iteration', type=int,
        default=1000,
        help='max iterations of training')

    # 10 is good for l-bfgs interface?
    parser.add_argument('--log_iteration', type=int,
        default=10,
        help='Number of iterations to print loss. For adam, also save intermediate result. For L-BFGS, don\'t larger than 10')

    parser.add_argument('--output_dir', type=str,
        default='./output',
        help='Directory to save result')

    return parser.parse_args()

