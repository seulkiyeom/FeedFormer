from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv import imwrite

import mmcv
import torch
from mmseg.ops import resize

from torchinfo import summary

"""
This script mainly adopts the functionality of the image_demo.py script i.e.,
given a config file it does an inference step on a certain query image. 
Furthermore it extract the interpretability measures and saves them into a dedicated
folder.
"""

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='ade20k',
        help='Color palette used for segmentation map')
    args = parser.parse_args()

# ------------- TOOK FROM test.py ------------------
# Goal is to apply data processing also for the given image path
# Otherwise input is not of sqaured shape and it is harder to
# compute the height and width from number of tokens in the 
# attention matrix
    # cfg = mmcv.Config.fromfile(args.config)
    # # if args.options is not None:
    # #     cfg.merge_from_dict(args.options)
    # # set cudnn_benchmark
    # if cfg.get('cudnn_benchmark', False):
    #     torch.backends.cudnn.benchmark = True
    # # if args.aug_test:
    # if cfg.data.test.type == 'CityscapesDataset':
    #     # hard code index
    #     cfg.data.test.pipeline[1].img_ratios = [
    #         0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
    #     ]
    #     cfg.data.test.pipeline[1].flip = True
    # elif cfg.data.test.type == 'ADE20KDataset':
    #     # hard code index
    #     cfg.data.test.pipeline[1].img_ratios = [
    #         0.75, 0.875, 1.0, 1.125, 1.25
    #     ]
    #     cfg.data.test.pipeline[1].flip = True
    # else:
    #     # hard code index
    #     cfg.data.test.pipeline[1].img_ratios = [
    #         0.5, 0.75, 1.0, 1.25, 1.5, 1.75
    #     ]
    #     cfg.data.test.pipeline[1].flip = True
# ------------- TOOK FROM test.py ------------------
    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)

# ------------ Try to use torch summary ------------    
    model.forward = model.forward_dummy 
    model_summary = summary(model.backbone, (1, 3, 512, 512))
# ------------ Try to use torch summary ------------
    
    # test a single image
    result = inference_segmentor(model, args.img)
    model.show_result(args.img, result, out_file="result2.png")
    
    # show the results
    # show_result_pyplot(model, args.img, result, get_palette(args.palette))
    stage_to_visualize = 0
    channel_to_visualize = 350

    h1, w1 = model.backbone.attention_maps[0].shape[2:]
    attention_matrix = model.backbone.attention_maps[stage_to_visualize]
    
    attention_matrix = resize(attention_matrix, size=(h1, w1), mode='bilinear', align_corners=False)
    # Average over attentin heads
    attention_matrix = attention_matrix.mean(0)
    # Select channel to vizualize
    attention_matrix_channel = attention_matrix[channel_to_visualize]
    # Normalize values
    attention_matrix_channel = (attention_matrix_channel - attention_matrix_channel.min()) / (attention_matrix_channel.max() - attention_matrix_channel.min())
    attention_matrix_channel = attention_matrix_channel.cpu().numpy() * 255
    imwrite(attention_matrix_channel, f'./visualization/attention_stage_{stage_to_visualize}_chan_{channel_to_visualize}.png')


if __name__ == '__main__':
    main()
