import os.path
import logging
import re

import numpy as np
from collections import OrderedDict

import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_model


'''
Spyder (Python 3.6)
PyTorch 1.1.0
Windows 10 or Linux

Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/KAIR

If you have any question, please feel free to contact with me.
Kai Zhang (e-mail: cskaizhang@gmail.com)
(github: https://github.com/cszn/KAIR)

by Kai Zhang (12/Dec./2019)
'''

"""
# --------------------------------------------
testing demo for RRDB-ESRGAN
https://github.com/xinntao/ESRGAN
@inproceedings{wang2018esrgan,
  title={Esrgan: Enhanced super-resolution generative adversarial networks},
  author={Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Qiao, Yu and Change Loy, Chen},
  booktitle={European Conference on Computer Vision (ECCV)},
  pages={0--0},
  year={2018}
}
# --------------------------------------------
|--model_zoo             # model_zoo
   |--rrdb_x4_esrgan     # model_name, optimized for perceptual quality      
   |--rrdb_x4_psnr       # model_name, optimized for PSNR
|--testset               # testsets
   |--set5               # testset_name
   |--srbsd68
|--results               # results
   |--set5_rrdb_x4_esrgan# result_name = testset_name + '_' + model_name
   |--set5_rrdb_x4_psnr 
# --------------------------------------------
"""


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    model_name = 'rrdb_x4_esrgan'        # 'rrdb_x4_esrgan' | 'rrdb_x4_psnr'
    testset_name = 'test_low'                # test set,  'set5' | 'srbsd68'
    need_degradation = True              # default: True
    x8 = False                           # default: False, x8 to boost performance
    sf = [int(s) for s in re.findall(r'\d+', model_name)][0]  # scale factor
    show_img = False                     # default: False




    task_current = 'sr'       # 'dn' for denoising | 'sr' for super-resolution
    n_channels = 3            # fixed
    model_pool = 'model_zoo'  # fixed
    testsets = 'testsets'     # fixed
    results = 'results'       # fixed
    noise_level_img = 0       # fixed: 0, noise level for LR image
    result_name = testset_name + '_' + model_name
    border = sf if task_current == 'sr' else 0     # shave boader to calculate PSNR and SSIM
    model_path = os.path.join(model_pool, model_name+'.pth')

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = os.path.join(testsets, testset_name) # L_path, for Low-quality images
    H_path = L_path                               # H_path, for High-quality images
    E_path = os.path.join(results, result_name)   # E_path, for Estimated images
    util.mkdir(E_path)

    if H_path == L_path:
        need_degradation = True
    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    need_H = True if H_path is not None else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # load model
    # ----------------------------------------

    from models.network_rrdb import RRDB as net
    model = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=23, gc=32, upscale=4, act_mode='L', upsample_mode='upconv')
    model.load_state_dict(torch.load(model_path), strict=True)  # strict=False
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    logger.info('Model path: {:s}'.format(model_path))
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []

    logger.info('model_name:{}, image sigma:{}'.format(model_name, noise_level_img))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)
    H_paths = util.get_image_paths(H_path) if need_H else None

    #---------------------------------------------------------------------------------------------------------#
    idx = 0
    #for idx, img in enumerate(L_paths):
    for img in util.get_image_paths(L_path):
        idx += 1
        img_name, ext = os.path.splitext(os.path.basename(img))
        logger.info('{:->4d}--> {:>10s}'.format(idx, img_name+ext))

        img_L = util.imread_uint(img, n_channels=3)
        img_L = util.uint2tensor4(img_L)
        torch.cuda.empty_cache()
        img_L = img_L.to(device)

        #start.record()
        img_E = model(img_L)
        # img_E = utils_model.test_mode(model, img_L, mode=2, min_size=480, sf=sf)  # use this to avoid 'out of memory' issue.
        # logger.info('{:>16s} : {:<.3f} [M]'.format('Max Memery', torch.cuda.max_memory_allocated(torch.cuda.current_device())/1024**2))  # Memery
       # end.record()
        torch.cuda.synchronize()
        #test_results['runtime'].append(start.elapsed_time(end))  # milliseconds


#        torch.cuda.synchronize()
#        start = time.time()
#        img_E = model(img_L)
#        torch.cuda.synchronize()
#        end = time.time()
#        test_results['runtime'].append(end-start)  # seconds

        # --------------------------------
        # (2) img_E
        # --------------------------------
        img_E = util.tensor2uint(img_E)


        test,n,low = img_name.split('_')
        util.imsave(img_E, os.path.join(E_path, test+"_"+n+'_answer.tif'))
    #---------------------------------------------------------------------------------------------------------#

if __name__ == '__main__':

    main()
