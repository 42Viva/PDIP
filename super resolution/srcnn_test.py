'''
Test
'''

from __future__ import print_function
import argparse
import torch
import math
from torch.autograd import Variable
from PIL import Image

from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import numpy as np
import os

#from srcnn_model import SRCNN

#Settings
#Totally 5 test images you can use in this foler.
input_image = './set5/set5/img_005.png'
model = './checkpoint/model_epoch_500.pth'    #your model dir
output_filename = 'set5/set5/test1_5.png'
scale_factor = 3
use_cuda = 1

img1 = Image.open(input_image).convert('YCbCr')
y, cb, cr = img1.split()

max_size0 = img1.size[0] - (img1.size[0] % scale_factor)
max_size1 = img1.size[1] - (img1.size[1] % scale_factor)
img = img1.crop((0,0,max_size0,max_size1))

img = img.resize((int(img.size[0]//scale_factor),int(img.size[1]//scale_factor)),Image.BICUBIC)
img = img.resize((int(img.size[0]*scale_factor),int(img.size[1]*scale_factor)),Image.BICUBIC)

y, cb, cr = img.split()

model = torch.load(model)
input = Variable(ToTensor()(y)).view(1, -1, img.size[1], img.size[0])

if use_cuda:
    model = model.cuda(device = 0)
    input = input.cuda(device = 0)

out = model(input)
out = out.cpu()

print ("type = ",type(out))
tt = transforms.ToPILImage()

img_out = tt(out.data[0])

#img_out.save(output_filename)




out_img_y = out.data[0].numpy()
out_img_y *= 255.0
out_img_y = out_img_y.clip(0, 255)
out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

out_img.save(output_filename)
# out_img.save(opt.output_filename)
print('output image saved to ', output_filename)
