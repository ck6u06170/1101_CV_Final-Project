from super_image import HanModel, ImageLoader
from PIL import Image
import requests
import matplotlib.pyplot as plt
import os
import numpy as np
file = os.listdir()

for i in file:
    image = Image.open(i)

    model = HanModel.from_pretrained('eugenesiow/han', scale=4)     # scale 2, 3 and 4 models available
    inputs = ImageLoader.load_image(image)
    preds = model(inputs)
    
    ImageLoader.save_image(preds, r'./HAN_train/{}.png'.format(i[:i.find('.')]))  
    print(i + 'save')
