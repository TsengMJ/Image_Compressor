import matplotlib.pyplot as plt
import PIL.Image as pil_image
import numpy as np
import torchvision
import argparse
import torch
import io

from Model import *

### Get Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', '-mp',  type=str,  required=True, help='Path to your trained model.')
parser.add_argument('--img_path',   '-ip',  type=str,  required=True, help='Path to testing image.')

## Read the parameters
args = parser.parse_args()
MODEL_PATH = args.model_path
IMG_PATH   = args.img_path
##
### Get Parameters


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### Load Model
model = My_Model().to(device)
model.load_state_dict(torch.load(MODEL_PATH))
### Load Model

### Read Image and Pre-processing
label = pil_image.open(IMG_PATH).convert('RGB')

buffer = io.BytesIO()
label.save(buffer, format='jpeg', quality=10)
input = pil_image.open(buffer)

input = np.array(input).astype(np.float32)
label = np.array(label).astype(np.float32)
input = np.transpose(input, axes=[2, 0, 1])
label = np.transpose(label, axes=[2, 0, 1])

## Normalize
input /= 255.0
label /= 255.0
## Normalize
### Read Image and Pre-processing

### Predict
input = torch.tensor(input).to(device)
label = torch.tensor(label).to(device)
input = input.unsqueeze(0)
pred = model(input)
### Predict

### De-normalize
input = (input*255).cpu().type(torch.ByteTensor)
label = (label*255).cpu().type(torch.ByteTensor)
pred = (pred*255).cpu().type(torch.ByteTensor)
### De-normalize


### Show Comparison Results
plt.figure(figsize=(15, 15))
plt.subplot(131);plt.imshow(torchvision.utils.make_grid(input,nrow=1).permute(1, 2, 0));plt.title("JPG");plt.xticks([]);plt.yticks([]);
plt.subplot(132);plt.imshow(torchvision.utils.make_grid(label,nrow=1).permute(1, 2, 0));plt.title("Ori");plt.xticks([]);plt.yticks([]);
plt.subplot(133);plt.imshow(torchvision.utils.make_grid(pred,nrow=1).permute(1, 2, 0));plt.title("Fixed");plt.xticks([]);plt.yticks([]);
plt.show()
### Show Comparison Results