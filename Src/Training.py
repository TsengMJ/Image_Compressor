from torch.utils.data.dataloader import DataLoader
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tqdm import tqdm
import argparse
import torch
import os

from Dataset import *
from Model import *

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### Get Parameters
parser = argparse.ArgumentParser()
# parser.add_argument('--train_path',     '-tp',     type=str,   required=True, 	            help='Path to your training data folder.')
# parser.add_argument('--valid_path',     '-vp',     type=str,   required=True, 	            help='Path to your validation data folder.')
# parser.add_argument('--output_path',    '-op',     type=str,   required=True, 	            help='Path for saving our model.')
parser.add_argument('--learning_rate',  '-lr',     type=float, required=False, default=0.5, help='Learning rate.(Don\' be too large)')
parser.add_argument('--batch_size',     '-bz',     type=int,   required=False, default=128, help='Batch zise.')
parser.add_argument('--n_epochs',       '-epochs', type=int,   required=False, default=100, help='Numbers of tranining epoch')
parser.add_argument('--scale_epochs',   '-se',     type=int,   required=False, default=20,  help='Rescale learning rate for every ? epochs')
parser.add_argument('--scale_percent',  '-sp',     type=float, required=False, default=0.5, help='Scale learning rate with ? time')





USE_AUGMENTATION = True
USE_FAST_LOADER = False
### Get Parameters