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
parser.add_argument('--train_path',     '-tp',     type=str,   required=True, 	            help='Path to your training data folder.')
parser.add_argument('--valid_path',     '-vp',     type=str,   required=True, 	            help='Path to your validation data folder.')
parser.add_argument('--output_path',    '-op',     type=str,   required=False, default="../Output",	help='Path for saving our model.')
parser.add_argument('--log_path',       '-lp',     type=str,   required=False, default="../log",    help='Path for saving our model.')
parser.add_argument('--batch_size',     '-bs',     type=int,   required=False, default=8,           help='Batch size.')
parser.add_argument('--patch_size',     '-ps',     type=int,   required=False, default=128,         help='Patch szie')
parser.add_argument('--epochs',         '-epochs', type=int,   required=False, default=100,         help='Numbers of tranining epoch')
parser.add_argument('--jpeg_quality',   '-jq',     type=int,   required=False, default=10,          help='Compressrion rate of JPEG.')
parser.add_argument('--threads',        '-t',      type=int,   required=False, default=4,           help='Number of threads')
parser.add_argument('--scale_percent',  '-sp',     type=float, required=False, default=0.5,         help='Scale learning rate with ? time')
parser.add_argument('--learning_rate',  '-lr',     type=float, required=False, default=5e-4,        help='Learning rate.(Don\' be too large)')


## Read the parameters
args = parser.parse_args()
TRAIN_PATH    = args.train_path
VALID_PATH    = args.valid_path
OUTPUT_PATH   = args.output_path
LOG_PATH      = args.log_path
BATCH_SIZE    = args.batch_size
PATCH_SIZE    = args.patch_size
EPOCHS        = args.epochs
JPEG_QUALITY  = args.jpeg_quality
THREADS       = args.threads
SCALE_PERCENT = args.scale_percent
LEARNING_RATE = args.learning_rate
##

USE_AUGMENTATION = True
USE_FAST_LOADER = False
### Get Parameters

###
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)
###

### Training 
model = My_Model()
model = model.to(device)
loss_function = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

## Read training data
train_dataset = Dataset(TRAIN_PATH, PATCH_SIZE, JPEG_QUALITY, USE_AUGMENTATION, USE_FAST_LOADER)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=THREADS, pin_memory=True, drop_last=True)

## Read validiation data
valid_dataset = Dataset(VALID_PATH, PATCH_SIZE, JPEG_QUALITY, USE_AUGMENTATION, USE_FAST_LOADER)
valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=THREADS, pin_memory=True, drop_last=True)


## Start training
with SummaryWriter(LOG_PATH) as writer:
    for epoch in range(EPOCHS):
        train_loss  = AverageMeter()
        valid_loss  = AverageMeter()
        target_loss = AverageMeter()
        
        ## Caculate how many batchs
        n_batchs = (len(train_dataset) - len(train_dataset) % BATCH_SIZE)
        with tqdm(total=n_batchs) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, EPOCHS))
            for batch, data in enumerate(train_dataloader):
                inputs, labels = data[0].to(device), data[1].to(device)

                preds = model(inputs)

                loss = loss_function(preds, labels)
                target = loss_function(inputs, labels)
                train_loss.update(loss.item(), len(inputs))
                target_loss.update(target.item(), len(inputs))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _tqdm.set_postfix(loss='{:.6f}, target{:.6f}'.format(train_loss.avg, target_loss.avg))
                _tqdm.update(len(inputs))
                
                ### Record every batch traing loss
                writer.add_scalar('Train//Loss', train_loss.avg, epoch*n_batchs + batch)
                writer.flush()
                ### Record every batch traing loss
                
            
            for data in valid_dataloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                
                preds = model(inputs)
                
                loss = loss_function(preds, labels)
                valid_loss.update(loss.item(), len(inputs))
                

        ### Record every epoch validation loss
        writer.add_scalar('Valid//Loss', valid_loss.avg, epoch)
        writer.flush()
        ### Record every epoch validation loss
        
        torch.save(model.state_dict(), os.path.join(OUTPUT_PATH, 'Epoch_{}.pth'.format(epoch)))
        
        if((epoch+1)%(EPOCHS//5)):
            LEARNING_RATE = SCALE_PERCENT*LEARNING_RATE
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Finished!")
### Training