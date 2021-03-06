import PIL.Image as pil_image
import tensorflow as tf
import numpy as np
import random
import glob
import io
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.enable_eager_execution(config=config)

## Dataset -> called by Dataloader
class Dataset(object):
    def __init__(self, images_dir, patch_size, jpeg_quality, use_augmentation=False, use_fast_loader=False):
        self.image_files = sorted(glob.glob(images_dir + '/*'))
        self.patch_size = patch_size
        self.jpeg_quality = jpeg_quality
        self.use_augmentation = use_augmentation
        self.use_fast_loader = use_fast_loader

    def __getitem__(self, idx):
        if self.use_fast_loader:
            label = tf.read_file(self.image_files[idx])
            label = tf.image.decode_jpeg(label, channels=3)
            label = pil_image.fromarray(label.numpy())
        else:
            label = pil_image.open(self.image_files[idx]).convert('RGB')

        if self.use_augmentation:
            # randomly rescale image
            if random.random() <= 0.5:
                scale = random.choice([0.9, 0.8, 0.7, 0.6])
                label = label.resize((int(label.width * scale), int(label.height * scale)), resample=pil_image.BICUBIC)

            # randomly rotate image
            if random.random() <= 0.5:
                label = label.rotate(random.choice([90, 180, 270]), expand=True)
                

        # randomly crop patch from training set
        crop_x = random.randint(0, label.width - self.patch_size)
        crop_y = random.randint(0, label.height - self.patch_size)
        label = label.crop((crop_x, crop_y, crop_x + self.patch_size, crop_y + self.patch_size))

        # additive jpeg noise
        buffer = io.BytesIO()
        label.save(buffer, format='jpeg', quality=self.jpeg_quality)
        input = pil_image.open(buffer)

        input = np.array(input).astype(np.float32)
        label = np.array(label).astype(np.float32)
        input = np.transpose(input, axes=[2, 0, 1])
        label = np.transpose(label, axes=[2, 0, 1])

        # normalization
        input /= 255.0
        label /= 255.0

        return input, label

    def __len__(self):
        return len(self.image_files)

## Use for count traing loss
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count