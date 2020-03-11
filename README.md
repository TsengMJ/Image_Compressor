# Image Compressor
這個專案目的是嘗試使用深度學習修復經過 JPEG 壓縮後的影像，實做框架使用 pytorch 進行訓練。其中程式主要分為3個部份
* 資料讀取 - DataLoader ([code](https://github.com/TsengMJ/Image_Compressor/blob/master/Src/DataLoader.ipynb))
* 建立模型 - Model ([code](https://github.com/TsengMJ/Image_Compressor/blob/master/Src/Model.ipynb))
* 訓練 - Training ([code](https://github.com/TsengMJ/Image_Compressor/blob/master/Src/Training.ipynb))

The purpose of this project is trying to use deep learning to repair JPEG compressed images, the using framework is pytorch. The whole process is mainly divided into 3 parts. 
* Loading data - DataLoader ([code](https://github.com/TsengMJ/Image_Compressor/blob/master/Src/DataLoader.ipynb))
* Building model - Model ([code](https://github.com/TsengMJ/Image_Compressor/blob/master/Src/Model.ipynb))
* Training model - Training ([code](https://github.com/TsengMJ/Image_Compressor/blob/master/Src/Training.ipynb))

## 資料讀取 (Loading Data)
**資料集 (Dataset)**
這次 **[Open Images 2019 (Google)](https://www.kaggle.com/c/open-images-2019-object-detection)** **[BSR](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)**, and **[DIV2k](https://data.vision.ee.ethz.ch/cvl/DIV2K/)** 作為訓練資料，其中訓練影像將近10萬張，包含多種情況，並且測試資料大約為1萬張2k圖像。

The training dataset we used are **[Open Images 2019 (Google)](https://www.kaggle.com/c/open-images-2019-object-detection)**, **[BSR](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)**, and **[DIV2k](https://data.vision.ee.ethz.ch/cvl/DIV2K/)**. The training data is near 100 thousand images,contain lots of kinds of situations, and the size of testing date is about 10 thousand 2k images. 

**步驟 (Steps)**
1. 為了加速訓練，每張影像在每個 Epoch 都隨機擷取一個方形區域進行訓練，並且隨機進行旋轉 (In order to speed up the training, each image randomly fetched a square area for training in each Epoch, and rotates randomly.)
```
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

```

2. 生成經 JPEG 壓縮後的影像，也就是實際餵進模型裡的資料 (Then generate JPEG compressed image, which is actually fed into the model)
```
  # additive jpeg noise
  buffer = io.BytesIO()
  label.save(buffer, format='jpeg', quality=self.jpeg_quality)
  input = pil_image.open(buffer)
  
  input = np.array(input).astype(np.float32)
  label = np.array(label).astype(np.float32)
  input = np.transpose(input, axes=[2, 0, 1])
  label = np.transpose(label, axes=[2, 0, 1])
```

3. 正歸化訓練資料（Normalize the training data
```
  # normalization
  input /= 255.0
  label /= 255.0
```
**[Note]**：

其實第一步最好方式，是將原本影像拆解成多張小影像都丟進去一起訓練，但是總訓練會變很久，所以這邊呈現結果就不那樣弄，有興趣自己試試看吧～～

In fact, the best way for the first step is to split the original image into multiple small images and feed them into the training together, but the total training will take a long time, so the results presented here are not so tricky, try yourself if you are have any interest ~~



下列是

## Introduction
  Recent years has seen an increased amount of research in image compression. Most of effort, however, has focused on how to use convolution neural network (CNN) to enhance image compression. One of them effort to use neural network to decrease the artifact of lossy image compression.  
  
  Ringing and blocking effect are two common, important, and unavoidable artifact of lossy image compression . Even though there are complex deblocking filters in recently compression techniques, for example H.264, there isn’t similar deblocking approach in wide used JPEG
  
  We will propose a convolution neural network which input is a decompressed image and output is a restored image to suppress ringing and blocking effect  and also have a satisfying performance.

## Network Architecture
Conv Unit which contains a convolutions layer with 3x3 kernel size followed by an 1x1 convolutions layer and a PReLU then followed by another 3x3 a convolutions layer and a PReLU. 

After three sequential Conv Unit, connect a 1x1 convolutions layer, 5x5 convolutions layer, and a PReLU.

![](/Result/Model.png)

### Loss Function
The mean-square error (MSE) is commonly used to compare image compression quality. We use MSE as our comparison standard and as our loss function, the formula is defined as:
MSE=1/WH sum(*xi*-*xi**)^2

Where *xi* and *x*i* are the values of the i-th pixel in *X* and *X**, *W* and *H* are the width and height of *X*. *X* denotes the original image and *X** denotes the restored image. 

## Dataset
The training dataset we used are **[Open Images 2019 (Google)](https://www.kaggle.com/c/open-images-2019-object-detection)**, **[BSR](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)**, and **[DIV2k](https://data.vision.ee.ethz.ch/cvl/DIV2K/)**. The training data is more than 10 million,contain lots of kinds of situations, and the size of validation date is about 10 thousand 2k images. 

At the training phase we also augmented the images with rotation and resizing for trying to make the training more comprehensive.

## Result
![](/Result/1.png)
![](/Result/7.png)
![](/Result/4.png)
![](/Result/6.png)

## Reference
1. [Compression Artifacts Reduction by a deep Convolutional Network](https://arxiv.org/abs/1504.06993)
2. [Deep Convolution Networks for Compression Artifacts Reduction](https://www.researchgate.net/publication/306185963_Deep_Convolution_Networks_for_Compression_Artifacts_Reduction)
3. [Near-lossless *l*`∞-constrained Image Decompression
via Deep Neural Network](https://arxiv.org/pdf/1801.07987.pdf)

