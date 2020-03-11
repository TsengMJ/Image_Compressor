# Image Restorer
這個專案目的是嘗試使用深度學習修復經過 JPEG 壓縮後的影像，實做框架使用 pytorch 進行訓練。其中程式主要分為3個部份。

The purpose of this project is trying to use deep learning to repair JPEG compressed images, the using framework is pytorch. The whole process is mainly divided into 3 parts. 
* 資料讀取 (Loading data) - DataLoader
* 建立模型 (Building model) - Model
* 訓練模型 (Training model) - Training

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


## 建立模型 (Building model)：
**Conv Unit**:

Contains a convolutions layer with 3x3 kernel size followed by an 1x1 convolutions layer and a PReLU then followed by another 3x3 a convolutions layer and a PReLU.

```
Conv Unit EX:

  nn.Conv2d(3, 128, kernel_size=3, padding=1)
  nn.Conv2d(128, 128, kernel_size=1)
  nn.PReLU()

  nn.Conv2d(128, 128, kernel_size=3, padding=1)
  nn.PReLU()
```

**完整架構 (Full Architecture)**:
```
self.base = nn.Sequential(
  nn.Conv2d(3, 128, kernel_size=3, padding=1),
  nn.Conv2d(128, 128, kernel_size=1),
  nn.PReLU(),

  nn.Conv2d(128, 128, kernel_size=3, padding=1),
  nn.PReLU(),

  nn.Conv2d(128, 64, kernel_size=3, padding=1),
  nn.Conv2d(64, 64, kernel_size=1),
  nn.PReLU(),

  nn.Conv2d(64, 64, kernel_size=3, padding=1),
  nn.PReLU(),

  nn.Conv2d(64, 32, kernel_size=3, padding=1),
  nn.Conv2d(32, 32, kernel_size=1),
  nn.PReLU(),

  nn.Conv2d(32, 32, kernel_size=3, padding=1),
  nn.PReLU(),

  nn.Conv2d(32, 16, kernel_size=1),
  nn.PReLU()
)
self.last = nn.Conv2d(16, 3, kernel_size=5, padding=2)

self._initialize_weights()
```

架構圖

![](/Result/Model.png)

## 訓練模型 (Training model)
**步驟 (Steps)**:
1. 初始化參數 (Initialize hyper parameters)
```
batch_size = 8
threads = 4
lr = 5e-4
num_epochs = 100
patch_size = 128
jpeg_quality = 10
```
2. 讀取模型並初始化優化器(Loading model and initialize optimizer)
```
model = My_Model()

model = model.to(device)
criterion = nn.MSELoss(reduction='sum')

optimizer = optim.Adam([
    {'params': model.base.parameters()},
    {'params': model.last.parameters(), 'lr': lr * 0.5},
], lr=lr)
```
3. 初始化 DataLoader (Initialize dataloader)
```
dataset = Dataset(images_dir, patch_size, jpeg_quality, use_augmentation, use_fast_loader)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                        num_workers=threads, pin_memory=True,drop_last=True)
```

4. 正式開訓練 (Start training!)
```
for epoch in range(num_epochs):
    epoch_losses = AverageMeter()
    target_losses = AverageMeter()

    with tqdm(total=(len(dataset) - len(dataset) % batch_size)) as _tqdm:
        _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, num_epochs))
        for data in dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs)

            loss = criterion(preds, labels)
            target = criterion(inputs, labels)

            epoch_losses.update(loss.item(), len(inputs))
            target_losses.update(target.item(), len(inputs))


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _tqdm.set_postfix(loss='{:.6f}, target{:.6f}'.format(epoch_losses.avg, target_losses.avg))
            _tqdm.update(len(inputs))
```

**[Note]**

這邊訓練沒有用驗證資料是因為篤定資料量夠大且變化多種，所以比較不會過度擬合，但訓練時最好還是乖乖用 Validation data 追蹤訓練過程～

The validation data is not used for training here because it is determined that the amount of data is large enough and diverse, so it is less likely to overfit, but it is best to use it well during training.

## 結果 (Result):

![](/Result/1.png)
![](/Result/7.png)
![](/Result/4.png)

失敗範例 (Faliure example)

![](/Result/5.png)


## Reference:
1. [Compression Artifacts Reduction by a deep Convolutional Network](https://arxiv.org/abs/1504.06993)
2. [Deep Convolution Networks for Compression Artifacts Reduction](https://www.researchgate.net/publication/306185963_Deep_Convolution_Networks_for_Compression_Artifacts_Reduction)
3. [Near-lossless *l*`∞-constrained Image Decompression
via Deep Neural Network](https://arxiv.org/pdf/1801.07987.pdf)

