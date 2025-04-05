``python main.py --model <model.pt> --data-dir <DATA_DIR>``
example: ``python main.py --model best.pt --data-dir NGD_HACK``
Use PROBLEM2/model/best.pt


## Experimentation

#### Classification
We implemented data augmentation using Keras, dynamically enhancing the diversity of our dataset during training. However, we faced challenges due to class imbalance within the dataset, which motivated us to explore several strategies to mitigate this issue.

One approach we experimented with was Synthetic Minority Over-sampling Technique (SMOTE). SMOTE generates synthetic data points in feature space by interpolating between existing examples and their nearest neighbors, effectively balancing the representation of minority classes. To apply SMOTE effectively, it was necessary to first embed the image data into a suitable feature space.

Although this approach showed promise in addressing imbalance, it introduced practical limitations. Specifically, embedding the images meant we could no longer utilize models like YOLO, which inherently require raw image inputs. As YOLO processes pixel-level data directly, our feature-space augmentation using SMOTE was incompatible with its architecture. Consequently, despite the theoretical benefits of SMOTE, the incompatibility with critical components of our pipeline rendered it ultimately ineffective for our purposes.

### Object detection

For object detection we decided to use YOLOv11 small. We tried the nano model, but it turned out to be quite limited. We achived better results by employing a small model which ran for 50 epochs with a batch size of 64. We avoided the YOLOv12 series as they use a architecture which involves transformers. Transformers require lots of data and trainingtime. 

To do object detection we used data augmentation at trainingtime. We used the following data augmentation parameters:
```
    mosaic=1.0,
    mixup=0.5,
    copy_paste=0.3,copy-paste augmentation (copies objects between images)
    degrees=20.0,
    shear=10.0,
    perspective=0.0015,
    scale=0.5,
    fliplr=0.5,
```
Combining these yielded quite good results. 

To fix the class imbalance in the dataset we set a loss multiplier for every class making our model focus on making good predicions for every class. This improved results.

We tried to use different models or train longer. However we soon realized that a smaller model which ran for fewer epochs outperformed a larger model.


### CenterNet pose estimation

In the last hours of the hackathon we realized we (and chatgpt) approached the entire hackathon the wrong way. We should have employed pose estimation to predict the position of different classes. This would likely yield better results while being more performant.