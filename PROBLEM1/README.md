``python main.py --model <model.pt> --data-dir <DATA_DIR>``
example: ``python main.py --model best.pt --data-dir NGD_HACK``


## Experimentation

### Data augmentation
We implemented real-time data augmentation using Keras, dynamically enhancing the diversity of our dataset during training. However, we faced challenges due to class imbalance within the dataset, which motivated us to explore several strategies to mitigate this issue.

One approach we experimented with was Synthetic Minority Over-sampling Technique (SMOTE). SMOTE generates synthetic data points in feature space by interpolating between existing examples and their nearest neighbors, effectively balancing the representation of minority classes. To apply SMOTE effectively, it was necessary to first embed the image data into a suitable feature space.

Although this approach showed promise in addressing imbalance, it introduced practical limitations. Specifically, embedding the images meant we could no longer utilize models like YOLO, which inherently require raw image inputs. As YOLO processes pixel-level data directly, our feature-space augmentation using SMOTE was incompatible with its architecture. Consequently, despite the theoretical benefits of SMOTE, the incompatibility with critical components of our pipeline rendered it ultimately ineffective for our purposes.

Despite these limitations, our experiments provided valuable insights into data augmentation techniques and underscored the importance of considering model-specific requirements when preprocessing data.

