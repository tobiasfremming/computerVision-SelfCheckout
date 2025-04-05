``python main.py --model <model.pt> --data-dir <DATA_DIR>``
example: ``python main.py --model best.pt --data-dir NGD_HACK``



We tried training on a few different models, but got the best result with YOLO models. We started of with yolov11nano, but later increased to the small model. 
We did a few differnet augmentations, both with albumentations, and with yolos built in augmentations. 