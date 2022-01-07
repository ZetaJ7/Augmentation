import albumentations as A
import cv2
import numpy as np


def aug(image,width,height):
    # Declare an augmentation pipeline
    transform=A.Compose([A.RandomCrop(width=int(0.8*width),height=int(0.8*height),p=0.5),A.HorizontalFlip(p=0.5),
                         A.VerticalFlip(p=0.5),A.RandomRotate90(p=0.5),A.ChannelShuffle(p=0.5),A.RGBShift(p=0.8),
                         A.Blur(p=0.5),A.RandomBrightnessContrast(p=0.2),],bbox_params=A.BboxParams(format='yolo'))
    # BBox available: Affine, CenterCrop, Crop, CropAndPad,CropNoneEmptyMaskIfExists, Flip, HorizentalFlip, Lambda,
    #                 LongestMaxSize, NoOp, PadIfNeeded, Perspective, PiecewiseAffine, RandomCrop, RandomCropNearBBox,
    #                 RandomSizedBBoxSafeCrop, RandomSizedCrop, Resize, Rotate, SafeRotate, ShiftScaleRotate,
    #                 SmallestMaxSize, Transpose, VerticalFlip

    cv2.imshow('origin',image)
    cv2.waitKey(2000)
    # Read an image and convert to RGB colorspace

    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    # Augment an image
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    img = transformed_image[:,:,::-1]
    cv2.imshow('test',img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    path = '000053.jpg'
    image = cv2.imread(path)
    width=image.shape[1]
    height=image.shape[0]
    aug(image=image,width=width,height=height)

