import os

import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch.utils.data as data
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
  return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_csv_lines(csv_file):
  with open(csv_file) as f:
    lines = [line.strip() for line in f.readlines()]
  return lines


def find_classes(csv_lines):
  _lines = [line.split(',')[-1] for line in csv_lines]
  classes = sorted(list(set(_lines)))
  # with open('classes.txt', 'w') as f:
  #   for cls in classes:
  #     f.write('{}\n'.format(cls))
  class_to_idx = {cls: i for i, cls in enumerate(classes)}
  return classes, class_to_idx


def make_dataset(csv_lines, class_to_idx):
  images = []
  for line in csv_lines:
    filename, target = line.split(',')

    if is_image_file(filename):
      item = (filename, class_to_idx[target])
      images.append(item)

  return images


def default_loader(path, phase):
  return Image.open(path).convert('RGB')


def my_loader(path, phase):
  img = cv2.imread(path)
  large = np.min(img.shape[:-1]) > 400
  if large:
    img = cv2.resize(img, (int(img.shape[1] * 0.3), int(img.shape[0] * 0.3)))
  img = cv2.resize(img, (224, 224))

  if phase == 'train' and np.random.random() < 0.2:
    img = cv2.flip(img, 1)

  if phase == 'train' and np.random.random() < 0.95:
    seq = iaa.SomeOf((3, 5), [
        iaa.AdditiveGaussianNoise(loc=(0.8, 1.2), scale=(0, 3)),
        iaa.Add((-10, 10), per_channel=0.5),
        iaa.Multiply((0.9, 1.1), per_channel=0.5),
        iaa.GaussianBlur((0.9, 1.1)),
        iaa.ContrastNormalization((0.7, 1.3), per_channel=0.5)
    ])
    img = seq.augment_image(img)

  if phase == 'train' and np.random.random() < 0.9 and large:
    seq = iaa.Sequential([
        iaa.Crop(percent=((0.1, 0.35), (0.1, 0.35), (0.1, 0.35), (0.1, 0.35)), keep_size=False)
    ])
    img = seq.augment_image(img)

  try:
    image = np.zeros((224, 224, 3), dtype=np.float)
    image[:, :, :] = [104, 117, 124]
    height = img.shape[0]
    width = img.shape[1]
    if max(height, width) > 224:
      if height > width:
        ratio = 224 / height
        width = int(ratio * width)
        img = cv2.resize(img, (width, 224))
      else:
        ratio = 224 / width
        height = int(ratio * height)
        img = cv2.resize(img, (224, height))
    height = img.shape[0]
    width = img.shape[1]
    if height >= width:
      ratio = 224 / height
      width = int(ratio * width)
      img = cv2.resize(img, (width, 224))
      image[:, int((224 - width) / 2):int((224 - width) / 2) + width, :] = img
    else:
      ratio = 224 / width
      height = int(ratio * height)
      img = cv2.resize(img, (224, height))
      image[int((224 - height) / 2):int((224 - height) / 2) + height, :, :] = img
  except IOError:
    print('Read image `{}` error.'.format(path))
    return -1, -1, -1

  image = 2 * (image / 255. - 0.5)

  return image


class ImageCSV(data.Dataset):
  def __init__(self, csv_file, phase, transform=None, target_transform=None,
               loader=default_loader):
    self.csv_lines = get_csv_lines(csv_file)
    classes, class_to_idx = find_classes(self.csv_lines)
    imgs = make_dataset(self.csv_lines, class_to_idx)

    self.imgs = imgs
    self.classes = classes
    self.class_to_idx = class_to_idx
    self.phase = phase
    self.transform = transform
    self.target_transform = target_transform
    self.loader = loader

  def __getitem__(self, index):
    path, target = self.imgs[index]
    img = self.loader(path, self.phase)
    if self.transform:
      img = self.transform(img)
    if self.target_transform:
      target = self.target_transform(target)

    return img, target

  def __len__(self):
    return len(self.imgs)
