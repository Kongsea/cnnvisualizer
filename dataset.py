import torch.utils.data as data
from PIL import Image

IMG_EXTENSIONS = ['.png', '.jpg']


class Dataset(data.Dataset):

  def __init__(self, imglist, transform=None):

    if not imglist:
      raise(RuntimeError("Found 0 images in subfolders.\n"
                         "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

    self.imgs = imglist
    self.transform = transform

  def __getitem__(self, index):
    path = self.imgs[index]
    img = Image.open(path).convert('RGB')
    if self.transform is not None:
      img = self.transform(img)
    return img, path

  def __len__(self):
    return len(self.imgs)
