import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.segmentation import slic


class Focaccia:
  def __init__ (self, path: str, target_point: tuple[int, int], lambda_: float, inside: bool):
    self.lambda_ = lambda_
    self.inside = inside
    self.attach_image(path)
    self.attach_mask(target_point)

  def attach_image (self, path: str):
    self.img = imread(path)[:, :, :3] / 255
  
  def attach_mask (self, target_point: tuple[int, int]):
    segment = slic(self.img, n_segments=100, compactness=10, sigma=1, start_label=1)
    self.mask = (segment == segment[target_point[0], target_point[1]])

  def result_show (self):
    fig = plt.figure()
    fig.add_subplot(3, 2, 1)
    plt.imshow(self.img)
    fig.add_subplot(3, 2, 2)
    plt.imshow(self.saliency_map(self.img))
    fig.add_subplot(3, 2, 3)
    plt.imshow(self.applied)
    fig.add_subplot(3, 2, 4)
    plt.imshow(self.saliency_map(self.applied))
    fig.add_subplot(3, 2, 5)
    plt.imshow(self.img - self.applied)
    fig.add_subplot(3, 2, 6)
    plt.imshow(self.saliency_map(self.applied) - self.saliency_map(self.img))
    plt.show()

  def show (self):
    plt.imshow(self.img)
    plt.show()

  def toBgr (self):
    return self.img[:, :, ::-1]
  
  def rgb2bgr (self, img):
    return img[:, :, ::-1]
  
  def bgr2rgb (self, img):
    return img[:, :, ::-1]

  def saliency_map (self, img):
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    return saliency.computeSaliency(self.rgb2bgr(img).astype(np.float32))[1]
  
  def saliency_diff (self):
    saliency_map = self.saliency_map(self.img)
    return (saliency_map[self.mask == 1]).mean() - (saliency_map[self.mask == 0]).mean()

  def score (self, output):
    return -self.saliency_diff() + self.lambda_ * np.linalg.norm(output - self.img)
