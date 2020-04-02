from torch.utils.data import Dataset
#from scipy.misc import imread # scipy (v0.18)
from scipy.ndimage import imread
import os


class Miniplaces(Dataset):
	def __init__(self, data_path, split='train', transform=None):
		self.imgs = []
		self.labels = []
		if split == 'train' or split == 'val':
			with open(data_path + split + '.txt') as f:
				for line in f:
					img_path, label = line.split()
					self.labels.append(int(label))
					self.imgs.append(data_path + 'images/' + img_path)
		else:
			test_path = data_path + 'images/test/'
			for img in os.listdir(test_path):
				self.imgs.append(test_path + img)
			self.images = sorted(self.images)
		self.transform = transform

	def __getitem__(self, index):
		img = imread(self.imgs[index], mode='RGB').astype(float)
		label = None
		if len(self.labels) != 0:
			label = self.labels[index]
		else:
			label = -1
		if self.transform != None:
			img = self.transform(img)
		return img, label

	def __len__(self):
return len(self.imgs)

