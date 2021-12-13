import PIL
import load_data
from tqdm import tqdm

from load_data import *
import gc
import matplotlib.pyplot as plt
from torch import autograd
import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter
import subprocess

import patch_config
import sys
import time

class WaymoPatchApplier(object):
	def __init__(self, mode):
		self.config = patch_config.patch_configs[mode]()

		self.darknet_model = Darknet(self.config.cfgfile)
		self.darknet_model.load_weights(self.config.weightfile)
		self.darknet_model = self.darknet_model.eval().cuda() # TODO: Why eval?
		self.patch_applier = PatchApplier().cuda()
		self.patch_transformer = PoseEstimationPatchTransformer().cuda()

		self.writer = self.init_tensorboard(mode)

	def init_tensorboard(self, name=None):
		if name is not None:
			time_str = time.strftime("%Y%m%d-%H%M%S")
			return SummaryWriter(f'runs/{time_str}_{name}')
		else:
			return SummaryWriter()

	def apply_patch_and_save(self):
		img_size = self.darknet_model.height
		batch_size = self.config.batch_size
		max_lab = 1

		time_str = time.strftime("%Y%m%d-%H%M%S")
		print(f'batch_size: {batch_size}')
		# Generate stating point
		# adv_patch_cpu = self.generate_patch("gray")
		# adv_patch_cpu = self.read_image("saved_patches/patch_image_waymo_2.png")
		adv_patch_cpu = self.read_image("patches/class_detection.png")

		adv_patch_cpu.requires_grad_(True)

		train_loader = torch.utils.data.DataLoader(InriaDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size),
												   batch_size=batch_size,
												   num_workers=1)
		self.number_of_images = len(train_loader)
		for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running Loading',
													total=self.number_of_images):
			img_batch = img_batch.cuda()
			lab_batch = lab_batch.cuda()
			adv_patch = adv_patch_cpu.cuda()
			adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)
			p_img_batch = self.patch_applier(img_batch, adv_batch_t)
			# self.writer.add_image('test_img'+str(i_batch), p_img_batch[0])
			
			print(f'writing image {i_batch}')
			image_name = self.config.dir_to_store+'og/'+str(i_batch)+'_og_img.jpg'
			adv_image_name = self.config.dir_to_store+'patched/'+str(i_batch)+'_patched_img.jpg'

			# og_img_tensor = self.resize_img(p_img_batch[0], 1280, 1920)
			# adv_img_tensor = self.resize_img(img_batch[0], 1280, 1920)
			self.writer.add_image('test_img'+str(i_batch), p_img_batch[0])

			torchvision.utils.save_image(p_img_batch[0], adv_image_name)
			torchvision.utils.save_image(img_batch[0], image_name)

	def generate_patch(self, type):
		"""
		Generate a random patch as a starting point for optimization.

		:param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
		:return:
		"""
		print(f'patch size: {self.config.patch_size}')
		if type == 'gray':
			adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)
		elif type == 'random':
			adv_patch_cpu = torch.rand((3, self.config.patch_size, self.config.patch_size))

		return adv_patch_cpu

	def read_image(self, path):
		"""
		Read an input image to be used as a patch

		:param path: Path to the image to be read.
		:return: Returns the transformed patch as a pytorch Tensor.
		"""
		patch_img = Image.open(path).convert('RGB')
		tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
		patch_img = tf(patch_img)
		tf = transforms.ToTensor()
		adv_patch_cpu = tf(patch_img)
		return adv_patch_cpu
	
	def resize_img(self, img_tensor, width, length):
		tf = transforms.Resize((length, width))
		img_tensor = tf(img_tensor)
		return img_tensor


def main():
	if len(sys.argv) != 2:
		print('You need to supply (only) a configuration mode.')
		print('Possible modes are:')
		print(patch_config.patch_configs)

	waymo_patch_applier = WaymoPatchApplier(sys.argv[1])
	waymo_patch_applier.apply_patch_and_save()

if __name__ == '__main__':
	main()


