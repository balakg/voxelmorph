# py imports
import os
import sys
import glob

# third party
import tensorflow as tf
import scipy.io as sio
import numpy as np
from keras.backend.tensorflow_backend import set_session
from scipy.interpolate import interpn

# project
sys.path.append('../ext/medipy-lib')
import medipy
import networks
from medipy.metrics import dice
import datagenerators


def test(model_name, gpu_id, iter_num, vol_size=(160,192,224), nf_enc=[16,32,32,32], nf_dec=[32,32,32,32,8,8,3]):
	"""
	test

	nf_enc and nf_dec
	#nf_dec = [32,32,32,32,32,16,16,3]
    # This needs to be changed. Ideally, we could just call load_model, and we wont have to
    # specify the # of channels here, but the load_model is not working with the custom loss...
    """  

	gpu = '/gpu:' + str(gpu_id)

	# Test file and anatomical labels we want to evaluate
	test_brain_file = open('../data/test_examples.txt')
	test_brain_strings = test_brain_file.readlines()
	test_brain_strings = [x.strip() for x in test_brain_strings]
	good_labels = sio.loadmat('../data/test_labels.mat')['labels'][0]

	atlas = np.load('../data/atlas_norm.npz')
	atlas_vol = atlas['vol']
	atlas_seg = atlas['seg']
	atlas_vol = np.reshape(atlas_vol, (1,)+atlas_vol.shape+(1,))

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	set_session(tf.Session(config=config))

	# load weights of model
	with tf.device(gpu):
		net = networks.unet(vol_size, nf_enc, nf_dec)
		net.load_weights('../models/' + model_name +
                         '/' + str(iter_num) + '.h5')

	n_batches = len(test_brain_strings)
	xx = np.arange(vol_size[1])
	yy = np.arange(vol_size[0])
	zz = np.arange(vol_size[2])
	grid = np.rollaxis(np.array(np.meshgrid(xx, yy, zz)), 0, 4)

	dice_vals = np.zeros((len(good_labels), n_batches))

	np.random.seed(17)

	for k in range(0, n_batches):
		vol_name, seg_name = test_brain_strings[k].split(",")
		X_vol, X_seg = datagenerators.load_example_by_name(vol_name, seg_name)

		with tf.device(gpu):
			pred = net.predict([X_vol, atlas_vol])

		# Warp segments with flow
		flow = pred[1][0, :, :, :, :]
		sample = flow+grid
		sample = np.stack(
			(sample[:, :, :, 1], sample[:, :, :, 0], sample[:, :, :, 2]), 3)
		warp_seg = interpn(
			(yy, xx, zz), X_seg[0, :, :, :, 0], sample, method='nearest', bounds_error=False, fill_value=0)

		vals, labels = dice(warp_seg, atlas_seg, labels=good_labels, nargout=2)
		dice_vals[:, k] = vals
		print np.mean(dice_vals[:, k])

		#sio.savemat('../results/test/' + model_name + '_' + iter_num + '.mat', {'dice_vals': dice_vals, 'labels': good_labels})


if __name__ == "__main__":
	test(sys.argv[1], sys.argv[2], sys.argv[3])
