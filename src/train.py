"""
train atlas-based alignment with voxelmorph
"""

# python imports
import os
import glob
import sys
import random

# third-party imports
import tensorflow as tf
import numpy as np
import scipy.io as sio
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.models import load_model, Model

# project imports
sys.path.append('../ext/medipy-lib')
import medipy
import datagenerators
import networks
import losses

# configuration
n_iterations = 200001
lr = 1e-4
reg_param = 1.0
model_save_iter = 5000

# UNET filters
nf_enc = [16, 32, 32, 32]
# VM-1
nf_dec = [32, 32, 32, 32, 8, 8, 3]
# VM-2
#nf_dec = [32,32,32,32,32,16,16,3]

vol_size = (160, 192, 224)
base_data_dir = '/insert/your/path/here/'
train_vol_names = glob.glob(base_data_dir + 'train/vols/*.npz')
random.shuffle(train_vol_names)

#val_vol_names = glob.glob(base_data_dir + 'test/vols/*.npz')
#train_seg_dir = base_data_dir + 'train/asegs/'
#val_seg_dir = base_data_dir + 'test/asegs/'

atlas = np.load('../data/atlas_norm.npz')
atlas_vol = atlas['vol']
atlas_vol = np.reshape(atlas_vol, (1,) + atlas_vol.shape+(1,))


def train(model_name, gpu_id):

    model_dir = '../models/' + model_name
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    gpu = '/gpu:' + str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    with tf.device(gpu):
        model = networks.unet(vol_size, nf_enc, nf_dec)
        model.compile(optimizer=Adam(lr=lr), loss=[
                      losses.cc3D(), losses.gradientLoss('l2')], loss_weights=[1.0, reg_param])
        # model.load_weights('../models/udrnet2/udrnet1_1/120000.h5')

    train_example_gen = datagenerators.example_gen(train_vol_names)
    zero_flow = np.zeros((1, vol_size[0], vol_size[1], vol_size[2], 3))

    for step in xrange(0, n_iterations):

        X = train_example_gen.next()[0]
        train_loss = model.train_on_batch(
            [X, atlas_vol], [atlas_vol, zero_flow])

        if not isinstance(train_loss, list):
            train_loss = [train_loss]

        printLoss(step, 1, train_loss)

        if(step % model_save_iter == 0):
            model.save(model_dir + '/' + str(step) + '.h5')


def printLoss(step, training, train_loss):
    s = str(step) + "," + str(training)

    if(isinstance(train_loss, list) or isinstance(train_loss, np.ndarray)):
        for i in xrange(len(train_loss)):
            s += "," + str(train_loss[i])
    else:
        s += "," + str(train_loss)

    print s
    sys.stdout.flush()


if __name__ == "__main__":
    if(len(sys.argv) != 3):
        print "Need model name and gpu id as command line arguments."
    else:
        train(sys.argv[1], sys.argv[2])
