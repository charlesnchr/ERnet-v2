import argparse
import glob
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import random
import parser
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pickle

from skimage import io
import cv2

def noisy(noise_typ,image,opts=[0,0.005]):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = opts[0]
        var = opts[1]
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
        for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
        for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(opts[0]))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy/np.max(noisy)*np.max(image)
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = image + image * gauss
        return noisy

def remove_isolated_pixels(image):
    connectivity = 8

    output = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)

    num_stats = output[0]
    labels = output[1]
    stats = output[2]

    new_image = image.copy()

    for label in range(num_stats):
        if stats[label,cv2.CC_STAT_AREA] < 25:
            new_image[labels == label] = 0

    return new_image


def degrade(img,dim):
    # gaussian darkness
    X,Y = np.meshgrid(np.linspace(0,1,dim),np.linspace(0,1,dim))
    mu_x, mu_y = np.random.rand(), np.random.rand()
    var_x = np.max( [0, 0.05*np.random.randn() + 0.5] )
    var_y = np.max( [0, 0.05*np.random.randn() + 0.5] )
    Z = np.exp( -(X - mu_x)**2 / (2*var_x) ) * np.exp( -(Y - mu_y)**2 / (2*var_y) )
    Z = np.expand_dims(Z, 2)

    darkimg = Z*img
    # darkimg = (0.2*np.random.rand()+0.8)*darkimg  # overall level between 0.5 and 0.1

    # poisson_param = np.max([0,3*np.random.randn() + 10])
    # noisyimg = noisy('poisson',darkimg,[poisson_param])

    # gauss_param = np.max([0,0.0001*np.random.randn() + 0.0005])
    # noisyimg = noisy('gauss',darkimg,[0,gauss_param])

    # noisyimg = np.clip(noisyimg,0,1)
    return darkimg


def partitionDataset(imgs,outdir,nreps,dim,degradeBool=True):
    try:
        os.makedirs(outdir)
    except OSError:
        pass

    for i in range(0,len(imgs),2):

        inp_img_stack = io.imread(imgs[i])
        gt_img_stack = io.imread(imgs[i+1])
        nch = inp_img_stack.shape[0]

        for c_idx in range(nch):

            src_img = inp_img_stack[c_idx]
            if len(src_img.shape) > 2:
                src_img = src_img[:,:,0]
            h,w = src_img.shape
            src_gt_img = gt_img_stack[c_idx]
            # src_gt_img = src_gt_img[:,:,3]

            # get rid of gba channels and invert
            # src_gt_img = src_gt_img[:,:,0]

            # foreground = src_gt_img > 10
            # background = src_gt_img < 10

            background = src_gt_img[:] == 0
            foreground = src_gt_img[:] == 1
            sheet = src_gt_img[:] == 2
            tubuleOnsheet = src_gt_img[:] == 3

            src_gt_grayscale = np.zeros((h,w))
            src_gt_grayscale[background] = 0
            src_gt_grayscale[foreground] = 85
            src_gt_grayscale[sheet] = 170
            src_gt_grayscale[tubuleOnsheet] = 255
            src_gt_img = src_gt_grayscale

            # remove isolated pixels ?
            src_gt_img = remove_isolated_pixels(src_gt_img.astype('uint8'))


            # normalize and add dimension
            # print(src_img.shape,np.max(src_img))
            src_img = 255 * (src_img - np.min(src_img)) / (np.max(src_img) - np.min(src_img))


            j = 0
            while j < nreps:
                print(imgs[i], j)
                r_rand = np.random.randint(0,h-dim)
                c_rand = np.random.randint(0,w-dim)
                img = src_img[r_rand:r_rand+dim,c_rand:c_rand+dim]
                gt_img = src_gt_img[r_rand:r_rand+dim,c_rand:c_rand+dim]

                if np.mean(gt_img) < 0.05*255:
                    continue

                # adding random brightness
                brightness = 1 + 0.3*np.random.randn()
                img = np.clip(img* brightness,0,255)

                # img = (img - np.min(img)) / (np.max(img) - np.min(img))
                # gt_img = (gt_img - np.min(gt_img)) / (np.max(gt_img) - np.min(gt_img))

                # img = np.expand_dims(img, 2)
                # if degradeBool:
                #     img = degrade(img,dim)
                # img = img.squeeze()

                filename = '%s/%d-%d-%d.npy' % (outdir,c_idx,i,j)

                img = Image.fromarray(img.astype('uint8'))
                gt_img = Image.fromarray(gt_img.astype('uint8'))
                pickle.dump((img,gt_img), open(filename,'wb'))

                # img.save(filename.replace(".npy","_in.jpg"))
                # gt_img.save(filename.replace(".npy","_gt.jpg"))

                combined = np.concatenate((np.array(img),np.array(gt_img)),axis=1)
                io.imsave(filename.replace(".npy",".png"),combined)

                j += 1

            print('[%d/%d]' % (c_idx+i+1,nch*len(imgs)))


# --------------------------------------------

nreps = 10
dim = 256


# two image files: one for input and one for ground truth

allimgs = [
    "tubule on sheet/C1-561ER 640lysotracker mbcd 1s-inter1.cxd-fairSIM-1 sequence.tif",
    "tubule on sheet/Classification result-sequence.tif",
]

outdir = '4class_partitioned_' + str(dim)
print('Training data')
partitionDataset(allimgs,outdir,nreps,dim,False)


# test
# imgs = [allimgs[-1]]
# outdir = 'G:/Data/segmentation/partitioned_testset_' + str(dim)
# print('Test data')
# partitionDataset(imgs,outdir,nreps,dim)


# experimentally degraded
# imgs = ["G:\Data\segmentation\ER488 LYSO640 10s int-3.cxd-fairSIM.tif"]
# imgs = ["G:/Data/segmentation/ER488 SIRLYSO640 vapa 1s-inter24.cxd-fairSIM.tif"]
# outdir = 'G:/Data/segmentation/partitioned_valset2_' + str(dim)
# print('Val data')
# partitionDataset(imgs,outdir,nreps,dim,False)
