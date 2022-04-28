import os

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import glob
from PIL import Image, ImageOps
import random

import numpy as np

from skimage import io, exposure, transform


def PSNR(I0,I1):
    MSE = torch.mean( (I0-I1)**2 )
    PSNR = 20*torch.log10(1/torch.sqrt(MSE))
    return PSNR


#normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
#                               std = [0.229, 0.224, 0.225])

scale = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize(48),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                std = [0.229, 0.224, 0.225])
                            ])

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
unnormalize = transforms.Normalize(mean = [-2.118, -2.036, -1.804], std = [4.367, 4.464, 4.444])

normalize2 = transforms.Normalize(mean = [0.69747254,0.53480325,0.68800158], std = [0.23605522,0.27857294,0.21456957])
unnormalize2 = transforms.Normalize(mean = [-2.9547, -1.9198, -3.20643], std = [4.2363, 3.58972, 4.66049])


toTensor = transforms.ToTensor()  
toPIL = transforms.ToPILImage()      


def GetDataloaders(opt):

    # if no separate validation set has been provided
    if opt.rootValidation is None:
        opt.rootValidation = opt.root

    # dataloaders
    if opt.dataset.lower() == 'div2k':
        dataloader = load_DIV2K_dataset(opt.root,'train',opt)
        validloader = load_DIV2K_dataset(opt.rootValidation,'valid',opt)
    elif opt.dataset.lower() == 'div2k_raw':
        dataloader = load_DIV2K_dataset(opt.root + '/DIV2K','train',opt)
        validloader = load_DIV2K_dataset(opt.rootValidation + '/DIV2K','valid',opt)
    elif opt.dataset.lower() == 'pcam': 
        dataloader = load_HDF5_dataset(opt.root + '/camelyonpatch_level_2_split_valid_x.h5','train',opt)
        validloader = load_HDF5_dataset(opt.rootValidation + '/camelyonpatch_level_2_split_valid_x.h5','valid',opt)
    elif opt.dataset.lower() == 'imagedataset': 
        dataloader = load_image_dataset(opt.root,'train',opt)
        validloader = load_image_dataset(opt.rootValidation,'valid',opt)
    elif opt.dataset.lower() == 'imageclassdataset': 
        dataloader = load_imageclass_dataset(opt.root,'train',opt)
        validloader = load_imageclass_dataset(opt.rootValidation,'valid',opt)
    elif opt.dataset.lower() == 'doubleimagedataset': 
        dataloader = load_doubleimage_dataset(opt.root,'train',opt)
        validloader = load_doubleimage_dataset(opt.rootValidation,'valid',opt)
    elif opt.dataset.lower() == 'pickledataset': 
        dataloader = load_GenericPickle_dataset(opt.root,'train',opt)
        validloader = load_GenericPickle_dataset(opt.rootValidation,'valid',opt)
    elif opt.dataset.lower() == 'sim': 
        dataloader = load_SIM_dataset(opt.root,'train',opt)
        validloader = load_SIM_dataset(opt.rootValidation,'valid',opt)
    elif opt.dataset.lower() == 'realsim': 
        dataloader = load_real_SIM_dataset(opt.root,'train',opt)
        validloader = load_real_SIM_dataset(opt.rootValidation,'valid',opt)        
    elif opt.dataset.lower() == 'fouriersim': 
        dataloader = load_fourier_SIM_dataset(opt.root,'train',opt)
        validloader = load_fourier_SIM_dataset(opt.rootValidation,'valid',opt)                
    elif opt.dataset.lower() == 'fourierfreqsim': 
        dataloader = load_freq_fourier_SIM_dataset(opt.root,'train',opt)
        validloader = load_freq_fourier_SIM_dataset(opt.rootValidation,'valid',opt)
    elif opt.dataset.lower() == 'ntiredenoising': 
        dataloader = load_NTIREDenoising_dataset(opt.root,'train',opt)
        validloader = load_NTIREDenoising_dataset(opt.rootValidation,'valid',opt)                
    elif opt.dataset.lower() == 'ntireestimatenl': 
        dataloader = load_EstimateNL_dataset(opt.root,'train',opt)
        validloader = load_EstimateNL_dataset(opt.rootValidation,'valid',opt)                
    elif opt.dataset.lower() == 'er':
        dataloader = load_ER_dataset(opt.root,category='train',batchSize=opt.batchSize,num_workers=opt.workers)
        validloader = load_ER_dataset(opt.rootValidation,category='valid',shuffle=False,batchSize=opt.batchSize,num_workers=0)        
    else:
        print('unknown dataset')
        return None,None
    return dataloader, validloader


class ImageDataset(Dataset):

    def __init__(self, root, category, opt):
        self.images = glob.glob(root + '/**/*.*', recursive=True)

        if category == 'train':
            self.images = self.images[:-10]
        else:
            self.images = self.images[-10:]

        self.croptransform = transforms.Compose([transforms.RandomCrop(opt.imageSize*opt.scale),transforms.ToTensor()])
        self.scaletransform = transforms.Compose([transforms.ToPILImage(),transforms.Resize(opt.imageSize),transforms.ToTensor()])
        self.imageSize = opt.imageSize
        self.scale = opt.scale
        self.len = len(self.images)
        
    def __getitem__(self, index):
        with open(self.images[index], 'rb') as f:
            hr = Image.open(f)
            hr.LOAD_TRUNCATED_IMAGES = True
            hr = hr.convert('RGB')

        hr = self.croptransform(hr)
        lr = self.scaletransform(hr)

        return lr, hr

    def __len__(self):
        return self.len


def load_image_dataset(root,category,opt):
        
    dataset = ImageDataset(root, category, opt)
    if category == 'train':
        dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)
    else:
        dataloader = DataLoader(dataset, batch_size=opt.batchSize_test, shuffle=False, num_workers=0)
    return dataloader



class ImageClassDataset(Dataset):

    def __init__(self, root, category, opt):
        self.images = glob.glob(root + '/**/*.*', recursive=True)
        
        # filter zero classes out
        newimages = []
        for img in self.images:
            if not '_0.tif' in img:
                newimages.append(img)
        self.images = newimages

        random.seed(1234)
        random.shuffle(self.images)

        if category == 'train':
            self.images = self.images[:opt.ntrain]
        else:
            self.images = self.images[-opt.ntest:]

        self.croptransform = transforms.Compose([transforms.RandomCrop(opt.imageSize*opt.scale),transforms.ToTensor()])
        self.scaletransform = transforms.Compose([transforms.ToPILImage(),transforms.Resize(opt.imageSize),transforms.ToTensor()])
        self.imageSize = opt.imageSize
        self.nch_in = opt.nch_in
        self.scale = opt.scale
        self.len = len(self.images)
        
    def __getitem__(self, index):
        stack = io.imread(self.images[index])
        stack[:,:,0] = stack[:,:,1]
        stack[:,:,2] = stack[:,:,1]
        dims = (self.imageSize,self.imageSize,self.nch_in)
        stack = transform.resize(stack,dims)

        # for i in range(self.nch_in):
        #     stack[i] = (stack[i] - np.min(stack[i])) / (np.max(stack[i]) - np.min(stack[i]))
        
        stack = torch.tensor(stack).float().permute(2,0,1)

        label = os.path.basename(self.images[index])

        label = label.split('_')[1]
        label = label.split('.tif')[0]
        label = int(label)
        if label < 3:
            label = 0
        else:
            label = 1

        # label = label.split('_')[0]
        # if 'cats' in label:
        #     label = 0
        # elif 'dogs' in label:
        #     label = 1
        # else:
        #     label = 2 # pandas
            

        return stack, label

    def __len__(self):
        return self.len


def load_imageclass_dataset(root,category,opt):
        
    dataset = ImageClassDataset(root, category, opt)
    if category == 'train':
        dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)
    else:
        dataloader = DataLoader(dataset, batch_size=opt.batchSize_test, shuffle=False, num_workers=0)
    return dataloader




class DoubleImageDataset(Dataset):

    def __init__(self, root, category, opt):
        self.lq = glob.glob(root.split(',')[0] + '/**/*.tif', recursive=True)
        self.hq = glob.glob(root.split(',')[1] + '/**/*.tif', recursive=True)
        random.seed(1234)
        random.shuffle(self.lq)
        random.seed(1234)
        random.shuffle(self.hq)
        print(self.lq[:3])
        print(self.hq[:3])


        if category == 'train':
            self.lq = self.lq[:-10]
            self.hq = self.hq[:-10]
        else:
            self.lq = self.lq[-10:]
            self.hq = self.hq[-10:]

        self.imageSize = opt.imageSize
        self.scale = opt.scale
        self.nch_in = opt.nch_in
        self.len = len(self.lq)
        
    def __getitem__(self, index):
        with open(self.lq[index], 'rb') as f:
            img = Image.open(f)
            img = np.array(img)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            lq = Image.fromarray(img)
        with open(self.hq[index], 'rb') as f:
            img = Image.open(f)
            img = np.array(img)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            hq = Image.fromarray(img)
        
        # random crop
        w,h = lq.size
        ix = random.randrange(0,w-self.imageSize+1)
        iy = random.randrange(0,h-self.imageSize+1)

        lq = lq.crop((ix,iy,ix+self.imageSize,iy+self.imageSize))
        hq = hq.crop((ix,iy,ix+self.imageSize,iy+self.imageSize))

        lq, hq = toTensor(lq), toTensor(hq)
        
        # rotate and flip?
        if random.random() > 0.5:
            lq = lq.permute(0, 2, 1)
            hq = hq.permute(0, 2, 1)
        if random.random() > 0.5:
            lq = torch.flip(lq, [1])
            hq = torch.flip(hq, [1])
        if random.random() > 0.5:
            lq = torch.flip(lq, [2])
            hq = torch.flip(hq, [2])
                
        if self.nch_in == 1:
            lq = torch.mean(lq,0,keepdim=True)
            hq = torch.mean(hq,0,keepdim=True)
        
        return lq,hq,hq

    def __len__(self):
        return self.len


def load_doubleimage_dataset(root,category,opt):
        
    dataset = DoubleImageDataset(root, category, opt)
    if category == 'train':
        dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)
    else:
        dataloader = DataLoader(dataset, batch_size=opt.batchSize_test, shuffle=False, num_workers=0)
    return dataloader



# class DIV2KDataset(Dataset):

#     def get_patch(self, lr, hr, patch_size=96, scale=4):
#         iw, ih = lr.size
#         p = 1
#         tp = patch_size
#         ip = tp // scale
        
#         ix = random.randrange(0, iw - ip + 1)
#         iy = random.randrange(0, ih - ip + 1)
        
        
#         tx, ty = scale * ix, scale * iy
        
#         # lr = transforms.functional.crop(lr, iy, ix, ip, ip)
#         # hr = transforms.functional.crop(hr, ty, tx, tp, tp)
#         lr = lr.crop((ix,iy,ix+ip,iy+ip))
#         hr = hr.crop((tx,ty,tx+tp,ty+tp))
        
#     #     return [
#     #         lr[:,iy:iy + ip, ix:ix + ip],
#     #         hr[:,ty:ty + tp, tx:tx + tp]
#     #     ]
#         return lr, hr    


#     def __init__(self, root, category, opt):
#         self.HRimages = glob.glob(root + '/DIV2K_' + category + '_HR/*')
#         self.LRimages = glob.glob(root + '/DIV2K_' + category + '_LR_bicubic/X4/*')

#         self.imageSize = imageSize
#         self.scale = scale
#         self.len = len(self.HRimages)


#     def __getitem__(self, index):
#         with open(self.HRimages[index], 'rb') as f:
#             hr = Image.open(f)
#             hr = hr.convert('RGB')
#         with open(self.LRimages[index], 'rb') as f:
#             lr = Image.open(f)
#             lr = lr.convert('RGB')
        
#         lr, hr = self.get_patch(lr,hr,self.imageSize*4, self.scale)
#         return toTensor(lr), toTensor(hr)

#     def __len__(self):
#         return self.len


# class DIV2KDataset(Dataset):
    
#     def __init__(self, root, category, opt):
#         self.images = glob.glob(root + '/DIV2K_' + category + '_HR/*')

#         self.croptransform = transforms.Compose([
#             transforms.RandomCrop(opt.imageSize*opt.scale),
#             transforms.ToTensor() ])
#         self.scaletransform = transforms.Compose([transforms.ToPILImage(),transforms.Resize(opt.imageSize),transforms.ToTensor()])
#         self.imageSize = opt.imageSize
#         self.scale = opt.scale
#         self.len = len(self.images)


#     def __getitem__(self, index):
#         with open(self.images[index], 'rb') as f:
#             hr = Image.open(f)
#             hr = hr.convert('RGB')
        
#         hr = self.croptransform(hr)

#         # rotate and flip?
#         if random.random() > 0.5:
#             hr = hr.permute(0, 2, 1)
#         if random.random() > 0.5:
#             hr = torch.flip(hr, [1])

#         # hr_img = toPIL(hr)
#         lr = self.scaletransform(hr)

#         # minval = torch.min(torch.tensor(hr.size))
#         # hr = ImageOps.fit(hr, (minval,minval), Image.ANTIALIAS)
#         # hr = hr.resize((self.imageSize*self.scale,self.imageSize*self.scale))
#         # lr = hr.resize((self.imageSize,self.imageSize))
#         # hr, lr = toTensor(hr), toTensor(lr)


#         # lr, hr = get_patch(lr,hr,self.imageSize*4, self.scale)
#         return lr, hr

#     def __len__(self):
#         return self.len        

# def load_DIV2K_dataset(root,category,opt):
        
#     dataset = DIV2KDataset(root, category, opt)
#     if category == 'train':
#         dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)
#     else:
#         dataloader = DataLoader(dataset, batch_size=opt.batchSize_test, shuffle=False, num_workers=0)
#     return dataloader


class DIV2KDataset(Dataset):

    def __init__(self, root, category,opt): # highres images not currently scaled, optdefault
        # fid = open(root + '/DIV2K_' + category + '_HR_%dx%d.pkl' % (imageSize,imageSize),'rb')
        # self.images = glob.glob(root + '/DIV2K_' + category + '_HR_bins%dx%d/*' % (opt.imageSize,opt.imageSize))
        
        # self.lqimg = glob.glob(root + '/*in.png')
        self.hqimg = glob.glob(root + '/*gt.png')

        # random.seed(1234)
        # random.shuffle(self.lqimg)
        random.seed(1234)
        random.shuffle(self.hqimg)

        self.scaletransform = transforms.Compose([transforms.Resize(opt.imageSize),transforms.ToTensor()])

        if category == 'train':
            # self.lqimg = self.lqimg[:opt.ntrain]
            self.hqimg = self.hqimg[:opt.ntrain]
        else:
            # self.lqimg = self.lqimg[-opt.ntest:]
            self.hqimg = self.hqimg[-opt.ntest:]

        self.imageSize = opt.imageSize
        self.scale = opt.scale
        self.nch = opt.nch_in
        self.len = len(self.hqimg)

    def __getitem__(self, index):

        # lq, hq = Image.open(self.lqimg[index]), Image.open(self.hqimg[index])
        # lq, hq = toTensor(lq), toTensor(hq)

        hq = Image.open(self.hqimg[index])
        
        hq = hq.resize((self.scale*self.imageSize,self.scale*self.imageSize),Image.BICUBIC)
        lq = hq.resize((self.imageSize,self.imageSize),Image.BICUBIC)
        lq, hq = toTensor(lq), toTensor(hq)
        
        # rotate and flip?
        if random.random() > 0.5:
            lq = lq.permute(0, 2, 1)
            hq = hq.permute(0, 2, 1)
        if random.random() > 0.5:
            lq = torch.flip(lq, [1])
            hq = torch.flip(hq, [1])
        if random.random() > 0.5:
            lq = torch.flip(lq, [2])
            hq = torch.flip(hq, [2])

            
        return lq, hq # hq, lq, lq

    def __len__(self):
        return self.len       


def load_DIV2K_dataset(root, category,opt):

    dataset = DIV2KDataset(root, category, opt)
        
    if category == 'train':
        dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)
    else:
        dataloader = DataLoader(dataset, batch_size=opt.batchSize_test, shuffle=False, num_workers=0)
    return dataloader





class HDF5Dataset(Dataset):
    from noise import noisy

    def __init__(self, root, category, opt): # highres images not currently scaled, optdefault
        import h5py 
        h5_file = h5py.File(root)

        if category == 'train':
            self.data = h5_file.get('x')[:opt.ntrain]
        else:
            self.data = h5_file.get('x')[-opt.ntest:]

        # self.trans = transforms.Compose([transforms.ToPILImage(),transforms.Resize(opt.imageSize),transforms.ToTensor()])
                
        self.len = len(self.data)
        
        if len(opt.noise) > 0:
            self.poisson = [float(opt.noise.split(',')[0])]
            self.gaussian = [0,float(opt.noise.split(',')[1])]
        else:
            self.poisson = [5]
            self.gaussian = [0,0.035]

    def __getitem__(self, index):
        hr = self.data[index]
        # hr = toTensor(arr)
        # lr = self.trans(arr)

        lr = hr.astype('float32') / 255.0
        lr = noisy('poisson',lr,self.poisson).clip(0,1) 
        lr = noisy('gauss',lr,self.gaussian).clip(0,1) 

        lr = toTensor(lr).float()
        hr = toTensor(hr).float()
        
        return lr, hr # hr, lr

    def __len__(self):
        return self.len        


def load_HDF5_dataset(root, category,opt):

    dataset = HDF5Dataset(root, category, opt)
    if category == 'train':
        dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)
    else:
        dataloader = DataLoader(dataset, batch_size=opt.batchSize_test, shuffle=False, num_workers=0)
    return dataloader
    



import pickle
class PickleDataset(Dataset):

    def __init__(self, root, category): # highres images not currently scaled, optdefault
        # fid = open(root + '/DIV2K_' + category + '_HR_%dx%d.pkl' % (imageSize,imageSize),'rb')
        self.images = glob.glob(root + '/DIV2K_' + category + '_HR_bins%dx%d/*' % (opt.imageSize,opt.imageSize))

        self.trans = transforms.Compose([transforms.ToPILImage(),transforms.Resize(opt.imageSize),transforms.ToTensor()])
                
        self.len = len(self.images)

    def __getitem__(self, index):
        img = pickle.load(open(self.images[index],'rb'))
        hr = toTensor(img)

        # rotate and flip?
        if random.random() > 0.5:
            hr = hr.permute(0, 2, 1)
        if random.random() > 0.5:
            hr = torch.flip(hr, [1])

        lr = self.trans(hr)
        
        return lr, hr

    def __len__(self):
        return self.len        


def load_Pickle_dataset(root, category,opt):

    dataset = PickleDataset(root, category, opt)
    if category == 'train':
        dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)
    else:
        dataloader = DataLoader(dataset, batch_size=opt.batchSize_test, shuffle=False, num_workers=0)
    return dataloader



import pickle
class GenericPickleDataset(Dataset):

    def __init__(self, root, category,opt): # highres images not currently scaled, optdefault
        # fid = open(root + '/DIV2K_' + category + '_HR_%dx%d.pkl' % (imageSize,imageSize),'rb')
        # self.images = glob.glob(root + '/DIV2K_' + category + '_HR_bins%dx%d/*' % (opt.imageSize,opt.imageSize))
        
        self.images = glob.glob(root + '/*.npy')

        random.seed(1234)
        random.shuffle(self.images)

        if category == 'train':
            self.images = self.images[:opt.ntrain]
        else:
            self.images = self.images[-opt.ntest:]

        self.trans = transforms.Compose([transforms.ToPILImage(),transforms.Resize(opt.imageSize),transforms.ToTensor()])
        
        self.scale = opt.scale
        self.nch = opt.nch_in
        self.len = len(self.images)
        self.category = category

    def __getitem__(self, index):

        if self.scale > 1:
            # img = pickle.load(open(self.images[index],'rb'))
            img = np.load(self.images[index],allow_pickle=True)
            hr = toTensor(img)

            # rotate and flip?
            # if random.random() > 0.5:
            #     hr = hr.permute(0, 2, 1)
            # if random.random() > 0.5:
            #     hr = torch.flip(hr, [1])

            lr = self.trans(hr)
            return lr, hr
        elif self.nch == 1:
            # lq, hq = pickle.load(open(self.images[index], 'rb'))
            inputTuple = np.load(self.images[index],allow_pickle=True)

            if len(inputTuple) == 2:
                lq, hq = inputTuple
                lq, hq = toTensor(lq).float(), toTensor(hq).float()
            elif len(inputTuple) == 4: ## assuming time sequence of 3 adjacent frames for input
                lq, hq = inputTuple[1], inputTuple[3]
                lq, hq = toTensor(lq).float(), toTensor(hq).float()
            
            # multi-image input?
            # if lq.shape[0] > self.nch:
            #     lq = lq[lq.shape[0] // 2].unsqueeze(0)
            #     hq = hq[hq.shape[0] // 2].unsqueeze(0)
            
            # rotate and flip?
            if self.category == 'train':
                if random.random() > 0.5:
                    lq = lq.permute(0, 2, 1)
                    hq = hq.permute(0, 2, 1)
                if random.random() > 0.5:
                    lq = torch.flip(lq, [1])
                    hq = torch.flip(hq, [1])
                if random.random() > 0.5:
                    lq = torch.flip(lq, [2])
                    hq = torch.flip(hq, [2])

            
            return lq, hq # hq, lq, lq
        elif self.nch == 3:
            img_in_pre,img_in,img_in_pos, img_gt = np.load(self.images[index],allow_pickle=True)
            img_in_pre,img_in,img_in_pos, hq = toTensor(img_in_pre).float(),toTensor(img_in).float(),toTensor(img_in_pos).float(), toTensor(img_gt).float()
            lq = torch.cat((img_in_pre,img_in,img_in_pos), 0)
            
            # rotate and flip?
            if self.category == 'train':
                if random.random() > 0.5:
                    lq = lq.permute(0, 2, 1)
                    hq = hq.permute(0, 2, 1)
                if random.random() > 0.5:
                    lq = torch.flip(lq, [1])
                    hq = torch.flip(hq, [1])
                if random.random() > 0.5:
                    lq = torch.flip(lq, [2])
                    hq = torch.flip(hq, [2])
                                
            return lq, hq

        else:
            print('datahandler error')

    def __len__(self):
        return self.len       


def load_GenericPickle_dataset(root, category,opt):

    dataset = GenericPickleDataset(root, category, opt)
        
    if category == 'train':
        dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)
    else:
        dataloader = DataLoader(dataset, batch_size=opt.batchSize_test, shuffle=False, num_workers=0)
    return dataloader



class SIM_dataset(Dataset):

    def __init__(self, root, category, opt):
        import scipy.io
        mat = scipy.io.loadmat(root)
        xdim,ydim,nsamples = mat['simdata'].shape

        self.inputimages = mat['inputdata'].transpose(3,0,1,2) 
        self.targetimages = mat['simdata'].reshape(xdim,ydim,1,nsamples).transpose(3,0,1,2) # no channels by default
        self.gtimages = mat['gtdata'].reshape(xdim,ydim,1,nsamples).transpose(3,0,1,2) # no channels by default

        if category == 'train':
            self.inputimages = self.inputimages[:-10]
            self.targetimages = self.targetimages[:-10]
            self.gtimages = self.gtimages[:-10]
        else:
            self.inputimages = self.inputimages[-10:]
            self.targetimages = self.targetimages[-10:]
            self.gtimages = self.gtimages[-10:]

        if opt.nch_in == 1: 
            self.widefield = True
        else:
            self.widefield = False
        self.len = len(self.inputimages)

    def __getitem__(self, index):
        inputimg = self.gtimages[index]
        targetimg = self.targetimages[index]
        gtimg = self.gtimages[index]

        # shrink
        inputimg = inputimg[::2,::2,:]
        inputimg = inputimg[::2,::2,:]
        targetimg = targetimg[::2,::2,:]
        targetimg = targetimg[::2,::2,:]
        gtimg = gtimg[::2,::2,:]
        gtimg = gtimg[::2,::2,:]

        # for i in range(inputimg.shape[2]):
        #     inputimg[:,:,i] = (inputimg[:,:,i]) / (np.max(inputimg[:,:,i]))
        #     inputimg[:,:,i] = np.log(np.abs(np.fft.fftshift(np.fft.fft2(inputimg[:,:,i])))+1)
        #     inputimg[:,:,i] = (inputimg[:,:,i]) / (np.max(inputimg[:,:,i]))
        # inputimg = torch.tensor(inputimg).permute(2,0,1).float()
        
        inputimg = inputimg.astype(float) / 255.0
        f = np.fft.fftshift(np.fft.fft2(inputimg))

        # % components
        freal = np.real(f)
        fimag = np.imag(f)

        frp = freal
        frp[frp < 0] = 0
        fcp = fimag
        fcp[fcp < 0] = 0

        frm = -freal
        frm[frm < 0] = 0
        fcm = -fimag
        fcm[fcm < 0] = 0

        # should learn this:
        # frec = (frp - frm) + 1i*(fcp - fcm)
        # frec = ifft2(fftshift(frec))  
        #   i.e.:
        # inputimg = (frp - frm) + 1j*(fcp - fcm)
        # inputimg = (np.fft.ifft2(np.fft.fftshift(inputimg))).astype(np.float)

        frp = torch.tensor(frp)
        fcp = torch.tensor(fcp)
        frm = torch.tensor(frm)
        fcm = torch.tensor(fcm)

        inputimg = torch.cat((frp,fcp,frm,fcm), 2)
        inputimg = torch.tensor(inputimg).permute(2,0,1).float()

        # inputimg = torch.tensor(inputimg).permute(2,0,1).float() / 255.0
        targetimg = torch.tensor(targetimg).permute(2,0,1).float() / 255.0
        gtimg = torch.tensor(gtimg).permute(2,0,1).float() / 255.0
        

        # inputimg = toTensor(inputimg)
        # targetimg = toTensor(targetimg)
        # gtimg = toTensor(gtimg) 

        if self.widefield:
            inputimg = torch.mean(inputimg,0).unsqueeze(0) # widefield

        return inputimg,targetimg,targetimg#targetimg

    def __len__(self):
        return self.len        

def load_SIM_dataset(root, category,opt):

    dataset = SIM_dataset(root, category, opt)
    if category == 'train':
        dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)
    else:
        dataloader = DataLoader(dataset, batch_size=opt.batchSize_test, shuffle=False, num_workers=0)
    return dataloader



class real_SIM_dataset(Dataset):

    def __init__(self, root, category, opt):

        self.imagestacks = []

        for folder in glob.glob(root + '/partitioned*'):
            images = glob.glob(folder + '/*.npy')
            if category == 'train':
                self.imagestacks.extend(images[:-1])
            else:
                self.imagestacks.extend(images[-1:])

        if opt.nch_in == 1: 
            self.widefield = True
        else:
            self.widefield = False
        self.len = len(self.imagestacks)
        self.scale = opt.scale

    def __getitem__(self, index):
        mat = np.load(self.imagestacks[index])
        
        inputimg = mat[:,:,:-4]

        # target has same dimensions
        # targetimg = mat[:,:,-1].reshape(mat.shape[0],mat.shape[1],1)  # no channels by default
        
        toprow = np.hstack((mat[:,:,-4],mat[:,:,-2]))
        botrow = np.hstack((mat[:,:,-3],mat[:,:,-1]))
        targetimg = np.vstack((toprow,botrow)).reshape(2*mat.shape[0],2*mat.shape[1],1)

        inputimg = toTensor(inputimg).float()
        targetimg = toTensor(targetimg).float()

        # fix scale if model is not for SR
        if self.scale == 1:
            targetimg = toPIL(targetimg)
            targetimg = toTensor(targetimg.resize((inputimg.shape[2],inputimg.shape[1]),Image.BICUBIC))

        if self.widefield:
            inputimg = torch.mean(inputimg,0).unsqueeze(0) # widefield

        # rotate and flip?
        if random.random() > 0.5:
            inputimg = inputimg.permute(0, 2, 1)
            targetimg = targetimg.permute(0, 2, 1)
        if random.random() > 0.5:
            inputimg = torch.flip(inputimg, [1])
            targetimg = torch.flip(targetimg, [1])
        if random.random() > 0.5:
            inputimg = torch.flip(inputimg, [2])
            targetimg = torch.flip(targetimg, [2])

        return inputimg,targetimg,targetimg

    def __len__(self):
        return self.len        

def load_real_SIM_dataset(root, category,opt):

    dataset = real_SIM_dataset(root, category, opt)
    if category == 'train':
        dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)
    else:
        dataloader = DataLoader(dataset, batch_size=opt.batchSize_test, shuffle=False, num_workers=0)
    return dataloader    




class Fourier_SIM_dataset(Dataset):

    def __init__(self, root, category, opt):

        self.images = []

        for folder in root.split(','):
            if ".tif" in folder:
                self.images.append(folder) # not a folder, but file (used for --test)
            else:
                folderimgs = sorted(glob.glob(folder + '/*.tif'))
                self.images.extend(folderimgs)

        random.seed(1234)
        random.shuffle(self.images)

        if category == 'train':
            self.images = self.images[:opt.ntrain]
        else:
            self.images = self.images[-opt.ntest:]

        self.len = len(self.images)
        self.scale = opt.scale
        self.task = opt.task
        self.nch_in = opt.nch_in
        self.nch_out = opt.nch_out
        self.norm = opt.norm
        self.out = opt.out
        self.imageSize = opt.imageSize

    def __getitem__(self, index):
        
        stack = io.imread(self.images[index])

        if len(stack.shape) > 3:
            stack = stack[0,:,:,:]

        ## Resize if too large? Generally not a good idea
        # if stack.shape[-1] > self.imageSize or stack.shape[-2] > self.imageSize:
        #     stack = transform.resize(stack,(stack.shape[0],self.imageSize,self.imageSize), order=3)

        if 'subset' in self.task and self.nch_in == 6:
            inputimg = stack[[0,1,3,4,6,7]]
        elif 'subset' in self.task and self.nch_in == 3:
            inputimg = stack[[0,4,8]]
        elif 'subset' in self.task and self.nch_in == 1: # don't do it if widefield input is expected
            inputimg = stack[[8]] # used for sequential SIM - first tests from 20201215 have GT as 9th frame
        elif 'last' in self.task and self.nch_in == 6:
            inputimg = stack[[3,4,5,6,7,8]]
        elif 'last' in self.task and self.nch_in == 3:
            inputimg = stack[[6,7,8]]
        elif 'last' in self.task and self.nch_in == 1: # don't do it if widefield input is expected
            inputimg = stack[[8]] # used for sequential SIM - first tests from 20201215 have GT as 9th frame
        elif 'wfin' in self.task:
            inputimg = stack[:9] 
        else:
            inputimg = stack[:self.nch_in]


        # adding noise
        # if 'noiseRetraining' in self.out:
        #     noisefrac = np.linspace(0,1,10)
        #     idx = np.random.randint(0,10)
        #     inputimg = inputimg + noisefrac[idx]*np.std(I)*np.random.randn(*inputimg.shape)
        #     inputimg = np.clip(inputimg,0,255).astype('uint16')


        if len(stack) > 9:
            # otf = stack[9]
            if self.scale == 2:
                toprow = np.hstack((stack[-4,:,:],stack[-2,:,:]))
                botrow = np.hstack((stack[-3,:,:],stack[-1,:,:]))
                gt = np.vstack((toprow,botrow)).reshape(2*stack.shape[1],2*stack.shape[2])
            elif self.nch_out > 1:
                gt = stack[-self.nch_out:]
            else:
                gt = stack[-1] # used to be index self.nch_in+1
        else:
            gt = stack[0] # if it doesn't exist, doesn't matter


        # widefield = stack[12]

        # print('max before:',end=' ')
        # print('%0.2f %0.2f %0.2f %0.2f %0.2f' % (np.max(inputimg),np.max(otf),np.max(gt),np.max(simimg),np.max(widefield)))

        if self.norm == 'convert': # raw img from microscope, needs normalisation and correct frame ordering
            print('Raw input assumed - converting')
            # NCHW
            # I = np.zeros((9,opt.imageSize,opt.imageSize),dtype='uint16')

            # for t in range(9):
            #     frame = inputimg[t]
            #     frame = 120 / np.max(frame) * frame
            #     frame = np.rot90(np.rot90(np.rot90(frame)))
            #     I[t,:,:] = frame
            # inputimg = I

            inputimg = np.rot90(inputimg,axes=(1,2))
            inputimg = inputimg[[6,7,8,3,4,5,0,1,2]] # could also do [8,7,6,5,4,3,2,1,0]
            for i in range(len(inputimg)):
                inputimg[i] = 100 / np.max(inputimg[i]) * inputimg[i]
        elif 'convert' in self.norm:
            fac = float(self.norm[7:])
            inputimg = np.rot90(inputimg,axes=(1,2))
            inputimg = inputimg[[6,7,8,3,4,5,0,1,2]] # could also do [8,7,6,5,4,3,2,1,0]
            for i in range(len(inputimg)):
                inputimg[i] = fac * 255 / np.max(inputimg[i]) * inputimg[i]


        inputimg = inputimg.astype('float') / np.max(inputimg) # used to be /255
        gt = gt.astype('float') / np.max(gt) # used to be /255
        widefield = np.mean(inputimg,0)

        if len(stack) > self.nch_in+2:
            simimg = stack[self.nch_in+2] # sim reference image
            simimg = simimg.astype('float') / np.max(simimg)
        else:
            simimg = np.mean(inputimg,0) # same as widefield
        
        if self.norm == 'adapthist':
            for i in range(len(inputimg)):
                inputimg[i] = exposure.equalize_adapthist(inputimg[i],clip_limit=0.001)
            widefield = exposure.equalize_adapthist(widefield,clip_limit=0.001)
            gt = exposure.equalize_adapthist(gt,clip_limit=0.001)
            simimg = exposure.equalize_adapthist(simimg,clip_limit=0.001)

            inputimg = torch.tensor(inputimg).float()
            gt = torch.tensor(gt).unsqueeze(0).float()
            widefield = torch.tensor(widefield).unsqueeze(0).float()
            simimg = torch.tensor(simimg).unsqueeze(0).float()
        else:
            inputimg = torch.tensor(inputimg).float()
            gt = torch.tensor(gt).float()
            if self.nch_out == 1:
                gt = gt.unsqueeze(0)
            widefield = torch.tensor(widefield).unsqueeze(0).float()
            simimg = torch.tensor(simimg).unsqueeze(0).float()

            # normalise 
            gt = (gt - torch.min(gt)) / (torch.max(gt) - torch.min(gt))
            simimg = (simimg - torch.min(simimg)) / (torch.max(simimg) - torch.min(simimg))
            widefield = (widefield - torch.min(widefield)) / (torch.max(widefield) - torch.min(widefield))

            if self.norm == 'minmax':
                for i in range(len(inputimg)):
                    inputimg[i] = (inputimg[i] - torch.min(inputimg[i])) / (torch.max(inputimg[i]) - torch.min(inputimg[i]))
            elif 'minmax' in self.norm:
                fac = float(self.norm[6:])
                for i in range(len(inputimg)):
                    inputimg[i] = fac * (inputimg[i] - torch.min(inputimg[i])) / (torch.max(inputimg[i]) - torch.min(inputimg[i]))
        

        if 'simin_simout' in self.task:
            return inputimg,simimg,gt,widefield,self.images[index]   # sim input, sim output
        elif 'wfin_simout' in self.task:
            return widefield,simimg,gt,widefield,self.images[index]   # wf input, sim output
        elif 'wfin_gtout' in self.task:
            return widefield,gt,simimg,widefield,self.images[index]  # wf input, gt output
        else: # simin_gtout
            return inputimg,gt,simimg,widefield,self.images[index]  # sim input, gt output



    def __len__(self):
        return self.len        

def load_fourier_SIM_dataset(root, category,opt):

    dataset = Fourier_SIM_dataset(root, category, opt)
    if category == 'train':
        dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)
    else:
        dataloader = DataLoader(dataset, batch_size=opt.batchSize_test, shuffle=False, num_workers=0)
    return dataloader    




    
class NTIREDenoisingdataset(Dataset):

    def __init__(self, root, category, opt):
        # self.lq = glob.glob(root + '/**/*_NOISY_*.png', recursive=True)
        # self.hq = glob.glob(root + '/**/*_GT_*.png', recursive=True)
        # random.seed(1234)
        # random.shuffle(self.lq)
        # random.seed(1234)
        # random.shuffle(self.hq)
        # print(self.lq[:3])
        # print(self.hq[:3])

        self.images = glob.glob(root + '/*.npy')
        random.seed(1234)
        random.shuffle(self.images)
        print(self.images[:3])


        if category == 'train':
            self.images = self.images[:opt.ntrain]
        else:
            self.images = self.images[-opt.ntest:]

        self.imageSize = opt.imageSize
        self.scale = opt.scale
        self.nch_in = opt.nch_in
        self.len = len(self.images)
        
    def __getitem__(self, index):
        
                
        lq, hq = pickle.load(open(self.images[index], 'rb'))
        lq, hq = toTensor(lq), toTensor(hq)

        # rotate and flip?
        if random.random() > 0.5:
            lq = lq.permute(0, 2, 1)
            hq = hq.permute(0, 2, 1)
        if random.random() > 0.5:
            lq = torch.flip(lq, [1])
            hq = torch.flip(hq, [1])
        if random.random() > 0.5:
            lq = torch.flip(lq, [2])
            hq = torch.flip(hq, [2])
            
        return lq, hq

    def __len__(self):
        return self.len


def load_NTIREDenoising_dataset(root,category,opt):
        
    dataset = NTIREDenoisingdataset(root, category, opt)
    if category == 'train':
        dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)
    else:
        dataloader = DataLoader(dataset, batch_size=opt.batchSize_test, shuffle=False, num_workers=0)
    return dataloader




class EstimateNLdataset(Dataset):
    
    def __init__(self, root, category, opt):
        # self.lq = glob.glob(root + '/**/*_NOISY_*.png', recursive=True)
        # self.hq = glob.glob(root + '/**/*_GT_*.png', recursive=True)
        # random.seed(1234)
        # random.shuffle(self.lq)
        # random.seed(1234)
        # random.shuffle(self.hq)
        # print(self.lq[:3])
        # print(self.hq[:3])

        self.images = glob.glob(root + '/*.npy')
        random.seed(1234)
        random.shuffle(self.images)
        print(self.images[:3])


        if category == 'train':
            self.images = self.images[:-100]
        else:
            self.images = self.images[-100:]

        self.imageSize = opt.imageSize
        self.scale = opt.scale
        self.nch_in = opt.nch_in
        self.len = len(self.images)
        
    def __getitem__(self, index):
        
        # rotate and flip?
        # if random.random() > 0.5:
        #     lq = lq.permute(0, 2, 1)
        #     hq = hq.permute(0, 2, 1)
        # if random.random() > 0.5:
        #     lq = torch.flip(lq, [1])
        #     hq = torch.flip(hq, [1])
        # if random.random() > 0.5:
        #     lq = torch.flip(lq, [2])
        #     hq = torch.flip(hq, [2])
                
        lq, hq = pickle.load(open(self.images[index], 'rb'))
        lq, hq = toTensor(lq), toTensor(hq)

        # rndstd = random.random()*0.3 + 0.1 # 0.1-0.4

        # noise = torch.FloatTensor(hq.size()).normal_(mean=0, std=rndstd)
        # hq = hq + noise

        noise = lq - hq
        noise = torch.clamp(noise,0,1)
        noisestd = torch.std(noise)

        return lq, noisestd.unsqueeze(0)

    def __len__(self):
        return self.len


def load_EstimateNL_dataset(root,category,opt):
        
    dataset = EstimateNLdataset(root, category, opt)
    if category == 'train':
        dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)
    else:
        dataloader = DataLoader(dataset, batch_size=opt.batchSize_test, shuffle=False, num_workers=0)
    return dataloader    








class ERdataset(Dataset):

    def __init__(self, root, category, opt): # highres images not currently scaled, just 96 by default
        
        self.img = Image.open(root)
        self.img = np.array(self.img)

        if category == 'train':
            self.len = 100
        else:
            self.len = 10

    def __getitem__(self, index):

        # random crop
        dim = 384
        r,c = self.img.shape
        r_rand = np.random.randint(0,r-dim)
        c_rand = np.random.randint(0,c-dim)
        img = self.img[r_rand:r_rand+dim,c_rand:c_rand+dim]

        # rotate and flip?
        img = torch.from_numpy(img) # use torch
        if random.random() > 0.5:
            img = img.permute(1, 0)
        if random.random() > 0.5:
            img = torch.flip(img, [1])
        img = img.numpy() # use numpy again

        # normalize and add dimension
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) 
        img = np.expand_dims(img, 2)

        # gaussian darkness
        X,Y = np.meshgrid(np.linspace(0,1,dim),np.linspace(0,1,dim))
        mu_x, mu_y = np.random.rand(), np.random.rand()
        var_x, var_y = 0.1, 0.1
        Z = np.exp( -(X - mu_x)**2 / (2*var_x) ) * np.exp( -(Y - mu_y)**2 / (2*var_y) )
        Z = np.expand_dims(Z, 2)


        noisyimg = Z*img
        noisyimg = (0.2*np.random.rand()+0.8)*noisyimg  # overall level between 0.5 and 0.1
        noisyimg = noisy('gauss',noisyimg,[0,0.002])

        
        x = np.clip(noisyimg,0,1)
        x,y = toTensor(x.astype('float32')), toTensor(img.astype('float32'))
        return x,y

    def __len__(self):
        return self.len           
        
def load_ER_dataset(root, category,shuffle=True,batchSize=6,num_workers=0):

    dataset = ERdataset(root, category)
    dataloader = DataLoader(dataset, batch_size=batchSize,
                                            shuffle=shuffle, num_workers=num_workers)
    return dataloader

