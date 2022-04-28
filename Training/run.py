import math
import os
import torch
import torch.nn as nn
import time
import numpy as np
import torch.optim as optim
import torchvision
from torch.autograd import Variable
import subprocess
from models import GetModel, ESRGAN_Discriminator, ESRGAN_FeatureExtractor
from datahandler import GetDataloaders


from plotting import testAndMakeCombinedPlots, generate_convergence_plots
import testfunctions

# from torch.utils.tensorboard import SummaryWriter

from options import parser
import traceback
import socket
from datetime import datetime
import shutil
import sys
import glob
import wandb

def options():

    opt = parser.parse_args()

    if opt.norm == '':
        opt.norm = opt.dataset
    elif opt.norm.lower() == 'none':
        opt.norm = None

    if len(opt.basedir) > 0:
        opt.root = opt.root.replace('basedir', opt.basedir)
        opt.weights = opt.weights.replace('basedir', opt.basedir)
        opt.out = opt.out.replace('basedir', opt.basedir)

    if opt.out[:4] == 'root':
        opt.out = opt.out.replace('root', opt.root)


    # convenience function
    if len(opt.weights) > 0 and not os.path.isfile(opt.weights):
        # folder provided, trying to infer model options

        logfile = opt.weights + '/log.txt'
        opt.weights += '/final.pth'
        if not os.path.isfile(opt.weights):
            opt.weights = opt.weights.replace('final.pth', 'prelim.pth')

        if os.path.isfile(logfile):
            fid = open(logfile, 'r')
            optstr = fid.read()
            optlist = optstr.split(', ')

            def getopt(optname, typestr):
                opt_e = [e.split('=')[-1].strip("\'")
                        for e in optlist if (optname.split('.')[-1] + '=') in e]
                return eval(optname) if len(opt_e) == 0 else typestr(opt_e[0])

            opt.model = getopt('opt.model', str)
            opt.task = getopt('opt.task', str)
            opt.nch_in = getopt('opt.nch_in', int)
            opt.nch_out = getopt('opt.nch_out', int)
            opt.n_resgroups = getopt('opt.n_resgroups', int)
            opt.n_resblocks = getopt('opt.n_resblocks', int)
            opt.n_feats = getopt('opt.n_feats', int)

    return opt



def remove_dataparallel_wrapper(state_dict):
    r"""Converts a DataParallel model to a normal one by removing the "module."
    wrapper in the module dictionary

    Args:
            state_dict: a torch.nn.DataParallel state dictionary
    """
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, vl in state_dict.items():
        name = k[7:]  # remove 'module.' of DataParallel
        new_state_dict[name] = vl

    return new_state_dict


def ESRGANtrain(opt, dataloader, validloader, generator):

    start_epoch = 0

    discriminator = ESRGAN_Discriminator(input_shape=(
        opt.nch_in, opt.imageSize, opt.imageSize)).cuda()
    feature_extractor = ESRGAN_FeatureExtractor().cuda()

    # Set feature extractor to inference mode
    feature_extractor.eval()

    # Losses
    criterion_GAN = torch.nn.BCEWithLogitsLoss().cuda()
    # criterion_content = torch.nn.L1Loss().cuda()
    # criterion_pixel = torch.nn.L1Loss().cuda()

    if opt.task == 'segment' or opt.task == 'classification':
        criterion_pixel = nn.CrossEntropyLoss()
    else:
        criterion_pixel = nn.L1Loss()

    # Optimizers
    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=opt.lr, betas=(0.9, 0.999))

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    criterion_pixel.cuda()

    opt.t0 = time.perf_counter()

    for epoch in range(start_epoch, opt.nepoch):
        mean_loss_D = 0
        mean_loss_G = 0
        mean_loss_GAN = 0
        mean_loss_pixel = 0

        for i, bat in enumerate(dataloader):
            lr, hr = bat[0], bat[1]

            hr = torch.round((opt.nch_out-1)*hr).long()
            # hr = hr.long().squeeze()

            # Configure model input
            imgs_lr = Variable(lr.type(Tensor))
            imgs_hr = Variable(hr.cuda())

            # Adversarial ground truths
            valid = Variable(Tensor(
                np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
            fake = Variable(Tensor(
                np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)

            # Measure pixel-wise loss against ground truth
            loss_pixel = criterion_pixel(gen_hr.squeeze(), imgs_hr.squeeze())

            if epoch < 3:
                # Warm-up (pixel-wise loss only)
                loss_pixel.backward()
                optimizer_G.step()
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
                    % (epoch, opt.nepoch, i, len(dataloader), loss_pixel.item())
                )
                continue

            m = nn.LogSoftmax(dim=1)
            gen_hr = m(gen_hr)
            gen_hr = gen_hr.argmax(dim=1, keepdim=True)

            # Extract validity predictions from discriminator

            pred_real = discriminator(imgs_hr.type(Tensor)).detach()
            pred_fake = discriminator(gen_hr.type(Tensor))

            # Adversarial loss (relativistic average GAN)
            loss_GAN = criterion_GAN(
                pred_fake - pred_real.mean(0, keepdim=True), valid)

            # Content loss
            # gen_features = feature_extractor(gen_hr)
            # real_features = feature_extractor(imgs_hr).detach()
            # loss_content = criterion_content(gen_features, real_features)

            # Total generator loss
            loss_G = opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel
            # loss_G = 0.001 * loss_GAN + 0.01 * loss_pixel

            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            pred_real = discriminator(imgs_hr.type(Tensor))
            pred_fake = discriminator(gen_hr.type(Tensor).detach())

            # Adversarial loss for real and fake images (relativistic average GAN)
            loss_real = criterion_GAN(
                pred_real - pred_fake.mean(0, keepdim=True), valid)
            loss_fake = criterion_GAN(
                pred_fake - pred_real.mean(0, keepdim=True), fake)

            # Total loss
            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            optimizer_D.step()

            ######### Status and display #########
            mean_loss_D += loss_D.item()
            mean_loss_G += loss_G.item()
            mean_loss_GAN += loss_GAN.item()
            mean_loss_pixel += loss_pixel.item()

            print(
                "\r[%d/%d][%d/%d] [D loss: %f] [G loss: %f, adv: %f, pixel: %f]"
                % (
                    epoch+1,
                    opt.nepoch,
                    i+1,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_pixel.item(),
                ), end=''
            )

        if epoch < 3:
            continue

        # ---------------- Printing -----------------
        mean_loss_D = loss_D / len(dataloader)
        mean_loss_G = loss_G / len(dataloader)
        mean_loss_GAN = loss_GAN / len(dataloader)
        mean_loss_pixel = loss_pixel / len(dataloader)
        outstr = '\nEpoch %d done, [D loss: %0.6f] [G loss: %0.6f, adv: %0.6f, pixel: %0.6f]' % (
            epoch, mean_loss_D, mean_loss_G, mean_loss_GAN, mean_loss_pixel)
        print(outstr)
        print(outstr, file=opt.fid)
        opt.fid.flush()
        if opt.log:
            t1 = time.perf_counter() - opt.t0
            mem = torch.cuda.memory_allocated()
            opt.writer.add_scalar('loss/mean_loss_D', mean_loss_D, epoch)
            opt.writer.add_scalar('loss/mean_loss_G', mean_loss_G, epoch)
            opt.writer.add_scalar('loss/mean_loss_GAN', mean_loss_GAN, epoch)
            opt.writer.add_scalar('loss/mean_loss_pixel', mean_loss_pixel, epoch)
            opt.writer.add_scalar('data/time', t1, epoch)
            opt.writer.add_scalar('data/mem', mem, epoch)

        # ---------------- TEST -----------------
        if (epoch + 1) % opt.testinterval == 0:
            testAndMakeCombinedPlots(net, validloader, opt, epoch)

            if opt.testFunction is not None:
                testfunctions.parse(net,opt, opt.testFunction, epoch)
            # if opt.scheduler:
            # scheduler.step(mean_loss / len(dataloader))

        if (epoch + 1) % opt.saveinterval == 0:
            # torch.save(net.state_dict(), opt.out + '/prelim.pth')
            checkpoint = {'epoch': epoch + 1,
                          'state_dict': net.state_dict(),
                          'optimizer_G': optimizer_G.state_dict(),
                          'optimizer_D': optimizer_D.state_dict()
                          }
            # if len(opt.scheduler) > 0:
            #     checkpoint['scheduler'] = scheduler.state_dict()
            torch.save(checkpoint, '%s/prelim%d.pth' % (opt.out, epoch+1))

    checkpoint = {'epoch': nepoch,
                  'state_dict': net.state_dict(),
                  'optimizer_G': optimizer_G.state_dict(),
                  'optimizer_D': optimizer_D.state_dict()
                  }
    # if len(opt.scheduler) > 0:
    #     checkpoint['scheduler'] = scheduler.state_dict()
    torch.save(checkpoint, opt.out + '/final.pth')


#from topologylayer.nn import LevelSetLayer2D, SumBarcodeLengths, TopKBarcodeLengths, BarcodePolyFeature
#layer = LevelSetLayer2D(size=(256,256), sublevel=False, maxdim=1)

## feat = SumBarcodeLengths(dim=1)
## feat = TopKBarcodeLengths(dim=1, k=2)
#feat = BarcodePolyFeature(dim=0, p=1, q=1)


def train(opt, dataloader, validloader, net):

    start_epoch = 0
    if opt.task == 'segment' or opt.task == 'classification':
        loss_function = nn.CrossEntropyLoss()
    else:
        loss_function = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    loss_function.cuda()
    if len(opt.weights) > 0:  # load previous weights?
        checkpoint = torch.load(opt.weights)
        print('loading checkpoint', opt.weights)
        if opt.undomulti:
            checkpoint['state_dict'] = remove_dataparallel_wrapper(
                checkpoint['state_dict'])
        if opt.modifyPretrainedModel:
            pretrained_dict = checkpoint['state_dict']
            model_dict = net.state_dict()
            # 1. filter out unnecessary keys
            for k, v in list(pretrained_dict.items()):
                print(k)
            pretrained_dict = {k: v for k, v in list(
                pretrained_dict.items())[:-2]}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            net.load_state_dict(model_dict)

            # optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
        else:
            net.load_state_dict(checkpoint['state_dict'])
            if opt.lr == 1:  # continue as it was
                optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']

    if len(opt.scheduler) > 0:
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=5, min_lr=0, eps=1e-08)
        stepsize, gamma = int(opt.scheduler.split(
            ',')[0]), float(opt.scheduler.split(',')[1])
        scheduler = optim.lr_scheduler.StepLR(optimizer, stepsize, gamma=gamma)
        if len(opt.weights) > 0:
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])

    opt.t0 = time.perf_counter()

    for epoch in range(start_epoch, opt.nepoch):
        count = 0
        mean_loss = 0

        # for param_group in optimizer.param_groups:
        #     print('\nLearning rate', param_group['lr'])

        for i, bat in enumerate(dataloader):
            lr, hr = bat[0], bat[1]

            optimizer.zero_grad()
            if opt.model == 'ffdnet':
                stdvec = torch.zeros(lr.shape[0])
                for j in range(lr.shape[0]):
                    noise = lr[j] - hr[j]
                    stdvec[j] = torch.std(noise)
                noise = net(lr.cuda(), stdvec.cuda())
                sr = torch.clamp(lr.cuda() - noise, 0, 1)
                gt_noise = lr.cuda() - hr.cuda()
                loss = loss_function(noise, gt_noise)
            elif opt.task == 'residualdenoising':
                noise = net(lr.cuda())
                gt_noise = lr.cuda() - hr.cuda()
                loss = loss_function(noise, gt_noise)
            elif opt.task == 'classification':
                inputimg = lr
                label = hr
                pred = net(inputimg.cuda())
                loss = loss_function(
                    pred.squeeze(), label.squeeze().cuda())
            else:
                if opt.cpu:
                    sr = net(lr)
                else:
                    sr = net(lr.cuda())
                if opt.task == 'segment':
                    if opt.nch_out > 2:
                        hr_classes = torch.round((opt.nch_out-1)*hr).long()
                        if opt.cpu:
                            loss = loss_function(
                                sr.squeeze(), hr_classes.squeeze())
                        else:
                            loss = loss_function(
                                sr.squeeze(), hr_classes.squeeze().cuda())
                    else:
                        if opt.cpu:
                            loss = loss_function(
                                sr.squeeze(), hr.long().squeeze())
                        else:
                            loss = loss_function(
                            sr.squeeze(), hr.long().squeeze().cuda())
                elif opt.task == 'persistenthomology':
                    # loss = loss_function(sr, hr.cuda())

                    #temp
                    loss_pix = loss_function(
                        sr, hr.cuda())

                    hr = hr.view(1,256,256)
                    dgm1,issub = layer(hr.float().cuda())

                    # y = sr.float().requires_grad_(True)

                    # a = torch.exp(20*sr)
                    # b = torch.sum(a)
                    # softmax = a/b
                    # softmax[:,0,:,:] = softmax[:,0,:,:]*0
                    # softmax[:,1,:,:] = softmax[:,1,:,:]*1
                    # pred_labels = torch.sum(softmax,1,keepdim=True) # softargmax

                    dgm2,issub = layer(sr)

                    loss_topo = torch.sum((dgm1[1]-dgm2[1])**2) / (256*256)

                    loss = loss_pix + loss_topo

                    print('\nloss pix/topo %0.6f/%0.6f\n' % (loss_pix.data.item(),loss_topo.data.item()))
                else:
                    loss = loss_function(sr, hr.cuda())

            loss.backward()
            optimizer.step()

            ######### Status and display #########
            mean_loss += loss.data.item()
            print('\r[%d/%d][%d/%d] Loss: %0.6f' % (epoch+1, opt.nepoch,
                                                    i+1, len(dataloader), loss.data.item()), end='')

            count += 1
            if opt.log and count*opt.batchSize // 1000 > 0:
                t1 = time.perf_counter() - opt.t0
                mem = torch.cuda.memory_allocated()
                opt.writer.add_scalar(
                    'data/mean_loss_per_1000', mean_loss / count, epoch)
                opt.writer.add_scalar('data/time_per_1000', t1, epoch)
                print(epoch, count*opt.batchSize, t1, mem,
                      mean_loss / count, file=opt.train_stats)
                opt.train_stats.flush()
                count = 0

        # ---------------- Scheduler -----------------
        if len(opt.scheduler) > 0:
            scheduler.step()
            for param_group in optimizer.param_groups:
                print('\nLearning rate', param_group['lr'])
                opt.wandb.log({'lr': param_group['lr']},step=epoch+1)
                break

        # ---------------- Printing -----------------
        mean_loss = mean_loss / len(dataloader)
        t1 = time.perf_counter() - opt.t0
        eta = (opt.nepoch - (epoch + 1)) * t1 / (epoch + 1)
        ostr = '\nEpoch [%d/%d] done, mean loss: %0.6f, time spent: %0.1fs, ETA: %0.1fs' % (
            epoch+1, opt.nepoch, mean_loss, t1, eta)
        opt.wandb.log({'epoch':epoch+1,'mean_loss': mean_loss},step=epoch+1)
        print(ostr)
        print(ostr, file=opt.fid)
        opt.fid.flush()
        if opt.log:
            opt.writer.add_scalar(
                'data/mean_loss', mean_loss / len(dataloader), epoch)

        # ---------------- TEST -----------------
        if (epoch + 1) % opt.testinterval == 0:
            testAndMakeCombinedPlots(net, validloader, opt, epoch)

            if opt.testFunction is not None:
                testfunctions.parse(net,opt, opt.testFunction, epoch)
            # if opt.scheduler:
            # scheduler.step(mean_loss / len(dataloader))

        if (epoch + 1) % opt.saveinterval == 0:
            # torch.save(net.state_dict(), opt.out + '/prelim.pth')
            checkpoint = {'epoch': epoch + 1,
                          'state_dict': net.state_dict(),
                          'optimizer': optimizer.state_dict()}
            if len(opt.scheduler) > 0:
                checkpoint['scheduler'] = scheduler.state_dict()
            torch.save(checkpoint, '%s/prelim%d.pth' % (opt.out, epoch+1))

    checkpoint = {'epoch': opt.nepoch,
                  'state_dict': net.state_dict(),
                  'optimizer': optimizer.state_dict()}
    if len(opt.scheduler) > 0:
        checkpoint['scheduler'] = scheduler.state_dict()
    torch.save(checkpoint, opt.out + '/final.pth')




def main(opt):


    opt.device = torch.device('cuda' if torch.cuda.is_available() and not opt.cpu else 'cpu')

    os.makedirs(opt.out,exist_ok=True)
    shutil.copy2('options.py',opt.out)

    opt.fid = open(opt.out + '/log.txt', 'w')

    ostr = 'ARGS: ' + ' '.join(sys.argv[:])
    print(opt, '\n')
    print(opt, '\n', file=opt.fid)
    print('\n%s\n' % ostr)
    print('\n%s\n' % ostr, file=opt.fid)


    print('getting dataloader', opt.root)
    dataloader, validloader = GetDataloaders(opt)

    if opt.log:
        # opt.writer = SummaryWriter(log_dir=opt.out, comment='_%s_%s' % (
        #     opt.out.replace('\\', '/').split('/')[-1], opt.model))
        opt.train_stats = open(opt.out.replace(
            '\\', '/') + '/train_stats.csv', 'w')
        opt.test_stats = open(opt.out.replace(
            '\\', '/') + '/test_stats.csv', 'w')
        print('iter,nsample,time,memory,meanloss', file=opt.train_stats)
        print('iter,time,memory,psnr,ssim', file=opt.test_stats)

    t0 = time.perf_counter()
    if opt.model.lower() == 'esrgan':
        net = GetModel(opt)
        ESRGANtrain(opt, dataloader, validloader, net)
    elif opt.model.lower() == 'wgan_binary':
        import WGAN_binary
        WGAN_binary.train(opt, dataloader, validloader)
    elif opt.model.lower() == 'wgan':
        import WGAN
        WGAN.train(opt, dataloader, validloader)
    else:
        net = GetModel(opt)

    if not opt.test:
        # opt.wandb.watch(net, log_freq=100)
        train(opt, dataloader, validloader, net)
        # torch.save(net.state_dict(), opt.out + '/final.pth')
    else:
        if len(opt.weights) > 0:  # load previous weights?
            checkpoint = torch.load(opt.weights,map_location=opt.device)
            print('loading checkpoint', opt.weights)
            if opt.undomulti:
                checkpoint['state_dict'] = remove_dataparallel_wrapper(
                    checkpoint['state_dict'])
            net.load_state_dict(checkpoint['state_dict'])
            print('time: %0.1f' % (time.perf_counter()-t0))
        testAndMakeCombinedPlots(net, validloader, opt)

    opt.fid.close()
    if not opt.test:
        generate_convergence_plots(opt,opt.out + '/log.txt')


    print('time: %0.1f' % (time.perf_counter()-t0))

    # optional clean up
    if opt.disposableTrainingData and not opt.test:
        print('deleting training data')
        # preserve a few samples
        os.makedirs('%s/training_data_subset' % opt.out, exist_ok=True)

        samplecount = 0
        for file in glob.glob('%s/*' % opt.root):
            if os.path.isfile(file):
                basename = os.path.basename(file)
                shutil.copy2(file, '%s/training_data_subset/%s' % (opt.out,basename))
                samplecount += 1
                if samplecount == 10:
                    break
        shutil.rmtree(opt.root)



if __name__ == '__main__':
    opt = options()
    wandb.init(project="phd")
    wandb.config.update(opt)
    opt.wandb = wandb
    main(opt)
