import torch
import matplotlib.pyplot as plt
import torchvision
import skimage
from skimage.metrics import structural_similarity
# from skimage import measure
import torchvision.transforms as transforms
import numpy as np
import time
from PIL import Image
import scipy.ndimage as ndimage
import torch.nn as nn
import os
import wandb

plt.switch_backend('agg')

toTensor = transforms.ToTensor()
toPIL = transforms.ToPILImage()

def testAndMakeCombinedPlots(net,loader,opt,idx=0):

    def PSNR_numpy(p0,p1):
        I0,I1 = np.array(p0)/255.0, np.array(p1)/255.0
        MSE = np.mean( (I0-I1)**2 )
        PSNR = 20*np.log10(1/np.sqrt(MSE))
        return PSNR

    def SSIM_numpy(p0,p1):
        I0,I1 = np.array(p0)/255.0, np.array(p1)/255.0
        return structural_similarity(I0, I1, multichannel=True)
        # return measure.compare_ssim(I0, I1, multichannel=True)

    def calcScores(img, hr=None, makeplotBool=False, plotidx=0, title=None):
        if makeplotBool:
            plt.subplot(1,4,plotidx)
            plt.gca().axis('off')
            plt.xticks([], [])
            plt.yticks([], [])
            plt.imshow(img,cmap='gray')
        if not hr == None:
            psnr,ssim = PSNR_numpy(img,hr),SSIM_numpy(img,hr)
            if makeplotBool: plt.title('%s (%0.2fdB/%0.3f)' % (title,psnr,ssim))
            return psnr,ssim
        if makeplotBool: plt.title(r'GT ($\infty$/1.000)')


    count, mean_bc_psnr, mean_sr_psnr, mean_bc_ssim, mean_sr_ssim = 0,0,0,0,0

    if opt.test:
        bc_ssim_arr = []
        sr_ssim_arr = []
        bc_psnr_arr = []
        sr_psnr_arr = []

    for i, bat in enumerate(loader):
        lr_bat, hr_bat = bat[0], bat[1]
        with torch.no_grad():
            if opt.model == 'ffdnet':
                stdvec = torch.zeros(lr_bat.shape[0])
                for j in range(lr_bat.shape[0]):
                    noise = lr_bat[j] - hr_bat[j]
                    stdvec[j] = torch.std(noise)
                noise_bat = net(lr_bat.cuda(), stdvec.cuda())
                sr_bat = torch.clamp( lr_bat.cuda() - noise_bat,0,1 )
            elif opt.task == 'residualdenoising':
                for j in range(lr_bat.shape[0]):
                    noise = lr_bat[j] - hr_bat[j]
                noise_bat = net(lr_bat.cuda())
                sr_bat = torch.clamp( lr_bat.cuda() - noise_bat,0,1 )
            else:
                if not opt.cpu:
                    sr_bat = net(lr_bat.cuda())
                else:
                    sr_bat = net(lr_bat)
        sr_bat = sr_bat.cpu()

        for j in range(len(lr_bat)): # loop over batch
            makeplotBool = (idx < 5 or (idx+1) % opt.plotinterval == 0 or idx == opt.nepoch - 1) and count < opt.nplot
            if opt.logimage: makeplotBool = False

            lr, sr, hr = lr_bat.data[j], sr_bat.data[j], hr_bat.data[j]

            if opt.task == 'segment':
                if opt.model == 'wgan':
                    lr, sr, hr = toPIL(lr), toPIL(sr.float() / (opt.nch_out - 1)), toPIL(hr.float())
                    if makeplotBool: plt.figure(figsize=(10,5))
                    calcScores(lr, hr, makeplotBool, plotidx=1, title='ns')
                    bc_psnr, bc_ssim = calcScores(lr, hr, makeplotBool, plotidx=2, title='bc')
                    sr_psnr, sr_ssim = calcScores(sr, hr, makeplotBool, plotidx=3, title='re')
                    calcScores(hr, None, makeplotBool, plotidx=4)
                elif opt.model == 'wgan_binary':

                    m = nn.LogSoftmax(dim=0)
                    sr = m(sr)
                    # print(sr)
                    sr = sr.argmax(dim=0, keepdim=True)
                    # print(sr.shape)

                    lr, sr, hr = toPIL(lr), toPIL(sr.float() / (opt.nch_out - 1)), toPIL(hr.float())

                    if makeplotBool: plt.figure(figsize=(10,5))
                    calcScores(lr, hr, makeplotBool, plotidx=1, title='ns')
                    bc_psnr, bc_ssim = calcScores(lr, hr, makeplotBool, plotidx=2, title='bc')
                    sr_psnr, sr_ssim = calcScores(sr, hr, makeplotBool, plotidx=3, title='re')
                    calcScores(hr, None, makeplotBool, plotidx=4)
                else:
                    if torch.max(hr.long()) == 0:
                        continue # all black, ignore
                    m = nn.LogSoftmax(dim=0)
                    sr = m(sr)
                    # print(sr)
                    sr = sr.argmax(dim=0, keepdim=True)
                    # print(sr.shape)

                    # multi-image input?
                    if lr.shape[0] > hr.shape[0]:
                        lr = lr[lr.shape[0] // 2].unsqueeze(0) # take center frame

                    lr, sr, hr = toPIL(lr), toPIL(sr.float() / (opt.nch_out - 1)), toPIL(hr.float())

                    if makeplotBool: plt.figure(figsize=(10,5))
                    calcScores(lr, hr, makeplotBool, plotidx=1, title='ns')
                    bc_psnr, bc_ssim = calcScores(lr, hr, makeplotBool, plotidx=2, title='bc')
                    sr_psnr, sr_ssim = calcScores(sr, hr, makeplotBool, plotidx=3, title='re')
                    calcScores(hr, None, makeplotBool, plotidx=4)
            elif opt.task == 'classification':
                predclass = sr.argmax(dim=0,keepdim=True)
                if predclass.numpy() != hr.numpy():
                    mean_bc_psnr += 1 # misclassification
                else:
                    mean_sr_psnr += 1

                if i + 1 == len(loader):
                    print('successes (%d/%d)' % (mean_sr_psnr,mean_bc_psnr+mean_sr_psnr))

                continue
            else:

                if 'sim' not in opt.dataset and opt.scale > 1:
                    sr = torch.clamp(sr,min=0,max=1)

                    if lr.shape[0] > 3:
                        lr = lr[lr.shape[0] // 2] # channels are not for colours but separate grayscale frames, take middle
                        hr = hr[hr.shape[0] // 2]

                    img = toPIL(lr)
                    lr = toTensor(img.resize((hr.shape[2],hr.shape[1]),Image.NEAREST))
                    bc = toTensor(img.resize((hr.shape[2],hr.shape[1]),Image.BICUBIC))

                    # ---- Plotting -----

                    lr, bc, sr, hr = toPIL(lr), toPIL(bc), toPIL(sr), toPIL(hr)

                    if makeplotBool: plt.figure(figsize=(10,5))
                    calcScores(lr, hr,makeplotBool, plotidx=1, title='lr')
                    bc_psnr, bc_ssim = calcScores(bc, hr,makeplotBool, plotidx=2, title='bc')
                    sr_psnr, sr_ssim = calcScores(sr, hr,makeplotBool, plotidx=3, title='sr')
                    calcScores(hr, None, makeplotBool, plotidx=4)
                elif 'sim' in opt.dataset: # SIM dataset

                    if 'simin_simout' in opt.task or 'wfin_simout' in opt.task:
                        ## sim target
                        gt_bat = bat[2]
                        wf_bat = bat[3]
                        bc, hr, lr = hr_bat.data[j], gt_bat.data[j], wf_bat.data[j]
                        sr = torch.clamp(sr,min=0,max=1)
                    else:
                        ## gt target
                        sim_bat = bat[2]
                        wf_bat = bat[3]
                        bc, hr, lr = sim_bat.data[j], hr_bat.data[j], wf_bat.data[j]
                        sr = torch.clamp(sr,min=0,max=1)


                    # fix to deal with 3D deconvolution
                    if opt.nch_out > 1:
                        lr = lr[lr.shape[0] // 2] # channels are not for colours but separate grayscale frames, take middle
                        sr = sr[sr.shape[0] // 2]
                        hr = hr[hr.shape[0] // 2]

                    ### Common commands
                    lr, bc, sr, hr = toPIL(lr), toPIL(bc), toPIL(sr), toPIL(hr)

                    if opt.scale == 2:
                        lr = lr.resize((1024,1024), resample=Image.BICUBIC)
                        bc = bc.resize((1024,1024), resample=Image.BICUBIC)
                        hr = hr.resize((1024,1024), resample=Image.BICUBIC)

                    if makeplotBool: plt.figure(figsize=(10,5))
                    calcScores(lr, hr, makeplotBool, plotidx=1, title='WF')
                    bc_psnr, bc_ssim = calcScores(bc, hr, makeplotBool, plotidx=2, title='SIM')
                    sr_psnr, sr_ssim = calcScores(sr, hr, makeplotBool, plotidx=3, title='SR')
                    calcScores(hr, None, makeplotBool, plotidx=4)
                else: # denoising / restoration
                    sr = torch.clamp(sr,min=0,max=1)

                    if lr.shape[0] > 3:
                        lr = lr[lr.shape[0] // 2].unsqueeze(0) # channels are not for colours but separate grayscale frames, take middle
                        sr = sr[sr.shape[0] // 2].unsqueeze(0)
                        hr = hr[hr.shape[0] // 2].unsqueeze(0)

                    img = toPIL(lr)

                    if lr.shape[0] == 1:
                        bc = ndimage.gaussian_filter(img, sigma=(0.6, 0.6), order=0)
                        bc = np.expand_dims(bc, 2)
                    else:
                        bc = ndimage.gaussian_filter(img, sigma=(0.5, 0.5, 0.2), order=0)

                    # ---- Plotting -----
                    lr, bc, sr, hr = toPIL(lr), toPIL(bc), toPIL(sr), toPIL(hr)

                    if makeplotBool: plt.figure(figsize=(10,5))
                    calcScores(lr, hr, makeplotBool, plotidx=1, title='ns')
                    bc_psnr, bc_ssim = calcScores(bc, hr, makeplotBool, plotidx=2, title='sm')
                    sr_psnr, sr_ssim = calcScores(sr, hr, makeplotBool, plotidx=3, title='re')
                    calcScores(hr, None, makeplotBool, plotidx=4)

            mean_bc_psnr += bc_psnr
            mean_sr_psnr += sr_psnr
            mean_bc_ssim += bc_ssim
            mean_sr_ssim += sr_ssim

            if opt.test:
                bc_psnr_arr.append(bc_psnr)
                sr_psnr_arr.append(sr_psnr)
                bc_ssim_arr.append(bc_ssim)
                sr_ssim_arr.append(sr_ssim)

            if opt.log and not opt.test:
                opt.writer.add_scalar('testimage_sr_psnr/%d' % count, sr_psnr,idx+1)
                opt.writer.add_scalar('testimage_sr_ssim/%d' % count, sr_ssim,idx+1)
                opt.writer.add_scalar('testimage_bc_psnr/%d' % count, bc_psnr,idx+1)
                opt.writer.add_scalar('testimage_bc_ssim/%d' % count, bc_ssim,idx+1)

            if makeplotBool:
                plt.tight_layout()
                plt.subplots_adjust(wspace=0.01, hspace=0.01)
                # plt.savefig('%s/combined_%d.png' % (opt.out,count), dpi=300, bbox_inches = 'tight', pad_inches = 0)
                plt.savefig('%s/combined_epoch%d_%d.png' % (opt.out,idx+1,count), dpi=300, bbox_inches = 'tight', pad_inches = 0)
                plt.close()
            if opt.test:
                # lr.save('%s/lr_epoch%d_%d.png' % (opt.out,idx+1,count))
                # sr.save('%s/sr_epoch%d_%d.png' % (opt.out,idx+1,count))
                # hr.save('%s/hr_epoch%d_%d.png' % (opt.out,idx+1,count))
                orig_filename = os.path.basename(bat[-1][0])
                sr.save('%s/%s.png' % (opt.out,orig_filename))

            if True: # wandb imaging logging
                if opt.task == 'segment':
                    opt.wandb.log({'valid_img_lr_%d' % count: wandb.Image(lr)},step=idx+1)
                    opt.wandb.log({'valid_img_sr_%d' % count: wandb.Image(sr)},step=idx+1)
                    opt.wandb.log({'valid_img_hr_%d' % count: wandb.Image(hr)},step=idx+1)
                else:
                    opt.wandb.log({'valid_img_lr_%d' % count: wandb.Image(lr)},step=idx+1)
                    opt.wandb.log({'valid_img_bc_%d' % count: wandb.Image(bc)},step=idx+1)
                    opt.wandb.log({'valid_img_sr_%d' % count: wandb.Image(sr)},step=idx+1)
                    opt.wandb.log({'valid_img_hr_%d' % count: wandb.Image(hr)},step=idx+1)


            if opt.logimage:
                if opt.task == 'segment':
                    opt.writer.add_image('lr/%d' % count, toTensor(lr),idx+1)
                    opt.writer.add_image('sr/%d' % count, toTensor(sr),idx+1)
                    opt.writer.add_image('hr/%d' % count, toTensor(hr),idx+1)
                else:
                    opt.writer.add_image('lr/%d' % count, toTensor(lr),idx+1)
                    opt.writer.add_image('bc/%d' % count, toTensor(bc),idx+1)
                    opt.writer.add_image('sr/%d' % count, toTensor(sr),idx+1)
                    opt.writer.add_image('hr/%d' % count, toTensor(hr),idx+1)

            count += 1
            if opt.test: print('[%d/%d]' % (count,min(len(loader),opt.ntest)),end='\r')
            if count == opt.ntest: break
        if count == opt.ntest: break

    if opt.test: print('\n')
    summarystr = ""
    if count == 0:
        summarystr += 'Warning: all test samples skipped - count forced to 1 -- '
        count = 1
    summarystr += 'Testing of %d samples complete. bc: %0.2f dB / %0.4f, sr: %0.2f dB / %0.4f' % (count, mean_bc_psnr / count, mean_bc_ssim / count, mean_sr_psnr / count, mean_sr_ssim / count)
    opt.wandb.log({'valid_bc_psnr':mean_bc_psnr / count, 'valid_bc_ssim': mean_bc_ssim / count, 'valid_sr_psnr':mean_sr_psnr / count, 'valid_sr_ssim': mean_sr_ssim / count},step=idx+1)
    print(summarystr)
    print(summarystr,file=opt.fid)
    opt.fid.flush()

    if opt.test:
        np.save('%s/test_scores.npy' % opt.out,{'bc_ssim_arr':bc_ssim_arr,'sr_ssim_arr':sr_ssim_arr,'bc_psnr_arr':bc_psnr_arr,'sr_psnr_arr':sr_psnr_arr})

    if opt.log and not opt.test:
        opt.writer.add_scalar('test/psnr', mean_sr_psnr / count,idx+1)
        opt.writer.add_scalar('test/ssim', mean_sr_ssim / count,idx+1)
        t1 = time.perf_counter() - opt.t0
        mem = torch.cuda.memory_allocated()
        print(idx,t1,mem,mean_sr_psnr / count, mean_sr_ssim / count, file=opt.test_stats)
        opt.test_stats.flush()


def generate_convergence_plots(opt,filename):
    fid = open(filename,'r')
    psnrlist = []
    ssimlist = []

    for line in fid:
        if 'sr: ' in line:
            psnrlist.append(float(line.split('sr: ')[1].split(' dB')[0]))
            ssimlist.append(float(line.split('sr: ')[1].split(' dB / ')[1]))

    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.plot(psnrlist,'.-')
    plt.title('PSNR')

    plt.subplot(122)
    plt.plot(ssimlist,'.-')
    plt.title('SSIM')

    plt.savefig('%s/convergencePlot.png' % opt.out, dpi=300)
