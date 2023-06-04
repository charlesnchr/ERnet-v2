import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import glob
from skimage import io, exposure, img_as_ubyte
from PIL import Image

from tqdm.notebook import tqdm
from swinir_rcab_arch import SwinIR_RCAB
from rcan_arch import RCAN

toTensor = transforms.ToTensor()
toPIL = transforms.ToPILImage()


from graph_processing import performGraphProcessing


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


def changeColour(I):  # change colours (used to match WEKA output, request by Meng)
    Inew = np.zeros(I.shape + (3,)).astype("uint8")
    for rowidx in range(I.shape[0]):
        for colidx in range(I.shape[1]):
            if I[rowidx][colidx] == 0:
                Inew[rowidx][colidx] = [198, 118, 255]
            elif I[rowidx][colidx] == 127:
                Inew[rowidx][colidx] = [79, 255, 130]
            elif I[rowidx][colidx] == 255:
                Inew[rowidx][colidx] = [255, 0, 0]
    return Inew


def AssembleStacks(basefolder):
    # export to tif

    folders = []
    folders.append(basefolder + "/in")
    folders.append(basefolder + "/out")

    for subfolder in ["in", "out"]:
        folder = basefolder + "/" + subfolder
        if not os.path.isdir(folder):
            continue
        imgs = glob.glob(folder + "/*.jpg")
        imgs.extend(glob.glob(folder + "/*.png"))
        n = len(imgs)

        shape = io.imread(imgs[0]).shape
        h = shape[0]
        w = shape[1]

        if len(shape) == 2:
            I = np.zeros((n, h, w), dtype="uint8")
        else:
            c = shape[2]
            I = np.zeros((n, h, w, c), dtype="uint8")

        for nidx, imgfile in enumerate(imgs):
            img = io.imread(imgfile)
            I[nidx] = img

            print("%s : [%d/%d] loaded imgs" % (folder, nidx + 1, len(imgs)), end="\r")
        print("")

        stackname = os.path.basename(basefolder)
        stackfilename = "%s/%s_%s.tif" % (basefolder, stackname, subfolder)
        io.imsave(stackfilename, I)
        print("saved stack: %s" % stackfilename)


def processImage_tiled(net, opt, imgid, img, savepath_in, savepath_out):
    imageSize = opt.imageSize

    h, w = img.shape[0], img.shape[1]
    if imageSize == 0:
        imageSize = 250
        while imageSize + 250 < h and imageSize + 250 < w:
            imageSize += 250
        print("Set imageSize to", imageSize)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not opt.cpu else "cpu"
    )

    # img_norm = (img - np.min(img)) / (np.max(img) - np.min(img))
    images = []

    images.append(img[:imageSize, :imageSize])
    images.append(img[h - imageSize :, :imageSize])
    images.append(img[:imageSize, w - imageSize :])
    images.append(img[h - imageSize :, w - imageSize :])

    proc_images = []
    for idx, sub_img in enumerate(images):
        # sub_img = (sub_img - np.min(sub_img)) / (np.max(sub_img) - np.min(sub_img))
        pil_sub_img = Image.fromarray((sub_img * 255).astype("uint8"))

        # sub_tensor = torch.from_numpy(np.array(pil_sub_img)/255).float().unsqueeze(0)
        sub_tensor = toTensor(pil_sub_img)

        sub_tensor = sub_tensor.unsqueeze(0)

        with torch.no_grad():
            sr = net(sub_tensor.to(device))

            sr = sr.cpu()
            # sr = torch.clamp(sr,min=0,max=1)

            m = nn.LogSoftmax(dim=0)
            sr = m(sr[0])
            sr = sr.argmax(dim=0, keepdim=True)

            # pil_sr_img = Image.fromarray((255*(sr.float() / (opt.nch_out - 1)).squeeze().numpy()).astype('uint8'))
            pil_sr_img = toPIL(sr.float() / (opt.nch_out - 1))

            # pil_sr_img.save(opt.out + '/segmeneted_output_' + str(i) + '_' + str(idx) + '.png')
            # pil_sub_img.save(opt.out + '/imageinput_' + str(i) + '_' + str(idx) + '.png')

            proc_images.append(pil_sr_img)

    # stitch together
    img1 = proc_images[0]
    img2 = proc_images[1]
    img3 = proc_images[2]
    img4 = proc_images[3]

    woffset = (2 * imageSize - w) // 2
    hoffset = (2 * imageSize - h) // 2

    img1 = np.array(img1)[: imageSize - hoffset, : imageSize - woffset]
    img3 = np.array(img3)[: imageSize - hoffset, woffset:]
    top = np.concatenate((img1, img3), axis=1)

    img2 = np.array(img2)[hoffset:, : imageSize - woffset]
    img4 = np.array(img4)[hoffset:, woffset:]
    bot = np.concatenate((img2, img4), axis=1)

    oimg = np.concatenate((top, bot), axis=0)
    # crop?
    # oimg = oimg[10:-10,10:-10]
    # img = img[10:-10,10:-10]
    # remove boundaries?
    # oimg[:10,:] = 0
    # oimg[-10:,:] = 0
    # oimg[:,:10] = 0
    # oimg[:,-10:] = 0

    if opt.stats_tubule_sheet:
        npix1 = np.sum(oimg == 170)  # tubule
        npix2 = np.sum(oimg == 255)  # sheet
        npix3 = np.sum(oimg == 85)  # SBT

        npix = npix1 + npix2 + npix3
        opt.csvfid.write(
            "%s,%0.4f,%0.4f,%0.4f\n" % (imgid, npix1 / npix, npix2 / npix, npix3 / npix)
        )
    if opt.weka_colours:
        oimg = changeColour(oimg)

    Image.fromarray(oimg).save(savepath_out)
    if opt.save_input:
        io.imsave(savepath_in, img_as_ubyte(img))

    # Image.fromarray((img*255).astype('uint8')).save('%s/input_%04d.png' % (opt.out,i))


def processImage(net, opt, imgid, img, savepath_in, savepath_out):
    imageSize = opt.imageSize

    h, w = img.shape[0], img.shape[1]

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not opt.cpu else "cpu"
    )

    # img_norm = (img - np.min(img)) / (np.max(img) - np.min(img))
    pil_sub_img = Image.fromarray((img * 255).astype("uint8"))

    # sub_tensor = torch.from_numpy(np.array(pil_sub_img)/255).float().unsqueeze(0)
    sub_tensor = toTensor(pil_sub_img)

    if "swin" in opt.weights and "nch1" not in opt.weights:
        sub_tensor = torch.cat((sub_tensor, sub_tensor, sub_tensor), 0)
    sub_tensor = sub_tensor.unsqueeze(0)

    with torch.no_grad():
        sr = net(sub_tensor.to(device))

        sr = sr.cpu()
        # sr = torch.clamp(sr,min=0,max=1)

        m = nn.LogSoftmax(dim=0)
        sr = m(sr[0])
        sr = sr.argmax(dim=0, keepdim=True)

        # pil_sr_img = Image.fromarray((255*(sr.float() / (opt.nch_out - 1)).squeeze().numpy()).astype('uint8'))
        pil_sr_img = toPIL(sr.float() / (opt.nch_out - 1))

        # pil_sr_img.save(opt.out + '/segmeneted_output_' + str(i) + '_' + str(idx) + '.png')
        # pil_sub_img.save(opt.out + '/imageinput_' + str(i) + '_' + str(idx) + '.png')

    oimg = np.array(pil_sr_img)

    # workaround for new order of classes
    sheet_ind = oimg == 255
    SBT_ind = oimg == 85
    tubule_ind = oimg == 170
    oimg[sheet_ind] = 85
    oimg[SBT_ind] = 170
    oimg[tubule_ind] = 255

    if opt.stats_tubule_sheet:
        npix1 = np.sum(oimg == 255)  # tubule
        npix2 = np.sum(oimg == 85)  # sheet
        npix3 = np.sum(oimg == 170)  # SBT

        npix = npix1 + npix2 + npix3
        opt.csvfid.write(
            "%s,%0.4f,%0.4f,%0.4f\n" % (imgid, npix1 / npix, npix2 / npix, npix3 / npix)
        )
    if opt.weka_colours:
        oimg = changeColour(oimg)

    Image.fromarray(oimg).save(savepath_out)
    if opt.save_input:
        io.imsave(savepath_in, img_as_ubyte(img))

    # Image.fromarray((img*255).astype('uint8')).save('%s/input_%04d.png' % (opt.out,i))


def EvaluateModel(opt):
    if opt.stats_tubule_sheet:
        # if opt.out == 'root':
        #     if opt.root[0].lower() in ['jpg','png','tif']:
        #         pardir = os.path.abspath(os.path.join(opt.root,os.pardir))
        #         opt.csvfid = open('%s/stats_tubule_sheet.csv' % pardir,'w')
        #     else:
        #         opt.csvfid = open('%s/stats_tubule_sheet.csv' % opt.root,'w')
        # else:
        #     opt.csvfid = open('%s/stats_tubule_sheet.csv' % opt.out,'w')
        opt.csvfid.write("Filename,Tubule fraction,Sheet fraction,SBT fraction\n")

    if opt.graph_metrics:
        opt.graphfid.write(
            "Filename,no_nodes,no_edges,assortativity, clustering, compo, ratio_nodes, \
        ratio_edges, degree 1, degree 2, degree 3, degree 4, degree 5, degree 6\n"
        )

    if "swin" not in opt.weights:
        print("LOADING: CNN architecture")
        # RCAN model
        net = RCAN(opt)
    elif "swin3d" in opt.weights:
        print("LOADING: Transformer architecture")
        # Swin model
        patch_size_t = 1 if "nch1" in opt.weights else 3
        opt.task = "segment"
        net = SwinTransformer3D_RCAB(
            opt,
            patch_size=(patch_size_t, 4, 4),
            in_chans=1,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=(2, 7, 7),
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            upscale=1,
            frozen_stages=-1,
            use_checkpoint=False,
            vis=False,
        )
    elif "swinir" in opt.weights:
        print("LOADING: Transformer architecture")
        # Swin model
        opt.task = "segment"
        net = SwinIR_RCAB(
            opt, img_size=128, in_chans=1, upscale=1, use_checkpoint=False, vis=False
        )
    else:
        print("model architecture not inferred")

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not opt.cpu else "cpu"
    )
    net.to(device)

    checkpoint = torch.load(opt.weights, map_location=device)
    print("loading checkpoint", opt.weights)
    net.load_state_dict(checkpoint["state_dict"])

    if opt.root[0].split(".")[-1].lower() in ["png", "jpg", "tif"]:
        imgs = opt.root
    else:
        imgs = []
        for ext in opt.ext:
            # imgs.extend(glob.glob(opt.root + '/*.jpg')) # scan only folder
            if len(imgs) == 0:  # scan everything
                for dir in opt.root:
                    imgs.extend(glob.glob(dir + "/**/*.%s" % ext, recursive=True))

    # find total number of images to process
    nimgs = 0
    for imgidx, imgfile in enumerate(imgs):
        basepath, ext = os.path.splitext(imgfile)

        if ext.lower() == ".tif":
            img = io.imread(imgfile)
            if len(img.shape) == 2:  # grayscale
                nimgs += 1
            elif img.shape[2] <= 3:
                nimgs += 1
            else:  # t or z stack
                nimgs += img.shape[0]
        else:
            nimgs += 1

    outpaths = []
    imgcount = 0

    # primary loop
    for imgidx, imgfile in enumerate(tqdm(imgs)):
        basepath, ext = os.path.splitext(imgfile)

        if ext.lower() == ".tif":
            img = io.imread(imgfile)
        else:
            img = np.array(Image.open(imgfile))

        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = img.astype("float")

        if len(img.shape) > 2 and img.shape[2] <= 3:
            print("removing colour channel")
            img = np.max(img, 2)  # remove colour channel

        # img = io.imread(imgfile)
        # img = (img - np.min(img)) / (np.max(img) - np.min(img))

        # filenames for saving
        idxstr = "%04d" % imgidx
        if opt.out == "root":  # save next to orignal
            savepath_out = imgfile.replace(ext, "_out_" + idxstr + ".png")
            savepath_in = imgfile.replace(ext, "_in_" + idxstr + ".png")
            basesavepath_graphfigures = imgfile.replace(ext, "")
        else:
            pass  # not implemented

        # process image
        if len(img.shape) == 2:
            p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
            imgnorm = exposure.rescale_intensity(img, in_range=(p1, p99))
            imgid = "%s:%s" % (os.path.basename(imgfile), idxstr)
            processImage(net, opt, imgid, imgnorm, savepath_in, savepath_out)

            # send result
            outpaths.append(savepath_out)

            # graph processing
            if opt.graph_metrics:
                graph_out_paths = performGraphProcessing(
                    savepath_out, opt, basesavepath_graphfigures, imgid
                )

                outpaths.extend(graph_out_paths)

            imgcount += 1

        else:  # more than 3 channels, assuming stack
            basefolder = basepath
            os.makedirs(basefolder, exist_ok=True)
            if opt.save_input:
                os.makedirs(basefolder + "/in", exist_ok=True)
            if opt.graph_metrics:
                os.makedirs(basefolder + "/graph", exist_ok=True)
            os.makedirs(basefolder + "/out", exist_ok=True)

            for subimgidx in tqdm(range(img.shape[0])):
                idxstr = "%04d_%04d" % (imgidx, subimgidx)
                savepath_in = "%s/in/%s.png" % (basefolder, idxstr)
                savepath_out = "%s/out/%s.png" % (basefolder, idxstr)
                basesavepath_graphfigures = "%s/graph/%s" % (basefolder, idxstr)
                p1, p99 = np.percentile(img[subimgidx], 1), np.percentile(
                    img[subimgidx], 99
                )
                imgnorm = exposure.rescale_intensity(img[subimgidx], in_range=(p1, p99))
                imgid = "%s:%s" % (os.path.basename(imgfile), idxstr)
                processImage(net, opt, imgid, imgnorm, savepath_in, savepath_out)

                # send result
                outpaths.append(savepath_out)

                # graph processing
                if opt.graph_metrics:
                    graph_out_paths = performGraphProcessing(
                        savepath_out, opt, basesavepath_graphfigures, imgid
                    )

                    outpaths.extend(graph_out_paths)

                imgcount += 1
            AssembleStacks(basefolder)

    if opt.stats_tubule_sheet:
        opt.csvfid.close()

    if opt.graph_metrics:
        opt.graphfid.close()

    print("Saved", outpaths)
    return outpaths
