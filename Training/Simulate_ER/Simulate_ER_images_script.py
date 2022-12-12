""" ----------------------------------------
* Creation Time : Mon Dec 12 10:35:34 2022
* Author : Charles N. Christensen
* Github : github.com/charlesnchr
----------------------------------------"""


# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from datetime import date
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

# from skimage.measure import compare_ssim
from skimage import io, img_as_ubyte
from delaunay2D import Delaunay2D


# this function was used for noise test of 20221030
def calculate_snr(image, gt):
    signal = np.mean(image[gt > 100])
    noise = np.std(image[gt < 50])
    snr = signal / noise
    return snr


def generate_ER_images(
    radius=500,
    dpi=128,
    breakEdgesProbability=0.2,
    inch=4,
    N=10,
    numSeeds=100,
    foldername="ER_brokenEdges_" + date.today().strftime("%Y%m%d"),
    noise=[0.0001, 0.0005],
    darkness=True,
    sigma=2,
    postfix="",
):
    def voronoi(P):
        delauny = Delaunay(P)
        triangles = delauny.points[delauny.vertices]

        lines = []

        # Triangle vertices
        A = triangles[:, 0]
        B = triangles[:, 1]
        C = triangles[:, 2]
        lines.extend(zip(A, B))
        lines.extend(zip(B, C))
        lines.extend(zip(C, A))
        lines = matplotlib.collections.LineCollection(lines, color="r")
        plt.gca().add_collection(lines)

        circum_centers = np.array([triangle_csc(tri) for tri in triangles])

        segments = []
        for i, triangle in enumerate(triangles):
            circum_center = circum_centers[i]
            for j, neighbor in enumerate(delauny.neighbors[i]):
                if neighbor != -1:
                    segments.append((circum_center, circum_centers[neighbor]))
                else:
                    ps = triangle[(j + 1) % 3] - triangle[(j - 1) % 3]
                    ps = np.array((ps[1], -ps[0]))

                    middle = (triangle[(j + 1) % 3] + triangle[(j - 1) % 3]) * 0.5
                    di = middle - triangle[j]

                    ps /= np.linalg.norm(ps)
                    di /= np.linalg.norm(di)

                    if np.dot(di, ps) < 0.0:
                        ps *= -1000.0
                    else:
                        ps *= 1000.0
                    segments.append((circum_center, circum_center + ps))
        return segments

    def triangle_csc(pts):
        rows, cols = pts.shape

        A = np.bmat(
            [
                [2 * np.dot(pts, pts.T), np.ones((rows, 1))],
                [np.ones((1, rows)), np.zeros((1, 1))],
            ]
        )

        b = np.hstack((np.sum(pts * pts, axis=1), np.ones((1))))
        x = np.linalg.solve(A, b)
        bary_coords = x[:-1]
        return np.sum(
            pts * np.tile(bary_coords.reshape((pts.shape[0], 1)), (1, pts.shape[1])),
            axis=0,
        )

    if __name__ == "__main__":
        P = np.random.random((300, 2))

        X, Y = P[:, 0], P[:, 1]

        fig = plt.figure(figsize=(4.5, 4.5))
        axes = plt.subplot(1, 1, 1)

        plt.scatter(X, Y, marker=".")
        plt.axis([-0.05, 1.05, -0.05, 1.05])

        segments = voronoi(P)
        lines = matplotlib.collections.LineCollection(segments, color="k")
        axes.add_collection(lines)
        plt.axis([-0.05, 1.05, -0.05, 1.05])

    # %% Implementation from Github

    def noisy(noise_typ, image, opts=[0, 0.005]):
        if noise_typ == "gauss":
            row, col, ch = image.shape
            mean = opts[0]
            var = opts[1]
            sigma = var**0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = image + gauss
            return noisy
        elif noise_typ == "s&p":
            row, col, ch = image.shape
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            out[coords] = 0
            return out
        elif noise_typ == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(opts[0]))
            noisy = np.random.poisson(image * vals) / float(vals)
            return noisy / np.max(noisy) * np.max(image)
        elif noise_typ == "speckle":
            row, col, ch = image.shape
            gauss = np.random.randn(row, col, ch)
            gauss = gauss.reshape(row, col, ch)
            noisy = image + image * gauss
            return noisy

    def degrade(img, dim, noise, darkness):

        if darkness:
            # gaussian darkness
            X, Y = np.meshgrid(np.linspace(0, 1, dim), np.linspace(0, 1, dim))
            mu_x, mu_y = np.random.rand(), np.random.rand()
            var_x = np.max([0, 0.05 * np.random.randn() + 0.2])
            var_y = np.max([0, 0.05 * np.random.randn() + 0.2])
            Z = np.exp(-((X - mu_x) ** 2) / (2 * var_x)) * np.exp(
                -((Y - mu_y) ** 2) / (2 * var_y)
            )
            Z = np.expand_dims(Z, 2)

            # img = Z*img # outcommented, 2022 may
            img = Z * img  # reintroduced, 2022 october
            # darkimg = (0.2*np.random.rand()+0.8)*darkimg  # overall level between 0.5 and 0.1

        # these two lines were not actually applied, now outcommented
        # poisson_param = np.max([0,3*np.random.randn() + 10])
        # img = noisy('poisson',img,[poisson_param])

        if noise is not None:
            gauss_param = np.max([0, noise[0] * np.random.randn() + noise[1]])
            img = noisy("gauss", img, [0, gauss_param])

        img = np.clip(img, 0, 1)
        return img

    # %%

    def GetVoronoi():

        ###########################################################
        # Generate 'numSeeds' random seeds in a square of size 'radius'

        seeds = radius * np.random.random((numSeeds, 2))
        # print("seeds:\n", seeds)
        #     print("BBox Min:", np.amin(seeds, axis=0),
        #           "Bbox Max: ", np.amax(seeds, axis=0))
        """
        Compute our Delaunay triangulation of seeds.
        """
        # It is recommended to build a frame taylored for our data
        # dt = D.Delaunay2D() # Default frame
        center = np.mean(seeds, axis=0)
        dt = Delaunay2D(center, 50 * radius)

        # Insert all seeds one by one
        for s in seeds:
            dt.addPoint(s)

        # Dump number of DT triangles
        #     print (len(dt.exportTriangles()), "Delaunay triangles")
        """
        Demostration of how to plot the data.
        """
        import matplotlib.pyplot as plt
        import matplotlib.tri
        import matplotlib.collections

        # Create a plot with matplotlib.pyplot
        fig, ax = plt.subplots()
        ax.margins(0.1)
        ax.set_aspect("equal")
        plt.axis([-1, radius + 1, -1, radius + 1])

        plt.switch_backend("agg")

        # Plot our Delaunay triangulation (plot in blue)
        cx, cy = zip(*seeds)
        dt_tris = dt.exportTriangles()
        ax.triplot(matplotlib.tri.Triangulation(cx, cy, dt_tris), "bo--")

        # Plot annotated Delaunay vertex (seeds)
        """
        for i, v in enumerate(seeds):
            plt.annotate(i, xy=v)
        """

        # Build Voronoi diagram as a list of coordinates and regions
        vc, vr = dt.exportVoronoiRegions()

        return vc, vr, fig

    # %%
    vc, vr, fig = GetVoronoi()

    # %%
    fig, ax = plt.subplots()
    ax.margins(0.1)
    ax.set_aspect("equal")
    plt.axis([-1, radius + 1, -1, radius + 1])

    for r in vr:
        polygon = [vc[i] for i in vr[r]]  # Build polygon for each region
        plt.plot(*zip(*polygon), color="red")  # Plot polygon edges in red

    fig.savefig("testVoronoi.png")
    # %%
    # fig, ax = plt.subplots()
    # ax.margins(0.1)
    # ax.set_aspect('equal')
    # plt.axis([-1, radius+1, -1, radius+1])
    from scipy import interpolate

    def GetNetwork(
        edges,
        breakEdgesProbability=0,
        seed=None,
        noise=[0.0001, 0.0005],
        darkness=True,
        sigma=2,
    ):
        fig = plt.figure(figsize=(inch, inch))

        np.random.seed(seed)

        for edge in edges:
            xold, x, yold, y = edge

            xmid = (xold + x) / 2
            ymid = (yold + y) / 2

            xmid0 = (xmid + xold) / 2
            xmid2 = (xmid + x) / 2

            ymid0 = (ymid + yold) / 2
            ymid2 = (ymid + y) / 2

            X = [xold, xmid0, xmid, xmid2, x]
            Y = [yold, ymid0, ymid, ymid2, y]
            fac = [0.05, 0.1, 0.05]

            for i in range(3):
                xr = np.random.rand() * 10 - 5
                yr = np.random.rand() * 10 - 5

                xterm = xr * (xmid - x) * fac[i]
                yterm = yr * (xmid - x) * fac[i]

                if X[i + 1] + xterm > 0 and X[i + 1] + xterm < radius:
                    X[i + 1] += xterm
                if Y[i + 1] + yterm > 0 and Y[i + 1] + yterm < radius:
                    Y[i + 1] += yterm

            r = np.linspace(0, 1, len(X))
            rnew = np.linspace(0, 1, 20)
            xinterp = interpolate.make_interp_spline(r, X)
            Xnew = xinterp(rnew)
            yinterp = interpolate.make_interp_spline(r, Y)
            Ynew = yinterp(rnew)

            if breakEdgesProbability > np.random.rand():
                plt.plot(Xnew[:8], Ynew[:8], color="red")
                plt.plot(Xnew[11:], Ynew[11:], color="red")
            else:
                plt.plot(Xnew, Ynew, color="red")

        plt.xlim([-1, radius + 1])
        plt.ylim([-1, radius + 1])
        plt.xticks([], [])
        plt.yticks([], [])
        plt.gca().axis("off")
        plt.tight_layout()
        plt.savefig("temp.png", dpi=dpi)
        plt.close()

        network = io.imread("temp.png")
        I = network.copy()
        I = -I.mean(axis=2, keepdims=True) + 255
        GT = ndimage.gaussian_filter(I, sigma=sigma)
        GT = (GT - np.min(GT)) / (np.max(GT) - np.min(GT))
        GT = GT.astype("float32")

        GT[GT > 0.5] = 1
        GT[GT <= 0.5] = 0

        I = (I - np.min(I)) / (np.max(I) - np.min(I))
        I = I.astype("float32")
        I = ndimage.gaussian_filter(I, sigma=sigma)
        I = degrade(I, I.shape[0], noise, darkness)

        return network, GT, I, fig

    # %%
    def GetEdges(vc, vr):

        edges = []

        # Plot voronoi diagram edges (in red)
        for r in vr:
            polygon = [vc[i] for i in vr[r]]  # Build polygon for each region
            #     plt.plot(*zip(*polygon), color="red", linewidth=2)  # Plot polygon edges in red
            count = 0
            xold = 0
            yold = 0
            for x, y in polygon:
                if count == 0:
                    count += 1
                    xold = x
                    yold = y
                    continue
                if xold < x:
                    edge = [xold, x, yold, y]
                else:
                    edge = [x, xold, y, yold]
                if not edge in edges:
                    edges.append(edge)
                count += 1
                xold = x
                yold = y

        # break up some edges
        new_edges = []

        for edge in edges:

            if np.random.rand() > 0.5:
                xold, x, yold, y = edge

                X = np.linspace(xold, x, 11)
                Y = np.linspace(yold, y, 11)

                new_edges.append([X[0], X[4], Y[0], Y[4]])
                new_edges.append([X[6], X[-1], Y[6], Y[-1]])
            else:
                new_edges.append(edge)

        return edges, new_edges

    # %%
    from skimage import io
    import scipy.misc
    from scipy import ndimage
    import os

    os.makedirs(foldername, exist_ok=True)

    plt.switch_backend("agg")

    for i in range(N):
        vc, vr, fig = GetVoronoi()

        # Delaunay figure
        fig.gca().axis("off")
        plt.tight_layout()
        fig.savefig("%s/%d_Delaunay.pdf" % (foldername, i))

        # Voronoi figure
        fig, ax = plt.subplots()
        ax.margins(0.1)
        ax.set_aspect("equal")
        plt.axis([-1, radius + 1, -1, radius + 1])

        for r in vr:
            polygon = [vc[i] for i in vr[r]]  # Build polygon for each region
            plt.plot(*zip(*polygon), color="red")  # Plot polygon edges in red

        fig.gca().axis("off")
        plt.tight_layout()
        fig.savefig("%s/%d_Voronoi.pdf" % (foldername, i))

        edges, broken_edges = GetEdges(vc, vr)
        seed = np.random.randint(0, 1000000)
        network1, GT1, I1, fig = GetNetwork(
            edges, seed=seed, noise=None, darkness=darkness, sigma=sigma
        )
        network2, GT2, I2, _ = GetNetwork(
            edges,
            breakEdgesProbability=breakEdgesProbability,
            seed=seed,
            noise=noise,
            darkness=darkness,
            sigma=sigma,
        )

        # Cubic spline interp figure
        fig.gca().axis("off")
        plt.tight_layout()
        fig.savefig("%s/%d_Cubicspline.pdf" % (foldername, i))

        # plt.figure(figsize=(20,10))

        # plt.subplot(121)

        # GT
        # plt.imshow(GT1[:,:,0],cmap='gray')
        # io.imsave('%s/%d_GT%s.png' % (foldername, i, postfix), img_as_ubyte(GT1[:, :, 0]))
        # for noise test 20221030
        GT = img_as_ubyte(I1[:, :, 0])
        io.imsave("%s/%d_GT%s.png" % (foldername, i, postfix), GT)

        # IN
        IN = img_as_ubyte(I2[:, :, 0])
        snr = calculate_snr(IN, GT)
        io.imsave("%s/%d_IN_snr%0.2f%s.png" % (foldername, i, snr, postfix), IN)

        # plt.savefig('%s/%d.jpg' % (foldername,i))
        # np.save('%s/%d.npy' % (foldername,i),(I2,GT1))
        print("[%d/%d]" % (i + 1, N), end="\r")
    print("")


if __name__ == "__main__":

    # -----------
    # Examples
    # -----------

    # generate_ER_images( radius=500, dpi=128, breakEdgesProbability=0.5, numSeeds=400, foldername = 'ER_500_' + date.today().strftime("%Y%m%d") )
    # generate_ER_images( radius=500, dpi=128, breakEdgesProbability=0.4, numSeeds=400, foldername = 'ER_highly_broken_tubules_500_' + date.today().strftime("%Y%m%d") )
    # generate_ER_images( radius=500, dpi=128, breakEdgesProbability=0.0, numSeeds=400,
    #                    foldername = 'ER_no_broken_tubules_500_' + date.today().strftime("%Y%m%d") )
    # generate_ER_images( radius=500, inch=8, dpi=64, breakEdgesProbability=0.0, numSeeds=400,
    #                    foldername = 'ER_no_broken_tubules_500_halfwidth_' + date.today().strftime("%Y%m%d") )

    # generate_ER_images( radius=250, dpi=64, breakEdgesProbability=0.5, foldername = 'ER_250_' + date.today().strftime("%Y%m%d") )
    # generate_ER_images( radius=1000, inch=8, dpi=128, breakEdgesProbability=0.5, numSeeds=1600, foldername = 'ER_1000_' + date.today().strftime("%Y%m%d") )

    # 20221008 -- updated generation pipeline figure for NatMeth paper
    generate_ER_images(
        radius=250,
        dpi=64,
        breakEdgesProbability=0,
        foldername="ER_250_" + date.today().strftime("%Y%m%d"),
    )
