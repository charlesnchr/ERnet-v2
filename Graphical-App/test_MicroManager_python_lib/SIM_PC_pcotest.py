import pco
import matplotlib.pyplot as plt
import time

t0 = time.perf_counter()

with pco.Camera() as cam:
    # cam.set_exposure_time(0.001)
    cam.configuration = {'trigger_mode': 'external exposure control',
                     'aquire_mode_ex':'external modulated',
                     'roi': (1, 1, 512, 512),
                     'acquire': 'auto',
                     'metadata': 'on',
                     'binning': (1, 1)}
    for i in range(100):
        cam.record()
        images, meta = cam.images()

        fpsval = 1 / (time.perf_counter() - t0)
        t0 = time.perf_counter()
        print(len(images),images[0].shape,fpsval)

    # plt.imshow(image, cmap='gray')
    # plt.show()