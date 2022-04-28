import sys
import os
from skimage import io
import matplotlib.pyplot as plt
import time

initdir = os.getcwd()
MM_path = 'C:\\Program Files\\Micro-Manager-1.4'
sys.path.append(MM_path)
import MMCorePy # requires the above path
os.chdir(MM_path) # necessary


mmc = MMCorePy.CMMCore()  # Instance micromanager core
print mmc.getVersionInfo()
print mmc.getAPIVersionInfo()
mmc.loadDevice("OpenCVgrabber","OpenCVgrabber","OpenCVgrabber")
mmc.initializeAllDevices()
mmc.setCameraDevice('OpenCVgrabber')


# Save image
# img8 = (img.astype('float') / (256**2)).astype('uint8')
# print img8.max(), img8.min()
# io.imsave('test.png',img8)
t0 = time.clock()

fig, ax = plt.subplots()


os.chdir(initdir)

for i in range(10):
    # mmc.snapImage()
    img = mmc.getImage()

    plt.cla()
    plt.imshow(img,cmap='magma')

    fpsval = 1 / (time.clock() - t0)
    
    fps = 'FPS: %0.1f' % fpsval
    t0 = time.clock()
    print(fps)
    plt.title(fps)
    plt.pause(0.01)


print('done')
