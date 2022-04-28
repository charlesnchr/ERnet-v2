import sys
import time
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg
from skimage import data

app = QtGui.QApplication([])
w = pg.GraphicsView()
w.show()
w.resize(600,600)
w.setWindowTitle('pyqtgraph example: Draw')

view = pg.ViewBox(enableMouse=True)
w.setCentralItem(view)

## lock the aspect ratio
view.setAspectLocked(True)
view.invertY()

## Create image item
imgitem = pg.ImageItem(axisOrder='row-major')
view.addItem(imgitem)

labelitem = pg.LabelItem()
view.addItem(labelitem)

img = data.astronaut()
# img = np.rot90(data.astronaut(), k=-1, axes=(0,1))

print('img is',img.shape)
imgitem.setImage(img)
view.autoRange(padding=0)

labelitem.setText("hej",color='CCFF00')



## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
