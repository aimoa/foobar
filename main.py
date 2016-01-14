import sys
import numpy as np

from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
from scipy.spatial import Delaunay
from skimage.draw import polygon

image = plt.imread('./%s.jpg' % sys.argv[1])/255.
r_grad, c_grad = np.gradient(np.dot(image, [0.299, 0.587, 0.114]))[:2]
grad = r_grad**2 + c_grad**2

coords = peak_local_max(grad, min_distance=4, threshold_rel=0,\
        num_peaks=grad.size/400)
coords = np.vstack((coords,[0,0],[0,image.shape[1]],[image.shape[0],0],\
        [image.shape[0],image.shape[1]]))
tri = Delaunay(coords)
for triangle in tri.simplices:
    pts = coords[triangle]
    rr, cc = polygon(pts[:,0], pts[:,1])
    image[rr, cc, 0] = np.mean(image[rr, cc, 0])
    image[rr, cc, 1] = np.mean(image[rr, cc, 1])
    image[rr, cc, 2] = np.mean(image[rr, cc, 2])

plt.imsave('./%sflat.jpg' % sys.argv[1], image)
