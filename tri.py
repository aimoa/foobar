import sys
import numpy as np

from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
from scipy.spatial import Delaunay
from skimage.draw import polygon

image = plt.imread('./{}.jpg'.format(sys.argv[1]))/255.
grad = np.gradient(np.dot(image, [0.299, 0.587, 0.114]))
grad = grad[0]**2 + grad[1]**2

coords = peak_local_max(grad, min_distance=8, threshold_rel=0)
coords = np.vstack((coords,[0,0],[0,image.shape[1]],[image.shape[0],0],\
        [image.shape[0],image.shape[1]]))
tri = Delaunay(coords)

if (sys.argv[2] == 'mean'):
    for triangle in tri.simplices:
        pts = coords[triangle]
        rr, cc = polygon(pts[:,0], pts[:,1])
        if rr.size:
            image[rr, cc, 0] = np.mean(image[rr, cc, 0])
            image[rr, cc, 1] = np.mean(image[rr, cc, 1])
            image[rr, cc, 2] = np.mean(image[rr, cc, 2])

elif (sys.argv[2] == 'median'):
    for triangle in tri.simplices:
        pts = coords[triangle]
        rr, cc = polygon(pts[:,0], pts[:,1])
        if rr.size:
            image[rr, cc, 0] = np.median(image[rr, cc, 0])
            image[rr, cc, 1] = np.median(image[rr, cc, 1])
            image[rr, cc, 2] = np.median(image[rr, cc, 2])

plt.imsave('./{0}{1}.jpg'.format(*sys.argv[1:]), image)
