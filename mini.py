import sys
import numpy as np

from matplotlib import pyplot as plt

image = plt.imread('./{}.jpg'.format(sys.argv[1]))[...,:3]/255.
grad = np.gradient(image[...,0])
grad = grad[0]**2 + grad[1]**2
grad = np.floor(1.001-grad)

image[...,0] = (np.floor(image[...,0]+0.5)+np.floor(image[...,0]+0.6))/2
image[...,1] = (np.floor(image[...,1]+0.5)+np.floor(image[...,1]+0.6))/2
image[...,2] = (np.floor(image[...,2]+0.5)+np.floor(image[...,2]+0.6))/2
image[...,1] = image[...,2] = np.mean(image[...,1:], axis=2)

plt.imsave('./{}fail.jpg'.format(sys.argv[1]), image*np.dstack((grad,grad,grad)))
