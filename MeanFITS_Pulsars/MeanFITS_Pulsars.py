# Write your mean_fits function here:
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

def get_data(image):
  hdulist = fits.open(image)
  return hdulist[0].data

def mean_fits(images):
  res = []
  for image in images:
    data = get_data(image)
    if res == []:
      res = data
    else:
      res += data
  res /= len(images)  
  return res
  
if __name__ == '__main__':
  
  # Test your function with examples from the question
  data  = mean_fits(['image0.fits', 'image1.fits', 'image2.fits'])
  print(data[100, 100])
  
  data = mean_fits(['image0.fits', 'image1.fits', 'image3.fits'])
  print(data[100, 100])
  
  data = mean_fits(['image0.fits', 'image1.fits', 'image2.fits', 'image3.fits', 'image4.fits'])
  print(data[100, 100])

  # You can also plot the result:
  
  # plt.imshow(data.T, cmap=plt.cm.viridis)
  # plt.colorbar()
  # plt.show()