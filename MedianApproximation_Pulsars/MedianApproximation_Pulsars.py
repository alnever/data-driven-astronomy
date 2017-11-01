# Import the running_stats function
from helper import running_stats
import numpy as np
from astropy.io import fits
# Write your median_bins_fits and median_approx_fits here:

def get_data(images):
  a = np.array([])

  for image in images:
    hdulist = fits.open(image)
    data = hdulist[0].data
    if a == []:
      a = data
    else:
      a = np.append(a, data)
   
  a = np.reshape(a, (len(images), len(data), len(data[0])))
  
  return a
  

def median_bins_fits(images, bins):
  mean, stdv = running_stats(images)
  a = get_data(images)

  minval = mean - stdv
  maxval = mean + stdv
  width = 2 * stdv / bins

  left = a[0:len(a)] < minval
  left_bin = np.sum(left, axis = (0))
  
  in_bins = []
  x = minval
  i = 0
  while  i < bins:
    passend = (a[0:len(a)] >= x) & (a[0:len(a)] < x + width)
    s = np.sum(passend, axis = (0))
    if in_bins == []:
       in_bins = s
    else: 
       in_bins = np.dstack((in_bins, s)) 
    x += width
    i += 1

  
  return (mean, stdv, left_bin, np.array(in_bins))

def median_approx_fits(images, bins):
  mean, std, left_bin, in_bins = median_bins_fits(images, bins)

  minval = mean - std
  maxval = mean + std
  width = 2 * std / bins

  x = minval
  total = left_bin
  i = 0
  #while np.all((x < maxval) & (total < (len(images) + 1)/2)):
  if (len(images) % 2 == 0):
    t = len(images) // 2
  else:
    t = len(images) // 2 + 1
  
  
  while (i < bins) & np.all(total <= t):
    total += np.sum(in_bins[:,:,i])
    i += 1
    x += width
  
  if (bins % 2 == 0):
    return x + width/2
  else:
    return x - width/2





# You can use this to test your function.
# Any code inside this `if` statement will be ignored by the automarker.
if __name__ == '__main__':
  # Run your function with examples from the question.
  mean, std, left_bin, bins = median_bins_fits(['image0.fits', 'image1.fits', 'image2.fits'], 5)
  median = median_approx_fits(['image0.fits', 'image1.fits', 'image2.fits'], 5)
  print(mean[100,100])
  print(std[100,100])
  print(left_bin[100,100])
  print(bins[100,100,:])
  print(median[100, 100])
  
  mean, std, left_bin, bins = median_bins_fits(['image{}.fits'.format(str(i)) for i in range(11)], 4)
  median = median_approx_fits(['image{}.fits'.format(str(i)) for i in range(11)], 4)
  print(mean[100,100])
  print(std[100,100])
  print(left_bin[100,100])  
  print(bins[100,100,:])
  print(median[100, 100])
