# Write your function median_FITS here:

import sys
import time
import numpy as np
from astropy.io import fits


def median_fits(images):
  start = time.perf_counter()
  a = np.array([])

  for image in images:
    hdulist = fits.open(image)
    data = hdulist[0].data
    if a == []:
      a = data
    else:
      a = np.append(a, data)
   
  a = np.reshape(a, (len(images), len(data), len(data[0])))

  res = np.median(a, axis = (0))
  return (res, time.perf_counter()-start, a.nbytes / 1024 )


# You can use this to test your function.
# Any code inside this `if` statement will be ignored by the automarker.
if __name__ == '__main__':
  # Run your function with first example in the question.
  result = median_fits(['image0.fits', 'image1.fits'])
  print(result[0][100, 100], result[1], result[2])
  
  # Run your function with second example in the question.
  result = median_fits(['image{}.fits'.format(str(i)) for i in range(11)])
  print(result[0][100, 100], result[1], result[2])