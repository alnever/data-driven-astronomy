# Write your crossmatch function here.
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
import time


def crossmatch(cat1, cat2, max_dist):
  start = time.perf_counter()
  sky_cat1 = SkyCoord(cat1*u.degree, frame='icrs')
  sky_cat2 = SkyCoord(cat2*u.degree, frame='icrs')
  closest_ids, closest_dists, closest_dists3d = sky_cat1.match_to_catalog_sky(sky_cat2)
  idx_cat1 = np.array(range(0,len(sky_cat1)))
  closest_dists = closest_dists.value
  pass_indexes = np.where(closest_dists <= max_dist)
  no_pass_indexes = np.where(closest_dists > max_dist)
  matches = tuple(np.stack((idx_cat1[pass_indexes],closest_ids[pass_indexes],closest_dists[pass_indexes]), axis = -1))
  no_matches = idx_cat1[no_pass_indexes]
  return matches, no_matches, time.perf_counter() - start
  


# You can use this to test your function.
# Any code inside this `if` statement will be ignored by the automarker.
if __name__ == '__main__':
  # The example in the question
  cat1 = np.array([[180, 30], [45, 10], [300, -45]])
  cat2 = np.array([[180, 32], [55, 10], [302, -44]])
  matches, no_matches, time_taken = crossmatch(cat1, cat2, 5)
  print('matches:', matches)
  print('unmatched:', no_matches)
  print('time taken:', time_taken)

  # A function to create a random catalogue of size n
  def create_cat(n):
    ras = np.random.uniform(0, 360, size=(n, 1))
    decs = np.random.uniform(-90, 90, size=(n, 1))
    return np.hstack((ras, decs))

  # Test your function on random inputs
  np.random.seed(0)
  cat1 = create_cat(10)
  cat2 = create_cat(20)
  matches, no_matches, time_taken = crossmatch(cat1, cat2, 5)
  print('matches:', matches)
  print('unmatched:', no_matches)
  print('time taken:', time_taken)
