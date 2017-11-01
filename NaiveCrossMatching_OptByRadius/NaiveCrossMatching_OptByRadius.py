# Write your crossmatch function here.
import numpy as np
import time

def angular_dist(ra1, dec1, ra2, dec2):
  ra1 = np.radians(ra1)
  dec1 = np.radians(dec1)
  ra2 = np.radians(ra2)
  dec2 = np.radians(dec2)
  
  a = np.sin(np.abs(dec1-dec2)/2) ** 2
  b = np.cos(dec1) * np.cos(dec2) * np.sin(np.abs(ra1-ra2)/2) ** 2
  return np.degrees(2 * np.arcsin(np.sqrt(a + b)))  

def find_closest(cat, ra, dec, max_radius):
  x = cat[0]
  min_dist = angular_dist(ra, dec, x[0], x[1])
  cat = cat[(cat[:,1] > dec - max_radius) & (cat[:,1] < dec + max_radius), :]
  min_idx = -1
  for galaxy in cat:
    cur_dist = angular_dist(ra, dec, galaxy[0], galaxy[1])
    if cur_dist <= min_dist:
      x = galaxy
      min_dist = cur_dist
      min_idx = galaxy[2]
  return (min_idx, min_dist)

def crossmatch(bss_cat, super_cat, max_dist):
  start = time.perf_counter()
  indexes = np.array(range(0,len(super_cat)))
  super_cat = np.stack((super_cat[:,0], super_cat[:,1],indexes), axis = -1)
  sort_id = np.argsort(super_cat[:, 1])
  super_cat = super_cat[sort_id,:]
  matches = []
  no_matches = []
  i = 0
  for first in bss_cat:
    second_id, cur_dist = find_closest(super_cat, first[0], first[1], max_dist)
    if cur_dist < max_dist:
      matches.append((i, second_id, cur_dist))
    else:
      no_matches.append(i)
    i += 1
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
