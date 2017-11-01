import numpy as np

def angular_dist(ra1, dec1, ra2, dec2):
  ra1 = np.radians(ra1)
  dec1 = np.radians(dec1)
  ra2 = np.radians(ra2)
  dec2 = np.radians(dec2)
  
  a = np.sin(np.abs(dec1-dec2)/2) ** 2
  b = np.cos(dec1) * np.cos(dec2) * np.sin(np.abs(ra1-ra2)/2) ** 2
  return np.degrees(2 * np.arcsin(np.sqrt(a + b))) 


def angular_dist_rad(ra1, dec1, ra2, dec2):
  a = np.sin(np.abs(dec1-dec2)/2) ** 2
  b = np.cos(dec1) * np.cos(dec2) * np.sin(np.abs(ra1-ra2)/2) ** 2
  return 2 * np.arcsin(np.sqrt(a + b)) 

ra1, dec1 = np.radians([180, 30])
cat2 = [[180, 32], [55, 10], [302, -44]]
cat2 = np.radians(cat2)
ra2s, dec2s = cat2[:,0], cat2[:,1]
dists = angular_dist_rad(ra1, dec1, ra2s, dec2s)
print(dists, np.degrees(dists))

ra1, dec1 = [180, 30]
cat2 = np.array([[180, 32], [55, 10], [302, -44]])
ra2s, dec2s = cat2[:,0], cat2[:,1]
dists = angular_dist(ra1, dec1, ra2s, dec2s)
print(np.radians(dists), dists)