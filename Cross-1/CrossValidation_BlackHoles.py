# Write your crossmatch function here.
import numpy as np

def hms2dec(h, m, s):
  return 15 * (h + m/60 + s/(60 * 60))

def dms2dec(g, m, s):
  return np.sign(g) * (np.sign(g) * g + m/60 + s/(60 * 60))

def import_bss():
   data = np.loadtxt('bss.dat', usecols=range(1, 7))
   res = []
   i = 1
   for d in data:
      ra = hms2dec(d[0], d[1], d[2]) 
      dec = dms2dec(d[3], d[4], d[5])
      res.insert(i-1, (i, ra, dec))
      i += 1
   return res
    
  
def import_super(): 
  data = np.loadtxt('super.csv', delimiter=',', skiprows=1, usecols=[0, 1])
  i = 1
  res = []
  for d in data:
    res.insert( i-1, (i, d[0], d[1]) )
    i += 1
  return res    
    
def angular_dist(ra1, dec1, ra2, dec2):
  ra1 = np.radians(ra1)
  dec1 = np.radians(dec1)
  ra2 = np.radians(ra2)
  dec2 = np.radians(dec2)
  
  a = np.sin(np.abs(dec1-dec2)/2) ** 2
  b = np.cos(dec1) * np.cos(dec2) * np.sin(np.abs(ra1-ra2)/2) ** 2
  return np.degrees(2 * np.arcsin(np.sqrt(a + b)))  

def find_closest(cat, ra, dec):
  x = cat[0]
  min_dist = angular_dist(ra, dec, x[1], x[2])
  for galaxy in cat:
    cur_dist = angular_dist(ra, dec, galaxy[1], galaxy[2])
    if cur_dist < min_dist:
      x = galaxy
      min_dist = cur_dist
  return (x[0], min_dist)

def crossmatch(bss_cat, super_cat, max_dist):
  matches = []
  no_matches = []
  i = 1
  for first in bss_cat:
    second_id, cur_dist = find_closest(super_cat, first[1], first[2])
    if cur_dist < max_dist:
      matches.append((first[0], second_id, cur_dist))
    else:
      no_matches.append(first[0])
    i += 1
  return matches, no_matches


# You can use this to test your function.
# Any code inside this `if` statement will be ignored by the automarker.
if __name__ == '__main__':
  bss_cat = import_bss()
  super_cat = import_super()

  # First example in the question
  max_dist = 40/3600
  matches, no_matches = crossmatch(bss_cat, super_cat, max_dist)
  print(matches[:3])
  print(no_matches[:3])
  print(len(no_matches))

  # Second example in the question
  max_dist = 5/3600
  matches, no_matches = crossmatch(bss_cat, super_cat, max_dist)
  print(matches[:3])
  print(no_matches[:3])
  print(len(no_matches))
