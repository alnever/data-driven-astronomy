import psycopg2
import numpy as np

def select_all(table_name):
  conn = psycopg2.connect(dbname='db', user='grok')
  cursor = conn.cursor()
  cursor.execute("select * from "+table_name)
  return cursor.fetchall()

def column_stats(table_name, field_name):
  conn = psycopg2.connect(dbname='db', user='grok')
  cursor = conn.cursor()
  cursor.execute("select "+field_name+" from "+table_name)
  records = np.array(cursor.fetchall())
  return np.mean(records), np.median(records)



def query1(filename):
  data = np.loadtxt(filename, delimiter=',', skiprows=0, usecols=[0, 2])
  data = data[data[:,1] > 1,:]
  idxs = np.argsort(data[:,1])
  data = data[idxs,:]
  return data

def query2(stars_csv, planets_csv):
  stars = np.loadtxt(stars_csv, delimiter=',', skiprows=0, usecols=[0, 2])
  planets = np.loadtxt(planets_csv, delimiter=',', skiprows=0, usecols=[0, 5])
  stars = stars[stars[:,1] > 1,:]
  idxs = np.argsort(stars[:,1])
  stars = stars[idxs,:]
  data = np.array([])
  for star in stars:
    planet_idx = np.where(np.ceil(planets[:,0]) == np.floor(star[0]))
    x_planets = planets[planet_idx]
    for planet in x_planets:
      data = np.append(data, (planet[1] / star[1]))
  data.sort();
  data = np.reshape(data, (len(data), 1))
  return data