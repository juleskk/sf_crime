from dateutil import parser
import datetime
import pandas as pd
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.cross_validation import (train_test_split, cross_val_score, 
                                      cross_val_predict)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import scipy as sp
import networkx as nx
import time
import pickle
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from shapely.geometry import Point, mapping, shape
import fiona
from fiona import collection
import shapely.geometry
from pyproj import Proj, transform




train = pd.read_csv('../datafiles/train.csv', parse_dates = ['dates'])
fc = fiona.open("../sfpd_plots/sfpd_plots.shp")

inProj = Proj(init='epsg:4326') # WGS84
outProj = Proj(init='epsg:2227', preserve_units=True) # CA State Plane Coordinate System 

train['x1'] , train['y1'] = transform(inProj, outProj, train['x'].values, train['y'].values) #convert units 

points = []
for row in train.values:
	points.append(Point(float(row[-2]), float(row[-1])))

train['shape'] = points

temp = []
for each in train['shape'].values:
     for feature in fc:
        if shape(feature['geometry']).contains(each):
            temp.append(feature['properties']['OBJECTID'])
train['clusters'] = temp

pickle.dump( train, open( "train_cl.p", "wb" ) )
