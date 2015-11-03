
import pandas as pd
import pickle
from shapely.geometry import Point, mapping, shape
import fiona
from fiona import collection
import shapely.geometry
from pyproj import Proj, transform



df = pd.read_csv('/datafiles/train.csv', parse_dates = ['dates'])
fc = fiona.open('/sfpd_plots/sfpd_plots.shp')

################################
# Functions 
################################

# Need to convert units-- the longitude and latitude are in WGS8
# while the shapefile is in California State Plane Coordinate System
def convert_coords(dataframe):
    inProj = Proj(init='epsg:4326') # WGS84
    outProj = Proj(init='epsg:2227', preserve_units=True) # CA State Plane Coordinate System
    longi, lati = transform(inProj, outProj, dataframe['longitude'].values, dataframe['latitude'].values) 
    return longi, lati
    
    
# Converting incidents to points    
def make_points(dataframe):
    points = []
    for row in dataframe[['lati', 'longi']].values:
        points.append(Point(float(row[0]), float(row[1])))
    return points 


# Assign each point to SFPD plot
def point_in_poly(dataframe):
    results = {}
    for index, each in enumerate(dataframe['shape'].values):
        for feature in fc:
            if shape(feature['geometry']).contains(each):
                results[index] = feature['properties']['OBJECTID']
            break 
    
    if len(results) % 50 == 0:
        with open('../datafiles/my_data_partial.pkl', 'w') as picklefile:
            pickle.dump(results.items(), picklefile)
            print "dumped " + str(len(results.items))
    return results 


#################################
pickle.dump( results, open( "../datafiles/results.p", "wb" ) )
df['longi'], df['lati'] = convert(df)
df['shape'] = make_points(df)
results = point_in_poly(df)
merged = df.merge(results, how='left', left_index=True, right_index=True)


