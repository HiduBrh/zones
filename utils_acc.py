import pandas as pd
import requests
import json
import dateutil.parser
from pandas_gbq import read_gbq
import numpy as np
from matplotlib.pyplot import plot, show
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gmplot import gmplot
import json, codecs


def get_traffic_dataset():
    projectid = "vaarta-202211"
    referentiel = read_gbq('SELECT fields.id_arc_tra, fields.geo_point_2d FROM Trafic.sensor_data_spec GROUP BY fields.id_arc_tra,fields.geo_point_2d', projectid, private_key='VAARTA-dcdd2e85b841.json')
    referentiel.columns = ['id_arc_tra','geo_point_2d']
    referentiel['id_arc_tra'] = pd.to_numeric(referentiel['id_arc_tra'], errors='coerce')
    referentiel['geo_point_2d']= referentiel['geo_point_2d'].str.strip("[]")
    url='https://opendata.paris.fr/api/records/1.0/search/?dataset=comptages-routiers-permanents&q=horodate:[2017/01/01 TO 2017/12/31]&rows=-1'
    response = requests.get(url)
    res = json.loads(response.text)
    id_arcs=    []
    taux =  []
    horodate = []
    for record in res['records']:
        if 'taux' in record['fields']:
            id_arcs.append(record['fields']['id_arc_trafic'])
            taux.append((record['fields']['taux']))
            date = dateutil.parser.parse(record['fields']['horodate'])
            horodate.append(date)
    d={'id_arc_tra':id_arcs,'horodate':horodate,'taux':taux}
    comptages = pd.DataFrame(data=d)
    comptages = comptages.groupby(['id_arc_tra'])['taux'].mean()
    comptages = comptages.to_frame()
    comptages = comptages.reset_index(level=['id_arc_tra'])
    dataset = pd.merge(comptages, referentiel, on=['id_arc_tra'])
    dataset['lat'], dataset['lon'] = zip(*dataset['geo_point_2d'].map(lambda x: x.split(', ')))
    dataset = dataset.sort_values('taux')
    dataset = dataset[dataset.taux > 15]
    #filtered = dataset.as_matrix(columns=['lat', 'lon','taux'])
    #filtered = np.array(filtered,dtype='float').tolist()
    dataset = dataset.drop(['geo_point_2d'], axis=1)
    my_json = dataset.to_json(orient='records')
    return np.array(dataset,dtype='float')


def plot_2d_geo(mat :np.numarray):
    gmap = gmplot.GoogleMapPlotter(48.866667, 2.333333, 12, apikey='AIzaSyBVglNw778QV74T3wmtX1wkcZuv1ZT0ECo')
    gmap.scatter(mat[:,0], mat[:,1], '#3B0B39', size=40, marker=False)
    gmap.draw("my_map.html")

