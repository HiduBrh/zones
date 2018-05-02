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


def get_dataset():
    mush = pd.read_csv("accidents-corporels-de-la-circulation-routiere.csv", header=0, delimiter=';')
    mush['lat'], mush['lon'] = zip(*mush['geo_point_2d'].map(lambda x: x.split(', ')))
    coord = mush.as_matrix(columns=['lat', 'lon'])
    return np.array(coord,dtype='float')

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
    filtered = dataset.as_matrix(columns=['lat', 'lon','taux'])
    return np.array(filtered,dtype='float')

def get_traffic_dataset_bkp():
    referentiel = pd.read_csv("referentiel-comptages-routiers.csv", header=0, delimiter=';')
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
    filtered = dataset.as_matrix(columns=['lat', 'lon','taux'])
    return np.array(filtered,dtype='float')


def sorted_eigen_values_and_vectors(array : np.numarray):
    """
    array rows:samples
    array columns:features
    :param array:
    :return: eigenValues, eigenVectors
    """
    var_matrix= np.var(np.transpose(array),axis=1)
    cov_matrix = np.cov(np.transpose(array))
    values, vectors = np.linalg.eig(cov_matrix)
    values = np.real(values)
    vectors = np.real(vectors)
    sorted_indexes = np.argsort(np.abs(values))[::-1]
    values = values[sorted_indexes]
    vectors = vectors[sorted_indexes]
    return values, vectors

def plot_2d(mat :np.numarray, labels :np.numarray):

    mat = mat.transpose()
    plt.scatter(mat[:, 0], mat[:, 1], s=3, c=labels)
    show()

def plot_3d(mat :np.numarray, labels :np.numarray):
    y = set(labels)
    mat = mat.transpose()
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in y:
        idxs = np.where(labels == i)[0]
        matrice=mat[idxs]
        ax.scatter3D(matrice[:,0], matrice[:,1], matrice[:,2])
    show()

def plot_2d_noLab(mat :np.numarray):
    mat = mat.transpose()
    plt.scatter(mat[:,0],mat[:,1],s=1,color='green')
    show()

def plot_2d_geo(mat :np.numarray):
    gmap = gmplot.GoogleMapPlotter(48.866667, 2.333333, 12, apikey='AIzaSyBVglNw778QV74T3wmtX1wkcZuv1ZT0ECo')
    gmap.scatter(mat[:,0], mat[:,1], '#3B0B39', size=40, marker=False)
    gmap.draw("my_map.html")

def plot_2d_geo_with_labels(mat :np.numarray):
    gmap = gmplot.GoogleMapPlotter(48.866667, 2.333333, 12, apikey='AIzaSyBVglNw778QV74T3wmtX1wkcZuv1ZT0ECo')
    gmap.scatter(mat[:,0], mat[:,1], '#3B0B39', size=40, marker=False)
    gmap.draw("my_map.html")

def plot_3d_noLab(mat :np.numarray):
    mat = mat.transpose()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter3D(mat[:,0], mat[:,1], mat[:,2])
    show()