import tifffile as tiff
from tqdm import tqdm
import numpy as np
import cv2
import os
import configparser
import re
from shapely.geometry import Polygon
import pandas as pd
import matplotlib.pyplot as plt
import trackpy as tp
from matplotlib.patches import Ellipse
from scipy import stats
import clickpoints

def removePar(traj,ids):
    for id in ids:
        traj = traj.drop(traj[traj.particle == id].index)
    return traj

def plot(traj):
    fig, axs = plt.subplots(ncols=3, figsize=(16, 5))
    # check if area of ellipsoids stays constant
    for par, d in traj.groupby('particle'):
        # plt.plot(d['traveled_dist'],d['poly_guck_strain'])
        axs[0].plot(d['traveled_dist'], d['strain'])
        axs[0].set_xlabel('euclidean distance from ejection point [px]')
        axs[0].set_ylabel('strain')

        axs[1].plot(d['traveled_dist'], d['el_area'],label=f'{par}')
        axs[1].set_xlabel('euclidean distance from ejection point [px]')
        axs[1].set_ylabel('area [px]')
        axs[1].legend()
        # shear stress = bubble strain * surface tension (air-liquid) / radius of the undisturbed bubble
        r0 = d['radius'].iloc[-1]
        stress = d['strain'] / r0
        axs[2].plot(d['traveled_dist'], stress)
        axs[2].set_xlabel('euclidean distance from ejection point [px]')
        axs[2].set_ylabel('stress')
    plt.show()

def write2cdb(cdb,traj, config, ej=False):
    deleteAll(cdb)
    # plots tracks
    mtype = cdb.setMarkerType(name="track", mode=cdb.TYPE_Track, color="#ff8800")
    cdb.deleteMarkers(type=mtype)
    trackIds = traj["particle"].unique()
    cdbTracks = dict([[i, cdb.setTrack(type=mtype)] for i in trackIds])
    for trackId in traj["particle"].unique():
        mask = traj["particle"] == trackId
        x, y, f = traj[mask][["x", "y", "frame"]].values.T
        cdb.setMarkers(x=x, y=y, frame=list(f.astype(int)), type=mtype, track=cdbTracks[trackId])

    # plot ellipses
    etype = cdb.setMarkerType(name="trackEll", mode=cdb.TYPE_Ellipse, color="#ff0000")
    cdb.deleteEllipses(type=etype)
    for frameId in traj["frame"].unique():
        mask = traj["frame"] == frameId
        x, y, l, s, phi = traj[mask][["x", "y", "l", "s", "phi"]].values.T
        cdb.setEllipses(x=x, y=y, width=l, height=s, angle=phi, image=cdb.getImage(frame=frameId), type=etype)

    if ej:
        # plot ejection point
        ejtype = cdb.setMarkerType(name="ejPoint", mode=cdb.TYPE_Normal, color="#ff0000")
        cdb.deleteMarkers(type=ejtype)
        for frameId in range(cdb.getImageCount()):
            cdb.setMarker(x=config['ej_x'], y=config['ej_y'], image=cdb.getImage(frame=frameId), type=ejtype)
    cdb.db.close()

def deleteAll(cdb):
    cdb.deleteMarkers()
    cdb.deleteTracks()
    cdb.deleteEllipses()

def assignOneParticleID(traj):
    traj['particle'] = 1
    return traj

def filterTrajectories(traj, min_det=100, y_last=400, y_first=np.inf, area_tol=0.2):
    '''
    min_det: minimum detections to count as trajectory
    y0_min/ y0_max: first detection has to be in this range
    #area_tol: area of detected object should not change more than tolarance
    y_last hast to be higher than this value
    '''
    # filter df that only reasonable tracks are in it
    remove = []

    for particle, g in traj.groupby('particle'):

        # , y-coord of first detection should be higher than y_0 and smaller than y0_max
        # last detection should be higher than 1500px to filter trajectories leaving image immediately
        # or g['y'].iloc[0] < g['ej_y'][0] or g['y'].iloc[0] > y0_max

        # paticle should have at least min_det detections in trajectory
        if len(g) < min_det:
            #print('min_det')
            remove.append(particle)
        #first detection should be lower than x px to filter trajectories starting in the middle
        if g['y'].iloc[0] > y_first:
            remove.append(particle)
        # last detection should be higher than 400px to filter trajectories leaving image immediately
        if g['y'].iloc[-1] < y_last:
            #print('ylast')
            remove.append(particle)

        # area shouldnt change drasticly over time
        mode = float(stats.mode(g['poly_area'])[0])
        area_tolerance = area_tol * mode
        if (g['poly_area'] < mode - area_tolerance).any():  # or :
            #print('area2small')
            remove.append(particle)
        if (g['poly_area'] > mode + area_tolerance).any():
            #print('area2big')
            remove.append(particle)

        # bubble should not travel backwards
        #if (g['traveled_dist'].sort_values(ascending=True).reset_index(drop=True) != g['traveled_dist'].reset_index(
        #        drop=True)).any():
        #    print('backwards')
        #    remove.append(particle)

    for r in remove:
        traj = traj.drop(traj[traj.particle == r].index)

    print('trajectories left: ', np.unique(traj['particle']))

    return traj



def plot_traj(traj, series):
    fig, axs = plt.subplots(figsize=(15, 10), nrows=1, ncols=2)
    n = 0
    print(series[n].shape)
    axs[0].imshow(series[n])
    axs[0].set_xlim(0, series.shape[2])
    axs[0].set_ylim(series.shape[1], 0)
    for par, d in traj.groupby('particle'):
        axs[0].scatter(d['x'], d['y'])

    axs[1].imshow(series[n])
    #m = df_['frame'] == n
    #ells = traj[m]
    for index, row in traj.iterrows():
        e = Ellipse(xy=(row['x'], row['y']), width=row['l'], height=row['s'], angle=row['phi'])
        axs[1].add_artist(e)
        e.set_facecolor("none")
        e.set_edgecolor("red")
    plt.show()

def deleteAll(cdb):
    cdb.deleteMarkers()
    cdb.deleteTracks()
    cdb.deleteEllipses()

def link(df,max_px = 300,memory = 0):
    '''
    set parameters for trajectories
        - max_px:       the maximum distance features can move between frames, optionally per dimension
        - memory:       the maximum number of frames during which a feature can vanish, then reappear nearby,
        and be considered the same particle. 0 by default.
        - pos_columns:  default: ['y','x'] or ['z','y','x'] if z is provided, using 'area_poly' as 'z' coordinate to
        consider the area of the polygons
    '''
    traj = tp.link(df, max_px, memory=memory)
                   #,pos_columns=['area_poly','x','y'])
    return traj

def filterObjects(df,config,l=(5,50),min_area=0):
    #TODO spheroid max size of needle opening
    # y has to have a higher y value than the ejection point


    mask = df['y'] > config['ej_y']
    # set range of long-axis l
    mask &= df['l'] > l[0]
    mask &= df['l'] < l[1]
    # neglect all detections near boundaries, 20px
    shape = config['shape']
    mask &= df['x'] < shape[2]-20
    mask &= df['x'] > 20
    mask &= df['y'] < shape[1]-20
    #spheroid max size of needle opening
    mask &= df['l'] < config['needle_width_px']
    mask &= df['poly_area'] > min_area
    #apply mask to DataFrame
    df = df[mask]
    return df

def filterTime(df,t):
    t0,t1 = t
    if t1 == None:
        t1 = np.inf
    mask = np.logical_and(df['frame'] > t0,df['frame'] < t1)
    df = df[mask]
    return df


def addProperties(df,config,meter=False):
    # define functions to calculate area, strain, deformation of ellipses

    def area_ell(l, s):
        return np.pi * l * s * 0.25

    def perimeter_ell(l, s):
        a = l / 2
        b = s / 2
        return np.pi * (3 * ((a + b) / 2) - np.sqrt(a * b))

    def guck_deform(area, perimeter):
        c = 2 * np.sqrt(np.pi * area) / perimeter
        d = 1 - c
        return d

    def strain(l, s):
        a = l / 2
        b = s / 2
        return (np.abs(a - b)) / np.sqrt(a * b)

    # add traveled distance
    def euclidean_dist(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        # add properties to pandas data frame

    df['el_guck_strain'] = guck_deform(area_ell(df['l'], df['s']), perimeter_ell(df['l'], df['s']))
    df['poly_guck_strain'] = guck_deform(df['poly_area'], df['poly_perimeter'])
    df['strain'] = strain(df['l'], df['s'])
    df['el_area'] = area_ell(df['l'], df['s'])
    df['radius'] = np.sqrt(df['el_area'] / np.pi)

    df['traveled_dist'] = euclidean_dist(df['x'], df['y'], config['ej_x'], config['ej_y'])
    if meter:
        # add properties in meter
        df['radius_m'] = df['scaling_factor'] * df['radius']
        df['l_m'] = df['scaling_factor'] * df['l']
        df['s_m'] = df['scaling_factor'] * df['s']
        df['el_area_m'] = area_ell(df['l_m'], df['s_m'])

        # add traveled distance (in meter)
        df['traveled_dist_m'] = df['scaling_factor'] * df['traveled_dist']

    return df


def findObjects(series):
    dic = {
        'frame': [],
        'x': [],
        'y': [],
        'l': [],
        's': [],
        'phi': [],
        'poly_area': [],
        'poly_perimeter': [],
        #'poly': [],
    }

    for frame in range(len(series)):
        img = series[frame]
        # thresholding, using automatic technique of Otsus thresholding
        ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # inverting the image
        inv = 255 - th2
        # find contours
        cnts, hiers = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        for cnt in cnts:
            if len(cnt) > 5:
                # fit ellipse to detected contour and save to DataFrame
                ellipse = cv2.fitEllipse(cnt)
                x, y, l, s, phi = ellipse[0][0], ellipse[0][1], ellipse[1][0], ellipse[1][1], ellipse[2]
                dic['frame'].append(frame)
                dic['x'].append(x)
                dic['y'].append(y)
                dic['l'].append(l)
                dic['s'].append(s)
                dic['phi'].append(phi)

                # get polygon and save properties to DataFrame
                contour = np.squeeze(cnt)
                polygon = Polygon(contour)
                dic['poly_area'].append(polygon.area)
                dic['poly_perimeter'].append(polygon.length)
                #dic['poly'].append(polygon)

    df = pd.DataFrame(dic)
    return df


def findNeedle(series,needle_width_um = 800,needle_max_extension=150,needle_max_area=5000):
    #take first image of series to determine where the ejection point of needle is
    #find contours of image
    img = series[0]
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - th2
    cnts, hiers = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    for cnt in cnts:
        if len(cnt) > 3:
            contour = np.squeeze(cnt)
            polygon = Polygon(contour)
            x, y = polygon.exterior.xy
            x,y = np.array(x), np.array(y)
            #criteria for finding needle: countour has to contain at least one value smaller
            if (y < needle_max_extension).any() and polygon.area > needle_max_area:
                #left and right point of needle tip
                xl = np.min(x)
                xr = np.max(x)
                # width of needle in px
                needle_width_px = (xr - xl)
                ej_x = xr - needle_width_px / 2
                ej_y = np.max(y)

                #scaling factor to convert to meter
                scaling_factor = needle_width_um*1e-6 / needle_width_px  # um/px
    return needle_width_px, ej_x, ej_y, scaling_factor



def getConfig(path,series):
    '''
    get pressure, number of cells in spheroid from config file
    :param
    :return: list of pressures
    '''
    configPath = path.replace('.tif','_config.txt')
    config = configparser.ConfigParser()
    config.read_file(open(configPath),'r')

    series = series.shape
    n_cells = config['CELL']['treatment']
    n_cells = int(re.search(r'\d+', n_cells).group())
    dic = {
        'file': path,
        'pressure': config['SETUP']['pressure'],
        'n_cells': n_cells,
        'shape': series,
    }
    return dic

def getImages(path):
    cdbPath = path.replace(".tif",".cdb")
    cdb = clickpoints.DataFile(cdbPath)
    series = [image.data for image in tqdm(cdb.getImages())]
    series = np.array(series)
    series = series.astype(np.uint8)
    series = series.squeeze()
    series = cv2.normalize(series, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cdb.db.close()

    return series

def load(path):
    '''
    create clickpoints database .cdb in same folder
    :return: cdb DataFile
    '''
    cdbPath = path.replace(".tif",".cdb")
    if os.path.exists(cdbPath):
        print("File exists! Skipping.")
    cdb = clickpoints.DataFile(cdbPath, "w")
    root, filename = os.path.split(path)
    cdbPath = cdb.setPath(path_string=root)
    with tiff.TiffFile(path) as tif:
        for j,page in enumerate(tif.pages):
            cdb.setImage(path=cdbPath, filename=filename, frame=j)
    cdb.db.close()
    return cdb