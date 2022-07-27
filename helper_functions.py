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



def filter_trajectories(traj,tifs,configs,min_det=3,y0_min=250,y0_max=1000,y_last=1500,area_tol=0.5,y_max=None,area_max=12000):
    '''
    min_det: minimum detections to count as trajectory
    y0_min/ y0_max: first detection has to be in this range
    #area_tol: area of detected object should not change more than tolarance
    y_last hast to be higher than this value
    y_max: cut everthing off right before image ends:last x pxs
    '''
    for series,config in zip(tifs,configs):
        #filter df that only reasonable tracks are in it
        remove = []
        if y_max != None:
            traj = traj[traj['y'] < series.shape[1] - y_max]

        mask = traj['file'] == config['file']
        traj_ = traj[mask]
        for particle,g in traj_.groupby('particle'):

            #paticle should have at least min_det detections in trajectory, y-coord of first detection should be higher than y_0 and smaller than y0_max
            #last detection should be higher than 1500px to filter trajectories leaving image immediately
            if len(g) < min_det or g['y'].iloc[0] < y0_min or g['y'].iloc[0] > y0_max or g['y'].iloc[-1] < y_last:
                remove.append(particle)
            mode = float(stats.mode(g['area_poly'])[0])
            #area should not be bigger than 12000 px and area shouldnt change drasticly over time
            if (g['area_poly']>area_max).any() or (g['area_el'] < mode*(1-area_tol)).any() or (g['area_el'] > mode*(1+area_tol)).any():
                remove.append(particle)
            #bubble should not travel backwards
            if (g['traveled_dist'].sort_values(ascending=True).reset_index(drop=True) != g['traveled_dist'].reset_index(drop=True)).any():
                remove.append(particle)


        for r in remove:
            traj = traj.drop(traj[traj.particle == r].index)

        n_par = np.unique(traj['particle'])

    return traj



def plot_traj(traj, tifs,configs):
    for series,config in zip(tifs,configs):
        fig, axs = plt.subplots(figsize=(15, 10), nrows=1, ncols=2)
        n = 0
        print(series[n].shape)
        axs[0].imshow(series[n])
        axs[0].set_xlim(0, series.shape[2])
        axs[0].set_ylim(series.shape[1], 0)
        mask = traj['file'] == config['file']
        df_ = traj[mask]
        for par, d in df_.groupby('particle'):
            axs[0].scatter(d['x'], d['y'])

        axs[1].imshow(series[n])
        #m = df_['frame'] == n
        #ells = traj[m]
        for index, row in df_.iterrows():
            e = Ellipse(xy=(row['x'], row['y']), width=row['l'], height=row['s'], angle=row['angle'])
            axs[1].add_artist(e)
            e.set_facecolor("none")
            e.set_edgecolor("red")
        plt.show()


def link(df,max_px = 300,memory = 0): #TODO might cause issues because each series has frames [0-x] so it may try to combine trajectories between series
    '''
    set parameters for trajectories
        - max_px:       the maximum distance features can move between frames, optionally per dimension
        - memory:       the maximum number of frames during which a feature can vanish, then reappear nearby,
        and be considered the same particle. 0 by default.
        - pos_columns:  default: ['y','x'] or ['z','y','x'] if z is provided, using 'area_poly' as 'z' coordinate to
        consider the area of the polygons
    '''
    traj = tp.link(df, max_px, memory=memory,pos_columns=['area_poly','y','x'])
    return traj

def filter_objects(df,min_l=20):
    '''
    filter detected objects: has to have a higher y value than the ejection point and has to have a minimum long-axis min_l
    :param df:
    :param min_l:
    :return: df
    '''
    mask = np.logical_and(df['l'] > min_l, df['y'] > df['ej_y'])
    df = df[mask]
    return df


def add_properties(df):
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

    df['guck_def'] = guck_deform(area_ell(df['l'], df['s']), perimeter_ell(df['l'], df['s']))
    df['guck_poly'] = guck_deform(df['area_poly'], df['perimeter_poly'])
    df['strain'] = strain(df['l'], df['s'])
    df['area_el'] = area_ell(df['l'], df['s'])
    df['radius'] = np.sqrt(df['area_el'] / np.pi)

    # add properties in meter
    df['radius_m'] = df['scaling_factor'] * df['radius']
    df['l_m'] = df['scaling_factor'] * df['l']
    df['s_m'] = df['scaling_factor'] * df['s']
    df['area_el_m'] = area_ell(df['l_m'], df['s_m'])

    # add traveled distance (in meter)
    df['traveled_dist'] = euclidean_dist(df['x'], df['y'], df['ej_x'], df['ej_y'])
    df['traveled_dist_m'] = df['scaling_factor'] * df['traveled_dist']

    return df

def determine_needle_property(series,min_px=80,needle_with_um = 800,check=False):
    img = series[0]
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - th2
    cnts, hiers = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    for cnt in cnts:
        if len(cnt) > min_px:
            contour = np.squeeze(cnt)
            polygon = Polygon(contour)
            x, y = polygon.exterior.xy
            x,y = np.array(x), np.array(y)
            #criteria for finding needle: countour has to contain at least one value smaller 250 and at least one == 0
            if (y < 250).any() and (y == 0).any() or (y == 10).any():
                mask = y == 0
                x_ = x[mask]
                xl = np.min(x_)
                xr = np.max(x_)
                needle_with_px = (xr - xl)
                ej_x = xr - needle_with_px / 2
                ej_y = np.max(y)

                # get calibration: needle = 800 mikron
                # width of needle in px
                scaling_factor = needle_with_um*1e-6 / needle_with_px  # um/px
    if check == True:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(inv)
        ax.plot(ej_x, ej_y, marker="x", markersize=10,color='red')
        ax.annotate('', xy=(xl, 20), xycoords='data',xytext=(xr, 20), textcoords='data',
                     arrowprops={'arrowstyle': '<->', 'color': 'red'})
        plt.show()
    return needle_with_px, ej_x, ej_y, scaling_factor


# ugly helper to find all the ellipses and insert into one dataframe TODO rewrite it
def find_objects(list_of_series,list_of_configs,min_px=80,check=False):
    '''
    get list of np.arrays with data and config and return one big pandas dataframe with all the information
    :return:
    '''
    dic = {}
    files,n_cells,pressures,frs, xs, ys, ls, ss, angles, area_poly, perimeter_poly,widths,ej_xs,ej_ys,scaling_factors = [], [], [], [], [], [], [], [], [], [], [], [],[],[],[]
    for series,config in zip(list_of_series,list_of_configs):
        print(config['file'])
        width,ej_x,ej_y,scaling_factor = determine_needle_property(series,check=check)
        for i in range(len(series)):
            img = series[i]
            # thresholding, using automatic technique of Otsus thresholding
            ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # inverting the image
            inv = 255 - th2
            # find contours
            cnts, hiers = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            for cnt in cnts:
                if len(cnt) > min_px:
                    ellipse = cv2.fitEllipse(cnt)
                    x, y, l, s, p = ellipse[0][0], ellipse[0][1], ellipse[1][0], ellipse[1][1], ellipse[2]
                    files.append(config['file'])
                    n_cells.append(config['n_cells'])
                    pressures.append(config['pressure'])
                    widths.append(width)
                    ej_xs.append(ej_x)
                    ej_ys.append(ej_y)
                    scaling_factors.append(scaling_factor)
                    frs.append(i)
                    xs.append(x)
                    ys.append(y)
                    ls.append(l)
                    ss.append(s)
                    angles.append(p)
                    contour = np.squeeze(cnt)
                    polygon = Polygon(contour)
                    area_poly.append(polygon.area)
                    perimeter_poly.append(polygon.length)
        # save data into dictionary
        dic['file'] = files
        dic['n_cells'] = n_cells
        dic['pressure'] = pressures
        dic['width'] = widths
        dic['ej_x'] = ej_xs
        dic['ej_y'] = ej_ys
        dic['scaling_factor'] = scaling_factor
        dic['frame'] = frs
        dic['x'] = xs
        dic['y'] = ys
        dic['l'] = ls
        dic['s'] = ss
        dic['angle'] = angles
        dic['area_poly'] = area_poly
        dic['perimeter_poly'] = perimeter_poly

    df = pd.DataFrame(dic)
    return df




def get_config(list_of_tifs):
    '''
    right now only extract pressure from each config and return in list
    :param list_of_tifs:
    :return: list of pressures
    '''
    out = []
    for tif in list_of_tifs:
        path = tif[:-4] + '_config.txt'
        config = configparser.ConfigParser()
        config.read_file(open(path),'r')
        dic = {}
        dic['file'] = tif
        dic['pressure'] = config['SETUP']['pressure']
        cells = config['CELL']['treatment']
        cells = int(re.search(r'\d+', cells).group())
        dic['n_cells'] = cells

        out.append(dic)
    return out


def find_tif(path_to_dir):
    out = []
    for root, dirs, files in os.walk(path_to_dir):
        for file in files:
            if file.endswith(".tif"):
                out.append(os.path.join(root, file))
    return out

def load(list):
    '''
    read all tiff files of list into numpy arrays
    :param list:
    :return: list of np.arrays each containing one "movie" with size (number-of-frames,img-height,img-width)
    '''
    out = []
    for path in list:
        with tiff.TiffFile(path) as tif:
            series = [page.asarray() for page in tqdm(tif.pages[0:])]
            series = np.array(series)
            series = series.astype(np.uint8)
            series = cv2.normalize(series, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        out.append(series)
    return out

