import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from pyproj import Proj, transform
from cartopy import crs as ccrs
from matplotlib import pyplot as plt

Y = 'MP10'

years = np.arange(2020, 2024)
months = np.arange(1, 13)

meta_sinca = pd.read_csv('data/SINCA/meta.csv', sep = ',')
lats = np.load('data/lats.npy')
lons = np.load('data/lons.npy')

def utm_to_latlon(x, y, zone_number):
    
    p = Proj(proj='utm', zone=zone_number, ellps='WGS84')
    lon, lat = transform(p, Proj(init='EPSG:4326'), x, y)
    return lat, lon

def get_sites(meta_sinca, Y):
    # list of column = y with 1 values
    sites = list(meta_sinca[meta_sinca[Y] == 1]['Estacion'])
    lat = [float(x.replace(',', '.')) for x in meta_sinca[meta_sinca[Y] == 1]['lat']]
    lon = [float(x.replace(',', '.')) for x in meta_sinca[meta_sinca[Y] == 1]['lon']]
    print(f'Found {len(sites)} sites')
    return sites, lat, lon


def get_index(lat, lon, lats, lons):
    # get index of the closest value
    lat_idx = np.abs(lats - lat).argmin()
    lon_idx = np.abs(lons - lon).argmin()
    return lat_idx, lon_idx

def get_value_fromsite_ds(site_pd, date):
    pd = site_pd[site_pd['FECHA (YYMMDD)'] == date]
    if not pd['Registros validados'].isna().all() and not pd['Registros validados'].empty:
        return float(str(pd['Registros validados'].values[0]).replace(',', '.'))
    elif not pd['Registros preliminares'].isna().all() and not pd['Registros preliminares'].empty:
        return float(str(pd['Registros preliminares'].values[0]).replace(',', '.'))
    elif not pd['Registros no validados'].isna().all() and not pd['Registros no validados'].empty:
        return float(str(pd['Registros no validados'].values[0]).replace(',', '.'))
    else:
        return np.nan
        

def get_value_from_layer(subproduct, month, year, index_lat, index_lon):
    if subproduct == 'AOD':
        layer = np.load(f'data/ABI-L2-AODF/ABI-L2-AODF__mean_{year}.npy')[month-1]
        climatology = np.load(f'data/climatologies/ABI-L2-AODF__climatology.npy')
    elif subproduct == 'dust':
        layer = np.load(f'data/ABI-L2-ADPF/ABI-L2-ADPF_dust_mean_{year}.npy')[month-1]
        climatology = np.load(f'data/climatologies/ABI-L2-ADPF_dust_climatology.npy')
    elif subproduct == 'smoke':
        layer = np.load(f'data/ABI-L2-ADPF/ABI-L2-ADPF_smoke_mean_{year}.npy')[month-1]
        climatology = np.load(f'data/climatologies/ABI-L2-ADPF_smoke_climatology.npy')   
    elif subproduct == 'aerosol':
        layer = np.load(f'data/ABI-L2-ADPF/ABI-L2-ADPF_aerosol_mean_{year}.npy')[month-1]
        climatology = np.load(f'data/climatologies/ABI-L2-ADPF_aerosol_climatology.npy')
    elif subproduct == 'band1':
        layer = np.load(f'data/ABI-L1b-RadF/ABI-L1b-RadF_1_mean_{year}.npy')[month-1]
        climatology = np.load(f'data/climatologies/ABI-L1b-RadF_1_climatology.npy')
    elif subproduct == 'band7':
        layer = np.load(f'data/ABI-L1b-RadF/ABI-L1b-RadF_7_mean_{year}.npy')[month - 1]
        climatology = np.load(f'data/climatologies/ABI-L1b-RadF_7_climatology.npy')
    elif subproduct == 'band12':
        layer = np.load(f'data/ABI-L1b-RadF/ABI-L1b-RadF_12_mean_{year}.npy')[month - 1]
        climatology = np.load(f'data/climatologies/ABI-L1b-RadF_12_climatology.npy')

    return layer[index_lat,index_lon] - climatology[index_lat,index_lon]

aerosol = []
dust = []
smoke = []
AOD = []
Ys = []
band1 = []
band7 = []
band12 = []
sites, latitudes, longitudes = get_sites(meta_sinca, Y)

for i in tqdm(range(0, len(sites))):
    site = sites[i]
    lat, lon = latitudes[i], longitudes[i]
    print(f'Processing {site} {lat}, {lon}')
    lat_idx, lon_idx = get_index(lat, lon, lats, lons)
    site_pd = pd.read_csv(f'data/SINCA/{Y}/{site}.csv', sep =';')

    for year in years:
        for month in months:
            date = int(f'{year-2000}{str(month).zfill(2)}01')
            if np.isnan(get_value_fromsite_ds(site_pd, date)):
                continue
            else:
                Ys.append(get_value_fromsite_ds(site_pd, date))
                dust.append(get_value_from_layer('dust', month, year, lat_idx, lon_idx))
                aerosol.append(get_value_from_layer('aerosol', month, year, lat_idx, lon_idx))
                smoke.append(get_value_from_layer('smoke', month, year, lat_idx, lon_idx))
                AOD.append(get_value_from_layer('AOD', month, year, lat_idx, lon_idx))
                band1.append(get_value_from_layer('band1', month, year, lat_idx, lon_idx))
                band7.append(get_value_from_layer('band7', month, year, lat_idx, lon_idx))
                band12.append(get_value_from_layer('band12', month, year, lat_idx, lon_idx))

print(np.shape(Ys), np.shape(aerosol), np.shape(dust), np.shape(smoke), np.shape(AOD), np.shape(band1), np.shape(band7), np.shape(band12))
df = pd.DataFrame({Y: Ys, 'aerosol': aerosol, 'dust': dust, 'smoke': smoke, 'AOD': AOD, 'band1': band1, 'band7': band7, 'band12': band12})
#save
df.to_csv(f'data/training_data/{Y}_train.csv', index=False)


