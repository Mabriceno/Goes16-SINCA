import os
from botocore import UNSIGNED
from botocore.client import Config
import boto3 
import numpy as np
from netCDF4 import Dataset
from datetime import datetime
from tqdm import tqdm
import re


def filtrar_archivos_por_banda(archivos, banda):
    # Asegurarse de que la banda sea un string con dos dígitos (e.g., "01" en lugar de "1")
    banda_str = f"{int(banda):02d}"
    
    # Crear una lista para almacenar los archivos filtrados
    archivos_filtrados = []
    
    # Crear un patrón de regex para buscar la banda en el nombre del archivo
    patron = re.compile(f'M6C{banda_str}')
    
    for archivo in archivos:
        if patron.search(archivo):
            archivos_filtrados.append(archivo)
    
    return archivos_filtrados


def how_many_days(year, month):
    if month == 12:
        return (datetime(year+1, 1, 1) - datetime(year, month, 1)).days
    else:
        return (datetime(year, month+1, 1) - datetime(year, month, 1)).days
  
download_dir = 'downloads'
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# Bucket name
bucket_name = 'noaa-goes16'

product = 'ABI-L1b-RadF'
band = '7'

if not os.path.exists(f'downloads/{product}'):
    os.makedirs(f'downloads/{product}')

years = np.arange(2020, 2024)
months = np.arange(1, 13)
hours = np.array([12,]) + 4 # local time to UTC


for year in years:
    n_day = 1
    for month in months:
        month_layer = np.zeros((5424, 5424))
        month_days = how_many_days(year, month)
        days = 0
        print(f'Downloading {product} for {year}/{month}')
        for day in tqdm(range(1, month_days+1, 8)):
            for hour in hours:
                if os.path.exists(f'downloads/{product}/{product}_mean_{year}_{month}.npy'):
                    print(f'File downloads/{product}/{product}_mean_{year}_{month}.npy already exists')
                else:
                    prefix= f'{product}/{year}/{str(n_day).zfill(3)}/{str(hour).zfill(2)}/'
                    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

                    try:
                        objs = response.get('Contents', []) #just first step of the hour    
                        file_keys = [obj['Key'] for obj in objs]
                        file_key = filtrar_archivos_por_banda(file_keys, band)[0]
                        
                        file_name = os.path.join(download_dir, file_key.split('/')[-1])
                        s3.download_file(bucket_name, file_key, f'downloads/{product}/temp.nc')
                        #get contents of the file
                        nc = Dataset(f'downloads/{product}/temp.nc', 'r')
                        layer = nc.variables['Rad'][:]
                        
                        #reduce resolution x2
                        if layer.shape[0] == 5424*2:
                            layer = layer[::2, ::2]
                        month_layer = np.nansum([month_layer, layer], axis=0)
                        nc.close()
                        days += 1
                    except:
                        print(f'No data for {year}/{month}/{n_day}/{hour}')

            n_day += 1
        month_mean_layer = month_layer / (days)
        #to float 16
        month_mean_layer = np.float16(month_mean_layer)
        month_mean_layer = month_mean_layer[len(month_mean_layer)//2:, :]
        np.save(f'downloads/{product}/{product}_{band}_mean_{year}_{month}.npy', month_mean_layer)
