import os
from botocore import UNSIGNED
from botocore.client import Config
import boto3 
import numpy as np
from netCDF4 import Dataset
from datetime import datetime
from tqdm import tqdm


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

product = 'ABI-L2-ADPF'
if not os.path.exists(f'downloads/{product}'):
    os.makedirs(f'downloads/{product}')

years = np.arange(2020, 2024)
months = np.arange(1, 13)
hours = np.array([12,]) + 4 # local time to UTC


for year in years:
    n_day = 1
    for month in months:
        month_layer_aerosol = np.zeros((5424, 5424))
        month_layer_dust = np.zeros((5424, 5424))
        month_layer_smoke = np.zeros((5424, 5424))
        month_days = how_many_days(year, month)
        days = month_days
        print(f'Downloading {product} for {year}/{month}')
        for day in tqdm(range(1, month_days+1)):
            for hour in hours:
                #pass if file exists
                if os.path.exists(f'downloads/{product}/{product}_aerosol_mean_{year}_{month}.npy'):
                    print(f'File downloads/{product}/{product}_aerosol_mean_{year}_{month}.npy already exists')
                else:   
                    prefix= f'{product}/{year}/{str(n_day).zfill(3)}/{str(hour).zfill(2)}/'
                    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
                    try:
                        obj = response.get('Contents', [])[0] #just first step of the hour
                        if n_day == 1 and hour == hours[0]: #save the first file to check the metadata
                            file_key = obj['Key']
                            file_name = os.path.join(f'{download_dir}/{product}', file_key.split('/')[-1])
                            print(f'Downloading {file_key} to {file_name} for metadata')
                            s3.download_file(bucket_name, file_key, file_name)
                            print(f'{file_name} downloaded successfully)')
                            
                        file_key = obj['Key']
                        file_name = os.path.join(download_dir, file_key.split('/')[-1])
                        s3.download_file(bucket_name, file_key, f'{download_dir}/{product}/temp.nc')
                        #get contents of the file
                        nc = Dataset(f'{download_dir}/{product}/temp.nc', 'r')
                        layer_aerosol = nc.variables['Aerosol'][:].data.astype(float)
                        layer_dust = nc.variables['Dust'][:].data.astype(float)
                        layer_smoke = nc.variables['Smoke'][:].data.astype(float)
                        layer_aerosol[layer_aerosol ==255] = np.nan
                        layer_dust[layer_dust ==255] = np.nan
                        layer_smoke[layer_smoke ==255] = np.nan
                        month_layer_aerosol = np.nansum([month_layer_aerosol, layer_aerosol], axis=0)
                        month_layer_dust = np.nansum([month_layer_dust, layer_dust], axis=0)
                        month_layer_smoke = np.nansum([month_layer_smoke, layer_smoke], axis=0)
                        nc.close()
                    except:
                        print(f'No data for {year}/{month}/{n_day}/{hour}')
                        days -= 1

            n_day += 1
        month_mean_layer_aerosol = month_layer_aerosol / (days)
        month_mean_layer_dust = month_layer_dust / (days)
        month_mean_layer_smoke = month_layer_smoke / (days)
        #to float 16
        month_mean_layer_aerosol = np.float16(month_mean_layer_aerosol)
        month_mean_layer_dust = np.float16(month_mean_layer_dust)
        month_mean_layer_smoke = np.float16(month_mean_layer_smoke)

        np.save(f'downloads/{product}/{product}_aerosol_mean_{year}_{month}.npy', month_mean_layer_aerosol)
        np.save(f'downloads/{product}/{product}_dust_mean_{year}_{month}.npy', month_mean_layer_dust)
        np.save(f'downloads/{product}/{product}_smoke_mean_{year}_{month}.npy', month_mean_layer_smoke)


