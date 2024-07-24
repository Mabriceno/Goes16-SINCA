import numpy as np
from scipy.interpolate import griddata
import os 
from netCDF4 import Dataset
from tqdm import tqdm


product = 'ABI-L1b-RadF'
if not os.path.exists(f'data/{product}'):
    os.makedirs(f'data/{product}')

subproduct = '12'
years = np.arange(2020, 2024)
months = np.arange(1, 13)

def calculate_degrees(file_id):
    
    # Read in GOES ABI fixed grid projection variables and constants
    x_coordinate_1d = file_id.variables['x'][:]  # E/W scanning angle in radians
    y_coordinate_1d = file_id.variables['y'][:]  # N/S elevation angle in radians
    projection_info = file_id.variables['goes_imager_projection']
    lon_origin = projection_info.longitude_of_projection_origin
    H = projection_info.perspective_point_height+projection_info.semi_major_axis
    r_eq = projection_info.semi_major_axis
    r_pol = projection_info.semi_minor_axis
    
    # Create 2D coordinate matrices from 1D coordinate vectors
    x_coordinate_2d, y_coordinate_2d = np.meshgrid(x_coordinate_1d, y_coordinate_1d)
    
    # Equations to calculate latitude and longitude
    lambda_0 = (lon_origin*np.pi)/180.0  
    a_var = np.power(np.sin(x_coordinate_2d),2.0) + (np.power(np.cos(x_coordinate_2d),2.0)*(np.power(np.cos(y_coordinate_2d),2.0)+(((r_eq*r_eq)/(r_pol*r_pol))*np.power(np.sin(y_coordinate_2d),2.0))))
    b_var = -2.0*H*np.cos(x_coordinate_2d)*np.cos(y_coordinate_2d)
    c_var = (H**2.0)-(r_eq**2.0)
    r_s = (-1.0*b_var - np.sqrt((b_var**2)-(4.0*a_var*c_var)))/(2.0*a_var)
    s_x = r_s*np.cos(x_coordinate_2d)*np.cos(y_coordinate_2d)
    s_y = - r_s*np.sin(x_coordinate_2d)
    s_z = r_s*np.cos(x_coordinate_2d)*np.sin(y_coordinate_2d)
    
    # Ignore numpy errors for sqrt of negative number; occurs for GOES-16 ABI CONUS sector data
    np.seterr(all='ignore')
    
    abi_lat = (180.0/np.pi)*(np.arctan(((r_eq*r_eq)/(r_pol*r_pol))*((s_z/np.sqrt(((H-s_x)*(H-s_x))+(s_y*s_y))))))
    abi_lon = (lambda_0 - np.arctan(s_y/(H-s_x)))*(180.0/np.pi)
    
    return abi_lat, abi_lon

def reproject(layer, latitud, longitud):
    ajuste = len(latitud)//2

    latitud = latitud[3500: 4800, 2600:3600]
    longitud = longitud[3500: 4800, 2600:3600]
    layer = layer[3500-ajuste: 4800-ajuste, 2600:3600]
    lat_diffs = np.diff(latitud, axis=0)
    lon_diffs = np.diff(longitud, axis=1)
    mean_lat_resolution = np.mean(np.abs(lat_diffs))
    mean_lon_resolution = np.mean(np.abs(lon_diffs))
    desired_resolution = min(mean_lat_resolution, mean_lon_resolution)
    lon_min, lon_max = np.min(longitud), np.max(longitud)
    lat_min, lat_max = np.min(latitud), np.max(latitud)
    aspect_ratio = (lon_max - lon_min) / (lat_max - lat_min)
    num_lat_steps = int((lat_max - lat_min) / desired_resolution)
    num_lon_steps = int(num_lat_steps * aspect_ratio)

    lon_grid = np.linspace(lon_min, lon_max, num_lon_steps)
    lat_grid = np.linspace(lat_min, lat_max, num_lat_steps)

    if not os.path.exists(f'data/lats.npy'):
        np.save('data/lats.npy', lat_grid)
    if not os.path.exists(f'data/lons.npy'):
        np.save('data/lons.npy', lon_grid)

    lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
    
    # Interpolación de los datos
    layer_rep = griddata((longitud.flatten(), latitud.flatten()), 
                                        layer.flatten(), 
                                        (lon_grid, lat_grid), 
                                        method='linear')

    return layer_rep

nc = Dataset('downloads/ABI-L2-AODF/OR_ABI-L2-AODF-M6_G16_s20200011600218_e20200011609525_c20200011614292.nc')
latitud, longitud = calculate_degrees(nc)



for year in years:
    year_monthly = []
    for month in tqdm(months):
        if os.path.exists(f'downloads/{product}/{product}_{subproduct}_mean_{year}_{month}.npy'):
            layer = np.load(f'downloads/{product}/{product}_{subproduct}_mean_{year}_{month}.npy')
            layer_rep = reproject(layer, latitud, longitud)
            year_monthly.append(layer_rep)
        else:
            print(f'File downloads/{product}/{product}_{subproduct}_mean_{year}_{month}.npy does not exist')
    
    year_monthly = np.array(year_monthly)
    np.save(f'data/{product}/{product}_{subproduct}_mean_{year}.npy', year_monthly)
