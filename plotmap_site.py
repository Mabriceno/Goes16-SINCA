import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import cartopy.io.img_tiles as cimgt

metadata = pd.read_csv('data/sinca/meta.csv')
lats_so2 =  [float(x.replace(',', '.')) for x in metadata[metadata['SO2'] == 1]['lat']]
lons_so2 =  [float(x.replace(',', '.')) for x in metadata[metadata['SO2'] == 1]['lon']]
lats_mp25 = [float(x.replace(',', '.')) for x in metadata[metadata['MP25'] == 1]['lat']]
lons_mp25 = [float(x.replace(',', '.')) for x in metadata[metadata['MP25'] == 1]['lon']]
lats_mp10 = [float(x.replace(',', '.')) for x in metadata[metadata['MP10'] == 1]['lat']]
lons_mp10 = [float(x.replace(',', '.')) for x in metadata[metadata['MP10'] == 1]['lon']]

# Crear la figura y los ejes con dos subplots
fig, ax1= plt.subplots(figsize=(7, 10), subplot_kw={'projection': ccrs.PlateCarree()})
fig.subplots_adjust(hspace=0.01)


extent = [-76, -60, -44, -15]  # [min_lon, max_lon, min_lat, max_lat]

ax1.set_extent(extent, crs=ccrs.PlateCarree())
ax1.add_feature(cfeature.COASTLINE)
ax1.add_feature(cfeature.BORDERS)
ax1.add_feature(cfeature.LAND.with_scale('10m'))
ax1.add_feature(cfeature.OCEAN)

ax1.scatter(lons_so2, lats_so2, color='#852F91', s=100, transform=ccrs.PlateCarree(), label='SO2 Estacion', zorder = 7, marker='*', alpha=0.8)
ax1.scatter(lons_mp25, lats_mp25, color='#FCB547', s=100, transform=ccrs.PlateCarree(), label='MP2.5 Estacion',  zorder = 6, marker='>',alpha=0.9)
ax1.scatter(lons_mp10, lats_mp10, color='#258BAE', s=100, transform=ccrs.PlateCarree(), label='MP10 Estacion',  zorder = 5, marker='o', alpha=0.8)
ax1.legend()
#lat and lon ticks
ax1.set_xticks([-74, -68, -62], crs=ccrs.PlateCarree())
ax1.set_yticks(np.arange(-42, -16, 8), crs=ccrs.PlateCarree())
# del tick upper and right
ax1.yaxis.set_tick_params(right=False)
ax1.xaxis.set_tick_params(top=False)
ax1.set_title('Sitios SINCA utilizados')

# Mostrar el mapa
plt.savefig(f'images/site_map.png', dpi=300, bbox_inches='tight')
plt.show()