import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

product = 'MP10'
predictions_mean = []
predictions_djf = []
predictions_jja = []
# Datos de ejemplo
for year in range(2020, 2024):
    layer = np.nanmean(np.load(f'data/predictions/{product}_2020.npy'), axis=0)
    layer_djf = np.nanmean(np.load(f'data/predictions/{product}_2020.npy')[[11, 0, 1], :, :], axis=0)
    layer_jja = np.nanmean(np.load(f'data/predictions/{product}_2020.npy')[[5, 6, 7], :, :], axis=0)
    predictions_mean.append(layer)
    predictions_djf.append(layer_djf)
    predictions_jja.append(layer_jja)

layer_mean = np.nanmean(predictions_mean, axis=0)
layer_djf = np.nanmean(predictions_djf, axis=0)
layer_jja = np.nanmean(predictions_jja, axis=0)

plt.hist(layer_mean.flatten(), bins=100)
plt.show()
lats = np.load('data/lats.npy')
lons = np.load('data/lons.npy')


# Parámetros ajustables
extent = [-76, -60, -44, -15]  # [min_lon, max_lon, min_lat, max_lat]
cmap = 'PuBu'
vmin = 30
vmax = 60


layer[layer>= vmax] = vmax
layer[layer<=vmin] = vmin
layer_djf[layer_djf>= vmax] = vmax
layer_djf[layer_djf<=vmin] = vmin
layer_jja[layer_jja>= vmax] = vmax
layer_jja[layer_jja<=vmin] = vmin

# Crear la figura y los ejes con dos subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,8), dpi=300, subplot_kw={'projection': ccrs.PlateCarree()})
fig.subplots_adjust(hspace=0.01)

# Ajustar el espacio entre subplots
fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.05, right=0.95, top=0.95, bottom=0.05)

# Anual promedio
ax1.set_extent(extent, crs=ccrs.PlateCarree())
ax1.add_feature(cfeature.COASTLINE)
ax1.add_feature(cfeature.BORDERS)
contour = ax1.contourf(lons, lats, layer, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
#lat and lon ticks
ax1.set_xticks([-74, -68, -62], crs=ccrs.PlateCarree())
ax1.set_yticks(np.arange(-42, -16, 8), crs=ccrs.PlateCarree())
# del tick upper and right
ax1.yaxis.set_tick_params(right=False)
ax1.xaxis.set_tick_params(top=False)
ax1.set_title('Promedio Anual')

# verano
ax2.set_extent(extent, crs=ccrs.PlateCarree())
ax2.add_feature(cfeature.COASTLINE)
ax2.add_feature(cfeature.BORDERS)
contour = ax2.contourf(lons, lats, layer_djf, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
ax2.set_title('Promedio Verano (DEF)')
#lat and lon ticks
ax2.set_xticks([-74, -68, -62], crs=ccrs.PlateCarree())
ax2.xaxis.set_tick_params(top=False)

# invierno
ax3.set_extent(extent, crs=ccrs.PlateCarree())
ax3.add_feature(cfeature.COASTLINE)
ax3.add_feature(cfeature.BORDERS)
contour = ax3.contourf(lons, lats, layer_jja, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
#lat and lon ticks
ax3.set_xticks([-74, -68, -62], crs=ccrs.PlateCarree())

# del tick upper and right
ax3.xaxis.set_tick_params(top=False)
ax3.set_title('Promedio Invierno (JJA)')

# Crear una barra de color común para todos los subplots
cbar = fig.colorbar(contour, ax=[ax1, ax2, ax3], orientation='horizontal', shrink=0.6, pad=0.1)
cbar.set_label(r'$\mu g/ m^3 N$')

# Mostrar el mapa
plt.savefig(f'images/{product}_map.png', dpi=300, bbox_inches='tight')
#plt.show()