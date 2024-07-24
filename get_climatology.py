import numpy as np
import os


product = 'ABI-L2-AODF'
subproducts = ['']

years = np.arange(2020, 2024)

for subproduct in subproducts:
    layers = []
    for year in years:
        filename = f'data/{product}/{product}_{subproduct}_mean_{year}.npy'
        layer = np.load(filename)
        layers.append(layer)
    layers = np.array(layers)
    layers = layers.reshape(len(years)*12, *layer[0].shape)
    climatology = np.mean(layers, axis=0)
    np.save(f'data/climatologies/{product}_{subproduct}_climatology.npy', climatology)

