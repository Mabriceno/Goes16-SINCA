import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

Y = 'MP25'
model = joblib.load(f'models/{Y}_model.pkl')

years  = np.arange(2020, 2024)
feature_names = ['AOD', 'dust', 'smoke', 'aerosol', 'band1', 'band7', 'band12']
oroginal_shape = np.load('data/ABI-L2-AODF/ABI-L2-AODF__mean_2020.npy')[0].shape

for year in years:
    predictions = []
    for month in tqdm(np.arange(1, 13)):
        aod = np.load(f'data/ABI-L2-AODF/ABI-L2-AODF__mean_{year}.npy')[month-1].flatten()
        dust = np.load(f'data/ABI-L2-ADPF/ABI-L2-ADPF_dust_mean_{year}.npy')[month-1].flatten()
        smoke = np.load(f'data/ABI-L2-ADPF/ABI-L2-ADPF_smoke_mean_{year}.npy')[month-1].flatten()
        aerosol = np.load(f'data/ABI-L2-ADPF/ABI-L2-ADPF_aerosol_mean_{year}.npy')[month-1].flatten()
        band1 = np.load(f'data/ABI-L1b-RadF/ABI-L1b-RadF_1_mean_{year}.npy')[month-1].flatten()
        band7 = np.load(f'data/ABI-L1b-RadF/ABI-L1b-RadF_7_mean_{year}.npy')[month-1].flatten()
        band12 = np.load(f'data/ABI-L1b-RadF/ABI-L1b-RadF_12_mean_{year}.npy')[month-1].flatten()
        X_new = np.array([aod, dust, smoke, aerosol, band1, band7, band12]).T
        scaler = StandardScaler()
        X_new_scaled = scaler.fit_transform(X_new)
        prediction = model.predict(X_new_scaled)
        prediction = prediction.reshape(oroginal_shape)
        predictions.append(prediction.reshape(oroginal_shape))

    np.save(f'data/predictions/{Y}_{year}.npy', np.array(predictions))


