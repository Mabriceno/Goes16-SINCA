import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from tqdm import tqdm
import joblib


def training(Y_name, feature_names):
    random_states = np.arange(1, 101)
    dataset= pd.read_csv(f'data/training_data/{Y_name}_train.csv')
    dataset = dataset.dropna()
    X = dataset[feature_names].values
    Y = dataset[Y_name].values
    idx = np.where(X != 0)[0]
    X = X[idx]
    Y = Y[idx]
    corrcoefs = []
    mses = []
    importancess = []
    best_model = None
    best_corrcoef = -np.inf
    for random_state in tqdm(random_states):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.6, random_state=random_state)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        corrcoef = np.corrcoef(y_test, y_pred)[0,1]
        mse = mean_squared_error(y_test, y_pred)
        importances = model.feature_importances_
        corrcoefs.append(corrcoef)
        mses.append(mse)
        importancess.append(importances)
        if corrcoef > best_corrcoef:
                best_corrcoef = corrcoef
                best_model = model
    corrcoefs = np.array(corrcoefs)
    mses = np.array(mses)
    importancess = np.array(importancess)
    joblib.dump(best_model, f'models/{Y_name}_model.pkl')  # Guardar el mejor modelo
    return corrcoefs, mses, importancess



feature_names = ['AOD','band1', 'band7', 'band12', 'dust', 'smoke', 'aerosol']
corrcoefs_so2, mses_so2, importancess_so2 = training('SO2', feature_names)
corrcoefs_mp2_5, mses_mp2_5, importancess_mp2_5 = training('MP25', feature_names)
corrcoefs_mp10, mses_mp10, importancess_mp10 = training('MP10', feature_names)



# Calcular promedios y desviaciones estándar
mean_corrcoefs = [np.mean(corrcoefs_so2), np.mean(corrcoefs_mp2_5), np.mean(corrcoefs_mp10)]
std_corrcoefs = [np.std(corrcoefs_so2), np.std(corrcoefs_mp2_5), np.std(corrcoefs_mp10)]
mean_mses = [np.mean(mses_so2), np.mean(mses_mp2_5), np.mean(mses_mp10)]
std_mses = [np.std(mses_so2), np.std(mses_mp2_5), np.std(mses_mp10)]

# Calcular promedios y desviaciones estándar de las importancias
mean_importances = [np.mean(importancess_so2, axis=0), np.mean(importancess_mp2_5, axis=0), np.mean(importancess_mp10, axis=0)]
std_importances = [np.std(importancess_so2, axis=0), np.std(importancess_mp2_5, axis=0), np.std(importancess_mp10, axis=0)]

# Crear figura y subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

# Primer subplot: scatter plot de corrcoef y mse
colors = ['#852F91', '#D3EEB3', '#258BAE']
names = ['SO2', 'MP2.5', 'MP10']
for i in range(3):
    ax1.errorbar(mean_mses[i], mean_corrcoefs[i], xerr=std_mses[i], yerr=std_corrcoefs[i], fmt='o', capsize=5, capthick=2, ecolor='black', color=colors[i], label=names[i]+'-model')
ax1.set_ylabel('Promedio de coeficiente de correlación')
ax1.set_xlabel('Promedio de MSE')
ax1.set_ylim([0.85, 1])
ax1.legend()

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Segundo subplot: gráfico de barras de las importancias
labels = feature_names
x = np.arange(len(labels))  # Posición de las etiquetas
width = 0.2  # Ancho de las barras

ax2.bar(x - width, mean_importances[0], width, yerr=std_importances[0], capsize=5, label = names[0]+'-model', color=colors[0])
ax2.bar(x, mean_importances[1], width, yerr=std_importances[1], capsize=5, label = names[1]+'-model', color=colors[1])
ax2.bar(x + width, mean_importances[2], width, yerr=std_importances[2], capsize=5, label = names[2]+'-model', color=colors[2])

ax2.set_xlabel('Características')
ax2.set_ylabel('Promedio de Importancia de Características')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend()
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('images/training_results.png')
plt.show()