import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import plotly.express as px

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

CIUDAD1 = "Barcelona"
CIUDAD2 = "Madrid"
ARCHIVO1 = "Barcelona_listings.csv"
ARCHIVO2 = "Madrid_listings.csv"

def cargar_y_limpia_datos(ruta_archivo, nombre_ciudad):
    df = pd.read_csv(ruta_archivo)
    df['ciudad'] = nombre_ciudad

    columnas = ['id', 'name', 'host_id', 'neighbourhood', 'latitude', 'longitude', 'room_type',
                'price', 'minimum_nights', 'availability_365', 'number_of_reviews', 
                'last_review', 'reviews_per_month']
    df = df[[col for col in columnas if col in df.columns] + ['ciudad']]

    df['price'] = df['price'].replace(r'[\$,]', '', regex=True).astype(float)
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
    df['minimum_nights'] = pd.to_numeric(df['minimum_nights'], errors='coerce')
    df['availability_365'] = pd.to_numeric(df['availability_365'], errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude', 'price', 'room_type'])

    percentil_99 = df['price'].quantile(0.99)
    df = df[df['price'] < percentil_99]

    return df

df1 = cargar_y_limpia_datos(ARCHIVO1, CIUDAD1)
df2 = cargar_y_limpia_datos(ARCHIVO2, CIUDAD2)
df = pd.concat([df1, df2], ignore_index=True)

df['tasa_ocupación'] = 1 - (df['availability_365'] / 365)

resumen_ciudad = df.groupby('ciudad').agg({
    'precio': ['promedio', 'mediana'],
    'Tasa_ocupación': 'promedio',
    'Estancia_mínima': 'mediana'
}).round(2)
print("Resumen por ciudad:\n", resumen_ciudad)

for c in [CIUDAD1, CIUDAD2]:
    correlacion = df[df['ciudad'] == c][['price', 'tasa_ocupación']].corr().iloc[0,1]
    print(f"Correlación precio-ocupación en {c}: {correlacion:.2f}")

plt.figure(figsize=(8,5))
sns.boxplot(x='ciudad', y='price', data=df)
plt.title('Distribución de Precios por Ciudad')
plt.ylabel('Precio por noche (€)')
plt.xlabel('Ciudad')
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='price', y='tasa_ocupacion', hue='ciudad', alpha=0.3)
plt.title('Precio vs. Tasa de Ocupación')
plt.xlabel('Precio por noche (€)')
plt.ylabel('Tasa de Ocupación')
plt.show()

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
for c in [CIUDAD1, CIUDAD2]:
    gdf_ciudad = gdf[gdf['ciudad'] == c]
    fig, ax = plt.subplots(figsize=(8,5))
    gdf_ciudad.plot(ax=ax, column='price', cmap='viridis', alpha=0.5, legend=True, markersize=2)
    plt.title(f'Ubicación de Airbnb en {c.capitalize()} (color según precio)')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.show()

fig = px.scatter_mapbox(df, lat="latitud", lon="longitud", color="precio", size="tasa_ocupación", hover_name="Barrio", 
    mapbox_style="carto-positron",
    zoom=5,
    title="Distribución Geográfica de Airbnb por Precio y Ocupación",
)
fig.show()

precio_medio = df.groupby('ciudad')['price'].mean()
print(f"\nPrecio promedio por ciudad:\n{precio_medio.to_string(index=True, header=False)}")

asequibilidad = df.groupby('ciudad').apply(lambda x: np.corrcoef(x['price'], x['tasa_ocupacion'])[0,1])
print(f"\nCorrelación precio-ocupación por ciudad (más alto => menos asequible):\n{asequibilidad.to_string(index=True, header=False)}")

resumen_ciudad.to_csv("resumen_ciudad.csv")
print("\nAnálisis completo. Gráficas y mapas generados. Revisa el archivo 'resumen_ciudad.csv' para el resumen estadístico.")