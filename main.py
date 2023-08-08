import pandas as pd
from fastapi import FastAPI, Query
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.linear_model import Ridge

app = FastAPI()

df = pd.read_csv('steam_games.csv')

@app.get('/')
def read_root():
    return {'message' : 'API para consultar datos de Juegos'}


@app.get("/genero/{anio}")
def genero(anio: str):
    # Filtrar los datos por el año especificado
    filtered_data = df[df['release_year'] == int(anio)]
    
    # Contar los géneros más ofrecidos en el año especificado
    generos_count = {}
    for genres in filtered_data['genres']:
        genre_list = ast.literal_eval(genres)  # Convierte la cadena de texto en lista
        for genre in genre_list:
            generos_count[genre] = generos_count.get(genre, 0) + 1

    # Ordenar los géneros por frecuencia (más ofrecidos primero)
    generos_ordenados = sorted(generos_count.items(), key=lambda x: x[1], reverse=True)

    # Tomar solo los 5 géneros más ofrecidos
    top_5_generos = [genero for genero, _ in generos_ordenados[:5]]

    return {'anio' : anio, 'top generos' : top_5_generos}

@app.get('/juegos/{anio}')
def juegos(anio: int):
    # Filtrar los datos por el año especificado
    filtered_data = df[df['release_year'] == anio]

    # Obtener la lista de juegos lanzados en el año
    juegos_lanzados = filtered_data['app_name'].tolist()

    return {'anio' : anio, 'Juegos Lanzados' : juegos_lanzados}


@app.get('/specs/{anio}')
def specs(anio: str):
    # Filtrar los datos por el año especificado
    filtered_data = df[df['release_year'] == int(anio)]
    
    # Contar los géneros más ofrecidos en el año especificado
    generos_count = {}
    for genres in filtered_data['genres']:
        genre_list = ast.literal_eval(genres)  # Convierte la cadena de texto en lista
        for genre in genre_list:
            generos_count[genre] = generos_count.get(genre, 0) + 1

    # Ordenar los géneros por frecuencia (más ofrecidos primero)
    generos_ordenados = sorted(generos_count.items(), key=lambda x: x[1], reverse=True)

    # Tomar solo los 5 géneros más ofrecidos
    top_5_generos = [genero for genero, _ in generos_ordenados[:5]]

    return {'anio' : anio, 'top generos' : top_5_generos}


@app.get("/earlyaccess/{anio}")
def earlyacces(anio: int):
    # Filtrar los datos por el año especificado y early access
    filtered_data = df[(df['release_year'] == anio) & (df['early_access'] == True)]

    # Contar la cantidad de juegos con early access
    cantidad_early_access = len(filtered_data)

    return {"año": anio, "cantidad_early_access": cantidad_early_access}


@app.get("/sentiment/{anio}")
def sentiment(anio: str):
    filtered_data = df[df['release_year'] == int(anio)]

    sentiment_counts = filtered_data['sentiment'].value_counts().to_dict()

    return sentiment_counts


@app.get("/metascore/{anio}")
def metascore(anio: str):
    filtered_data = df[df['release_year'] == int(anio)]

    registros_con_metascore = filtered_data.dropna(subset=['metascore'])

    if registros_con_metascore.empty:
        return {"message": "No hay registros con metascore para el año especificado"}

    top_juegos = registros_con_metascore.nlargest(5, 'metascore')[['app_name', 'metascore']]

    top_dict = top_juegos.set_index('app_name')['metascore'].to_dict()

    top_dict = {juego: str(metascore) for juego, metascore in top_dict.items()}

    return top_dict


df = df.dropna(subset=['release_date'])
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df = df.dropna(subset=['price'])
df = df.dropna(subset=['genres'])
df['release_date'] = pd.to_datetime(df['release_date'])
df['release_year'] = df['release_date'].dt.year
df = df.drop(columns=['release_date'])
df = df.dropna(subset=['developer'])

# Realiza el One-Hot Encoding para la columna 'genres'
mlb = MultiLabelBinarizer()
genre_encoded = pd.DataFrame(mlb.fit_transform(df['genres']), columns=mlb.classes_, index=df.index)

# Codifica la columna 'developer' usando Label Encoding
le = LabelEncoder()
df['developer_encoded'] = le.fit_transform(df['developer'])

# Combina el DataFrame con el One-Hot Encoding y la columna 'developer' codificada
df_encoded = pd.concat([df, genre_encoded], axis=1)

# Divide tus datos en conjuntos de entrenamiento y prueba
train_df, test_df = train_test_split(df_encoded, test_size=0.2, random_state=42)

# Entrena un modelo de Regresión Ridge
X_train = train_df[['early_access', 'release_year', 'developer_encoded'] + mlb.classes_.tolist()]
y_train = train_df['price']

alpha = 1.0  # Parámetro de regularización (ajustar según sea necesario)
model = Ridge(alpha=alpha)
model.fit(X_train, y_train)

def obtener_prediccion(
    genero: list = Query(..., description="Lista de géneros", example=["Free to Play", "Indie"]),
    earlyaccess: bool = Query(..., description="Acceso temprano"),
    release_year: int = Query(..., description="Año de lanzamiento"),
    developer: str = Query(..., description="Desarrollador")
):
    # Realiza la predicción
    genre_encoded = mlb.transform([genero])
    developer_encoded = le.transform([developer])
    X_pred = [[earlyaccess, release_year, developer_encoded[0]] + genre_encoded.tolist()[0]]
    precio_predicho = model.predict(X_pred)[0]

    # Calcula el RMSE en el conjunto de prueba
    X_test = test_df[['early_access', 'release_year', 'developer_encoded'] + mlb.classes_.tolist()]
    y_test = test_df['price']
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    return {"precio_predicho": precio_predicho, "rmse": rmse}