import pandas as pd
from fastapi import FastAPI, Query
import ast
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

app = FastAPI()

df = pd.read_csv('steam_games.csv')

@app.get('/')
def read_root():
    return {'message' : 'API para consultar datos de Juegos'}


@app.get('/genero/{anio}')
def genero(anio: str):
    # Limpia los valores faltantes en la columna 'release_date'
    df_cleaned = df.dropna(subset=['release_year'])

    # Elimina las filas con valores 'nan' en la columna 'genres'
    df_cleaned = df_cleaned.dropna(subset=['genres'])

    # Filtra los juegos del año ingresado
    juegos_año = df_cleaned[df_cleaned['release_year'].str.startswith(anio)]

    if juegos_año.empty:
        return {"message": f"No se encontraron juegos para el año {anio}"}

    # Contar los géneros más ofrecidos en el año especificado
    generos_count = {}
    for genres in juegos_año['genres']:
        genre_list = ast.literal_eval(genres)  # Convierte la cadena de texto en lista
        for genre in genre_list:
            generos_count[genre] = generos_count.get(genre, 0) + 1

    # Ordenar los géneros por frecuencia (más ofrecidos primero)
    generos_ordenados = sorted(generos_count.items(), key=lambda x: x[1], reverse=True)

    # Tomar solo los 5 géneros más ofrecidos
    top_5_generos = [genero for genero, _ in generos_ordenados[:5]]

    return {'anio' : anio, 'top generos' : top_5_generos}


@app.get('/juegos/{anio}')
def juegos(anio: str):
    # Filtra los juegos del año ingresado
    juegos_año = df[df['release_year'].str.startswith(anio, na=False)]['app_name'].tolist()

    if not juegos_año:
        return {"message": f"No se encontraron juegos para el año {anio}"}

    return {'anio' : anio, 'Juegos Lanzados' : juegos_año}


@app.get('/specs/{anio}')
def specs(anio: str):
    df_cleaned = df.dropna(subset=['release_year'])

    df_cleaned = df_cleaned.dropna(subset=['specs'])

    juegos_año = df_cleaned[df_cleaned['release_year'].str.startswith(anio)]

    if juegos_año.empty:
        return {"message": f"No se encontraron juegos para el año {anio}"}

    specs_count = {}
    for specs in juegos_año['specs']:
        specs_list = ast.literal_eval(specs)  
        for specs in specs_list:
            specs_count[specs] = specs_count.get(specs, 0) + 1

    specs_ordenados = sorted(specs_count.items(), key=lambda x: x[1], reverse=True)

    top_5_specs = [genero for genero, _ in specs_ordenados[:5]]

    return {'Anio' : anio, 'Top specs' : top_5_specs}


@app.get("/earlyaccess/{anio}")
def earlyacces(anio: str):
    juegos_early = df[(df['release_year'].str.startswith(anio, na=False)) & (df['early_access'] == True)]

    cantidad_juegos = len(juegos_early)

    return {"cantidad de juegos": cantidad_juegos}


@app.get("/sentiment/{anio}")
def sentiment(anio: str):
    registros_año = df[df['release_year'].str.startswith(anio, na=False)]

    sentiment_counts = registros_año['sentiment'].value_counts().to_dict()

    return sentiment_counts


@app.get("/metascore/{anio}")
def metascore(anio: str):
    registros_año = df[df['release_year'].str.startswith(anio, na=False)]

    registros_con_metascore = registros_año.dropna(subset=['metascore'])

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

# Entrena un modelo de regresión
X_train = train_df[['early_access', 'release_year', 'developer_encoded'] + mlb.classes_.tolist()]
y_train = train_df['price']
model = LinearRegression()
model.fit(X_train, y_train)

@app.get("/prediccion/")
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