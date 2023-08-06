import pandas as pd
from fastapi import FastAPI
import ast
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MultiLabelBinarizer


app = FastAPI()

df = pd.read_csv('steam_games.csv')

@app.get('/')
def read_root():
    return {'message' : 'API para consultar datos de Juegos'}


@app.get('/genero/{anio}')
def genero(anio: str):
    # Limpia los valores faltantes en la columna 'release_date'
    df_cleaned = df.dropna(subset=['release_date'])

    # Elimina las filas con valores 'nan' en la columna 'genres'
    df_cleaned = df_cleaned.dropna(subset=['genres'])

    # Filtra los juegos del año ingresado
    juegos_año = df_cleaned[df_cleaned['release_date'].str.startswith(anio)]

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
    juegos_año = df[df['release_date'].str.startswith(anio, na=False)]['app_name'].tolist()

    if not juegos_año:
        return {"message": f"No se encontraron juegos para el año {anio}"}

    return {'anio' : anio, 'Juegos Lanzados' : juegos_año}


@app.get('/specs/{anio}')
def specs(anio: str):
    df_cleaned = df.dropna(subset=['release_date'])

    df_cleaned = df_cleaned.dropna(subset=['specs'])

    juegos_año = df_cleaned[df_cleaned['release_date'].str.startswith(anio)]

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
    juegos_early = df[(df['release_date'].str.startswith(anio, na=False)) & (df['early_access'] == True)]

    cantidad_juegos = len(juegos_early)

    return {"cantidad de juegos": cantidad_juegos}


@app.get("/sentiment/{anio}")
def sentiment(anio: str):
    registros_año = df[df['release_date'].str.startswith(anio, na=False)]

    sentiment_counts = registros_año['sentiment'].value_counts().to_dict()

    return sentiment_counts


@app.get("/metascore/{anio}")
def metascore(anio: str):
    registros_año = df[df['release_date'].str.startswith(anio, na=False)]

    registros_con_metascore = registros_año.dropna(subset=['metascore'])

    if registros_con_metascore.empty:
        return {"message": "No hay registros con metascore para el año especificado"}

    top_juegos = registros_con_metascore.nlargest(5, 'metascore')[['app_name', 'metascore']]

    top_dict = top_juegos.set_index('app_name')['metascore'].to_dict()

    top_dict = {juego: str(metascore) for juego, metascore in top_dict.items()}

    return top_dict



# Convertir la columna 'genres' en columnas binarias usando MultiLabelBinarizer
mlb = MultiLabelBinarizer()
genres_encoded = pd.DataFrame(mlb.fit_transform(df['genres']), columns=mlb.classes_, index=df.index)

# Concatenar las columnas binarias con el DataFrame original
df = pd.concat([df, genres_encoded], axis=1)

# Definir las características y el objetivo
X = df.drop(["app_name", "release_date", "price", "genres"], axis=1)
y = df["price"]

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

@app.post("/prediccion/")
def prediccion(genero: list, earlyaccess: bool, metascore: int):
    # Preprocesar la entrada para hacer la predicción
    input_data = pd.DataFrame({
        "early_access": [earlyaccess],
        "metascore": [metascore]
    })

    # Convertir el género ingresado en columnas binarias
    input_genre = mlb.transform([genero])  # Usar el mismo MultiLabelBinarizer
    input_genre_df = pd.DataFrame(input_genre, columns=mlb.classes_)

    # Concatenar las características de entrada
    input_data = pd.concat([input_data, input_genre_df], axis=1)

    # Hacer la predicción
    predicted_price = model.predict(input_data)

    # Calcular RMSE en el conjunto de prueba
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    return {"predicted_price": predicted_price[0], "rmse": rmse}