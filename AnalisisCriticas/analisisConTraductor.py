import requests
import pandas as pd
from getpass import getpass
from ibm_cloud_sdk_core import IAMTokenManager
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator, BearerTokenAuthenticator

# Define la clase Prompt (copia la definición del código que proporcionaste)
class Prompt:
    def __init__(self, access_token, project_id):
        self.access_token = access_token
        self.project_id = project_id

    def generate(self, input, model_id, parameters):
        wml_url = "https://us-south.ml.cloud.ibm.com/ml/v1-beta/generation/text?version=2023-05-28"
        Headers = {
            "Authorization": "Bearer " + self.access_token,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        data = {
            "model_id": model_id,
            "input": input,
            "parameters": parameters,
            "project_id": self.project_id
        }
        response = requests.post(wml_url, json=data, headers=Headers)
        if response.status_code == 200:
            return response.json()["results"][0]["generated_text"]
        else:
            return response.text

# Función para traducir una crítica al idioma inglés
def traducir_al_ingles(critica, apikey):
    url_traduccion = "https://api.au-syd.language-translator.watson.cloud.ibm.com/instances/428be32b-34e8-41f7-a7d5-982889cf2a99"  # Reemplaza esto con la URL de tu servicio de traducción
    data = {
        "text": [critica],
        "model_id": "es-en"  # Indica que queremos traducir del español al inglés
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url_traduccion + "/v3/translate?version=2018-05-01", json=data, headers=headers, auth=("apikey", apikey))
    if response.status_code == 200:
        translated_data = response.json()
        critica_ingles = translated_data["translations"][0]["translation"]
        return critica_ingles
    else:
        print("Error en la traducción:", response.text)
        return None

# Lee el archivo CSV y obtén la lista de clientes y críticas
clientes_criticas_df = pd.read_csv("./criticas.csv", delimiter="|")
clientes_criticas = clientes_criticas_df.to_dict(orient="records")
# Define las variables del modelo y parámetros
modelo_id = "google/flan-t5-xxl"
parametros = {
    "decoding_method": "greedy",
    "max_new_tokens": 1,
    "repetition_penalty": 1
}


# Obtén el token de acceso
access_token = IAMTokenManager(
    apikey=getpass("Por favor, ingresa tu API key de IBM Cloud: "),
    url="https://iam.cloud.ibm.com/identity/token"
).get_token()


# Obtén el token de acceso para la API de traducción
traduccion_apikey = getpass("Por favor, ingresa tu API key de IBM Cloud para la traducción: ")

# Realiza la inferencia para cada cliente y almacena los resultados
resultados = []
for cliente in clientes_criticas:
    nombre_cliente = cliente["nombre"]
    critica_espanol = cliente["critica"]

    # Traduce la crítica al idioma inglés
    critica_ingles = traducir_al_ingles(critica_espanol, traduccion_apikey)
    if critica_ingles is None:
        continue

    prompt_input_espanol = f"Classify this review as positive or negative.\n\nReview:\n{critica_espanol}\n\nClassification:\n"
    prompt_input_ingles = f"Classify this review as positive or negative.\n\nReview:\n{critica_ingles}\n\nClassification:\n"

    prompt_espanol = Prompt(access_token, "bed3098a-5916-40a7-a3f8-db548121755d")
    prompt_ingles = Prompt(access_token, "bed3098a-5916-40a7-a3f8-db548121755d")

    resultado_inferencia_espanol = prompt_espanol.generate(prompt_input_espanol, modelo_id, parametros)
    resultado_inferencia_ingles = prompt_ingles.generate(prompt_input_ingles, modelo_id, parametros)

    resultados.append({"Cliente": nombre_cliente, "Critica Español": critica_espanol, "Critica Inglés": critica_ingles, "Resultado Español": resultado_inferencia_espanol, "Resultado Inglés": resultado_inferencia_ingles})

# Almacena los resultados en un DataFrame y guárdalo en un archivo CSV
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv("resultados_criticas.csv", index=False)
