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
# Resto de tu código ...
# ... el resto de tu código ...
# Define la clase Prompt (copia la definición del código que proporcionaste)

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

# Realiza la inferencia para cada cliente y almacena los resultados
resultados = []
for cliente in clientes_criticas:
    nombre_cliente = cliente["nombre"]
    critica = cliente["critica"]

    prompt_input = f"Classify this review as positive or negative.\n\nReview:\n{critica}\n\nClassification:\n"

    prompt = Prompt(access_token, "e73e9170-2ba3-421c-b8fd-6c78dd5e34be")
    resultado_inferencia = prompt.generate(prompt_input, modelo_id, parametros)

    resultados.append({"Cliente": nombre_cliente, "Critica": critica, "Resultado": resultado_inferencia})

# Almacena los resultados en un DataFrame y guárdalo en un archivo CSV
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv("resultados_criticas.csv", index=False)
