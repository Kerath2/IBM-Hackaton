import requests
import pandas as pd
from getpass import getpass
from ibm_cloud_sdk_core import IAMTokenManager

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

def read_input_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Obtén el token de acceso
access_token = IAMTokenManager(
    apikey=getpass("Por favor, ingresa tu API key de IBM Cloud: "),
    url="https://iam.cloud.ibm.com/identity/token"
).get_token()

model_id = "google/flan-ul2"
parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 50,
    "min_new_tokens": 1,
    "repetition_penalty": 1
}

project_id = "e73e9170-2ba3-421c-b8fd-6c78dd5e34be"

# Lee el texto de entrada del archivo "input.txt"
inputTXT = read_input_file("input.txt")

# Lee el ejemplo del archivo "ejemplo.txt"
ejemploTXT = read_input_file("ejemplo.txt")

# Define el prompt input combinando el contenido de ambos archivos
prompt_input = f"{ejemploTXT}\n\nInput:\n{inputTXT}\n\nNamed Entities:\n"

prompt = Prompt(access_token, project_id)

resultado_inferencia = prompt.generate(prompt_input, model_id, parameters)

# Procesar el resultado y extraer nombres y tipos de entidad
entities = resultado_inferencia.split(", ")
data = [entity.split(": ") for entity in entities]

# Crear DataFrame de pandas
df_resultados = pd.DataFrame(data, columns=["Nombre", "Tipo"])

# Guardar DataFrame como archivo CSV
df_resultados.to_csv("resultados_entities.csv", index=False)
