import requests
import pandas as pd
from getpass import getpass
from ibm_cloud_sdk_core import IAMTokenManager

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

# Leer el archivo CSV con las oraciones
def read_input_csv(input_file):
    df = pd.read_csv(input_file, delimiter='|')
    return df.to_dict(orient='records')

# Define las variables del modelo y parámetros
modelo_id = "google/flan-t5-xxl"
parametros = {
    "decoding_method": "greedy",
    "max_new_tokens": 1,
    "repetition_penalty": 1
}

def main():
    access_token = IAMTokenManager(
        apikey=getpass("Por favor, ingresa tu API key de IBM Cloud: "),
        url="https://iam.cloud.ibm.com/identity/token"
    ).get_token()

    input_file = "./criticas.csv"
    clientes_criticas = read_input_csv(input_file)

    resultados = []  # Lista para almacenar los resultados de cada cliente

    for cliente in clientes_criticas:
        nombre_cliente = cliente["nombre"]
        critica = cliente["critica"]

        prompt_input = f"Act as a webmaster who must extract structured information from sentences. Read each sentence and extract the product or service:\nInput:\nThe attention when applying for a bank loan was excellent, I am very satisfied.\n\nProduct or service:\nbank loan\n\nInput:\n{critica}\nProduct or service:"

        prompt = Prompt(access_token, "bed3098a-5916-40a7-a3f8-db548121755d")
        resultado_inferencia = prompt.generate(prompt_input, modelo_id, parametros)

        print(f"Cliente: {nombre_cliente}")
        print(f"Critica: {critica}")
        print(f"Respuesta del pront: {resultado_inferencia}")
        print("---------------------------------------")

        resultados.append({"nombre": nombre_cliente, "respuesta": resultado_inferencia})

    # Almacena los resultados en un archivo CSV
    df_resultados = pd.DataFrame(resultados)
    output_file = "resultados.csv"
    df_resultados.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()
