import requests
from ibm_cloud_sdk_core import IAMTokenManager
import os, getpass

project_id = "bed3098a-5916-40a7-a3f8-db548121755d"

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

def realizar_inferencia(transaccion):
    access_token = IAMTokenManager(
        apikey = getpass.getpass("Por favor, ingresa tu API key de IBM Cloud (presiona enter): "),
        url = "https://iam.cloud.ibm.com/identity/token"
    ).get_token()

    model_id = "google/flan-t5-xxl"

    parameters = {
        "decoding_method": "sample",
        "max_new_tokens": 50,
        "temperature": 1.74,
        "top_k": 50,
        "top_p": 1,
        "repetition_penalty": 1
    }

    prompt_input = f"""Act as a financial consultant to help me detect errors in transactions. Responding only if the transaction is correct or incorrect.

A transaction will be incorrect if:
-exceeds 500 dollars.
-It is done on a date after 10/28/2023
-The transaction is made to the same account

The transaction is the following:
{transaccion}

The transaction is:"""

    prompt = Prompt(access_token, project_id)

    resultado = prompt.generate(prompt_input, model_id, parameters)

    return resultado

if __name__ == "__main__":
    transaccion = input("Por favor, ingresa la transacción:\n")
    resultado_inferencia = realizar_inferencia(transaccion)
    print("La transaccion es "  + resultado_inferencia)
