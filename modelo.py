import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

modelo_nome = "meta-llama/Llama-2-7b-chat-hf"
token = os.getenv("HUGGINGFACE_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(modelo_nome, use_auth_token=token)
modelo = AutoModelForCausalLM.from_pretrained(modelo_nome, use_auth_token=token)
tokenizer.pad_token = tokenizer.eos_token
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo.to(device)

def gerar_resposta(prompt):
    entrada = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        output = modelo.generate(**entrada, max_new_tokens=200, num_return_sequences=1)
    resposta = tokenizer.decode(output[0], skip_special_tokens=True)
    return resposta
