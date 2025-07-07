import pdfplumber
import re
import ftfy
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

token = os.getenv("HUGGINGFACE_TOKEN")

def extraindo_arquivo(arquivo):
    texto_bruto = ""
    tabelas_extraidas = []

    with pdfplumber.open(arquivo) as pdf:
        for pagina in pdf.pages:
            if pagina.extract_text():
                texto_bruto += pagina.extract_text() + "\n"
            tabelas = pagina.extract_tables()
            for tabela in tabelas:
                tabelas_extraidas.append(pd.DataFrame(tabela))

    texto_bruto = ftfy.fix_text(texto_bruto)
    texto_bruto = re.sub(r'\s+', ' ', texto_bruto)
    texto_bruto = re.sub(r'[^a-zA-ZÀ-ÿ0-9,.!?;:()\-\– ]', '', texto_bruto)
    texto_bruto = re.sub(r'\s+([,.!?;:])', r'\1', texto_bruto)

    return texto_bruto.strip(), tabelas_extraidas

def dividir_chunk(texto):
    chunk = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    chunks = chunk.split_text(texto)
    return chunks

def criar_embedding(chunks):
    modelo = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = modelo.encode(chunks)
    return embeddings

def banco_vetorial(chunks, embeddings):
    chroma_cliente = chromadb.EphemeralClient()
    colecao = chroma_cliente.get_or_create_collection(name="meus_dados")
    ids = [str(i) for i in range(len(chunks))]
    colecao.add(ids=ids, embeddings=embeddings.tolist(), documents=chunks,
                metadatas=[{"indice": i} for i in range(len(chunks))])
    return colecao

def criar_prompt(pergunta, colecao):
    pergunta_embedding = criar_embedding([pergunta])
    resultado = colecao.query(query_embeddings=pergunta_embedding.tolist(), n_results=5)
    trechos_recuperados = resultado["documents"][0]
    contexto = "\n".join([f"{i+1}. {trecho}" for i, trecho in enumerate(trechos_recuperados)])
    prompt = f"""
    Você é um assistente virtual especializado em responder perguntas com base em documentos oficiais.

    **Pergunta do usuário:** "{pergunta}"

    **Informações relevantes extraídas do documento:**
    {contexto}

    **Sua tarefa:**
    Com base nas informações acima, **responda de forma objetiva e clara** à pergunta do usuário.
    - Se houver uma resposta direta no contexto, apresente-a de forma resumida e compreensível.
    - Se necessário, reformule a resposta para que fique mais natural e fácil de entender.
    - Se o documento não fornecer informações suficientes, informe que a resposta não está disponível.

    **Resposta:**
    """
    return prompt.strip()

def gerar_resposta(prompt):
    modelo_nome = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(modelo_nome, use_auth_token=token)
    modelo = AutoModelForCausalLM.from_pretrained(modelo_nome, use_auth_token=token)
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo.to(device)
    entrada = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        output = modelo.generate(**entrada, max_new_tokens=200, num_return_sequences=1)
    resposta = tokenizer.decode(output[0], skip_special_tokens=True)
    return resposta
