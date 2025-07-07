import os
import logging
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from modelo import extraindo_arquivo, dividir_chunk, criar_embedding, banco_vetorial, criar_prompt, gerar_resposta

app = FastAPI()

# Configuração básica de logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/responder/")
async def responder(pergunta: str = Form(...), pdf: UploadFile = File(...)):
    arquivo_temp = "cliente_temp.pdf"
    try:
        logger.info("Recebendo PDF e pergunta do cliente")

        conteudo_pdf = await pdf.read()

        with open(arquivo_temp, "wb") as f:
            f.write(conteudo_pdf)

        logger.info("Extraindo texto do PDF")
        texto, tabelas = extraindo_arquivo(arquivo_temp)

        logger.info("Dividindo texto em chunks")
        chunks = dividir_chunk(texto)

        logger.info("Criando embeddings")
        embeddings = criar_embedding(chunks)

        logger.info("Criando banco vetorial")
        colecao = banco_vetorial(chunks, embeddings)

        logger.info("Montando prompt")
        prompt = criar_prompt(pergunta, colecao)

        logger.info("Gerando resposta com LLaMA")
        resposta = gerar_resposta(prompt)

        return JSONResponse(content={"resposta": resposta})

    except Exception as e:
        logger.error(f"Erro ao processar requisição: {e}")
        return JSONResponse(content={"erro": str(e)}, status_code=500)

    finally:
        # Limpa o arquivo temporário para não acumular lixo
        if os.path.exists(arquivo_temp):
            os.remove(arquivo_temp)
            logger.info("Arquivo temporário removido")
