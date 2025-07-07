
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from modelo import extraindo_arquivo, dividir_chunk, criar_embedding, banco_vetorial, criar_prompt, gerar_resposta

app = FastAPI()

@app.post("/responder/")
async def responder(pergunta: str = Form(...), pdf: UploadFile = File(...)):
    try:
        conteudo_pdf = await pdf.read()

        with open("cliente_temp.pdf", "wb") as f:
            f.write(conteudo_pdf)

        texto, tabelas = extraindo_arquivo("cliente_temp.pdf")
        chunks = dividir_chunk(texto)
        embeddings = criar_embedding(chunks)
        colecao = banco_vetorial(chunks, embeddings)
        prompt = criar_prompt(pergunta, colecao)
        resposta = gerar_resposta(prompt)

        return JSONResponse(content={"resposta": resposta})
    
    except Exception as e:
        return JSONResponse(content={"erro": str(e)}, status_code=500)
