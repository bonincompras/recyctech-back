from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="EcoVision API")

# -------- MODELO DE DADOS --------
class NumeroRequest(BaseModel):
    numero: int

class NumeroResponse(BaseModel):
    original: int
    resultado: int


# -------- ENDPOINT --------
@app.post("/calcular", response_model=NumeroResponse)
def calcular_numero(dados: NumeroRequest):
    numero = dados.numero

    if numero % 2 == 0:
        resultado = numero + 5
    else:
        resultado = numero + 6

    return {
        "original": numero,
        "resultado": resultado
    }


# -------- HEALTH CHECK --------
@app.get("/")
def health_check():
    return {"status": "API rodando 🚀"}
