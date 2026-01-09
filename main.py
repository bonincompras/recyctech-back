from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="EcoVision API")

# -------- HABILITAR CORS --------
origins = [
    "https://bonincompras.github.io/recyctech-front/",  # seu front-end
    # "*"  # para testes, pode usar "*" mas não recomendado em produção
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # quem pode acessar
    allow_credentials=True,
    allow_methods=["*"],         # métodos permitidos (GET, POST, etc)
    allow_headers=["*"],         # headers permitidos
)

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
    resultado = numero + 5 if numero % 2 == 0 else numero + 6
    return {
        "original": numero,
        "resultado": resultado
    }

# -------- HEALTH CHECK --------
@app.get("/")
def health_check():
    return {"status": "API rodando 🚀"}

