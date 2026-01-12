from fastapi import FastAPI, File, UploadFile, Form  # Importando Form aqui
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from model_utils import process_objects  # Certifique-se de que esta função esteja no model_utils.py
import cv2
import numpy as np



# =========================
# App FastAPI
# =========================
app = FastAPI(title="EcoVision API")

# =========================
# CORS - Permite que o frontend local acesse a API
# =========================
origins = [
    "http://127.0.0.1:5500",
    "https://bonincompras.github.io/recyctech-front/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Permite acesso de origens específicas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Endpoints
# =========================


model = None

@app.on_event("startup")
async def load_model():
    global model
    from model_utils import load_model  # função que inicializa o modelo pesado
    model = load_model()  # Carrega só no startup event

# --- Health check ---
@app.get("/")
def health_check():
    return {"status": "API está rodando 🚀"}

# --- Analisar imagem (com IA) ---
@app.post("/analisar")
async def analisar_imagem(file: UploadFile = File(...)):
    try:
        # Salvar a imagem temporariamente
        img_data = await file.read()
        image = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Não foi possível decodificar a imagem.")

        # Realizar a previsão (analisando a imagem por inteiro ou por objeto)
        result = process_objects(image)

        return JSONResponse(content={"objetos": result})

    except Exception as e:
        # Logando o erro completo no console para ajudar a diagnosticar o problema
        print(f"Erro ao processar a imagem: {str(e)}")  # Detalhe do erro no backend
        return JSONResponse(
            {"erro": f"Erro ao analisar a imagem: {str(e)}"},
            status_code=500
        )

# --- Salvar feedback (retorno simples) ---
@app.post("/feedback")
async def salvar_feedback(
    categoria: str = Form(...),  # Categoria analisada (Plástico, Metal, etc.)
    feedback: str = Form(...),   # Feedback do usuário
):
    try:
        # Apenas retorna uma confirmação simples
        return {"status": "feedback salvo"}

    except Exception as e:
        return JSONResponse({"erro": str(e)}, status_code=500)



