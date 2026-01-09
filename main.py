# =========================
# Otimizações para CPU (Render)
# =========================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# =========================
# Imports
# =========================
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
import requests
from supabase import create_client, Client

# =========================
# App
# =========================
app = FastAPI(title="EcoVision API")

# =========================
# CORS
# =========================
origins = [
    "https://bonincompras.github.io",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Supabase
# =========================
SUPABASE_URL = os.environ.get("SUPABASE_URL")  # colocar sua URL do Supabase
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")  # colocar sua chave do Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Diretório para salvar imagens temporariamente (opcional)
IMAGES_DIR = "feedback_images"
os.makedirs(IMAGES_DIR, exist_ok=True)

# =========================
# Modelo YOLO
# =========================
MODEL_PATH = "yolov8n.pt"

if not os.path.exists(MODEL_PATH):
    print("📥 Modelo YOLO não encontrado. Baixando...")
    url = "https://ultralytics.com/assets/yolov8n.pt"
    r = requests.get(url, timeout=60)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("✅ Download concluído!")

print("🚀 Carregando modelo YOLO...")
model = YOLO(MODEL_PATH)
model.fuse()  # acelera inferência
print("✅ Modelo carregado!")

# =========================
# Funções utilitárias
# =========================
def resize_imagem(img, max_size=512):
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    scale = max_size / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))

# =========================
# Endpoints
# =========================

# --- Analisar imagem ---
@app.post("/analisar")
async def analisar_imagem(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse({"erro": "Imagem inválida"}, status_code=400)

        altura, largura, _ = img.shape
        img = resize_imagem(img, max_size=512)

        results = model.predict(
            source=img,
            imgsz=416,
            conf=0.45,
            iou=0.5,
            device="cpu",
            half=False,
            stream=False,
            verbose=False
        )

        objetos = []
        for r in results[0].boxes:
            objetos.append({
                "categoria": model.names[int(r.cls[0])],
                "confianca": round(float(r.conf[0]) * 100, 1),
                "bbox": [round(v, 1) for v in r.xyxy[0].tolist()]
            })

        return {
            "objetos": objetos,
            "largura_imagem": largura,
            "altura_imagem": altura
        }

    except Exception as e:
        return JSONResponse({"erro": str(e)}, status_code=500)

# --- Salvar feedback ---
@app.post("/feedback")
async def salvar_feedback(
    file: UploadFile = File(...),
    feedback: str = Form(...)
):
    try:
        # Salvar imagem temporariamente (opcional)
        caminho_imagem = os.path.join(IMAGES_DIR, file.filename)
        with open(caminho_imagem, "wb") as f:
            f.write(await file.read())

        # Salvar dados no Supabase
        supabase.table("feedbacks").insert({
            "nome_imagem": file.filename,
            "feedback": feedback
        }).execute()

        return JSONResponse({"status": "salvo", "imagem": file.filename})

    except Exception as e:
        return JSONResponse({"erro": str(e)}, status_code=500)

# --- Health check ---
@app.get("/")
def health_check():
    return {"status": "API rodando 🚀"}
