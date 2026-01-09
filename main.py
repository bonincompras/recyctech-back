from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI(title="EcoVision API")

# ---------- CORS ----------
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

# ---------- Carregar modelo YOLO ----------
# Pode usar yolov8n.pt (nano) para testes
model = YOLO("yolov8n.pt")  

@app.post("/analisar")
async def analisar_imagem(file: UploadFile = File(...)):
    # Ler imagem
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    altura, largura, _ = img.shape

    # Rodar detecção
    results = model.predict(img)
    
    objetos = []
    for r in results[0].boxes:
        x_min, y_min, x_max, y_max = r.xyxy[0].tolist()
        confianca = float(r.conf[0])
        categoria = model.names[int(r.cls[0])]
        objetos.append({
            "categoria": categoria,
            "confianca": round(confianca * 100, 1),
            "bbox": [x_min, y_min, x_max, y_max]
        })
    
    return JSONResponse({
        "objetos": objetos,
        "largura_imagem": largura,
        "altura_imagem": altura
    })

@app.get("/")
def health_check():
    return {"status": "API rodando 🚀"}
