import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# Carrega o modelo YOLOv8
model_yolo = YOLO("yolov8n.pt")

# Carrega o modelo de classificação de resíduos
model_recycling = tf.keras.models.load_model("modelo_reciclavel.h5")

# Função que processa a imagem e retorna os objetos detectados
def process_objects(frame):
    objeto_id = 0
    conf_threshold = 0.1
    iou_threshold = 0.2

    altura, largura = frame.shape[:2]
    area_imagem = altura * largura

    results = model_yolo(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
    boxes = results[0].boxes

    result_objs = []

    if len(boxes) == 0:
        print("Nenhum objeto detectado, analisando a imagem inteira.")
        
        image_resized = cv2.resize(frame, (224, 224))
        image_resized = image_resized / 255.0
        input_frame = np.expand_dims(image_resized, axis=0)
        
        predictions = model_recycling.predict(input_frame)[0]
        class_idx = np.argmax(predictions)
        categories = ['Plastico', 'Papel', 'Metal', 'Vidro']
        class_name = categories[class_idx]
        confidence = round(float(predictions[class_idx]) * 100, 2)

        result_objs.append({
            "objeto": "Imagem_inteira",
            "categoria": class_name,
            "confianca": confidence,
            "caixa": [0, 0, largura, altura]  # pixels absolutos - OK
        })

    else:
        print(f"{len(boxes)} objetos detectados, processando individualmente...")

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            conf = box.conf[0].item()

            if conf < conf_threshold:
                continue

            area_caixa = (x2 - x1) * (y2 - y1)
            if area_caixa / area_imagem > 0.7:
                print(f"Detecção ignorada: objeto muito grande ({area_caixa / area_imagem * 100:.1f}%)")
                continue

            # Recorte e classificação
            objeto = frame[y1:y2, x1:x2]
            objeto_resized = cv2.resize(objeto, (224, 224))
            objeto_resized = objeto_resized / 255.0
            input_frame = np.expand_dims(objeto_resized, axis=0)
            
            predictions = model_recycling.predict(input_frame)[0]
            class_idx = np.argmax(predictions)
            categories = ['Plastico', 'Papel', 'Metal', 'Vidro']
            class_name = categories[class_idx]
            confidence = round(float(predictions[class_idx]) * 100, 2)

            objeto_id += 1
            result_objs.append({
                "objeto": f"Obj_{objeto_id}",
                "categoria": class_name,
                "confianca": confidence,
                "caixa": [x1, y1, x2, y2]          # ← Correto! Pixels absolutos originais
            })

    # Ordenar por confiança (opcional, mas melhora a UX)
    result_objs.sort(key=lambda x: x["confianca"], reverse=True)
    
    return result_objs
