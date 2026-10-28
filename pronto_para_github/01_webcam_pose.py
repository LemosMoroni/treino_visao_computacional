"""Parte 2: primeiro contato com o YOLO em Python, usando o modelo de pose do CORPO."""
import cv2
from ultralytics import YOLO

from config import DEVICE, IMGSZ_INFERENCIA

modelo = YOLO("yolo26n-pose.pt")  # baixado automaticamente na primeira vez
camera = cv2.VideoCapture(0)

while True:
    ok, frame = camera.read()
    if not ok:
        print("Não consegui ler a câmera.")
        break

    resultados = modelo.predict(frame, imgsz=IMGSZ_INFERENCIA, device=DEVICE, verbose=False)
    r = resultados[0]  # um objeto Results por imagem; mandamos só uma

    # r.boxes     -> caixas: r.boxes.xyxy (cantos), r.boxes.conf (confiança), r.boxes.cls (classe)
    # r.keypoints -> pontos: r.keypoints.xy tem formato (n_pessoas, 17, 2)
    pontos = r.keypoints.xy.cpu().numpy()
    if len(pontos) > 0:
        nariz = pontos[0][0]  # pessoa 0, ponto 0 (no modelo de corpo, o ponto 0 é o nariz)
        print(f"Pessoas: {len(pontos)} | nariz da 1ª em x={nariz[0]:.0f}, y={nariz[1]:.0f}")

    cv2.imshow("YOLO pose (q = sair)", r.plot())
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
