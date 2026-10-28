"""Parte 3: baixa o dataset hand-keypoints e desenha os rótulos "na mão" para entender o formato."""
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics.data.utils import check_det_dataset

# Ligações entre os 21 pontos, para desenhar o "esqueleto" da mão
ESQUELETO = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # polegar
    (0, 5), (5, 6), (6, 7), (7, 8),          # indicador
    (5, 9), (9, 10), (10, 11), (11, 12),     # médio
    (9, 13), (13, 14), (14, 15), (15, 16),   # anelar
    (13, 17), (17, 18), (18, 19), (19, 20),  # mindinho
    (0, 17),
]


def caminho_rotulo(imagem):
    # .../hand-keypoints/images/train/x.jpg  ->  .../hand-keypoints/labels/train/x.txt
    return imagem.parents[2] / "labels" / imagem.parent.name / f"{imagem.stem}.txt"


dados = check_det_dataset("maos.yaml")  # baixa o dataset se ainda não existir
print("Classes:", dados["names"], "| kpt_shape:", dados["kpt_shape"])
imagens = sorted(p for p in Path(dados["train"]).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
print(f"{len(imagens)} imagens de treino em {dados['train']}")

while True:
    caminho = random.choice(imagens)
    img = cv2.imread(str(caminho))
    H, W = img.shape[:2]
    linhas = caminho_rotulo(caminho).read_text(encoding="utf-8").split("\n")
    linhas = [linha for linha in linhas if linha.strip()]
    print(f"\n{caminho.name} ({W}x{H}): {len(linhas)} mão(s)")

    for linha in linhas:
        valores = [float(v) for v in linha.split()]
        classe = int(valores[0])
        cx, cy, w, h = valores[1:5]
        pontos = np.array(valores[5:]).reshape(-1, 3)  # 21 linhas de (x, y, visibilidade)
        print(f"  {len(valores)} números | classe={classe} | centro=({cx:.3f}, {cy:.3f}) | tamanho=({w:.3f}, {h:.3f})")
        print(f"  ponta do indicador (ponto 8): x={pontos[8, 0]:.3f} y={pontos[8, 1]:.3f} v={pontos[8, 2]:.0f}")

        # Coordenadas normalizadas (0 a 1) -> pixels
        x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
        x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        px = [(int(x * W), int(y * H)) for x, y, _ in pontos]
        for a, b in ESQUELETO:
            cv2.line(img, px[a], px[b], (255, 200, 0), 2)
        for i, ((x, y), v) in enumerate(zip(px, pontos[:, 2])):
            if v > 0:  # 0 = ponto não anotado
                cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
                cv2.putText(img, str(i), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imshow("Rotulos (qualquer tecla = proxima, q = sair)", img)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
