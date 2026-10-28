"""Partes 6 e 7: desenhe no ar com a mão e transforme os traços em formas geométricas.

Controles: pinça (encostar polegar e indicador) = desenhar
           c = limpar | f = liga/desliga formas | p = mostra/esconde pontos | s = salvar | q = sair
"""
import argparse
import math
import time

import cv2
import numpy as np
from ultralytics import YOLO

from config import DEVICE, IMGSZ_INFERENCIA, ultimo_modelo

# Índices dos 21 pontos da mão (mesmo padrão do MediaPipe, usado no dataset)
PULSO, POLEGAR_PONTA, INDICADOR_PONTA, MEDIO_BASE = 0, 4, 8, 9

# Pinça com histerese: começa a desenhar abaixo de LIMIAR_FECHA e só para acima de LIMIAR_ABRE
LIMIAR_FECHA = 0.30
LIMIAR_ABRE = 0.45
SUAVIZACAO = 0.5          # 0 = sem suavização; perto de 1 = muito suave (e mais atrasado)
TOLERANCIA_SEM_MAO = 5    # frames seguidos sem ver a mão antes de encerrar o traço

COR_TRACO = (255, 0, 255)  # OpenCV usa BGR: isto é magenta
COR_FORMA = (0, 255, 255)  # amarelo
ESPESSURA = 4


def classificar_forma(traco):
    """Analisa um traço (lista de pontos) e devolve (nome, dados) ou None se não reconhecer."""
    pts = np.array(traco, dtype=np.float32)
    _, _, largura, altura = cv2.boundingRect(pts)
    diagonal = math.hypot(largura, altura)  # tamanho do traço (diagonal do retângulo que o envolve)
    if diagonal < 40:
        return None  # traço pequeno demais: provavelmente um toque acidental

    distancia_pontas = float(np.linalg.norm(pts[-1] - pts[0]))

    # 1) Traço ABERTO (o fim ficou longe do início): só vira forma se for quase reto,
    #    ou seja, se nenhum ponto se afasta muito da reta que liga o início ao fim
    if distancia_pontas > 0.3 * diagonal:
        direcao = pts[-1] - pts[0]
        relativo = pts - pts[0]
        desvios = np.abs(direcao[0] * relativo[:, 1] - direcao[1] * relativo[:, 0]) / distancia_pontas
        if desvios.max() < 0.1 * distancia_pontas:
            return "linha", (pts[0], pts[-1])
        return None

    # 2) Traço FECHADO: usamos o contorno convexo, que ignora as tremidas para dentro
    casco = cv2.convexHull(pts)
    perimetro = cv2.arcLength(casco, closed=True)
    area = cv2.contourArea(casco)
    if area < 400:
        return None
    circularidade = 4 * math.pi * area / perimetro**2  # círculo = 1.0 | quadrado ≈ 0.79 | triângulo ≈ 0.60
    vertices = cv2.approxPolyDP(casco, 0.04 * perimetro, closed=True).reshape(-1, 2)

    if circularidade > 0.92:
        momentos = cv2.moments(casco)
        centro = (momentos["m10"] / momentos["m00"], momentos["m01"] / momentos["m00"])
        return "circulo", (centro, math.sqrt(area / math.pi))
    if len(vertices) == 3:
        return "triangulo", vertices
    if len(vertices) == 4:
        retangulo = cv2.minAreaRect(casco)  # menor retângulo (possivelmente girado) que cobre o traço
        largura, altura = retangulo[1]
        nome = "quadrado" if min(largura, altura) / max(largura, altura) > 0.85 else "retangulo"
        return nome, cv2.boxPoints(retangulo)
    if 5 <= len(vertices) <= 8:
        return "poligono", vertices
    return None


def desenhar_forma(img, nome, dados):
    if nome == "linha":
        inicio, fim = dados
        cv2.line(img, tuple(map(int, inicio)), tuple(map(int, fim)), COR_FORMA, ESPESSURA, cv2.LINE_AA)
    elif nome == "circulo":
        centro, raio = dados
        cv2.circle(img, tuple(map(int, centro)), int(raio), COR_FORMA, ESPESSURA, cv2.LINE_AA)
    else:  # triângulo, quadrado, retângulo e polígono são listas de vértices
        cv2.polylines(img, [np.asarray(dados, dtype=np.int32)], True, COR_FORMA, ESPESSURA, cv2.LINE_AA)


def finalizar_traco(traco, canvas, modo_formas):
    """Chamado quando a pinça abre: converte o traço em forma ou mantém o desenho livre."""
    if len(traco) < 2:
        return None
    forma = classificar_forma(traco) if modo_formas else None
    if forma:
        desenhar_forma(canvas, *forma)
        return forma[0]
    cv2.polylines(canvas, [np.array(traco, dtype=np.int32)], False, COR_TRACO, ESPESSURA, cv2.LINE_AA)
    return "livre"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", help="padrão: best.pt mais recente em runs/")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=IMGSZ_INFERENCIA)
    args = parser.parse_args()

    modelo = YOLO(args.modelo or ultimo_modelo(), task="pose")
    camera = cv2.VideoCapture(args.camera)
    ok, frame = camera.read()
    if not ok:
        raise SystemExit("Não consegui abrir a câmera. Tente --camera 1.")

    canvas = np.zeros_like(frame)  # camada onde ficam os desenhos já finalizados
    traco = []                      # pontos do traço que está sendo desenhado agora
    desenhando = False
    caneta = None                   # posição suavizada da "ponta da caneta"
    frames_sem_mao = 0
    modo_formas, mostrar_pontos = True, True
    ultima_forma, abertura = "-", 0.0
    instante_anterior = time.perf_counter()

    while True:
        ok, frame = camera.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # espelho: mexer a mão para a direita move o cursor para a direita

        r = modelo.predict(frame, imgsz=args.imgsz, conf=0.5, max_det=1, device=DEVICE, verbose=False)[0]
        tem_mao = r.keypoints is not None and len(r.keypoints.xy) > 0

        if tem_mao:
            frames_sem_mao = 0
            mao = r.keypoints.xy[0].cpu().numpy()  # (21, 2): x, y em pixels de cada ponto

            # Distância polegar-indicador dividida pelo tamanho da mão (pulso -> base do médio),
            # assim o gesto funciona igual com a mão perto ou longe da câmera
            tamanho_mao = np.linalg.norm(mao[MEDIO_BASE] - mao[PULSO]) + 1e-6
            abertura = np.linalg.norm(mao[POLEGAR_PONTA] - mao[INDICADOR_PONTA]) / tamanho_mao

            # A caneta é o ponto médio entre as pontas do polegar e do indicador
            alvo = (mao[POLEGAR_PONTA] + mao[INDICADOR_PONTA]) / 2
            caneta = alvo if caneta is None else SUAVIZACAO * caneta + (1 - SUAVIZACAO) * alvo

            if not desenhando and abertura < LIMIAR_FECHA:
                desenhando, traco = True, []
            elif desenhando and abertura > LIMIAR_ABRE:
                desenhando = False
                ultima_forma = finalizar_traco(traco, canvas, modo_formas) or ultima_forma
                traco = []
            if desenhando:
                traco.append((int(caneta[0]), int(caneta[1])))
        else:
            frames_sem_mao += 1
            if frames_sem_mao > TOLERANCIA_SEM_MAO:  # a mão sumiu de verdade, não foi só um frame ruim
                if desenhando:
                    ultima_forma = finalizar_traco(traco, canvas, modo_formas) or ultima_forma
                desenhando, traco, caneta = False, [], None

        # Monta a imagem final: câmera (+ pontos da mão) + desenhos + traço atual + cursor
        tela = r.plot(boxes=False) if (mostrar_pontos and tem_mao) else frame.copy()
        mascara = canvas.any(axis=2)
        tela[mascara] = canvas[mascara]
        if len(traco) > 1:
            cv2.polylines(tela, [np.array(traco, dtype=np.int32)], False, COR_TRACO, ESPESSURA, cv2.LINE_AA)
        if caneta is not None:
            cv2.circle(tela, (int(caneta[0]), int(caneta[1])), 8, (0, 255, 0), -1 if desenhando else 2)

        agora = time.perf_counter()
        fps = 1 / max(agora - instante_anterior, 1e-6)
        instante_anterior = agora
        info = f"FPS {fps:4.1f} | abertura {abertura:.2f} | formas {'ON' if modo_formas else 'OFF'} | ultima: {ultima_forma}"
        cv2.putText(tela, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(tela, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Maos que desenham (q = sair)", tela)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if tecla == ord("c"):
            canvas[:] = 0
            ultima_forma = "-"
        elif tecla == ord("f"):
            modo_formas = not modo_formas
        elif tecla == ord("p"):
            mostrar_pontos = not mostrar_pontos
        elif tecla == ord("s"):
            nome_arquivo = f"desenho_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(nome_arquivo, tela)
            print("Salvo:", nome_arquivo)

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
