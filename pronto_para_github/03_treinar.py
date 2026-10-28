"""Parte 4: fine-tuning do YOLO pose no dataset hand-keypoints, adaptado ao seu hardware."""
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

from config import DEVICE, descrever_hardware

# imgsz = tamanho da imagem | batch = imagens por passo | epochs = voltas no dataset
# fraction = parte do dataset de treino usada | workers = processos que carregam imagens
# val = validar a cada época (False = só na última, economiza muito tempo na CPU)
PERFIS = {
    "gpu_grande":  dict(imgsz=640, batch=32, epochs=100, fraction=1.0,  workers=8),              # NVIDIA >= 8 GB
    "gpu_media":   dict(imgsz=640, batch=16, epochs=80,  fraction=1.0,  workers=4),              # NVIDIA 4-8 GB
    "gpu_pequena": dict(imgsz=480, batch=8,  epochs=60,  fraction=0.5,  workers=2),              # NVIDIA < 4 GB
    "apple":       dict(imgsz=512, batch=16, epochs=50,  fraction=0.5,  workers=4),              # Apple Silicon
    "cpu":         dict(imgsz=320, batch=8,  epochs=15,  fraction=0.1,  workers=2, val=False),   # só processador
    "teste":       dict(imgsz=320, batch=8,  epochs=2,   fraction=0.02, workers=0, val=False),   # confere se tudo roda
}


def perfil_automatico():
    if DEVICE == "mps":
        return "apple"
    if DEVICE == "cpu":
        return "cpu"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if vram_gb >= 7.5:
        return "gpu_grande"
    if vram_gb >= 3.5:
        return "gpu_media"
    return "gpu_pequena"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perfil", choices=PERFIS, help="padrão: escolhido pelo seu hardware")
    parser.add_argument("--modelo", default="yolo26n-pose.pt", help="modelo pré-treinado de partida")
    parser.add_argument("--epocas", type=int, help="sobrescreve o número de épocas do perfil")
    args = parser.parse_args()

    perfil = args.perfil or perfil_automatico()
    config = dict(PERFIS[perfil])
    if args.epocas:
        config["epochs"] = args.epocas
    print(f"Hardware: {descrever_hardware()} | perfil: {perfil} | {config}")

    modelo = YOLO(args.modelo)
    modelo.train(
        data="maos.yaml",
        device=DEVICE,
        project=str(Path("runs").resolve()),  # resultados em runs/maos_<perfil>/
        name=f"maos_{perfil}",
        patience=15,  # para sozinho se a validação não melhorar por 15 épocas
        **config,
    )


# Obrigatório no Windows: cada "worker" é um processo novo que reimporta este arquivo.
# Sem essa proteção, o treino recomeçaria dentro de cada worker.
if __name__ == "__main__":
    main()
