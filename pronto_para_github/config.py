"""Configurações compartilhadas: escolhe o hardware automaticamente e acha o último modelo treinado."""
from pathlib import Path

import torch


def escolher_device():
    """Devolve o melhor dispositivo disponível para o PyTorch/YOLO."""
    if torch.cuda.is_available():
        return 0  # primeira GPU NVIDIA (ou AMD com ROCm no Linux)
    if torch.backends.mps.is_available():
        return "mps"  # chip Apple Silicon (M1, M2, M3, M4...)
    return "cpu"


def descrever_hardware():
    if DEVICE == 0:
        props = torch.cuda.get_device_properties(0)
        return f"GPU {props.name} ({props.total_memory / 1024**3:.1f} GB)"
    if DEVICE == "mps":
        return "Apple Silicon (MPS)"
    return "CPU (sem aceleração)"


def ultimo_modelo():
    """Caminho do best.pt mais recente dentro de runs/."""
    pesos = sorted(Path("runs").glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    if not pesos:
        distribuido = Path(__file__).resolve().parent / "weights/maos_best.pt"
        if distribuido.is_file():
            return str(distribuido)
        raise SystemExit("Nenhum modelo treinado em runs/. Faça a Parte 4 ou use --modelo caminho/best.pt")
    return str(pesos[-1])


DEVICE = escolher_device()
# Tamanho da imagem na inferência em tempo real: menor = mais rápido, porém menos preciso
IMGSZ_INFERENCIA = 320 if DEVICE == "cpu" else 640


if __name__ == "__main__":
    print("PyTorch:", torch.__version__)
    print("Hardware escolhido:", descrever_hardware())
