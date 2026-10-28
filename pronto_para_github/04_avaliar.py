"""Parte 5: mede a qualidade do modelo treinado no conjunto de validação."""
import argparse

from ultralytics import YOLO

from config import DEVICE, ultimo_modelo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", help="padrão: best.pt mais recente em runs/")
    args = parser.parse_args()

    caminho = args.modelo or ultimo_modelo()
    print("Avaliando:", caminho)
    modelo = YOLO(caminho)
    m = modelo.val(data="maos.yaml", device=DEVICE, batch=8)

    print("\n=== Resultado ===")
    print(f"Caixa da mão  -> precisão {m.box.mp:.3f} | recall {m.box.mr:.3f} | mAP50 {m.box.map50:.3f} | mAP50-95 {m.box.map:.3f}")
    print(f"Pontos (pose) -> precisão {m.pose.mp:.3f} | recall {m.pose.mr:.3f} | mAP50 {m.pose.map50:.3f} | mAP50-95 {m.pose.map:.3f}")
    print(f"Velocidade por imagem (ms): {m.speed}")
    print(f"Gráficos salvos em: {m.save_dir}")


if __name__ == "__main__":
    main()
