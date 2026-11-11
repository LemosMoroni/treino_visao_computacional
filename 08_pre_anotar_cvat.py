"""Pré-anota fotos com o modelo de EPIs e gera um zip YOLO 1.1 para importar no CVAT.

Uso:
    python 08_pre_anotar_cvat.py [pasta_imagens] [--modelo weights/epis_v6_best.pt] [--conf 0.35]

No CVAT:
    1. Crie a task com as MESMAS imagens (mesmos nomes de arquivo) e os labels
       luvas, soldador, avental_soldagem, mascara_soldagem (nesta ordem).
    2. Na task: Actions > Upload annotations > formato "YOLO 1.1" > escolha o zip gerado.
    3. Revise: corrija caixas erradas, apague falsos positivos e marque o que o modelo não viu.

As caixas são só um ponto de partida; o modelo erra, então revise todas as imagens.
"""
import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEVICE = 0 if torch.cuda.is_available() else 'cpu'

NAMES = ['luvas', 'soldador', 'avental_soldagem', 'mascara_soldagem']
EXTENSOES = {'.jpg', '.jpeg', '.png', '.bmp'}

p = argparse.ArgumentParser()
p.add_argument('pasta', nargs='?', default='capturas_epis_limpas')
p.add_argument('--modelo', default=str(ROOT / 'weights/epis_v6_best.pt'))
p.add_argument('--conf', type=float, default=0.35, help='confiança mínima da caixa')
p.add_argument('--imgsz', type=int, default=640)
p.add_argument('--saida', default=None, help='zip de saída (padrão: <pasta>_pre_anotacoes_cvat.zip)')
args = p.parse_args()

pasta = Path(args.pasta)
imagens = sorted(f for f in pasta.iterdir() if f.suffix.lower() in EXTENSOES)
if not imagens:
    raise SystemExit(f'Nenhuma imagem em {pasta}')

modelo = YOLO(args.modelo)
nomes_modelo = [modelo.names[i] for i in range(len(modelo.names))]
if nomes_modelo != NAMES:
    raise SystemExit(f'Classes do modelo {nomes_modelo} diferentes do esperado {NAMES}')

saida = Path(args.saida or f'{pasta.name}_pre_anotacoes_cvat.zip')
contagem = [0] * len(NAMES)
sem_caixa = 0

with ZipFile(saida, 'w', ZIP_DEFLATED) as z:
    z.writestr('obj.names', '\n'.join(NAMES) + '\n')
    z.writestr('obj.data', f'classes = {len(NAMES)}\ntrain = data/train.txt\nnames = data/obj.names\nbackup = backup/\n')
    z.writestr('train.txt', ''.join(f'data/obj_train_data/{f.name}\n' for f in imagens))
    for i in range(0, len(imagens), 16):
        lote = imagens[i:i + 16]
        for img, r in zip(lote, modelo.predict([str(f) for f in lote], conf=args.conf, imgsz=args.imgsz,
                                                device=DEVICE, verbose=False)):
            linhas = []
            for c, (x, y, w, h) in zip(r.boxes.cls.tolist(), r.boxes.xywhn.tolist()):
                linhas.append(f'{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}')
                contagem[int(c)] += 1
            sem_caixa += not linhas
            z.writestr(f'obj_train_data/{img.stem}.txt', '\n'.join(linhas) + ('\n' if linhas else ''))
        print(f'{min(i + 16, len(imagens))}/{len(imagens)}', end='\r')

print(f'\nZip gerado: {saida.resolve()}')
print(f'{len(imagens)} imagens, {sem_caixa} sem nenhuma caixa')
for nome, n in zip(NAMES, contagem):
    print(f'  {nome}: {n} caixas')
