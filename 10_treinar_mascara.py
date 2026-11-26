"""Treina os classificadores usados no alerta de máscara levantada com arco aceso.

    --tarefa mascara (padrão): recortes da máscara, abaixada x levantada
                               (datasets/mascara_recortes, gerados pelo 09)
    --tarefa arco:             quadros inteiros, arco x sem_arco
                               (datasets/arco_quadros, gerados pelo 12)

Uso:
    python 10_treinar_mascara.py [--tarefa arco] [--epocas 60] [--versao 1]
    python 10_treinar_mascara.py [--tarefa arco] --empacotar   # só gera o zip para o Colab

No Colab, abra treinar_mascara_colab.ipynb (envie o zip para o Google Drive antes).

Lê as imagens já separadas por classe, monta datasets/<tarefa>_cls/{train,val}/ e
treina um YOLO de classificação. O melhor peso é copiado para
weights/<tarefa>_cls_v<versao>_best.pt, de onde o dashboard o carrega.

A validação usa sessões inteiras que ficaram fora do treino: fotos seguidas da
mesma sessão são quase iguais e inflariam a acurácia se fossem divididas ao acaso.
"""
import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent
TAREFAS = {  # classes, pasta das imagens separadas, tamanho de entrada
    'mascara': (['abaixada', 'levantada'], 'mascara_recortes', 224),
    'arco': (['arco', 'sem_arco'], 'arco_quadros', 320),
}

p = argparse.ArgumentParser()
p.add_argument('--tarefa', choices=TAREFAS, default='mascara')
p.add_argument('--recortes', default=None, help='pasta com uma subpasta por classe')
p.add_argument('--dataset', default=None)
p.add_argument('--base', default='yolo11n-cls.pt', help='modelo pré-treinado de partida')
p.add_argument('--epocas', type=int, default=60)
p.add_argument('--imgsz', type=int, default=None)
p.add_argument('--batch', type=int, default=32)
p.add_argument('--val', type=float, default=0.2, help='fração aproximada para validação')
p.add_argument('--versao', default='1')
p.add_argument('--pesos', default=str(ROOT / 'weights'), help='pasta onde o melhor peso é copiado')
p.add_argument('--empacotar', action='store_true',
               help='não treina: gera <pasta>_colab.zip com as imagens e este script')
args = p.parse_args()

CLASSES, PASTA, IMGSZ = TAREFAS[args.tarefa]
recortes = Path(args.recortes or ROOT / 'datasets' / PASTA)
dataset = Path(args.dataset or ROOT / 'datasets' / f'{args.tarefa}_cls')
imgsz = args.imgsz or IMGSZ

# Agrupa por sessão (prefixo antes de "__" no nome do recorte)
grupos = defaultdict(lambda: {c: [] for c in CLASSES})
for c in CLASSES:
    pasta = recortes / c
    if not pasta.is_dir():
        raise SystemExit(f'Pasta não encontrada: {pasta} (gere as imagens e separe com o 11)')
    for f in sorted(pasta.glob('*.jpg')):
        grupos[f.name.split('__')[0]][c].append(f)

totais = {c: sum(len(g[c]) for g in grupos.values()) for c in CLASSES}
print('Recortes:', ', '.join(f'{c}={n}' for c, n in totais.items()), f'em {len(grupos)} sessões')
if min(totais.values()) < 20:
    raise SystemExit('Poucos exemplos: separe pelo menos ~20 por classe (o ideal é algumas centenas)')
if min(totais.values()) < 100:
    print('Aviso: menos de 100 exemplos em alguma classe; o modelo tende a errar bastante')

if args.empacotar:
    pacote = ROOT / f'{PASTA}_colab.zip'
    with ZipFile(pacote, 'w', ZIP_DEFLATED) as z:
        z.write(Path(__file__).resolve(), Path(__file__).name)
        for g in grupos.values():
            for c in CLASSES:
                for f in g[c]:
                    z.write(f, f'{PASTA}/{c}/{f.name}')
    print(f'Pacote gerado: {pacote} ({pacote.stat().st_size / 2**20:.1f} MB)')
    print('Envie para o Google Drive e abra treinar_mascara_colab.ipynb no Colab')
    raise SystemExit

import torch
from ultralytics import YOLO

DEVICE = 0 if torch.cuda.is_available() else 'cpu'
print('Dispositivo:', torch.cuda.get_device_name(0) if DEVICE == 0 else 'CPU (lento)')

# Separa sessões inteiras para validação; com uma sessão só, divide as fotos ao acaso
rng = random.Random(0)
divisao = {'train': {c: [] for c in CLASSES}, 'val': {c: [] for c in CLASSES}}
sessoes = sorted(grupos)
if len(sessoes) >= 3:
    rng.shuffle(sessoes)
    alvo = args.val * sum(totais.values())
    em_val = 0
    for s in sessoes:
        n = sum(len(grupos[s][c]) for c in CLASSES)
        # Pula sessões grandes que passariam muito do alvo (evita tirar uma pessoa enorme do treino)
        parte = 'val' if em_val < 0.8 * alvo and em_val + n <= 1.2 * alvo else 'train'
        em_val += n if parte == 'val' else 0
        for c in CLASSES:
            divisao[parte][c] += grupos[s][c]
        print(f'  {parte:5s} {s}: ' + ', '.join(f'{c}={len(grupos[s][c])}' for c in CLASSES))
else:
    print('Aviso: menos de 3 sessões; validação sorteada entre fotos parecidas fica otimista')
    for c in CLASSES:
        todos = [f for g in grupos.values() for f in g[c]]
        rng.shuffle(todos)
        corte = max(1, int(len(todos) * args.val))
        divisao['val'][c], divisao['train'][c] = todos[:corte], todos[corte:]

for parte in divisao:
    for c in CLASSES:
        if not divisao[parte][c]:
            raise SystemExit(f'Sem exemplos de "{c}" em {parte}; separe mais recortes ou mais sessões')

if dataset.exists():
    shutil.rmtree(dataset)
for parte, por_classe in divisao.items():
    for c, arquivos in por_classe.items():
        (dataset / parte / c).mkdir(parents=True)
        for f in arquivos:
            shutil.copy2(f, dataset / parte / c / f.name)
    print(f'{parte}:', ', '.join(f'{c}={len(a)}' for c, a in por_classe.items()))

modelo = YOLO(args.base)
modelo.train(data=str(dataset), epochs=args.epocas, imgsz=imgsz, batch=args.batch, device=DEVICE,
             project=str(ROOT / f'runs/{args.tarefa}_cls'), name=f'v{args.versao}', exist_ok=True,
             fliplr=0.5, hsv_v=0.5, patience=20)

melhor = Path(modelo.trainer.best)
Path(args.pesos).mkdir(parents=True, exist_ok=True)
destino = Path(args.pesos) / f'{args.tarefa}_cls_v{args.versao}_best.pt'
shutil.copy2(melhor, destino)
print(f'Modelo salvo em {destino}')
print(f'Classes: {YOLO(destino).names}')
