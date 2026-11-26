"""Prepara fotos inteiras para treinar o detector de arco (arco x sem_arco).

Uso:
    python 12_quadros_arco.py datasets/soldatech_fotos capturas_epis_limpas ...

Copia as fotos (reduzidas para 640 px de largura) para
datasets/arco_quadros/a_classificar/. Depois separe com:
    python 11_separar_recortes.py --tarefa arco
e treine com:
    python 10_treinar_mascara.py --tarefa arco --empacotar   (Colab)

Só contar pixels claros não funciona nestas câmeras: a parede branca estoura sem
arco e, com a exposição baixa, o arco aparece só como faíscas laranja. Por isso o
arco é reconhecido por um classificador treinado com as fotos de vocês.

O nome de cada quadro começa pela pessoa/sessão (partes "-Parte02" juntas), para o
10 validar com pessoas que ficaram fora do treino. Rodar de novo não duplica.
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
EXTENSOES = {'.jpg', '.jpeg', '.png', '.bmp'}
PASTAS = ['a_classificar', 'arco', 'sem_arco', 'descartar']

p = argparse.ArgumentParser()
p.add_argument('pastas', nargs='+')
p.add_argument('--largura', type=int, default=640)
p.add_argument('--saida', default=str(ROOT / 'datasets/arco_quadros'))
args = p.parse_args()

saida = Path(args.saida)
for nome in PASTAS:
    (saida / nome).mkdir(parents=True, exist_ok=True)
existentes = {f.name for nome in PASTAS for f in (saida / nome).iterdir()}

novos = pulados = 0
for pasta in map(Path, args.pastas):
    if not pasta.is_dir():
        raise SystemExit(f'Pasta não encontrada: {pasta}')
    for arq in sorted(f for f in pasta.rglob('*') if f.is_file() and f.suffix.lower() in EXTENSOES):
        rel = arq.relative_to(pasta).parts
        sessao = re.split(r'[-_ ]parte', rel[0] if len(rel) > 1 else pasta.name, flags=re.IGNORECASE)[0]
        nome = f'{sessao}__{arq.stem}.jpg'
        if nome in existentes:
            pulados += 1
            continue
        dados = np.fromfile(str(arq), np.uint8)
        img = cv2.imdecode(dados, cv2.IMREAD_COLOR) if dados.size else None
        if img is None:
            print(f'Não foi possível ler {arq}')
            continue
        if img.shape[1] > args.largura:
            img = cv2.resize(img, (args.largura, round(img.shape[0] * args.largura / img.shape[1])),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if ok:
            buf.tofile(str(saida / 'a_classificar' / nome))
            existentes.add(nome)
            novos += 1

print(f'{novos} quadros novos em {(saida / "a_classificar").resolve()}' + (f' ({pulados} já existiam)' if pulados else ''))
print('Separe com: python 11_separar_recortes.py --tarefa arco')
