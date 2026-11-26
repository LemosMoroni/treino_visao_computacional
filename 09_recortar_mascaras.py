"""Recorta as máscaras de solda das fotos para montar o dataset do classificador de
máscara levantada/abaixada.

Uso:
    python 09_recortar_mascaras.py datasets/soldatech_fotos --anotacoes   # caixas do CVAT
    python 09_recortar_mascaras.py capturas_epis_limpas --recursivo       # caixas do modelo v7

Com --anotacoes, usa as caixas anotadas no CVAT (export YOLO 1.1): cada subpasta da
pasta indicada é uma task, com as fotos e um obj_train_data/<foto>.txt. Sem essa
opção, o modelo de EPIs detecta as máscaras.

Gera recortes quadrados (máscara + margem ao redor) em
datasets/mascara_recortes/a_classificar/. Depois, mova cada recorte para:
    abaixada/   máscara cobrindo o rosto (protegido)
    levantada/  máscara erguida, rosto exposto
    descartar/  recorte errado, cortado, borrado ou duvidoso
e rode o 10_treinar_mascara.py.

O nome do recorte começa pela pessoa/sessão (ex.: SoldaTech_Angelo__...): o 10 usa
isso para validar com pessoas que ficaram fora do treino. Partes de uma mesma pessoa
("-Parte02", "-Parte-3") ficam juntas.

Rodar de novo não duplica recortes que já foram separados (o nome do arquivo é
procurado em todas as subpastas). Inclua fotos tiradas COM o arco aceso: é nelas
que o classificador precisa acertar.
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
EXTENSOES = {'.jpg', '.jpeg', '.png', '.bmp'}
PASTAS_CLASSE = ['a_classificar', 'abaixada', 'levantada', 'descartar']
MASCARA, PESSOA = 3, 1

p = argparse.ArgumentParser()
p.add_argument('pastas', nargs='*', default=['capturas_epis_limpas'])
p.add_argument('--anotacoes', action='store_true',
               help='usa as caixas do CVAT (YOLO 1.1) em vez do modelo; procura em subpastas')
p.add_argument('--recursivo', action='store_true', help='procura imagens também nas subpastas')
p.add_argument('--modelo', default=str(ROOT / 'weights/epis_v7_best.pt'))
p.add_argument('--conf', type=float, default=0.35, help='confiança mínima da máscara (modo modelo)')
p.add_argument('--imgsz', type=int, default=640)
p.add_argument('--margem', type=float, default=0.35,
               help='margem em volta da caixa, em fração do maior lado (padrão 0.35)')
p.add_argument('--cabeca', action='store_true',
               help='quando a pessoa não tiver máscara, recorta também a região da cabeça')
p.add_argument('--saida', default=str(ROOT / 'datasets/mascara_recortes'))
args = p.parse_args()


def ler_imagem(caminho):
    # np.fromfile + imdecode aceita acentos no caminho (cv2.imread falha no Windows)
    dados = np.fromfile(str(caminho), np.uint8)
    return cv2.imdecode(dados, cv2.IMREAD_COLOR) if dados.size else None


def salvar_imagem(caminho, img):
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if ok:
        buf.tofile(str(caminho))
    return ok


def sessao_de(arq, pasta):
    """Nome da task/pessoa: primeira subpasta sob a pasta indicada, sem o sufixo de parte."""
    rel = arq.relative_to(pasta).parts
    nome = rel[0] if len(rel) > 1 else arq.parent.name
    return re.split(r'[-_ ]parte', nome, flags=re.IGNORECASE)[0]


# Lista as imagens (e, no modo anotações, o .txt correspondente)
itens = []  # (imagem, sessão, txt ou None)
for pasta in map(Path, args.pastas):
    if not pasta.is_dir():
        raise SystemExit(f'Pasta não encontrada: {pasta}')
    recursivo = args.recursivo or args.anotacoes
    arquivos = sorted(f for f in (pasta.rglob('*') if recursivo else pasta.iterdir())
                      if f.is_file() and f.suffix.lower() in EXTENSOES)
    if args.anotacoes:
        rotulos = {}
        for t in pasta.rglob('obj_train_data/*.txt'):
            rel = t.relative_to(pasta).parts
            rotulos[(rel[0] if len(rel) > 1 else '', t.stem)] = t
        sem = 0
        for f in arquivos:
            rel = f.relative_to(pasta).parts
            t = rotulos.get((rel[0] if len(rel) > 1 else '', f.stem))
            if t is None:
                sem += 1
            else:
                itens.append((f, sessao_de(f, pasta), t))
        if sem:
            print(f'{pasta}: {sem} imagens sem anotação foram ignoradas')
    else:
        itens += [(f, sessao_de(f, pasta), None) for f in arquivos]
if not itens:
    raise SystemExit('Nenhuma imagem encontrada (use --recursivo para incluir subpastas)')

saida = Path(args.saida)
for nome in PASTAS_CLASSE:
    (saida / nome).mkdir(parents=True, exist_ok=True)
existentes = {f.name for nome in PASTAS_CLASSE for f in (saida / nome).iterdir()}
destino = saida / 'a_classificar'

if args.anotacoes:
    modelo = None
else:
    import torch
    from ultralytics import YOLO
    DEVICE = 0 if torch.cuda.is_available() else 'cpu'
    modelo = YOLO(args.modelo)
    if modelo.names.get(MASCARA) != 'mascara_soldagem' or modelo.names.get(PESSOA) != 'soldador':
        raise SystemExit(f'Classes do modelo inesperadas: {modelo.names}')


def caixas_anotadas(txt, w, h):
    """Lê um .txt YOLO (classe cx cy bw bh normalizados) e devolve [(classe, xyxy em pixels)]."""
    caixas = []
    for linha in txt.read_text(encoding='utf-8').splitlines():
        v = linha.split()
        if len(v) != 5:
            continue
        c, cx, cy, bw, bh = int(v[0]), *map(float, v[1:])
        caixas.append((c, [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return caixas


def recorte_quadrado(img, cx, cy, lado):
    """Recorta um quadrado centrado em (cx, cy), deslocado para caber na imagem."""
    h, w = img.shape[:2]
    lado = int(min(lado, w, h))
    x1 = int(min(max(0, cx - lado / 2), w - lado))
    y1 = int(min(max(0, cy - lado / 2), h - lado))
    return img[y1:y1 + lado, x1:x1 + lado]


novos = pulados = 0
for i in range(0, len(itens), 16):
    lote = itens[i:i + 16]
    if modelo is not None:
        preditos = modelo.predict([str(f) for f, _, _ in lote], conf=min(args.conf, 0.25), imgsz=args.imgsz,
                                  device=DEVICE, verbose=False)
    for k, (arq, sessao, txt) in enumerate(lote):
        img = ler_imagem(arq)
        if img is None:
            print(f'\nNão foi possível ler {arq}')
            continue
        if modelo is None:
            caixas = caixas_anotadas(txt, img.shape[1], img.shape[0])
            mascaras = [b for c, b in caixas if c == MASCARA]
            pessoas = [b for c, b in caixas if c == PESSOA]
        else:
            r = preditos[k]
            caixas = list(zip(r.boxes.cls.int().tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()))
            mascaras = [b for c, s, b in caixas if c == MASCARA and s >= args.conf]
            pessoas = [b for c, s, b in caixas if c == PESSOA and s >= 0.25]
        # Nome: <sessão>__<foto>_<tipo><n>.jpg (o 10 usa a sessão para separar treino/validação)
        base = f'{sessao}__{arq.stem}'
        recortes = []
        for n, (x1, y1, x2, y2) in enumerate(mascaras):
            lado = max(x2 - x1, y2 - y1) * (1 + 2 * args.margem)
            recortes.append((f'{base}_m{n}.jpg', recorte_quadrado(img, (x1 + x2) / 2, (y1 + y2) / 2, lado)))
        if args.cabeca:
            for n, (x1, y1, x2, y2) in enumerate(pessoas):
                tem_mascara = any(x1 <= (a + c) / 2 <= x2 and y1 <= (b + d) / 2 <= y1 + .45 * (y2 - y1)
                                  for a, b, c, d in mascaras)
                if tem_mascara:
                    continue
                lado = min(x2 - x1, .35 * (y2 - y1)) * 1.2
                recortes.append((f'{base}_c{n}.jpg', recorte_quadrado(img, (x1 + x2) / 2, y1 + lado * .45, lado)))
        for nome, crop in recortes:
            if nome in existentes:
                pulados += 1
                continue
            if crop.size and min(crop.shape[:2]) >= 32 and salvar_imagem(destino / nome, crop):
                existentes.add(nome)
                novos += 1
    print(f'{min(i + 16, len(itens))}/{len(itens)} imagens, {novos} recortes', end='\r')

print(f'\n{novos} recortes novos em {destino.resolve()}')
if pulados:
    print(f'{pulados} já existiam e foram mantidos')
print('Agora mova cada recorte para abaixada/, levantada/ ou descartar/ e rode 10_treinar_mascara.py')
