"""Separa imagens pelo teclado, uma por vez.

Uso:
    python 11_separar_recortes.py                 # recortes de máscara (09)
    python 11_separar_recortes.py --tarefa arco   # quadros inteiros para o detector de arco (12)

Teclas (máscara):
    A  abaixada   (máscara cobrindo o rosto)
    L  levantada  (máscara erguida, rosto exposto)
    D  descartar  (recorte errado, cortado ou duvidoso)
Teclas (arco):
    A  arco       (arco aceso: clarão ou faíscas da solda)
    S  sem_arco   (nenhuma solda acontecendo)
    D  descartar  (duvidoso)
    Espaço  pular (o recorte volta para o fim da fila)
    Z  desfazer o último
    Q / Esc  sair (o progresso já fica salvo: os arquivos são movidos na hora)

Os recortes saem de a_classificar/ e vão para a pasta escolhida. Para revisar uma
pasta já separada, rode com --origem levantada (por exemplo): os recortes daquela
pasta são mostrados de novo e podem ser movidos para outra.
"""
import argparse
import shutil
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
TAREFAS = {  # teclas, pasta, lado padrão na tela
    'mascara': ({ord('a'): 'abaixada', ord('l'): 'levantada', ord('d'): 'descartar'}, 'mascara_recortes', 560),
    'arco': ({ord('a'): 'arco', ord('s'): 'sem_arco', ord('d'): 'descartar'}, 'arco_quadros', 960),
}
CORES = {'abaixada': (80, 200, 80), 'levantada': (60, 60, 230), 'arco': (0, 200, 255),
         'sem_arco': (200, 160, 80), 'descartar': (160, 160, 160)}

p = argparse.ArgumentParser()
p.add_argument('--tarefa', choices=TAREFAS, default='mascara')
p.add_argument('--recortes', default=None, help='pasta com a_classificar/ e as pastas das classes')
p.add_argument('--origem', default='a_classificar', help='pasta de onde os recortes são lidos')
p.add_argument('--tamanho', type=int, default=None, help='maior lado da imagem na tela, em pixels')
args = p.parse_args()
TECLAS, PASTA, LADO = TAREFAS[args.tarefa]
args.recortes = args.recortes or str(ROOT / 'datasets' / PASTA)
args.tamanho = args.tamanho or LADO
JANELA = 'Separar - ' + ' | '.join(f'{chr(k).upper()} {v}' for k, v in TECLAS.items()) + ' | Z desfazer | Q sair'

base = Path(args.recortes)
origem = base / args.origem
if not origem.is_dir():
    raise SystemExit(f'Pasta não encontrada: {origem} (rode o 09 antes)')
for nome in TECLAS.values():
    (base / nome).mkdir(exist_ok=True)

fila = deque(sorted(f for f in origem.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}))
if not fila:
    raise SystemExit(f'Nenhum recorte em {origem}')
total = len(fila)
feitos = []  # pilha de (caminho original, caminho novo) para desfazer


def contagem():
    return {n: sum(1 for _ in (base / n).iterdir()) for n in TECLAS.values()}


def tela(arq, cont):
    dados = np.fromfile(str(arq), np.uint8)
    img = cv2.imdecode(dados, cv2.IMREAD_COLOR) if dados.size else None
    if img is None:
        img = np.zeros((args.tamanho, args.tamanho, 3), np.uint8)
    escala = args.tamanho / max(img.shape[:2])
    img = cv2.resize(img, (round(img.shape[1] * escala), round(img.shape[0] * escala)), interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(img, 0, 0, 0, max(0, 560 - img.shape[1]), cv2.BORDER_CONSTANT)
    barra = np.full((90, img.shape[1], 3), 30, np.uint8)
    feitos_n = total - len(fila)
    cv2.putText(barra, f'{feitos_n}/{total}  ({len(fila)} restantes)', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
    x = 10
    for nome, n in cont.items():
        texto = f'[{nome[0].upper()}] {nome}: {n}'
        cv2.putText(barra, texto, (x, 55), cv2.FONT_HERSHEY_SIMPLEX, .55, CORES[nome], 1, cv2.LINE_AA)
        x += cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, .55, 1)[0][0] + 20
    nome_arq = arq.name.encode('ascii', 'replace').decode()
    cv2.putText(barra, nome_arq[:70], (10, 80), cv2.FONT_HERSHEY_SIMPLEX, .4, (150, 150, 150), 1, cv2.LINE_AA)
    return np.vstack([img, barra])


cv2.namedWindow(JANELA, cv2.WINDOW_AUTOSIZE)
cont = contagem()
while fila:
    atual = fila[0]
    cv2.imshow(JANELA, tela(atual, cont))
    tecla = cv2.waitKey(0) & 0xFF
    if cv2.getWindowProperty(JANELA, cv2.WND_PROP_VISIBLE) < 1:
        break
    tecla = ord(chr(tecla).lower()) if 65 <= tecla <= 90 else tecla
    if tecla in (ord('q'), 27):
        break
    if tecla in TECLAS:
        destino = base / TECLAS[tecla] / atual.name
        if destino != atual:
            shutil.move(str(atual), str(destino))
            cont[TECLAS[tecla]] += 1
            if atual.parent.name in cont:
                cont[atual.parent.name] -= 1
        feitos.append((atual, destino))
        fila.popleft()
    elif tecla == ord(' '):
        fila.rotate(-1)
    elif tecla == ord('z') and feitos:
        original, movido = feitos.pop()
        if movido != original:
            shutil.move(str(movido), str(original))
            cont[movido.parent.name] -= 1
            if original.parent.name in cont:
                cont[original.parent.name] += 1
        fila.appendleft(original)

cv2.destroyAllWindows()
print(f'Separados nesta sessão: {len(feitos)}. Restam {len(fila)} em {origem}')
print(', '.join(f'{n}: {c}' for n, c in contagem().items()))
if not fila:
    print(f'Tudo separado. Próximo passo: 10_treinar_mascara.py --tarefa {args.tarefa} --empacotar')
