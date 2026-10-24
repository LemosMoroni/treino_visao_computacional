# 🖐️ Mãos que Desenham: YOLO Pose + OpenCV

Projeto de estudo de **visão computacional** e **treinamento de modelos**: você treina um modelo YOLO para encontrar os
pontos da sua mão pela webcam e o usa para **desenhar no ar**. Faça uma pinça (polegar + indicador) para desenhar. Quando
você solta, o programa reconhece se o traço era um **círculo, triângulo, quadrado, retângulo, polígono ou linha** e o
redesenha perfeito.

O guia foi feito para **qualquer hardware**: GPU NVIDIA, Mac com chip Apple, GPU AMD ou só CPU. Onde o caminho muda,
há instruções separadas.

> **Como estudar com este guia:** não copie tudo de uma vez. Faça uma parte por vez, rode, observe e só depois
> avance. Cada parte tem uma seção **"Entendendo"** com a teoria e uma seção **"✅ Checkpoint"** com o que você
> deve conseguir ver ao terminar.

---

## Sumário

- [Visão geral do sistema](#visão-geral-do-sistema)
- [Parte 0: Conceitos: o que é o YOLO](#parte-0-conceitos-o-que-é-o-yolo)
- [Parte 1: Ambiente (adaptado ao seu hardware)](#parte-1-ambiente-adaptado-ao-seu-hardware)
- [Parte 2: Primeiro contato: YOLO pronto na webcam](#parte-2-primeiro-contato-yolo-pronto-na-webcam)
- [Parte 3: Dados: como uma máquina "lê" um rótulo](#parte-3-dados-como-uma-máquina-lê-um-rótulo)
- [Parte 4: Treinamento (fine-tuning)](#parte-4-treinamento-fine-tuning)
- [Parte 5: Avaliação: o modelo ficou bom?](#parte-5-avaliação-o-modelo-ficou-bom)
- [Parte 6: App de desenho com a mão](#parte-6-app-de-desenho-com-a-mão)
- [Parte 7: Reconhecendo formas geométricas](#parte-7-reconhecendo-formas-geométricas)
- [Parte 8 (bônus): Seu próprio dataset](#parte-8-bônus-seu-próprio-dataset)
- [Solução de problemas](#solução-de-problemas)
- [Glossário](#glossário)
- [Referências](#referências)

### Estrutura final do projeto

```
maos-yolo/
├── README.md
├── config.py               ← detecta seu hardware (GPU NVIDIA / Apple / CPU)
├── maos.yaml               ← Parte 3: "ficha" do dataset (com uma correção importante)
├── 01_webcam_pose.py       ← Parte 2: YOLO pronto na webcam
├── 02_explorar_dataset.py  ← Parte 3: abre o dataset e desenha os rótulos
├── 03_treinar.py           ← Parte 4: treina o modelo de mãos
├── 04_avaliar.py           ← Parte 5: mede a qualidade do modelo
├── 05_desenhar.py          ← Partes 6 e 7: o app final
├── 06_capturar_fotos.py    ← Parte 8: fotos para seu próprio dataset
├── runs/                   ← (gerado) resultados dos treinos
└── .venv/                  ← (gerado) ambiente virtual Python
```

---

## Visão geral do sistema

```
 ┌─────────┐   frame    ┌──────────────────┐  21 pontos  ┌────────────────┐  traço   ┌──────────────────┐
 │ Webcam  │──────────► │ YOLO pose (mão)  │───────────► │ Lógica do app  │────────► │ Reconhecimento   │
 │ OpenCV  │            │ treinado por você│  (x, y)     │ pinça = caneta │          │ de formas        │
 └─────────┘            └──────────────────┘             └────────────────┘          │ (OpenCV, sem IA) │
                                                                                     └────────┬─────────┘
                                                              tela com o desenho ◄────────────┘
```

Este projeto usa **dois tipos de "visão"**:

1. **Aprendizado de máquina (YOLO):** um modelo *aprendeu com exemplos* a encontrar mãos e dedos. Ninguém
   programou uma regra do tipo "dedo é uma coisa comprida cor de pele".
2. **Visão computacional clássica (OpenCV):** regras geométricas *escritas por nós* para decidir se um traço é um
   círculo ou um quadrado.

Saber quando usar cada um é uma das habilidades mais importantes da área.

---

## Parte 0: Conceitos: o que é o YOLO

### O nome

**YOLO = You Only Look Once** ("você só olha uma vez"). Detectores antigos analisavam a imagem várias vezes, região por
região. O YOLO passa a imagem **uma única vez** por uma rede neural e devolve todas as previsões juntas. Por isso ele é
rápido o suficiente para vídeo em tempo real.

### A arquitetura em 3 partes

```
Imagem ──► [ Backbone ] ──► [ Neck ] ──► [ Head ] ──► previsões
           extrai padrões   combina      gera caixas,
           (bordas →        escalas      classes, pontos
           texturas →       (objetos
           partes →         grandes e
           objetos)         pequenos)
```

- **Backbone:** camadas convolucionais que transformam pixels em "características". As primeiras camadas detectam
  bordas e cores. As mais profundas detectam coisas como "dedo" ou "olho".
- **Neck:** mistura as características de várias resoluções. Assim o modelo enxerga tanto uma mão perto da câmera
  (grande) quanto uma mão longe (pequena).
- **Head:** a "cabeça" que dá a resposta final: onde está o objeto, que classe ele é e, no nosso caso, onde estão os
  pontos-chave.

### A biblioteca Ultralytics e suas tarefas

A **Ultralytics** é a biblioteca Python mais usada para YOLO. A versão atual dos modelos é a **YOLO26**. A mesma
biblioteca faz várias tarefas:

| Tarefa | O que devolve | Modelo | Exemplo de uso |
|---|---|---|---|
| `detect` | caixa + classe | `yolo26n.pt` | contar carros |
| `segment` | contorno exato (máscara) | `yolo26n-seg.pt` | recortar pessoas do fundo |
| **`pose`** | **caixa + pontos-chave** | **`yolo26n-pose.pt`** | **postura, mãos ← nosso caso** |
| `classify` | uma classe para a imagem toda | `yolo26n-cls.pt` | "é gato ou cachorro?" |
| `obb` | caixa girada | `yolo26n-obb.pt` | imagens de satélite |

A letra após o número é o **tamanho**: `n` (nano) < `s` < `m` < `l` < `x`. Modelos maiores são mais precisos e mais
lentos. Vamos usar o **`n`**, que roda bem em qualquer computador.

> Se sua versão da Ultralytics for mais antiga e não encontrar `yolo26n-pose.pt`, atualize com
> `pip install -U ultralytics` ou troque por `yolo11n-pose.pt`. O resto do código é idêntico.

### Por que *pose* e não *detect*?

Para desenhar, não basta saber **onde está a mão** (uma caixa). Precisamos saber **onde está a ponta de cada dedo**. O
modelo de pose devolve **21 pontos-chave** por mão:

```
         8   12  16  20     ← pontas dos dedos
         │   │   │   │
    4    7   11  15  19
     ╲   │   │   │   │
      3  6   10  14  18
       ╲ │   │   │   │
        2 5───9───13──17
         ╲           ╱
          1         ╱
           ╲       ╱
             0            ← pulso

 0 pulso | 1-4 polegar | 5-8 indicador | 9-12 médio | 13-16 anelar | 17-20 mindinho
```

### Treinar do zero × Fine-tuning (transfer learning)

O `yolo26n-pose.pt` original foi treinado para o **corpo humano**: 17 pontos (nariz, ombros, cotovelos, joelhos…).
Ele **não conhece dedos**.

- **Treinar do zero** exigiria milhões de imagens e dias de GPU.
- **Fine-tuning:** partimos de um modelo que já "sabe enxergar" (bordas, formas, pele, partes do corpo) e o ensinamos
  só a tarefa nova. O backbone reaproveita quase tudo. A head é trocada para prever 21 pontos em vez de 17.

É assim que quase todo projeto real de visão computacional é feito.

### O ciclo de vida de um projeto de ML

```
 Dados ──► Treino ──► Avaliação ──► Inferência (uso real)
   ▲                      │
   └──── melhorar dados ◄─┘   (quase sempre é aqui que se ganha qualidade)
```

---

## Parte 1: Ambiente (adaptado ao seu hardware)

### 1.1 Descubra seu hardware

| Seu computador | Como verificar | `device` usado | Onde treinar | Perfil de treino |
|---|---|---|---|---|
| GPU NVIDIA com ≥ 8 GB | `nvidia-smi` mostra a placa | `0` | local | `gpu_grande` |
| GPU NVIDIA com 4–8 GB (ex.: RTX 3050/3060 laptop) | `nvidia-smi` | `0` | local | `gpu_media` |
| GPU NVIDIA com < 4 GB | `nvidia-smi` | `0` | local (reduzido) ou Colab | `gpu_pequena` |
| Mac com chip M1/M2/M3/M4 | Menu Apple → Sobre este Mac | `"mps"` | local | `apple` |
| GPU AMD no **Linux** (ROCm) | `rocm-smi` | `0` | local | `gpu_media` |
| GPU AMD/Intel no Windows, Mac Intel, ou só CPU | — | `"cpu"` | **Google Colab** (grátis) ou CPU reduzido | `cpu` |

No Windows, abra o PowerShell e rode `nvidia-smi`. Se aparecer uma tabela com o nome da placa, você tem GPU NVIDIA.

> **Por que a GPU importa?** Uma rede neural faz milhões de multiplicações independentes, e a GPU tem milhares de
> núcleos pequenos que as fazem em paralelo. Treinar na GPU costuma ser de 10 a 50 vezes mais rápido que na CPU. Já
> para **usar** o modelo nano (inferência), a CPU dá conta.

### 1.2 Crie a pasta do projeto

⚠️ **Use um caminho sem acentos e fora de pastas sincronizadas (OneDrive, iCloud, Dropbox).**
- No Windows, o OpenCV (`cv2.imread`) **falha sem avisar** quando há acentos no caminho (ex.: `TesteMãos`).
- O serviço de sincronização tentaria enviar gigabytes de dataset e resultados de treino para a nuvem.

**Windows (PowerShell):**
```powershell
mkdir C:\projetos\maos-yolo
cd C:\projetos\maos-yolo
```

**Linux / macOS:**
```bash
mkdir -p ~/projetos/maos-yolo
cd ~/projetos/maos-yolo
```

Copie este `README.md` para dentro dessa pasta.

### 1.3 Crie um ambiente virtual (venv)

Um **venv** é uma "caixa" isolada de bibliotecas só para este projeto, para não misturar versões com outros projetos.
Use **Python 3.10 a 3.12**: as bibliotecas de IA costumam demorar a dar suporte completo às versões mais novas.

**Windows:**
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```
> Se aparecer "a execução de scripts foi desabilitada", rode uma vez
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` e ative de novo.

**Linux / macOS:**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python --version
```

Com o venv ativo, o terminal mostra `(.venv)` no início da linha. **Ative o venv sempre que abrir um terminal novo
para este projeto.**

### 1.4 Instale o PyTorch para o SEU hardware

O **PyTorch** é o "motor" matemático por baixo do YOLO.

⚠️ **A ordem importa:** instale o PyTorch **antes** da Ultralytics. Se a Ultralytics vier primeiro, ela instala a
versão padrão do PyTorch, que no Windows é **só para CPU**, e sua GPU fica sem uso.

```bash
python -m pip install --upgrade pip
```

Depois, **escolha uma** das linhas abaixo:

| Hardware | Comando |
|---|---|
| NVIDIA (Windows/Linux) | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128` |
| Mac (Apple Silicon ou Intel) | `pip install torch torchvision` |
| Só CPU (Windows/Linux) | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |
| AMD no Linux (ROCm) | use o comando "ROCm" gerado em [pytorch.org/get-started](https://pytorch.org/get-started/locally/) |

> `cu128` = CUDA 12.8, que funciona com drivers NVIDIA recentes. Se a sua placa ou driver for antigo, gere o comando
> certo em [pytorch.org/get-started](https://pytorch.org/get-started/locally/) e atualize o driver NVIDIA.

### 1.5 Instale a Ultralytics

```bash
pip install ultralytics
```

Isso também instala o **OpenCV** (`cv2`, para câmera e desenho), o **NumPy** (contas com matrizes) e o comando `yolo`
no terminal.

### 1.6 Avise à Ultralytics onde guardar datasets

Por padrão, a Ultralytics escolhe a pasta de datasets na primeira vez que roda, e ela pode acabar dentro do OneDrive.
Defina a pasta você mesmo:

```bash
# Windows
yolo settings datasets_dir=C:/projetos/datasets
# Linux / macOS
yolo settings datasets_dir=~/projetos/datasets
```

Para ver todas as configurações: `yolo settings`.

### 1.7 Crie o `config.py`

Todos os scripts importam este arquivo, que escolhe o hardware **automaticamente**. É ele que faz o projeto rodar em
qualquer máquina.

```python
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
        raise SystemExit("Nenhum modelo treinado em runs/. Faça a Parte 4 ou use --modelo caminho/best.pt")
    return str(pesos[-1])


DEVICE = escolher_device()
# Tamanho da imagem na inferência em tempo real: menor = mais rápido, porém menos preciso
IMGSZ_INFERENCIA = 320 if DEVICE == "cpu" else 640


if __name__ == "__main__":
    print("PyTorch:", torch.__version__)
    print("Hardware escolhido:", descrever_hardware())
```

### ✅ Checkpoint 1

```bash
python config.py
yolo checks
```

O resultado esperado é algo como `Hardware escolhido: GPU NVIDIA GeForce RTX 3050 6GB Laptop GPU (6.0 GB)`,
`Apple Silicon (MPS)` ou `CPU (sem aceleração)`, conforme a sua máquina.

Se você **tem** GPU NVIDIA e apareceu "CPU", o PyTorch foi instalado errado. Veja
[Solução de problemas](#solução-de-problemas).

---

## Parte 2: Primeiro contato: YOLO pronto na webcam

### 2.1 Pela linha de comando

Rode um comando de cada vez. Para fechar, use `q` na janela ou `Ctrl+C` no terminal.

```bash
yolo predict model=yolo26n.pt source=0 show=True
yolo predict model=yolo26n-pose.pt source=0 show=True
```

- `model=`: qual modelo usar. Na primeira vez ele é baixado automaticamente (poucos MB).
- `source=0`: a primeira webcam. Também aceita um arquivo (`foto.jpg`, `video.mp4`), uma pasta ou uma URL.
- `show=True`: abre uma janela com o resultado.
- Sem GPU, acrescente `imgsz=320` para ficar mais fluido.

**Observe:** o primeiro modelo desenha caixas com nomes (person, cell phone, cup…). O segundo desenha um **esqueleto do
corpo**, mas **não marca os dedos**. É isso que vamos resolver treinando.

### 2.2 Pelo Python: `01_webcam_pose.py`

A linha de comando é ótima para testar, mas no app precisamos dos **números**. Crie o arquivo:

```python
"""Parte 2: primeiro contato com o YOLO em Python, usando o modelo de pose do CORPO."""
import cv2
from ultralytics import YOLO

from config import DEVICE, IMGSZ_INFERENCIA

modelo = YOLO("yolo26n-pose.pt")  # baixado automaticamente na primeira vez
camera = cv2.VideoCapture(0)

while True:
    ok, frame = camera.read()
    if not ok:
        print("Não consegui ler a câmera.")
        break

    resultados = modelo.predict(frame, imgsz=IMGSZ_INFERENCIA, device=DEVICE, verbose=False)
    r = resultados[0]  # um objeto Results por imagem; mandamos só uma

    # r.boxes     -> caixas: r.boxes.xyxy (cantos), r.boxes.conf (confiança), r.boxes.cls (classe)
    # r.keypoints -> pontos: r.keypoints.xy tem formato (n_pessoas, 17, 2)
    pontos = r.keypoints.xy.cpu().numpy()
    if len(pontos) > 0:
        nariz = pontos[0][0]  # pessoa 0, ponto 0 (no modelo de corpo, o ponto 0 é o nariz)
        print(f"Pessoas: {len(pontos)} | nariz da 1ª em x={nariz[0]:.0f}, y={nariz[1]:.0f}")

    cv2.imshow("YOLO pose (q = sair)", r.plot())
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
```

```bash
python 01_webcam_pose.py
```

### Entendendo

- **Um vídeo é uma sequência de imagens (frames).** O `while True` lê um frame, roda o modelo, mostra o resultado e
  repete.
- **Uma imagem é uma matriz de números.** `frame` é um array NumPy de formato `(altura, largura, 3)`. Cada pixel tem 3
  valores de 0 a 255 na ordem **B, G, R**: o OpenCV usa BGR, não RGB.
- **`predict` devolve uma lista de `Results`**, um para cada imagem enviada. Dentro dele:
  - `r.boxes.xyxy`: cantos das caixas em pixels; `r.boxes.conf`: confiança de 0 a 1.
  - `r.keypoints.xy`: **tensor** de formato `(N, K, 2)`, com N objetos, K pontos por objeto e (x, y) de cada ponto.
  - `r.keypoints.conf`: confiança de cada ponto.
- **Tensor** é o "array" do PyTorch. Ele pode estar na GPU, por isso usamos `.cpu().numpy()` antes de usar o valor
  com NumPy/OpenCV.
- **`conf` (limiar de confiança):** o modelo dá uma nota a cada detecção, e as notas abaixo do limiar são descartadas
  (o padrão é 0,25). Teste `conf=0.7` no `predict` e veja detecções sumirem.
- **`imgsz`:** a imagem é redimensionada antes de entrar na rede. 640 é mais preciso; 320 é cerca de 4 vezes mais
  rápido.

### ✅ Checkpoint 2

O esqueleto do corpo aparece sobre você, o terminal imprime a posição do nariz, e ela muda quando você mexe a cabeça.
Anote o FPS aproximado do seu computador.

---

## Parte 3: Dados: como uma máquina "lê" um rótulo

> "Um modelo é tão bom quanto os dados com que foi treinado." Esta é a parte mais importante para entender de verdade.

### 3.1 O dataset Hand Keypoints

Vamos usar o [Hand Keypoints](https://docs.ultralytics.com/datasets/pose/hand-keypoints) da Ultralytics:

- **26.768 imagens** de mãos: **18.776 de treino** e **7.992 de validação**;
- **1 classe** (`hand`) e **21 pontos por mão**;
- os rótulos foram gerados com o MediaPipe, do Google;
- download de cerca de **370 MB**, feito automaticamente.

### 3.2 O arquivo YAML: a "ficha" do dataset

Todo dataset YOLO é descrito por um arquivo YAML. A Ultralytics já traz um `hand-keypoints.yaml` pronto, mas **vamos
criar o nosso, `maos.yaml`**, por causa de um detalhe explicado logo abaixo. Crie o arquivo na pasta do projeto:

```yaml
# Cópia do hand-keypoints.yaml da Ultralytics, com o flip_idx corrigido.
# Espelhar uma mão não troca os dedos de lugar: a ponta do indicador continua sendo o ponto 8.
path: hand-keypoints # pasta do dataset, dentro do datasets_dir
train: images/train # 18776 imagens de treino
val: images/val # 7992 imagens de validação

kpt_shape: [21, 3] # 21 pontos, cada um com x, y, visibilidade
flip_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

names:
  0: hand

download: https://github.com/ultralytics/assets/releases/download/v0.0.0/hand-keypoints.zip
```

| Campo | Significado |
|---|---|
| `path` | pasta raiz do dataset. Se for relativa e não existir a partir da pasta atual, é procurada dentro do `datasets_dir` (Parte 1.6) |
| `train` / `val` | imagens de **treino** (o modelo aprende com elas) e de **validação** (só para medir; o modelo nunca aprende com elas) |
| `kpt_shape` | quantos pontos por objeto (21) e quantos valores por ponto (3: x, y, visibilidade) |
| `flip_idx` | ao espelhar a imagem, qual ponto vira qual (veja abaixo) |
| `names` | id da classe → nome |
| `download` | de onde baixar se a pasta não existir |

**Por que separar treino e validação?** Para medir se o modelo **generaliza**. Um aluno que decorou as respostas da
lista de exercícios vai bem na lista, mas mal na prova. A validação é a "prova" com questões que ele nunca viu.

#### O `flip_idx` e por que não usamos o YAML oficial

Durante o treino, metade das imagens é espelhada horizontalmente (parâmetro `fliplr=0.5`, um tipo de *data
augmentation*). Quando a imagem é espelhada, os rótulos são reordenados assim: **o ponto `i` novo recebe o ponto
`flip_idx[i]` antigo**.

- **No corpo humano** (dataset COCO), isso é necessário. Espelhada, a foto mostra o olho esquerdo onde estava o
  direito, por isso o `coco-pose.yaml` troca os pares esquerda ↔ direita: `[0, 2, 1, 4, 3, 6, 5, ...]`.
- **Numa mão**, espelhar não troca nenhum dedo de lugar: a ponta do indicador continua sendo a ponta do indicador.
  O correto é a **identidade**: `[0, 1, 2, ..., 20]`.

O `hand-keypoints.yaml` que vem com a Ultralytics (conferido na versão 8.4.188) traz este valor:

```
flip_idx: [0, 1, 2, 4, 3, 10, 11, 12, 13, 14, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20]
```

Com ele, em toda imagem espelhada, o rótulo do **ponto 8 (ponta do indicador)** receberia a posição do ponto 13 (base
do anelar), e o do **ponto 4 (ponta do polegar)** receberia a do ponto 3. Esses são justamente os dois pontos que nosso
app usa. Por isso criamos o `maos.yaml` com a identidade.

> 💡 **Lição de ML:** arquivos de configuração "oficiais" também podem ter erros. Antes de treinar, **leia o YAML** e
> pergunte se cada valor faz sentido para o seu problema. Você pode conferir o original com
> `python -c "import ultralytics, pathlib; print((pathlib.Path(ultralytics.__file__).parent / 'cfg/datasets/hand-keypoints.yaml').read_text(encoding='utf-8'))"`.
> Se a versão que você instalou já estiver corrigida, ótimo: o `maos.yaml` continua correto.

### 3.3 Estrutura de pastas

```
datasets/hand-keypoints/
├── images/
│   ├── train/  IMG_0001.jpg  IMG_0002.jpg ...
│   └── val/    ...
└── labels/
    ├── train/  IMG_0001.txt  IMG_0002.txt ...   ← mesmo nome da imagem, extensão .txt
    └── val/    ...
```

A regra é simples: **para cada imagem existe um `.txt` com o mesmo nome**, na pasta `labels` equivalente.

### 3.4 O formato do rótulo (pose)

Cada linha do `.txt` é **um objeto** (uma mão):

```
<classe> <cx> <cy> <w> <h>  <x0> <y0> <v0>  <x1> <y1> <v1>  ...  <x20> <y20> <v20>
   │      └──── caixa ───┘   └─────── 21 pontos × (x, y, visibilidade) ────────┘
   │
   └ 0 = hand
```

São 1 + 4 + 21 × 3 = **68 números por linha**.

- **Coordenadas normalizadas de 0 a 1:** `cx = 0.5` significa "no meio da largura", qualquer que seja a resolução. Para
  converter para pixels: `x_pixel = x * largura_da_imagem`.
- **A caixa** é dada pelo **centro** (`cx`, `cy`) e pelo tamanho (`w`, `h`), não pelos cantos.
- **Visibilidade `v`:** `0` = não anotado, `1` = anotado mas escondido (oculto), `2` = visível.

### 3.5 Explore o dataset: `02_explorar_dataset.py`

Este script baixa o dataset, se ainda não existir, e **desenha os rótulos lendo o `.txt` linha a linha**, sem usar
nenhuma função "mágica". Assim você vê que o rótulo é só uma lista de números.

```python
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
```

```bash
python 02_explorar_dataset.py
```

### Entendendo: data augmentation

Durante o treino, a Ultralytics **modifica as imagens aleatoriamente** a cada época: espelha, muda brilho, cor e
saturação, aumenta, diminui, desloca, e junta 4 imagens numa só (*mosaic*). Com isso o modelo vê "imagens novas" a
cada volta e aprende a reconhecer mãos em condições diferentes, em vez de decorar as fotos.

Os pontos são transformados **junto** com a imagem. É por isso que o `flip_idx` precisa estar certo.

### ✅ Checkpoint 3

Uma janela mostra fotos de mãos com caixa verde, esqueleto e pontos numerados, e o terminal mostra `68 números` por
mão. Confira que o **ponto 8 está na ponta do indicador**.

---

## Parte 4: Treinamento (fine-tuning)

### 4.1 O que acontece durante o treino

```
Para cada ÉPOCA (uma volta completa no dataset de treino):
    Para cada LOTE (batch) de imagens:
        1. Forward:       o modelo faz previsões para o lote
        2. Loss:          compara previsões × rótulos e calcula "o tamanho do erro"
        3. Backward:      calcula o quanto cada peso da rede contribuiu para o erro (gradientes)
        4. Otimizador:    ajusta levemente cada peso na direção que diminui o erro
    Validação: mede o desempenho em imagens que o modelo nunca usou para aprender
    Se for o melhor resultado até agora: salva best.pt
```

O modelo tem alguns **milhões de números (pesos)**. "Aprender" é ajustar esses números para que a loss fique pequena.

### 4.2 O script `03_treinar.py`

O script escolhe sozinho um **perfil** adequado ao seu hardware. Você pode forçar outro com `--perfil`.

```python
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
```

### 4.3 Primeiro, uma rodada de TESTE

Antes de deixar o computador horas treinando, confira se tudo funciona e **meça quanto tempo leva**:

```bash
python 03_treinar.py --perfil teste
```

Esse teste usa 2% do dataset por 2 épocas. O resultado vai ser ruim, e isso é esperado: o objetivo é só ver o
processo rodar até o fim e criar `runs/maos_teste/weights/best.pt`.

**Calcule o tempo do treino de verdade:** veja quanto tempo levou cada época e faça
`tempo_por_época × (fração do perfil ÷ 0,02) × épocas`. Esse cálculo mostra se vale treinar localmente ou no Colab.

### 4.4 O treino de verdade

```bash
python 03_treinar.py                   # perfil escolhido pelo hardware
python 03_treinar.py --perfil gpu_media --epocas 40   # ou escolha você
```

**Se o treino for interrompido** (queda de energia, fechou o terminal), continue de onde parou:

```bash
yolo train resume model=runs/maos_gpu_media/weights/last.pt
```

### Entendendo os parâmetros

| Parâmetro | O que é | Efeito de aumentar |
|---|---|---|
| `epochs` | quantas voltas completas no dataset | aprende mais, mas demora mais e pode decorar (overfitting) |
| `batch` | quantas imagens por passo de ajuste | usa mais memória da GPU; gradiente mais estável |
| `imgsz` | lado da imagem que entra na rede | mais detalhes (dedos pequenos), mas mais lento e usa mais memória |
| `fraction` | parte do dataset de treino usada | mais dados = modelo melhor, treino mais longo |
| `workers` | processos lendo imagens do disco em paralelo | alimenta a GPU mais rápido, mas usa mais RAM |
| `patience` | épocas sem melhora antes de parar (*early stopping*) | — |
| `device` | onde calcular: `0`, `"mps"` ou `"cpu"` | — |

Outros parâmetros que aparecem no log e valem conhecer: `lr0` (*learning rate*, o tamanho do "passo" de ajuste),
`warmup_epochs` (começa com passos pequenos), `close_mosaic` (desliga o mosaic nas últimas épocas), `amp`
(*mixed precision*: contas em 16 bits na GPU NVIDIA, mais rápidas e com menos memória) e `batch=-1` (a Ultralytics
escolhe o batch sozinha, só em GPU NVIDIA).

**Faltou memória de GPU (`CUDA out of memory`)?** Reduza o `batch` pela metade. Se não bastar, reduza o `imgsz`.

### Entendendo o log

```
      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss  ...  Instances       Size
      12/80      2.31G      1.024      3.871     0.4120     0.6532  ...         38        640
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95)
                   all       7992      ...
```

- **Losses (erros), que devem cair ao longo das épocas:**
  - `box_loss`: erro na posição da caixa;
  - `pose_loss`: erro na posição dos pontos;
  - `kobj_loss`: erro ao prever se cada ponto está visível;
  - `cls_loss`: erro de classe (aqui só existe "hand", então ela cai rápido).
  - Os nomes exatos variam um pouco entre versões. O que importa é a **tendência de queda**.
- **Box(…)** e **Pose(…)**: métricas de validação da caixa e dos pontos (explicadas na Parte 5).
- Na primeira linha do treino aparece `Transferred X/Y items from pretrained weights`. Esse é o **transfer learning**
  acontecendo: as camadas compatíveis foram copiadas do modelo de corpo e a head nova (21 pontos) começa do zero.

### 4.5 Sem GPU? Treine de graça no Google Colab

O Colab é um Jupyter Notebook na nuvem com GPU gratuita (com limite de uso diário). Kaggle Notebooks é uma
alternativa parecida.

1. Acesse [colab.research.google.com](https://colab.research.google.com) → **Novo notebook**.
2. Menu **Ambiente de execução → Alterar o tipo de ambiente de execução → GPU T4** → Salvar.
3. Cole e rode as células:

```python
# Célula 1: instala e salva os resultados no seu Google Drive (para não perder se a sessão cair)
!pip install -q ultralytics
from google.colab import drive
drive.mount("/content/drive")
```

```python
%%writefile maos.yaml
# Célula 2: cria o maos.yaml no Colab (o mesmo conteúdo da Parte 3.2)
path: hand-keypoints
train: images/train
val: images/val
kpt_shape: [21, 3]
flip_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
names:
  0: hand
download: https://github.com/ultralytics/assets/releases/download/v0.0.0/hand-keypoints.zip
```

```python
# Célula 3: treina
from ultralytics import YOLO

modelo = YOLO("yolo26n-pose.pt")
modelo.train(
    data="maos.yaml",
    epochs=60, imgsz=640, batch=32, device=0, patience=15,
    project="/content/drive/MyDrive/maos-yolo", name="maos_colab",
)
```

```python
# Célula 4 (só se a sessão cair no meio): continua de onde parou
modelo = YOLO("/content/drive/MyDrive/maos-yolo/maos_colab/weights/last.pt")
modelo.train(resume=True)
```

4. No final, baixe `MyDrive/maos-yolo/maos_colab/weights/best.pt` e coloque no seu projeto em
   `runs/maos_colab/weights/best.pt`. Os scripts das partes seguintes encontram esse arquivo sozinhos.

> A inferência do modelo nano roda bem em CPU. Treinar na nuvem e usar no seu computador é um fluxo muito comum.

### ✅ Checkpoint 4

Existe o arquivo `runs/maos_<perfil>/weights/best.pt`, e as losses caíram ao longo das épocas.

---

## Parte 5: Avaliação: o modelo ficou bom?

### 5.1 Olhe os gráficos

Abra a pasta `runs/maos_<perfil>/`:

| Arquivo | O que mostra |
|---|---|
| `results.png` | curvas de loss e métricas por época (**o mais importante**) |
| `results.csv` | os mesmos números em tabela |
| `val_batch0_labels.jpg` × `val_batch0_pred.jpg` | rótulo verdadeiro × previsão, lado a lado |
| `train_batch0.jpg` | como ficaram as imagens depois do augmentation (repare no mosaic) |
| `weights/best.pt` | pesos da melhor época na validação: **use este** |
| `weights/last.pt` | pesos da última época (serve para continuar o treino) |

### 5.2 Como ler as curvas: overfitting

```
 loss                                    loss
  │╲  treino                              │╲
  │ ╲___                                  │ ╲      ___── validação   ← sobe!
  │  ╲  ‾‾‾───___                         │  ╲__──
  │   ╲__ validação ‾‾──                  │   ╲___
  │      ‾‾‾───___                        │       ‾‾‾───___ treino
  └──────────────── épocas                └──────────────── épocas
     ✅ saudável: as duas caem              ⚠️ overfitting: decorou o treino
```

**Overfitting** acontece quando o modelo "decora" as imagens de treino: a loss de treino continua caindo, mas a de
validação para de cair ou sobe. Para evitar: mais dados, mais augmentation, menos épocas (o `patience` ajuda) ou um
modelo menor.

### 5.3 As métricas

- **Precisão (P):** de tudo que o modelo disse que é mão, quanto era mão de verdade? (poucos alarmes falsos)
- **Recall (R):** de todas as mãos que existiam, quantas ele encontrou? (poucas mãos perdidas)
- **IoU (Intersection over Union):** o quanto a caixa prevista se sobrepõe à verdadeira (0 = nada, 1 = perfeito).
- **OKS (Object Keypoint Similarity):** a versão do IoU para **pontos**. Mede o quão perto os pontos previstos estão
  dos verdadeiros, levando em conta o tamanho da mão.
- **mAP50:** precisão média considerando "acerto" quando IoU/OKS ≥ 0,5. É uma régua mais tolerante.
- **mAP50-95:** a média do mAP com régua de 0,50 até 0,95. É mais exigente e é **a métrica principal** para comparar
  modelos.

O resultado aparece em duas versões: **Box** (a caixa da mão) e **Pose** (os pontos). Para desenhar, a que mais
importa é a **Pose**.

### 5.4 Script de avaliação: `04_avaliar.py`

```python
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
```

```bash
python 04_avaliar.py
```

### 5.5 O teste que importa: a SUA mão

Métricas em dataset não garantem que o modelo funciona **na sua webcam, no seu quarto, com a sua luz**. Teste:

```bash
yolo predict model=runs/maos_gpu_media/weights/best.pt source=0 show=True
```

(troque `maos_gpu_media` pela pasta do seu treino)

Teste mão perto e longe, de lado, com pouca luz, com fundo bagunçado. Onde ele erra, os dados de treino tinham
poucos exemplos parecidos. É aí que entra a [Parte 8](#parte-8-bônus-seu-próprio-dataset).

### 5.6 (Opcional) Acelere a inferência na CPU: exportação

Um `.pt` é um modelo PyTorch, ótimo para treinar. Para **usar**, dá para convertê-lo para formatos otimizados:

```bash
# Intel (a maioria dos notebooks sem GPU NVIDIA): OpenVINO costuma ser o mais rápido
yolo export model=runs/maos_cpu/weights/best.pt format=openvino imgsz=320
# Qualquer CPU: ONNX
yolo export model=runs/maos_cpu/weights/best.pt format=onnx imgsz=320
```

Depois, rode o app com `--modelo runs/maos_cpu/weights/best_openvino_model` (ou o arquivo `best.onnx` gerado).

### ✅ Checkpoint 5

Você sabe dizer se seu modelo teve overfitting, qual é o mAP50-95 da Pose, e ele encontra os 21 pontos da sua mão
pela webcam.

---

## Parte 6: App de desenho com a mão

Agora juntamos tudo. Esta parte e a próxima formam um único arquivo, o `05_desenhar.py` (código completo no fim da
Parte 7). Antes do código, as ideias:

### 6.1 Espelhar a imagem

A webcam vê você "de frente", como outra pessoa veria. Sem espelhar, mover a mão para a direita moveria o cursor para a
esquerda. `cv2.flip(frame, 1)` resolve isso.

### 6.2 A "caneta" e o gesto de pinça

```
 desenhando (pinça fechada)        parado (pinça aberta)
        8  ●  4                       8 ●
          ╲╱  ← caneta = ponto médio    │       ● 4
         ╱  ╲                           │      ╱
```

- **Caneta:** o **ponto médio** entre o polegar (4) e o indicador (8). Quando os dedos se separam, o ponto médio quase
  não se mexe, e o fim do traço não "pula".
- **Pinça:** medimos a distância entre os pontos 4 e 8. Só que 50 pixels é "pinça fechada" com a mão longe da câmera e
  "aberta" com ela perto. Por isso **normalizamos**: dividimos pela distância pulso (0) → base do dedo médio (9), que é
  o "tamanho da mão" na imagem.

  ```
  abertura = distância(4, 8) / distância(0, 9)      → ~0,1-0,25 em pinça, ~0,8-1,2 com a mão aberta
  ```

### 6.3 Histerese: evitando o "pisca-pisca"

Os pontos tremem um pouco a cada frame. Com um limiar só (por exemplo, desenhar se `abertura < 0,35`), um valor
oscilando entre 0,34 e 0,36 ligaria e desligaria o desenho sem parar. A solução é usar **dois limiares**:

```
 abertura:  0.0 ─────── 0.30 ─────── 0.45 ─────── 1.0
                 │ começa a desenhar   │ para de desenhar
                 └─ quando cai abaixo  └─ só quando sobe acima
```

Entre 0,30 e 0,45, o estado anterior continua valendo. Termostatos usam a mesma ideia.

### 6.4 Suavização (média móvel exponencial)

```
caneta = 0.5 × caneta_anterior + 0.5 × posição_nova
```

Isso filtra o tremor, mas cria um pequeno atraso. Com `SUAVIZACAO` perto de 0 o traço fica nervoso; perto de 1 ele
fica liso, mas "atrasado".

### 6.5 Tolerância a falhas

Às vezes o modelo perde a mão em um único frame (borrão de movimento). Se o traço fosse encerrado nesse instante, o
desenho quebraria. Por isso só encerramos depois de `TOLERANCIA_SEM_MAO` frames seguidos sem mão.

### 6.6 Camadas

```
 tela final = câmera  +  pontos da mão  +  canvas (desenhos prontos)  +  traço atual  +  cursor
```

O **canvas** é uma imagem preta do tamanho do frame onde ficam os desenhos finalizados. A cada frame, copiamos para a
tela só os pixels **não pretos** do canvas (`mascara = canvas.any(axis=2)`).

---

## Parte 7: Reconhecendo formas geométricas

Quando a pinça abre, o traço (uma lista de pontos) é analisado com **geometria clássica do OpenCV**, sem IA:

### 7.1 Ferramentas

- **`cv2.boundingRect`:** o retângulo reto que envolve o traço. A **diagonal** dele mede o "tamanho" do desenho.
- **`cv2.arcLength`:** perímetro de um contorno.
- **`cv2.convexHull` (envoltória convexa):** o "elástico" esticado em volta de todos os pontos. Elimina as tremidas
  para dentro e torna o traço mais limpo.
- **Circularidade** = `4π × área / perímetro²`. É **1,0** para um círculo perfeito, **≈ 0,79** para um quadrado e
  **≈ 0,60** para um triângulo equilátero. Ela não depende do tamanho da forma.
- **`cv2.approxPolyDP` (algoritmo de Douglas-Peucker):** simplifica uma curva em poucos segmentos de reta. A
  tolerância (`0.04 × perímetro`) define o quanto ele pode "arredondar". O número de vértices que sobra diz a forma:
  3 → triângulo, 4 → quadrilátero.
- **`cv2.minAreaRect`:** o menor retângulo (possivelmente girado) que cobre o traço.

### 7.2 A árvore de decisão

```
diagonal do traço < 40 px? ─────────────────────────────────────► ignora (toque acidental)
fim longe do início (distância > 30% da diagonal) → ABERTO:
   ├─ nenhum ponto se afasta > 10% da reta início→fim? ─────────► LINHA
   └─ senão ────────────────────────────────────────────────────► mantém o desenho livre
senão → FECHADO:
   ├─ circularidade > 0,92 ─────────────────────────────────────► CÍRCULO
   ├─ 3 vértices ───────────────────────────────────────────────► TRIÂNGULO
   ├─ 4 vértices ──► lados parecidos? ─► QUADRADO / RETÂNGULO
   ├─ 5 a 8 vértices ───────────────────────────────────────────► POLÍGONO
   └─ senão ────────────────────────────────────────────────────► mantém o desenho livre
```

**Por que comparar com a diagonal e não com o comprimento do traço?** O tremor da mão faz o traço "zigue-zaguear", e o
comprimento medido fica bem maior que o real. Um "U" tremido parecia fechado. A diagonal do retângulo envolvente quase
não muda com o tremor.

#### Como os limiares foram escolhidos: com dados, não no chute

Os números 0,92, 0,04, 30% etc. são **hiperparâmetros**, como os do treino. Para escolher o da circularidade, foram
gerados 200 desenhos sintéticos de cada forma, com tremor aleatório de 3 e 6 pixels, e medida a circularidade:

| Forma (com tremor) | menor valor | mediana | maior valor |
|---|---|---|---|
| círculo | 0,931 | 0,98 | 0,991 |
| círculo achatado (oval) | 0,925 | 0,96 | 0,974 |
| quadrado / quadrado girado | 0,797 | 0,85 | 0,908 |
| quadrado de cantos arredondados | 0,847 | 0,89 | 0,937 |
| retângulo | 0,715 | 0,76 | 0,812 |

Todos os círculos ficaram acima de 0,925 e os quadrados normais abaixo de 0,908, então **0,92** separa os dois grupos.
O "quadrado de cantos bem arredondados" é a única zona de confusão, e faz sentido: ele é *de fato* meio círculo.

Com esses limiares, nos testes sintéticos, círculos, quadrados (mesmo girados), retângulos, triângulos e linhas foram
reconhecidos em 100% dos casos com tremor de 3 px. Arcos e "U" foram corretamente recusados.

> **Experimente:** coloque `print(f"{circularidade:.3f}", len(vertices))` dentro de `classificar_forma`, desenhe várias
> vezes cada forma **do seu jeito** e ajuste os limiares. É exatamente o processo de *ajuste de hiperparâmetros*.

### 7.3 O app completo: `05_desenhar.py`

```python
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
```

```bash
python 05_desenhar.py                      # usa o best.pt mais recente em runs/
python 05_desenhar.py --modelo caminho/para/best.pt --camera 1 --imgsz 320
```

### Controles

| Ação | Como |
|---|---|
| Desenhar | encoste o polegar no indicador e mova a mão |
| Finalizar traço / virar forma | separe os dedos |
| Limpar tudo | `c` |
| Ligar/desligar reconhecimento de formas | `f` |
| Mostrar/esconder os pontos da mão | `p` |
| Salvar imagem | `s` |
| Sair | `q` |

### Ajuste fino

- **O desenho liga e desliga sozinho?** Observe o valor `abertura` na tela com a pinça aberta e fechada, e ajuste
  `LIMIAR_FECHA` e `LIMIAR_ABRE`.
- **Traço tremido?** Aumente `SUAVIZACAO` (0,6 a 0,7).
- **Cursor atrasado?** Diminua `SUAVIZACAO` ou o `--imgsz`.
- **FPS baixo?** Use `--imgsz 320` e/ou exporte para OpenVINO/ONNX (Parte 5.6).
- **Formas erradas?** Ajuste os limiares de `classificar_forma` (veja a dica na 7.2).

### ✅ Checkpoint 6 e 7

Você desenha com a pinça, e um círculo tremido vira um círculo perfeito amarelo.

---

## Parte 8 (bônus): Seu próprio dataset

Até aqui você usou um dataset pronto. O ciclo completo de ML inclui **coletar e anotar dados**. Um exercício ótimo é
treinar um **detector de gestos** e usá-lo no app (por exemplo: mão aberta = pausa, punho = apagar, "V" com dois dedos
= trocar cor).

### 8.1 Colete fotos: `06_capturar_fotos.py`

```python
"""Parte 8: tira fotos da webcam para montar seu próprio dataset.

Uso: python 06_capturar_fotos.py pinca      (espaço = tirar foto, q = sair)
"""
import sys
import time
from pathlib import Path

import cv2

classe = sys.argv[1] if len(sys.argv) > 1 else "sem_nome"
pasta = Path("meu_dataset") / "fotos" / classe
pasta.mkdir(parents=True, exist_ok=True)
contador = len(list(pasta.glob("*.jpg")))
camera = cv2.VideoCapture(0)

while True:
    ok, frame = camera.read()
    if not ok:
        break
    tela = frame.copy()  # o texto vai só na cópia; a foto salva fica limpa
    cv2.putText(tela, f"{classe}: {contador} fotos (espaco = foto, q = sair)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Captura", tela)

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord(" "):
        cv2.imwrite(str(pasta / f"{classe}_{int(time.time() * 1000)}.jpg"), frame)
        contador += 1
    elif tecla == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
```

```bash
python 06_capturar_fotos.py mao_aberta
python 06_capturar_fotos.py punho
python 06_capturar_fotos.py dois_dedos
```

**Boas práticas de coleta:**
- **100 a 200 fotos por classe** para começar;
- **varie tudo:** luz, fundo, distância, ângulo, mão esquerda e direita, roupa, outras pessoas;
- inclua fotos **sem mão nenhuma** (as "imagens de fundo" ensinam o que **não** é mão).

### 8.2 Anote (rotule)

Use uma ferramenta de anotação e desenhe uma caixa em volta de cada mão, escolhendo a classe:

- **[Label Studio](https://labelstud.io)**: local e grátis (`pip install label-studio` e depois `label-studio start`);
- **[CVAT](https://www.cvat.ai)**: online ou local, muito usado na indústria;
- **[Roboflow](https://roboflow.com)**: online, com plano grátis, que também divide e exporta o dataset.

Exporte no formato **YOLO**. Para detecção, cada linha é só `classe cx cy w h` (sem pontos).

### 8.3 Organize e descreva

```
meu_dataset/yolo/
├── images/train/  images/val/     ← ~80% treino / ~20% validação
└── labels/train/  labels/val/
```

⚠️ **Vazamento de dados (data leakage):** fotos tiradas em sequência são quase idênticas. Se uma for para o treino e a
"gêmea" para a validação, a métrica fica artificialmente alta. Separe por sessão de fotos, não foto a foto.

Crie `gestos.yaml` (use o caminho **absoluto**):

```yaml
path: C:/projetos/maos-yolo/meu_dataset/yolo
train: images/train
val: images/val
names:
  0: mao_aberta
  1: punho
  2: dois_dedos
```

### 8.4 Treine e use

```bash
yolo detect train model=yolo26n.pt data=gestos.yaml epochs=100 imgsz=640 project=runs name=gestos
yolo predict model=runs/gestos/weights/best.pt source=0 show=True
```

Depois, carregue esse segundo modelo no `05_desenhar.py` e use `r.boxes.cls` para trocar a cor ou apagar o desenho
conforme o gesto.

> **Ideia avançada (auto-labeling):** anotar 21 pontos por mão à mão é muito trabalhoso. Uma técnica profissional é
> usar o modelo que você já treinou para **pré-anotar** fotos novas e depois só **corrigir** os erros na ferramenta de
> anotação.

---

## Solução de problemas

| Problema | Causa provável | Solução |
|---|---|---|
| `CUDA: False` / "CPU" com GPU NVIDIA | PyTorch de CPU instalado | `pip uninstall -y torch torchvision` e reinstale com o comando NVIDIA da 1.4 |
| `CUDA out of memory` | batch/imgsz grandes para a GPU | reduza `batch` (16 → 8 → 4), depois `imgsz` |
| Windows: `The paging file is too small` (WinError 1455) ou travamento no início | cada worker usa muita RAM | reduza `workers` (2 ou 0) |
| Windows: erro com `freeze_support` / treino que recomeça sozinho | falta o `if __name__ == "__main__":` | use o script como está no guia |
| `cv2.imread` devolve `None` | caminho com acento ou arquivo inexistente | use pasta sem acentos (Parte 1.2) |
| Câmera não abre / tela preta | outro app usando a câmera, ou índice errado | feche Teams/Zoom/navegador; tente `--camera 1`. No Windows, troque `cv2.VideoCapture(i)` por `cv2.VideoCapture(i, cv2.CAP_DSHOW)` |
| `cv2.imshow` → "The function is not implemented" | `opencv-python-headless` instalado | `pip uninstall -y opencv-python-headless` e `pip install opencv-python` |
| Download do dataset falha/lento | rede | rode de novo; ou baixe o `.zip` da URL `download:` do YAML e extraia no `datasets_dir` |
| `yolo26n-pose.pt` não encontrado | Ultralytics antiga | `pip install -U ultralytics`, ou use `yolo11n-pose.pt` |
| Mac: erro com `mps` durante o treino | operação não suportada no MPS | treine com `--perfil cpu` ou no Colab; para inferência, o MPS costuma funcionar |
| Modelo detecta mal a SUA mão | diferença entre o dataset e o seu ambiente | mais luz, fundo neutro, `conf` menor; ou complemente o dataset (Parte 8) |
| App lento (FPS < 10) | CPU fraca / `imgsz` alto | `--imgsz 320`, exportar para OpenVINO/ONNX (5.6) |

---

## Glossário

| Termo | Significado |
|---|---|
| **Visão computacional** | área que faz computadores extraírem informação de imagens e vídeos |
| **Rede neural convolucional (CNN)** | rede que analisa imagens com filtros que "deslizam" sobre os pixels |
| **Inferência** | usar um modelo já treinado para prever algo |
| **Pesos** | os números que o modelo ajusta ao aprender; ficam salvos no `.pt` |
| **Época (epoch)** | uma volta completa no dataset de treino |
| **Batch (lote)** | grupo de imagens processadas antes de cada ajuste dos pesos |
| **Loss (perda)** | número que mede o erro do modelo; o treino tenta diminuí-lo |
| **Gradiente / backpropagation** | cálculo de como cada peso deve mudar para reduzir a loss |
| **Learning rate** | o tamanho do passo de ajuste dos pesos |
| **Fine-tuning / transfer learning** | aproveitar um modelo pré-treinado e adaptá-lo a uma tarefa nova |
| **Data augmentation** | gerar variações das imagens (espelho, cor, escala…) para o modelo generalizar melhor |
| **Overfitting** | o modelo decora o treino e vai mal em dados novos |
| **Early stopping** | parar o treino quando a validação para de melhorar (`patience`) |
| **Keypoint** | ponto-chave de um objeto (ex.: ponta do dedo) |
| **Bounding box** | retângulo em volta de um objeto detectado |
| **Confiança (conf)** | nota de 0 a 1 que o modelo dá para cada detecção |
| **NMS** | etapa que remove detecções duplicadas do mesmo objeto (o YOLO26 foi desenhado para dispensá-la) |
| **IoU / OKS** | sobreposição entre caixas / proximidade entre pontos previstos e reais |
| **Precisão / Recall** | poucos alarmes falsos / poucas detecções perdidas |
| **mAP** | métrica-resumo de qualidade de detecção (média das precisões) |
| **Tensor** | array multidimensional do PyTorch (pode viver na GPU) |
| **CUDA / MPS** | tecnologia para rodar cálculos na GPU NVIDIA / Apple |
| **Histerese** | usar dois limiares diferentes para ligar e desligar, evitando oscilação |
| **Envoltória convexa** | o menor polígono convexo que contém todos os pontos |

---

## Referências

- [Documentação Ultralytics](https://docs.ultralytics.com): tarefas, parâmetros de treino (`/modes/train`) e de
  predição (`/modes/predict`)
- [Dataset Hand Keypoints](https://docs.ultralytics.com/datasets/pose/hand-keypoints)
- [Tarefa Pose na Ultralytics](https://docs.ultralytics.com/tasks/pose/)
- [Instalação do PyTorch](https://pytorch.org/get-started/locally/)
- [Tutoriais OpenCV: contornos](https://docs.opencv.org/4.x/d3/d05/tutorial_py_table_of_contents_contours.html)
- Artigo original: Redmon et al., *You Only Look Once: Unified, Real-Time Object Detection* (2016)
