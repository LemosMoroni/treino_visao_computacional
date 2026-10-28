# Guia de comandos do projeto

Comandos para Windows/PowerShell. Execute na pasta do projeto (a que contém
`07_testar_epis.py`). Os exemplos com arquivos externos exigem ajustar os caminhos.
Não execute todos em sequência: escolha a seção da tarefa desejada.

## 1. Python e ambiente virtual

Listar versões instaladas:
```powershell
py --list
```

Criar ambiente com Python 3.12 (precisa estar instalado):
```powershell
py -3.12 -m venv .venv
```

Ativar e sair do ambiente:
```powershell
.\.venv\Scripts\Activate.ps1
deactivate
```

Os demais exemplos usam diretamente o Python da `.venv`, sem precisar ativá-la.
Se `py -3.12` não encontrar um runtime, instale Python 3.12 antes de continuar.

## 2. Instalação em outro computador

Após criar a `.venv`, instale a configuração CPU para o primeiro teste:
```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

`requirements.txt` está no pacote `pronto_para_github`. Faça a instalação nessa
pasta quando usar a cópia distribuída. Não copie `.venv` de outra máquina.

Configuração CUDA 11.8 utilizada no computador original com MX250, caso queira
substituir a instalação CPU (não é uma escolha universal para outras GPUs):
```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y torch torchvision
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe -m pip check
```

Verificar driver e disponibilidade da GPU:
```powershell
nvidia-smi
.\.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA disponível:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
.\.venv\Scripts\python.exe config.py
```

`nvidia-smi` depende de GPU e driver NVIDIA; não instala ou atualiza drivers.

## 3. Câmera com detecção de EPIs

Câmera integrada, normalmente índice 0:
```powershell
.\.venv\Scripts\python.exe 07_testar_epis.py --fonte 0
```

Câmera USB, normalmente índice 1 (tente 2 se necessário):
```powershell
.\.venv\Scripts\python.exe 07_testar_epis.py --fonte 1
```

Captura automática de fotos limpas a cada 5 segundos:
```powershell
.\.venv\Scripts\python.exe 07_testar_epis.py --fonte 1 --intervalo 5
```

O modelo continua mostrando caixas na tela. As fotos automáticas são JPEG de
qualidade 95, sem caixas/textos, em `capturas_epis_limpas/sessao_.../`.
A primeira sai cerca de 5 segundos após o primeiro quadro. Em 4 horas são
aproximadamente 2.880 fotos. Não há encerramento automático após 4 horas.
Mantenha energia, espaço livre e suspensão automática desativada.

Sem janela, mantendo inferência e captura automática (Ctrl+C encerra):
```powershell
.\.venv\Scripts\python.exe 07_testar_epis.py --fonte 1 --intervalo 5 --sem-janela
```

Escolher modelo, tamanho de inferência e confiança:
```powershell
.\.venv\Scripts\python.exe 07_testar_epis.py --fonte 1 --modelo weights/epis_v3_best.pt --imgsz 640 --conf 0.40
```

Forçar CPU ou GPU NVIDIA disponível:
```powershell
.\.venv\Scripts\python.exe 07_testar_epis.py --fonte 1 --device cpu
.\.venv\Scripts\python.exe 07_testar_epis.py --fonte 1 --device 0
```

Espelhar a câmera (também espelha as fotos salvas):
```powershell
.\.venv\Scripts\python.exe 07_testar_epis.py --fonte 1 --espelhar
```

Teste curto, encerrando após 10 quadros:
```powershell
.\.venv\Scripts\python.exe 07_testar_epis.py --fonte 1 --limite 10
```

Analisar imagem ou vídeo existentes; substitua os caminhos:
```powershell
.\.venv\Scripts\python.exe 07_testar_epis.py --fonte "C:\imagens\foto.jpg" --saida "resultado.png" --sem-janela
.\.venv\Scripts\python.exe 07_testar_epis.py --fonte "C:\videos\teste.mp4"
```

`--saida` grava a última imagem **anotada**, para demonstração. Não é a captura
limpa para CVAT. `--intervalo` só pode ser usado com câmera, não com arquivos.

Atalho Windows equivalente:
```powershell
.\iniciar_epis.bat --fonte 1 --intervalo 5
```

Teclas na janela:

| Tecla | Ação |
| --- | --- |
| S | Salvar foto limpa extra em PNG na pasta capturas_epis_limpas |
| Q ou Esc | Encerrar |
| C | Abrir ajustes do driver da câmera, se suportado no Windows |
| + / - | Aumentar / diminuir confiança mínima |

A captura solicita 1280×720; `--imgsz 640` controla a inferência, não a resolução
da câmera. Execute apenas um programa usando a câmera por vez.

## 4. Dashboard

Abrir com câmera USB:
```powershell
.\.venv\Scripts\python.exe dashboard_epis/server.py --camera 1 --abrir
```

Ou pelo atalho:
```powershell
.\iniciar_dashboard.bat --camera 1
```

Exigir duas detecções de luvas, escolher modelo e porta:
```powershell
.\.venv\Scripts\python.exe dashboard_epis/server.py --camera 1 --modelo weights/epis_v3_best.pt --luvas 2 --port 8765 --abrir
```

Selecionar outro arquivo de histórico:
```powershell
.\.venv\Scripts\python.exe dashboard_epis/server.py --camera 1 --db dados_dashboard/teste.sqlite3 --abrir
```

Acesse http://127.0.0.1:8765 e ative a câmera na página. Ctrl+C encerra o servidor;
fechar o navegador não encerra. LEDs são simulados. O dashboard não possui
captura automática de fotos; essa função está no script 07. Duas detecções de
luvas não comprovam que são mãos distintas.

## 5. Estudo de pose e mãos — scripts 01 a 05

Não são comandos de treinamento de EPIs. Não há script 06 nesta versão.

```powershell
.\.venv\Scripts\python.exe 01_webcam_pose.py
.\.venv\Scripts\python.exe 02_explorar_dataset.py
```

O script 02 pode baixar o dataset de mãos. Treino pequeno de verificação:
```powershell
.\.venv\Scripts\python.exe 03_treinar.py --perfil teste
```

Treino com seleção automática de hardware ou perfil explícito:
```powershell
.\.venv\Scripts\python.exe 03_treinar.py
.\.venv\Scripts\python.exe 03_treinar.py --perfil gpu_pequena --epocas 60
```

Perfis: `teste`, `cpu`, `gpu_pequena`, `gpu_media`, `gpu_grande`, `apple`.
Treinos completos podem levar muitas horas. Ctrl+C interrompe; checkpoints já
salvos permanecem em `runs/`.

Avaliar e desenhar:
```powershell
.\.venv\Scripts\python.exe 04_avaliar.py
.\.venv\Scripts\python.exe 05_desenhar.py --camera 1
```

No pacote GitHub, é possível apontar explicitamente o modelo de mãos distribuído:
```powershell
.\.venv\Scripts\python.exe 04_avaliar.py --modelo weights/maos_best.pt
.\.venv\Scripts\python.exe 05_desenhar.py --modelo weights/maos_best.pt --camera 1
```

A avaliação precisa do dataset. `maos_best.pt` é de um treino local parcial.
No pacote, 04 e 05 procuram um treino em `runs/` e usam esse peso como alternativa.

## 6. Verificação e ajuda

```powershell
.\.venv\Scripts\python.exe 07_testar_epis.py --help
.\.venv\Scripts\python.exe dashboard_epis/server.py --help
.\.venv\Scripts\python.exe 03_treinar.py --help
.\.venv\Scripts\python.exe -m unittest discover -s dashboard_epis -p "test_*.py"
.\.venv\Scripts\python.exe -m pip check
```

Contar fotos limpas salvas, incluindo subpastas:
```powershell
(Get-ChildItem capturas_epis_limpas -Recurse -File | Where-Object { $_.Extension -in '.jpg', '.jpeg', '.png' } | Measure-Object).Count
```

## 7. Preparação e treinamento de EPIs no projeto completo

Os scripts de preparação, ZIPs e notebooks NÃO estão no pacote mínimo do GitHub.
Estes comandos só se aplicam ao projeto completo original, com os datasets de
origem nos caminhos esperados pelos scripts. Revise esses caminhos antes de executar.

```powershell
.\.venv\Scripts\python.exe treinamento/preparar_epis.py "C:\caminho\PPE Welding Dataset_v1.v1i.yolo26.zip"
.\.venv\Scripts\python.exe treinamento/preparar_epis_v2.py
.\.venv\Scripts\python.exe treinamento/preparar_epis_v3.py
```

O treinamento e avaliação de EPIs foram feitos no Google Colab pelos notebooks
`treinar_epis_colab.ipynb`, `treinar_epis_v2_colab.ipynb` e
`treinar_epis_v3_colab.ipynb`. Abra o notebook correspondente no Colab e execute
suas células conforme as instruções. Não execute o treino de mãos (03) para EPIs.

## 8. Enviar o pacote ao GitHub

Abra o terminal DENTRO de `pronto_para_github`. Se a pasta ainda não tiver seu
próprio repositório, inicialize-o; isso evita enviar o projeto completo por engano:

```powershell
git init
git status
git add .
git diff --cached --stat
git commit -m "Adiciona aplicacao local de EPIs e dashboard"
git branch -M main
```

Após criar um repositório vazio no GitHub, substitua USUARIO e REPOSITORIO:
```powershell
git remote add origin https://github.com/USUARIO/REPOSITORIO.git
git push -u origin main
```

Se `origin` já existir, confira com `git remote -v` antes de alterá-lo.
Para atualizações posteriores, revise o que será enviado:
```powershell
git status
git add .
git diff --cached --stat
git commit -m "Atualiza aplicacao e documentacao"
git push
```

O `.gitignore` exclui ambientes, fotos, datasets e bancos locais. Os modelos e
`dashboard_epis/static/` devem ser enviados. Estes comandos são instruções;
nenhum commit ou envio ao GitHub foi realizado automaticamente.
