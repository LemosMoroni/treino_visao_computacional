# Detector local de EPIs de soldagem

Pacote somente para execucao em computador: camera, modelo v3 e dashboard.
Nao inclui treinamento, datasets, fotos, historico ou modelos antigos.

## Instalar no Windows

Instale Python 3.12 de 64 bits e abra o PowerShell nesta pasta.
A instalacao abaixo utiliza CPU; a instalacao de PyTorch para GPU depende da maquina.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Nao precisa ativar a .venv para usar estes comandos. Nao copie ambientes de outra maquina.

## Abrir camera e modelo

```powershell
.\.venv\Scripts\python.exe 07_testar_epis.py --fonte 1
```

Indice 1 costuma ser USB; tente 0 ou 2 conforme a maquina. Feche outros programas
que estejam usando a camera. A janela mostra deteccoes do modelo v3.
S salva foto limpa; Q/Esc encerra. C solicita configuracoes do driver no Windows.
A captura solicita 1280x720; a inferencia usa tamanho 640 por padrao.

Capturas automaticas a cada 5 segundos:

```powershell
.\.venv\Scripts\python.exe 07_testar_epis.py --fonte 1 --intervalo 5
```

Fotos JPEG limpas ficam em capturas_epis_limpas/sessao_.../. Aproximadamente 2880
fotos em quatro horas; nao ha encerramento automatico. Mantenha energia, espaco
livre e suspensao desativada. Ctrl+C tambem encerra. O modelo continua funcionando.

## Abrir dashboard

```powershell
.\.venv\Scripts\python.exe dashboard_epis/server.py --camera 1 --abrir
```

Acesse http://127.0.0.1:8765 e ative a camera. Nao execute a camera do script 07
simultaneamente. Atalhos equivalentes:

```powershell
.\iniciar_epis.bat --fonte 1 --intervalo 5
.\iniciar_dashboard.bat --camera 1
```

O dashboard registra uma liberacao por periodo verde, apos 3 segundos de deteccoes
associadas a uma pessoa: mascara, avental e pelo menos uma luva. Nao mede soldas
fisicamente realizadas. --luvas 2 exige duas deteccoes, sem garantir maos distintas.
LEDs sao simulados, sem GPIO ou acionamento fisico. Deteccoes podem errar; o sistema
e um prototipo e nao comprova uso correto de EPI.
Historico local criado em dados_dashboard/. Fechar o navegador nao encerra o servidor;
use Ctrl+C. Captura automatica de fotos existe no script 07, nao no dashboard.

## Arquivos necessarios

- 07_testar_epis.py: camera, inferencia e captura; tambem importado pelo dashboard.
- dashboard_epis/server.py e core.py: servidor e regras.
- dashboard_epis/static/: interface HTML, CSS e JavaScript.
- weights/epis_v3_best.pt: unico modelo distribuido.
- requirements.txt: dependencias.
- iniciar_epis.bat e iniciar_dashboard.bat: atalhos opcionais.

Classes: 0 luvas, 1 soldador, 2 avental_soldagem, 3 mascara_soldagem.
Nao remova a classe soldador: as regras atuais dependem dela.
Esta distribuicao e para computador; Raspberry Pi 3 exige adaptacao e validacao.

## GitHub

Envie apenas o conteudo desta pasta, incluindo o .gitignore. Fotos, bancos e .venv
nao devem ser enviados. Nao adicione a pasta inteira do projeto de treinamento.
Dependencias e pesos de terceiros continuam sujeitos as suas respectivas licencas.
