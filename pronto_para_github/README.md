# Visao computacional: EPIs de soldagem e estudo de maos

Aplicacao local com camera, modelo YOLO e dashboard. Esta distribuicao contem
o necessario para executar em outro computador, sem fotos, datasets ou historico pessoal.

## Instalar no Windows

Instale Python 3.12 de 64 bits. Baixe ou clone este repositorio e abra o PowerShell
na pasta que contem este README. A instalacao precisa de internet.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Esses comandos instalam a versao CPU para o primeiro teste. Para usar uma GPU,
adapte a instalacao do PyTorch ao hardware e ao driver da outra maquina.
Nao copie o ambiente `.venv` de outro computador.

## Abrir o dashboard

```powershell
.\.venv\Scripts\python.exe dashboard_epis/server.py --camera 1 --abrir
```

Endereco: http://127.0.0.1:8765. Camera 0 costuma ser a integrada; teste 1 ou 2
para a USB. Ative a camera na pagina. Feche outros programas que estejam usando a camera.
Tambem pode executar `iniciar_dashboard.bat --camera 1`.

O modelo padrao e `weights/epis_v3_best.pt`. O painel mostra a camera, EPIs,
estado de liberacao, contagens, historico e exportacao CSV. O historico e criado
localmente em `dados_dashboard/epis.sqlite3`.

Uma liberacao e contada quando uma pessoa tem mascara, avental e pelo menos uma
luva detectados e associados geometricamente por 3 segundos continuos.
Uma falha na verificacao retorna o estado para vermelho. `--luvas 2` exige duas
deteccoes, sem comprovar que sao duas maos distintas. Cada periodo verde conta
uma liberacao, nao uma operacao de solda fisicamente realizada.

LEDs sao simulados na interface. Nao existe acionamento fisico de GPIO ou maquina.
E um prototipo de deteccao, que pode errar e nao comprova uso correto de EPI.
O servidor fica restrito ao computador local. Ctrl+C no terminal encerra o servidor;
fechar apenas o navegador nao o encerra.

## Camera e fotos limpas

```powershell
.\.venv\Scripts\python.exe 07_testar_epis.py --fonte 1
```

Solicita captura em 1280x720 e inferencia em 640; a resolucao efetiva depende da camera.
S salva fotos sem caixas/textos em `capturas_epis_limpas/`; Q ou Esc encerra.
O parametro `--saida arquivo.png` salva uma imagem ANOTADA para demonstracao.
As fotos limpas podem ser anotadas no CVAT para um treinamento futuro.

## Arquivos e modelos

| Arquivo | Finalidade |
| --- | --- |
| 01_webcam_pose.py | Demonstracao de pose na webcam com modelo generico |
| 02_explorar_dataset.py | Explorar dataset de maos; baixa dados se necessario |
| 03_treinar.py | Treinar pontos de maos; pode levar bastante tempo |
| 04_avaliar.py | Avaliar o modelo de maos com maos.yaml |
| 05_desenhar.py | Desenhar usando os pontos da mao |
| 07_testar_epis.py | Detectar EPIs em camera, video ou imagem |
| config.py, maos.yaml | Configuracao dos scripts de maos |
| dashboard_epis/ | Servidor, regras, testes e arquivos HTML/CSS/JS |
| weights/epis_v1_best.pt, epis_v2_best.pt, epis_v3_best.pt | Versoes do detector de EPIs |
| weights/maos_best.pt | Modelo do treino local parcial de maos |
| yolo26n-pose.pt | Modelo generico usado pelos scripts 01 e 03 |
| weights/yolo26n.pt | Modelo base de deteccao, opcional para execucao |

Nao existe script 06 nesta versao. Os scripts 01 a 05 sao o estudo de maos;
o sistema de EPIs usa 07 e dashboard_epis. Execute os comandos na raiz do projeto.
04 e 05 procuram um treino em runs/ e, se nao houver, usam weights/maos_best.pt.
Esse modelo veio de um treinamento parcial, sem garantia de qualidade.
Para avaliar maos e necessario obter o dataset; ele nao acompanha o repositorio.

Classes dos modelos de EPIs, em ordem: 0 luvas, 1 soldador,
2 avental_soldagem, 3 mascara_soldagem. Nao use esses modelos nos scripts de pose.
V3 manteve o conjunto de validacao da v2: 450 imagens, mAP50 0,8017 e
mAP50-95 0,6079. Nao ha resultado de teste final v3 registrado nesta distribuicao.

## Verificar a instalacao

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe 07_testar_epis.py --help
.\.venv\Scripts\python.exe -m unittest discover -s dashboard_epis -p "test_*.py"
```

O projeto foi desenvolvido em Windows/Python 3.12. A distribuicao ainda precisa
ser validada na outra maquina. Raspberry Pi 3 com 1 GB exige uma etapa separada
de instalacao e otimizacao; estes comandos nao garantem suporte ou desempenho nele.

## Enviar ao GitHub

Envie o CONTEUDO desta pasta, incluindo weights/ e dashboard_epis/static/.
Nao envie .venv, datasets, runs, fotos ou bancos de dados. O .gitignore exclui
esses arquivos gerados localmente. Os pesos desta distribuicao tem menos de
50 MB por arquivo. O arquivo manifesto_modelos.json registra seus hashes.

Os componentes e modelos de terceiros continuam sujeitos as respectivas licencas.
Este pacote nao atribui uma nova licenca aos pesos ou aos dados usados no treino.
