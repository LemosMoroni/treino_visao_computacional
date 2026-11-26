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
que estejam usando a camera. A janela mostra deteccoes do modelo v7.
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

O dashboard registra uma liberacao por periodo verde, apos 1 segundo de deteccoes
associadas a uma pessoa: mascara, avental e pelo menos uma luva. Depois de liberado,
o verde permanece ate a pessoa sair da camera (modo "Libera"). No modo "Continuo"
os EPIs seguem sendo conferidos e o sinal volta a vermelho se algum faltar por mais
de 1 segundo. O cabecalho tem seletores para o modelo (arquivos weights/epis_*.pt),
o minimo de luvas (1 ou 2) e o modo; os valores iniciais vem de --luvas 2 e
--continuo. Nao mede soldas fisicamente realizadas. Duas luvas sao duas deteccoes,
sem garantir maos distintas.
LEDs sao simulados, sem GPIO ou acionamento fisico. Deteccoes podem errar; o sistema
e um prototipo e nao comprova uso correto de EPI.
Historico local criado em dados_dashboard/. Fechar o navegador nao encerra o servidor;
use Ctrl+C. Captura automatica de fotos existe no script 07, nao no dashboard.

## Pre-anotar fotos para o CVAT

Em vez de desenhar todas as caixas do zero, o script 08 roda o modelo nas fotos
limpas e gera um ZIP no formato YOLO 1.1, que o CVAT importa como anotacao inicial.

```powershell
.\.venv\Scripts\python.exe 08_pre_anotar_cvat.py capturas_epis_limpas\sessao_XXXX
.\.venv\Scripts\python.exe 08_pre_anotar_cvat.py capturas_epis_limpas\sessao_XXXX --conf 0.25
```

Sem argumento, le `capturas_epis_limpas/` (sem subpastas). Gera
`<pasta>_pre_anotacoes_cvat.zip` e mostra quantas caixas de cada classe encontrou.
`--conf` (padrao 0.35) e a confianca minima: valores menores trazem mais caixas,
inclusive erradas. `--modelo` e `--saida` trocam o modelo e o nome do ZIP.

No CVAT:
1. Crie a task com as mesmas imagens, sem renomear, e os labels nesta ordem:
   luvas, soldador, avental_soldagem, mascara_soldagem.
2. Actions > Upload annotations > formato YOLO 1.1 > escolha o ZIP.
3. Revise todas as imagens: ajuste caixas, apague falsos positivos e marque o que
   faltou. Luvas sao pouco detectadas e normalmente precisam ser marcadas a mao.

As caixas sao so um ponto de partida. Aceita-las sem revisar faz o proximo treino
aprender os mesmos erros do modelo atual.

## Modelo v7

Ajuste do v6 somente com fotos proprias da cabine, anotadas no CVAT. No teste
(184 imagens), mAP50 geral foi de 0,803 (v6) para 0,857 (v7). Luvas foi de 0,515
para 0,774 e mascara de 0,916 para 0,959; soldador ficou igual (0,93). Avental caiu
de 0,845 para 0,764 por diferenca de criterio de anotacao em uma das sessoes, nao
por piora do modelo nas outras. Luvas continuam sendo o ponto mais fraco: o modelo
ainda deixa de detectar parte delas.

## Arquivos necessarios

- 07_testar_epis.py: camera, inferencia e captura; tambem importado pelo dashboard.
- dashboard_epis/server.py e core.py: servidor e regras.
- dashboard_epis/static/: interface HTML, CSS e JavaScript.
- weights/epis_v7_best.pt: modelo padrao (camera, dashboard e script 08).
- weights/epis_v6_best.pt e epis_v3_best.pt: modelos anteriores; use com --modelo weights/epis_v6_best.pt.
- 08_pre_anotar_cvat.py: pre-anotacao das fotos limpas para o CVAT.
- requirements.txt: dependencias.
- iniciar_epis.bat e iniciar_dashboard.bat: atalhos opcionais.

Classes: 0 luvas, 1 soldador, 2 avental_soldagem, 3 mascara_soldagem.
Nao remova a classe soldador: as regras atuais dependem dela.
Esta distribuicao e para computador; Raspberry Pi 3 exige adaptacao e validacao.

## GitHub

Envie apenas o conteudo desta pasta, incluindo o .gitignore. Fotos, bancos e .venv
nao devem ser enviados. Nao adicione a pasta inteira do projeto de treinamento.
Dependencias e pesos de terceiros continuam sujeitos as suas respectivas licencas.

## Alerta: mascara levantada com arco aceso

O dashboard emite um alerta (faixa vermelha piscando, som e borda vermelha na imagem)
quando ha arco aceso e a mascara da pessoa esta levantada por mais de 0,5 s. O alerta
termina 1 s depois que a mascara abaixa ou o arco apaga, e fica registrado no historico
(card "Alertas hoje" e botao "Alertas" para exportar CSV). O som so toca depois de
algum clique na pagina (regra do navegador).

Dois classificadores pequenos fazem isso, alem do detector de EPIs:

- `weights/mascara_cls_v*_best.pt`: recorte da mascara -> abaixada / levantada
- `weights/arco_cls_v*_best.pt`: imagem inteira -> arco / sem_arco

O dashboard carrega a versao mais alta de cada um (ou use `--mascara-modelo` e
`--arco-modelo`). Sem o modelo de arco, o botao **Simular arco** no cabecalho finge
que ha arco, para testar o alerta da mascara sem soldar. Mascara nao visivel durante o
arco nao gera alerta. Para testar sem camera, `--camera` aceita um arquivo de video.

So contar pixels claros nao detecta o arco nestas cameras: a parede branca estoura
sem arco e, com exposicao baixa, o arco aparece so como faiscas laranja. Por isso o
arco tambem e um classificador treinado com as fotos.

### Treinar os classificadores

1. Gerar as imagens (inclua sessoes com o arco aceso):

```powershell
# mascara: recortes a partir das caixas do CVAT (ou do v7, sem --anotacoes)
.\.venv\Scripts\python.exe 09_recortar_mascaras.py datasets/soldatech_fotos --anotacoes
# arco: fotos inteiras reduzidas
.\.venv\Scripts\python.exe 12_quadros_arco.py datasets/soldatech_fotos
```

2. Separar pelo teclado (o progresso fica salvo; Q sai):

```powershell
.\.venv\Scripts\python.exe 11_separar_recortes.py                 # A abaixada, L levantada, D descartar
.\.venv\Scripts\python.exe 11_separar_recortes.py --tarefa arco   # A arco, S sem arco, D descartar
```

   Espaco pula, Z desfaz. `--origem levantada` revisa uma pasta ja separada.
   No 09, `--cabeca` tambem recorta a cabeca de pessoas sem mascara detectada.

3. Treinar no Colab:

```powershell
.\.venv\Scripts\python.exe 10_treinar_mascara.py --tarefa mascara --empacotar   # ou --tarefa arco
```

   Envie o zip gerado (`mascara_recortes_colab.zip` ou `arco_quadros_colab.zip`) para
   `MyDrive/wmm_mascara/`, abra `treinar_mascara_colab.ipynb` no Colab com GPU, ajuste
   `TAREFA` na segunda celula e execute. Copie o `.pt` resultante do Drive para `weights/`.
   A validacao usa pessoas inteiras separadas do treino.
   (Treinar localmente tambem funciona, sem `--empacotar`; lento em CPU.)
