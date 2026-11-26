"""Sirene de alerta tocada pelo próprio programa (Windows), sem depender do navegador.

O som é uma sirene de dois sentidos (sobe e desce entre 650 e 1750 Hz) em onda
quadrada com amplitude máxima: é o mais alto que o arquivo de som permite. O volume
final depende do volume do Windows e das caixas de som do computador.
"""
from pathlib import Path
import struct
import sys
import threading
import wave

import numpy as np

TAXA = 44100


def gerar_wav(caminho, ciclo=1.0, repeticoes=2, f_min=650, f_max=1750):
    """Grava um WAV de sirene (varredura triangular de frequência, onda quadrada)."""
    t = np.arange(int(TAXA * ciclo * repeticoes)) / TAXA
    fase_ciclo = (t % ciclo) / ciclo
    tri = 1 - np.abs(2 * fase_ciclo - 1)                  # 0 → 1 → 0 em cada ciclo
    freq = f_min + (f_max - f_min) * tri
    fase = 2 * np.pi * np.cumsum(freq) / TAXA
    onda = np.sign(np.sin(fase))                          # onda quadrada: mais "alta" que a senoidal
    borda = int(.003 * TAXA)                              # rampas curtas evitam estalos no loop
    env = np.ones_like(onda); env[:borda] = np.linspace(0, 1, borda); env[-borda:] = np.linspace(1, 0, borda)
    dados = (onda * env * 32000).astype('<i2')
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(caminho), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(TAXA)
        w.writeframes(dados.tobytes())
    return caminho


class Sirene:
    def __init__(self, caminho_wav, ativa=True):
        self.lock = threading.Lock()
        self.tocando = False
        self.winsound = None
        self.erro = None
        if not ativa:
            return
        if sys.platform != 'win32':
            self.erro = 'Sirene local só no Windows'
            return
        try:
            import winsound
            self.wav = str(gerar_wav(caminho_wav))
            self.winsound = winsound
        except Exception as exc:
            self.erro = f'Sirene indisponível: {exc}'

    @property
    def disponivel(self):
        return self.winsound is not None

    def ligar(self, ligar=True):
        if not self.disponivel:
            return
        with self.lock:
            if ligar == self.tocando:
                return
            try:
                if ligar:
                    ws = self.winsound
                    ws.PlaySound(self.wav, ws.SND_FILENAME | ws.SND_ASYNC | ws.SND_LOOP | ws.SND_NODEFAULT)
                else:
                    self.winsound.PlaySound(None, 0)
                self.tocando = ligar
            except Exception as exc:
                self.erro = f'Falha ao tocar a sirene: {exc}'

    def desligar(self):
        self.ligar(False)
