"""Teste local dos EPIs: câmera USB, vídeo ou imagem, sem treinamento."""
import argparse
import math
from datetime import datetime
from collections import Counter
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
NAMES = {0: 'luvas', 1: 'soldador', 2: 'avental_soldagem', 3: 'mascara_soldagem'}
LABELS = {0: 'Luvas', 1: 'Soldador', 2: 'Avental', 3: 'Mascara'}
COLORS = {0: (70, 215, 245), 1: (240, 180, 70), 2: (140, 220, 110), 3: (220, 140, 230)}


def open_camera(index):
    """Tenta backends no mesmo indice, sem trocar silenciosamente a camera."""
    backends = [('DirectShow', cv2.CAP_DSHOW), ('Media Foundation', cv2.CAP_MSMF),
                ('automatico', cv2.CAP_ANY)] if __import__('sys').platform == 'win32' else [('automatico', cv2.CAP_ANY)]
    for name, backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, frame = cap.read()
            if ok and frame is not None:
                print(f'Camera {index} aberta via {name}.', flush=True)
                return cap
        cap.release()
        print(f'Camera {index}: {name} nao forneceu imagem.', flush=True)
    raise RuntimeError(f'Camera {index} indisponivel. Feche Camera/Teams/Zoom/dashboard, '
                       'confira a permissao de camera para aplicativos da area de trabalho '
                       'no Windows e teste outro indice com --fonte 0, 1 ou 2.')


class CaptureTimer:
    """Relogio monotono: sem sleep e sem rajadas para recuperar atrasos."""
    def __init__(self, interval):
        self.interval = interval
        self.next_at = None

    def due(self, now):
        if not self.interval:
            return False
        if self.next_at is None:
            self.next_at = now + self.interval
            return False
        if now < self.next_at:
            return False
        self.next_at = now + self.interval
        return True


class LatestCamera:
    """Lê continuamente a câmera e mantém apenas seu quadro mais recente."""
    def __init__(self, cap):
        self.cap = cap
        self.condition = threading.Condition()
        self.frame = None
        self.sequence = 0
        self.timestamp = 0.0
        self.ended = False
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        try:
            while not self.stop.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    break
                with self.condition:
                    self.frame = frame
                    self.timestamp = time.perf_counter()
                    self.sequence += 1
                    self.condition.notify_all()
        finally:
            with self.condition:
                self.ended = True
                self.condition.notify_all()

    def read(self, previous, timeout=5):
        with self.condition:
            ready = self.condition.wait_for(lambda: self.sequence > previous or self.ended, timeout)
            if not ready or self.sequence <= previous:
                return None
            return self.frame.copy(), self.sequence, self.timestamp

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        self.cap.release()
        self.thread.join(timeout=1)


def choose_device(request):
    if request != 'auto':
        return request
    return '0' if torch.cuda.is_available() else 'cpu'


def render(frame, result, fps, elapsed_ms, age_ms, confidence, model_name):
    canvas = frame.copy()
    counts = Counter()
    for box in result.boxes:
        cls = int(box.cls.item())
        counts[cls] += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color = COLORS[cls]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f'{LABELS[cls]} {float(box.conf.item()):.0%}'
        cv2.putText(canvas, label, (max(0, x1), max(18, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3)
        cv2.putText(canvas, label, (max(0, x1), max(18, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 1)
    # Informações ficam fora da imagem para não esconder EPIs próximos da borda.
    width = max(canvas.shape[1], 780)
    screen = np.full((canvas.shape[0]+145, width, 3), (27, 24, 22), np.uint8)
    screen[135:135+canvas.shape[0], :canvas.shape[1]] = canvas
    lines = [
        f'EPIs | {model_name} | conf {confidence:.2f}',
        f'{fps:.1f} analises/s | inferencia {elapsed_ms:.0f} ms | idade do quadro {age_ms:.0f} ms',
        '   '.join(f'{LABELS[i]}: {counts[i]}' for i in range(4)),
        'Contagem de objetos; nao confirma uso de EPI por pessoa.',
        'Q/Esc: sair | S: salvar | C: ajustes da camera | +/-: confianca',
    ]
    for i, line in enumerate(lines):
        cv2.putText(screen, line, (12, 23+i*24), cv2.FONT_HERSHEY_SIMPLEX, .48, (225, 225, 225), 1, cv2.LINE_AA)
    return screen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modelo', type=Path, default=ROOT/'weights/epis_v3_best.pt')
    parser.add_argument('--fonte', default='0', help='Índice da câmera (0, 1...) ou caminho de vídeo/imagem')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=.40)
    parser.add_argument('--device', default='auto', help='auto, cpu ou 0 (GPU)')
    parser.add_argument('--espelhar', action='store_true', help='Espelhar câmera, se desejado')
    parser.add_argument('--sem-janela', action='store_true')
    parser.add_argument('--limite', type=int, default=0, help='Máximo de quadros analisados; 0 = sem limite')
    parser.add_argument('--saida', type=Path, help='Salvar a última imagem anotada ao encerrar')
    parser.add_argument('--intervalo', type=float, default=0,
                        help='Salvar foto limpa automaticamente a cada N segundos; 0 desativa')
    args = parser.parse_args()
    if not math.isfinite(args.intervalo) or args.intervalo < 0:
        parser.error('--intervalo deve ser um numero finito >= 0.')
    if args.intervalo and not args.fonte.isdecimal():
        parser.error('--intervalo e destinado a camera: use --fonte 0, 1 ou 2.')
    if not args.modelo.is_file():
        parser.error(f'Modelo não encontrado: {args.modelo}')
    if not 0 < args.conf < 1 or args.imgsz < 32 or args.imgsz % 32 or args.limite < 0:
        parser.error('Use conf entre 0 e 1, imgsz múltiplo de 32 e limite >= 0.')
    device = choose_device(args.device)
    model = YOLO(str(args.modelo), task='detect')
    if model.names != NAMES or model.task != 'detect':
        parser.error(f'O modelo não tem as quatro classes esperadas: {model.names}')
    print(f'Modelo: {args.modelo.name} | dispositivo: {device} | resolução: {args.imgsz}', flush=True)
    # Aquece antes de abrir a câmera; falhas de CUDA aparecem sem prender a captura.
    model.predict(np.zeros((args.imgsz, args.imgsz, 3), np.uint8), imgsz=args.imgsz, device=device, verbose=False)
    is_camera = args.fonte.isdecimal()
    path = Path(args.fonte)
    is_image = not is_camera and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    if not is_camera and not path.is_file():
        parser.error(f'Fonte não encontrada: {path}')
    cap = camera = None
    screen = None
    sequence = count = 0
    durations = []
    begin = previous = time.perf_counter()
    title = 'EPIs - teste local'
    timer = CaptureTimer(args.intervalo)
    automatic_count = 0
    capture_folder = ROOT/'capturas_epis_limpas'/f'sessao_{datetime.now():%Y%m%d_%H%M%S}_{time.time_ns()}'
    if args.intervalo:
        print(f'Captura automatica: {args.intervalo:g}s | JPEG qualidade 95 | {capture_folder}', flush=True)
    try:
        if is_camera or not is_image:
            if is_camera:
                cap = open_camera(int(args.fonte))
            else:
                cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError('Não consegui abrir a fonte. Para câmera, feche outros apps ou tente --fonte 1.')
            if is_camera:
                camera = LatestCamera(cap)
        if not args.sem_janela:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        print(f'Fonte aberta: {args.fonte}. Q ou Esc na janela encerra o programa.', flush=True)
        while True:
            if camera:
                item = camera.read(sequence)
                if item is None:
                    raise RuntimeError('A câmera parou de enviar imagens.')
                frame, sequence, captured = item
            elif is_image:
                frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError('Não consegui ler a imagem.')
                captured = time.perf_counter()
            else:
                ok, frame = cap.read()
                if not ok:
                    if not count:
                        raise RuntimeError('O vídeo não possui quadros legíveis.')
                    break
                captured = time.perf_counter()
            if args.espelhar and is_camera:
                frame = cv2.flip(frame, 1)
            if timer.due(time.monotonic()):
                save_image(capture_folder/f'epis_{time.time_ns()}.jpg', frame)
                automatic_count += 1
            start = time.perf_counter()
            result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=device, verbose=False)[0]
            now = time.perf_counter()
            elapsed = (now-start)*1000
            durations.append(elapsed)
            # Guarda somente uma amostra limitada de tempos, mesmo em execuções longas.
            if len(durations) > 1000:
                durations.pop(0)
            fps = 1 / max(now-previous, 1e-6)
            previous = now
            screen = render(frame, result, fps, elapsed, (now-captured)*1000, args.conf, args.modelo.name)
            count += 1
            if count == 1:
                print(f'Resolução recebida: {frame.shape[1]} x {frame.shape[0]} pixels.', flush=True)
                print(f'Primeira análise OK: {elapsed:.0f} ms, {len(result.boxes)} objetos.', flush=True)
                if not args.sem_janela:
                    # Evita a janela pequena padrão do OpenCV, respeitando a tela.
                    available_width, available_height = 1280, 900
                    if __import__('sys').platform == 'win32':
                        import ctypes
                        available_width = ctypes.windll.user32.GetSystemMetrics(0) - 80
                        available_height = ctypes.windll.user32.GetSystemMetrics(1) - 120
                    scale = min(1.0, available_width / screen.shape[1], available_height / screen.shape[0])
                    cv2.resizeWindow(title, max(320, int(screen.shape[1]*scale)), max(240, int(screen.shape[0]*scale)))
            key = -1
            if not args.sem_janela:
                cv2.imshow(title, screen)
                key = cv2.waitKey(0 if is_image else 1) & 0xff
                if key in (27, ord('q')) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord('s'):
                    destination = ROOT/'capturas_epis_limpas'/f'epis_{time.time_ns()}.png'
                    save_image(destination, frame)
                if key == ord('c') and is_camera:
                    if __import__('sys').platform == 'win32':
                        cap.set(cv2.CAP_PROP_SETTINGS, 1)
                    else:
                        print('Ajustes do driver disponíveis por esta tecla somente no Windows.', flush=True)
                if key in (ord('+'), ord('=')):
                    args.conf = min(.95, round(args.conf+.05, 2))
                elif key == ord('-'):
                    args.conf = max(.05, round(args.conf-.05, 2))
            if is_image or (args.limite and count >= args.limite):
                break
    finally:
        if camera:
            camera.close()
        elif cap:
            cap.release()
        cv2.destroyAllWindows()
        if args.intervalo:
            print(f'Fotos automaticas salvas: {automatic_count} | {capture_folder}', flush=True)
    if args.saida and screen is not None:
        save_image(args.saida, screen)
    if count:
        print(f'Concluído: {count} análises em {time.perf_counter()-begin:.1f}s; '
              f'mediana de inferência {np.median(durations):.0f} ms (até 1000 últimas análises).', flush=True)


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if path.suffix.lower() in {'.jpg', '.jpeg'} else []
    ok, data = cv2.imencode(path.suffix or '.png', image, params)
    if not ok:
        raise RuntimeError(f'Falha ao salvar {path}')
    data.tofile(path)
    print(f'Imagem salva: {path}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nTeste encerrado.')
    except Exception as exc:
        raise SystemExit(f'Erro no teste: {exc}') from exc
