"""Dashboard local para detecção de EPIs e registro de sinalização simulada."""
import argparse
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import mimetypes
from pathlib import Path
import secrets
import sys
import threading
import time
import webbrowser
from core import History, ReleaseGate, assess, iou

ROOT=Path(__file__).resolve().parent.parent
STATIC=Path(__file__).resolve().parent/'static'


class Engine:
    def __init__(self, history, model, source, imgsz, gloves):
        self.history,self.model,self.source,self.imgsz,self.gloves=history,model,source,imgsz,gloves
        self.lock=threading.RLock();self.stop=threading.Event();self.thread=None
        self.gate=ReleaseGate(history,model.name)
        self.jpeg=None;self.last=0;self.person=None
        self.state=dict(camera='off',status='denied',reason='Câmera desligada',progress=0,
                        found={'mascara':False,'avental':False,'luvas':False},fps=0,latency=0,resolution='—',error=None)
        threading.Thread(target=self.watchdog,daemon=True).start()

    def deny(self, reason):
        self.state.update(status='denied',progress=0,reason=reason)
        self.gate.deny(time.monotonic(),reason)
        self.person=None

    def watchdog(self):
        while True:
            time.sleep(.25)
            with self.lock:
                if self.state['camera']=='on' and time.monotonic()-self.last>2:
                    try:self.deny('Sem imagem recente: sinal vermelho')
                    except Exception as exc:self.state['error']=str(exc)

    def start(self):
        with self.lock:
            if self.thread and self.thread.is_alive():return
            self.stop.clear();self.jpeg=None;self.person=None
            self.state.update(camera='starting',status='denied',reason='Carregando modelo e câmera',error=None)
            self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()

    def halt(self):
        self.stop.set()
        with self.lock:
            self.deny('Câmera desligada');self.state.update(camera='off',fps=0,found={k:False for k in self.state['found']})
            self.jpeg=None

    def run(self):
        camera=cap=None
        try:
            spec=importlib.util.spec_from_file_location('local_epis',ROOT/'07_testar_epis.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            cv2=module.cv2;np=module.np
            model=module.YOLO(str(self.model),task='detect')
            if model.names!=module.NAMES:raise RuntimeError('O modelo não tem as quatro classes esperadas.')
            device=module.choose_device('auto')
            model.predict(np.zeros((self.imgsz,self.imgsz,3),np.uint8),imgsz=self.imgsz,device=device,verbose=False)
            if self.stop.is_set():return
            cap=cv2.VideoCapture(self.source,cv2.CAP_DSHOW if sys.platform=='win32' else cv2.CAP_ANY)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
            if not cap.isOpened():raise RuntimeError('Câmera indisponível. Feche o programa antigo ou selecione outra câmera.')
            camera=module.LatestCamera(cap);sequence=0
            while not self.stop.is_set():
                item=camera.read(sequence,timeout=2)
                if item is None:raise RuntimeError('A câmera deixou de enviar imagens.')
                frame,sequence,captured=item
                begin=time.monotonic()
                result=model.predict(frame,imgsz=self.imgsz,conf=.40,device=device,verbose=False)[0]
                detections=[{'class':int(b.cls.item()),'confidence':round(float(b.conf.item()),3),
                             'box':list(map(float,b.xyxy[0].tolist()))} for b in result.boxes]
                ready,reason,found,person=assess(detections,self.gloves)
                annotated=frame.copy()
                for d in detections:
                    a,b,c,e=map(int,d['box']);cl=d['class'];color=module.COLORS[cl]
                    cv2.rectangle(annotated,(a,b),(c,e),color,2)
                    cv2.putText(annotated,f'{module.LABELS[cl]} {d["confidence"]:.0%}',(max(0,a),max(18,b-6)),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
                ok,jpeg=cv2.imencode('.jpg',annotated,[cv2.IMWRITE_JPEG_QUALITY,88])
                if not ok:raise RuntimeError('Falha ao preparar a imagem.')
                now=time.monotonic()
                with self.lock:
                    if self.stop.is_set():break
                    if self.person is not None and person is not None and iou(person,self.person)<.3:
                        self.deny('Pessoa mudou de posição; nova conferência')
                    self.person=person
                    if time.perf_counter()-captured>2:
                        ready=False;reason='Imagem atrasada: aguarde nova captura'
                    status,progress=self.gate.update(ready,now,reason,{'found':found,'detections':detections,'required_gloves':self.gloves})
                    self.state.update(camera='on',status=status,reason=reason,progress=progress,found=found,
                                      fps=round(1/max(now-begin,.001),1),latency=round((now-begin)*1000),
                                      resolution=f'{frame.shape[1]} × {frame.shape[0]}',error=None)
                    self.jpeg=jpeg.tobytes();self.last=now
        except Exception as exc:
            with self.lock:
                self.state.update(camera='error',error=str(exc),status='denied',progress=0,reason='Falha na câmera ou no processamento')
                try:self.deny('Falha na câmera ou no processamento')
                except Exception:pass
                self.jpeg=None
        finally:
            if camera:camera.close()
            elif cap:cap.release()

    def snapshot(self):
        with self.lock:
            result=dict(self.state)
            result.update(model=self.model.name,required_gloves=self.gloves,
                          active_id=self.gate.active,active_seconds=round(time.monotonic()-self.gate.started) if self.gate.started is not None else 0,
                          led_green=self.state['status']=='released',led_red=self.state['status']!='released')
            return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--modelo',type=Path,default=ROOT/'weights/epis_v3_best.pt')
    parser.add_argument('--camera',type=int,default=0)
    parser.add_argument('--imgsz',type=int,default=640)
    parser.add_argument('--luvas',type=int,choices=[1,2],default=1)
    parser.add_argument('--db',type=Path,default=ROOT/'dados_dashboard/epis.sqlite3')
    parser.add_argument('--abrir',action='store_true')
    args=parser.parse_args()
    if not args.modelo.is_file():parser.error('Modelo não encontrado.')
    history=History(args.db);engine=Engine(history,args.modelo,args.camera,args.imgsz,args.luvas)
    token=secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def local_request(self):
            return self.headers.get('Host') in {f'127.0.0.1:{args.port}',f'localhost:{args.port}'}
        def send(self,status,data,kind='application/json; charset=utf-8'):
            if isinstance(data,(dict,list)):data=json.dumps(data,ensure_ascii=False).encode()
            if isinstance(data,str):data=data.encode()
            self.send_response(status);self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff');self.end_headers()
            try:self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError):pass

        def do_GET(self):
            if not self.local_request():return self.send(403,{'error':'Acesso local apenas.'})
            path=self.path.split('?')[0]
            if path=='/api/status':return self.send(200,dict(engine.snapshot(),summary=history.summary(),history=history.rows(),token=token))
            if path=='/api/frame':
                with engine.lock:data=engine.jpeg
                return self.send(200,data,'image/jpeg') if data else self.send(204,b'')
            if path=='/api/export':
                out=io.StringIO();writer=csv.writer(out)
                writer.writerow(['id','inicio','fim','duracao_segundos','motivo_encerramento','modelo'])
                for r in history.rows(None):writer.writerow([r['id'],r['started_at'],r['ended_at'],r['duration'],r['reason'],r['model']])
                return self.send(200,'\ufeff'+out.getvalue(),'text/csv; charset=utf-8')
            files={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}
            if path not in files:return self.send(404,{'error':'Não encontrado'})
            file=STATIC/files[path]
            return self.send(200,file.read_bytes(),(mimetypes.guess_type(str(file))[0] or 'application/octet-stream')+'; charset=utf-8')

        def do_POST(self):
            if not self.local_request():return self.send(403,{'error':'Acesso local apenas.'})
            if self.headers.get('X-Local-Token')!=token:return self.send(403,{'error':'Atualize a página para continuar.'})
            try:
                if self.path=='/api/start':engine.start()
                elif self.path=='/api/stop':engine.halt()
                else:return self.send(404,{'error':'Não encontrado'})
                self.send(200,engine.snapshot())
            except Exception as exc:self.send(500,{'error':str(exc)})

    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    url=f'http://127.0.0.1:{args.port}'
    print(f'Dashboard local: {url}\nHistórico: {args.db}\nLEDs simulados; nenhum equipamento físico conectado.',flush=True)
    if args.abrir:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:engine.halt();server.server_close()


if __name__=='__main__':main()
