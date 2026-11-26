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
import re
from sirene import Sirene
from core import AlertGate, History, ReleaseGate, Smooth, assess, iou, person_mask, square_crop

ROOT=Path(__file__).resolve().parent.parent
STATIC=Path(__file__).resolve().parent/'static'


def latest_weight(prefix):
    """Peso mais recente weights/<prefix>_v<N>_best.pt (maior N), ou None."""
    found=[(int(m.group(1)),f) for f in (ROOT/'weights').glob(f'{prefix}_v*_best.pt') if (m:=re.search(r'_v(\d+)_best',f.name))]
    return max(found)[1] if found else None


class FileCamera:
    """Vídeo de arquivo tocado em tempo real e em repetição (para testes sem câmera)."""
    def __init__(self, cv2, path):
        self.cv2=cv2;self.cap=cv2.VideoCapture(path)
        self.delay=1/max(1,self.cap.get(cv2.CAP_PROP_FPS) or 15);self.next=time.monotonic()
    def isOpened(self):return self.cap.isOpened()
    def set(self,*a):return False
    def read(self):
        time.sleep(max(0,self.next-time.monotonic()));self.next=max(self.next+self.delay,time.monotonic()-1)
        ok,frame=self.cap.read()
        if not ok:
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES,0);ok,frame=self.cap.read()
        return ok,frame
    def release(self):self.cap.release()


class Engine:
    def __init__(self, history, model, source, imgsz, gloves, continuous=False, mask_model=None, arc_model=None, siren=None):
        self.history,self.model,self.source,self.imgsz,self.gloves=history,model,source,imgsz,gloves
        self.continuous=continuous;self.mask_model=mask_model;self.arc_model=arc_model;self.arc_test=False
        self.alert=AlertGate(history,'mascara_levantada_com_arco');self.siren=siren
        self.mask_smooth=Smooth(.5,1.0);self.arc_smooth=Smooth(.5,1.0)
        self.lock=threading.RLock();self.stop=threading.Event();self.thread=None
        self.gate=ReleaseGate(history,model.name,continuous=continuous)
        self.jpeg=None;self.last=0;self.person=None
        self.state=dict(camera='off',status='denied',reason='Câmera desligada',progress=0,
                        found={'mascara':False,'avental':False,'luvas':False},fps=0,latency=0,resolution='—',error=None,
                        mask_state=None,mask_prob=None,arc=False,arc_prob=None,alert_on=False)
        threading.Thread(target=self.watchdog,daemon=True).start()

    def deny(self, reason):
        self.state.update(status='denied',progress=0,reason=reason)
        self.gate.deny(time.monotonic(),reason)
        self.person=None

    def clear_safety(self):
        self.alert.reset(time.monotonic());self.mask_smooth.value=self.arc_smooth.value=None
        if self.siren:self.siren.desligar()
        self.state.update(mask_state=None,mask_prob=None,arc=False,arc_prob=None,alert_on=False)

    def watchdog(self):
        while True:
            time.sleep(.25)
            with self.lock:
                if self.state['camera']=='on' and time.monotonic()-self.last>2:
                    try:self.deny('Sem imagem recente: sinal vermelho');self.clear_safety()
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
            self.deny('Câmera desligada');self.clear_safety();self.state.update(camera='off',fps=0,found={k:False for k in self.state['found']})
            self.jpeg=None

    def models(self):
        return sorted(p.name for p in self.model.parent.glob('epis_*.pt'))

    def set_model(self, name):
        if name not in self.models():raise ValueError('Modelo não encontrado.')
        with self.lock:
            thread=self.thread;restart=self.state['camera'] in ('on','starting')
        self.halt()
        if thread:thread.join(15)
        if thread and thread.is_alive():raise RuntimeError('A câmera não parou a tempo; tente de novo.')
        with self.lock:
            self.model=self.model.parent/name
            self.gate=ReleaseGate(self.history,name,continuous=self.continuous)
        if restart:self.start()

    def set_continuous(self, continuous):
        with self.lock:
            self.continuous=self.gate.continuous=bool(continuous)

    def set_arc_test(self, on):
        with self.lock:self.arc_test=bool(on)

    def set_gloves(self, count):
        if count not in (1,2):raise ValueError('Mínimo de luvas deve ser 1 ou 2.')
        with self.lock:
            if count==self.gloves:return
            self.gloves=count
            if self.state['camera']=='on':self.deny(f'Mínimo de luvas alterado para {count}; nova conferência')

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
            mask_cls=arc_cls=None
            if self.mask_model:
                mask_cls=module.YOLO(str(self.mask_model),task='classify')
                if sorted(mask_cls.names.values())!=['abaixada','levantada']:raise RuntimeError(f'Classificador de máscara com classes inesperadas: {mask_cls.names}')
                mask_cls.predict(np.zeros((224,224,3),np.uint8),device=device,verbose=False)
            if self.arc_model:
                arc_cls=module.YOLO(str(self.arc_model),task='classify')
                if sorted(arc_cls.names.values())!=['arco','sem_arco']:raise RuntimeError(f'Detector de arco com classes inesperadas: {arc_cls.names}')
                arc_cls.predict(np.zeros((320,320,3),np.uint8),device=device,verbose=False)
            if self.stop.is_set():return
            if isinstance(self.source,str):cap=FileCamera(cv2,self.source)
            else:cap=cv2.VideoCapture(self.source,cv2.CAP_DSHOW if sys.platform=='win32' else cv2.CAP_ANY)
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
                gloves=self.gloves;ready,reason,found,person=assess(detections,gloves)
                stamp=time.monotonic()
                # Máscara levantada/abaixada: classifica o recorte da máscara da pessoa
                mask_box=person_mask(detections,person);mask_p=None
                if mask_cls is not None and mask_box is not None:
                    crop=square_crop(frame,mask_box)
                    if min(crop.shape[:2])>=32:
                        r=mask_cls.predict(crop,device=device,verbose=False)[0]
                        mask_p=float(r.probs.data[[k for k,v in mask_cls.names.items() if v=='levantada'][0]])
                mask_p=self.mask_smooth.update(mask_p,stamp)
                mask_state=None if mask_p is None else ('levantada' if mask_p>=.5 else 'abaixada')
                # Arco: classificador do quadro inteiro (ou simulação pelo botão)
                arc_p=None
                if arc_cls is not None:
                    r=arc_cls.predict(frame,imgsz=320,device=device,verbose=False)[0]
                    arc_p=self.arc_smooth.update(float(r.probs.data[[k for k,v in arc_cls.names.items() if v=='arco'][0]]),stamp)
                arc_test=self.arc_test
                arc=arc_test or (arc_p is not None and arc_p>=.5)
                annotated=frame.copy()
                for d in detections:
                    a,b,c,e=map(int,d['box']);cl=d['class'];color=module.COLORS[cl]
                    cv2.rectangle(annotated,(a,b),(c,e),color,2)
                    label=f'{module.LABELS[cl]} {d["confidence"]:.0%}'
                    if cl==3 and mask_state and d['box']==mask_box:
                        label=f'Mascara LEVANTADA {mask_p:.0%}' if mask_state=='levantada' else f'Mascara abaixada {1-mask_p:.0%}'
                    cv2.putText(annotated,label,(max(0,a),max(18,b-6)),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
                if arc:
                    cv2.putText(annotated,'ARCO (simulado)' if arc_test else f'ARCO {arc_p:.0%}',(12,annotated.shape[0]-16),cv2.FONT_HERSHEY_SIMPLEX,.9,(0,200,255),2)
                ok,jpeg=cv2.imencode('.jpg',annotated,[cv2.IMWRITE_JPEG_QUALITY,88])
                if not ok:raise RuntimeError('Falha ao preparar a imagem.')
                now=time.monotonic()
                with self.lock:
                    if self.stop.is_set():break
                    if gloves!=self.gloves:continue
                    if self.gate.active is None and self.person is not None and person is not None and iou(person,self.person)<.3:
                        self.deny('Pessoa mudou de posição; nova conferência')
                    self.person=person
                    alert_on=self.alert.update(arc and mask_state=='levantada',now,
                                               {'mask_prob':mask_p,'arc_prob':arc_p,'arc_test':arc_test})
                    if self.siren:self.siren.ligar(alert_on)
                    if alert_on:
                        cv2.rectangle(annotated,(0,0),(annotated.shape[1]-1,annotated.shape[0]-1),(0,0,255),24)
                        ok,jpeg=cv2.imencode('.jpg',annotated,[cv2.IMWRITE_JPEG_QUALITY,88])
                    if time.perf_counter()-captured>2:
                        ready=False;reason='Imagem atrasada: aguarde nova captura';self.deny(reason)
                    status,progress=self.gate.update(ready,now,reason,{'found':found,'detections':detections,'required_gloves':self.gloves},present=person is not None)
                    if status=='released':reason='Liberado enquanto todos os EPIs forem detectados' if self.continuous else 'Liberado até o soldador sair da câmera'
                    self.state.update(camera='on',status=status,reason=reason,progress=progress,found=found,
                                      fps=round(1/max(now-begin,.001),1),latency=round((now-begin)*1000),
                                      resolution=f'{frame.shape[1]} × {frame.shape[0]}',error=None,
                                      mask_state=mask_state,mask_prob=None if mask_p is None else round(mask_p,3),
                                      arc=arc,arc_prob=None if arc_p is None else round(arc_p,3),alert_on=alert_on)
                    self.jpeg=jpeg.tobytes();self.last=now
        except Exception as exc:
            with self.lock:
                self.state.update(camera='error',error=str(exc),status='denied',progress=0,reason='Falha na câmera ou no processamento')
                try:self.deny('Falha na câmera ou no processamento');self.clear_safety()
                except Exception:pass
                self.jpeg=None
        finally:
            if camera:camera.close()
            elif cap:cap.release()

    def snapshot(self):
        with self.lock:
            result=dict(self.state)
            result.update(model=self.model.name,stable=self.gate.stable,models=self.models(),required_gloves=self.gloves,continuous=self.continuous,
                          active_id=self.gate.active,active_seconds=round(time.monotonic()-self.gate.started) if self.gate.started is not None else 0,
                          led_green=self.state['status']=='released',led_red=self.state['status']!='released',
                          mask_model=self.mask_model.name if self.mask_model else None,
                          arc_model=self.arc_model.name if self.arc_model else None,arc_test=self.arc_test,
                          local_siren=bool(self.siren and self.siren.disponivel),siren_error=self.siren.erro if self.siren else None,
                          alert_seconds=round(time.monotonic()-self.alert.started) if self.alert.started is not None else 0)
            return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--modelo',type=Path,default=ROOT/'weights/epis_v7_best.pt')
    parser.add_argument('--camera',default='0',help='índice da câmera ou caminho de um vídeo (teste)')
    parser.add_argument('--mascara-modelo',type=Path,default=latest_weight('mascara_cls'),help='classificador abaixada/levantada')
    parser.add_argument('--arco-modelo',type=Path,default=latest_weight('arco_cls'),help='classificador arco/sem_arco')
    parser.add_argument('--imgsz',type=int,default=640)
    parser.add_argument('--luvas',type=int,choices=[1,2],default=1)
    parser.add_argument('--sem-sirene',action='store_true',help='não toca a sirene neste computador (só o aviso no navegador)')
    parser.add_argument('--continuo',action='store_true',help='Exige os EPIs o tempo todo, e não só na liberação')
    parser.add_argument('--db',type=Path,default=ROOT/'dados_dashboard/epis.sqlite3')
    parser.add_argument('--abrir',action='store_true')
    args=parser.parse_args()
    if not args.modelo.is_file():parser.error('Modelo não encontrado.')
    for extra in (args.mascara_modelo,args.arco_modelo):
        if extra and not extra.is_file():parser.error(f'Modelo não encontrado: {extra}')
    camera=int(args.camera) if args.camera.isdigit() else args.camera
    siren=Sirene(args.db.parent/'sirene.wav',ativa=not args.sem_sirene)
    history=History(args.db);engine=Engine(history,args.modelo,camera,args.imgsz,args.luvas,args.continuo,args.mascara_modelo,args.arco_modelo,siren)
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
            if path=='/api/status':return self.send(200,dict(engine.snapshot(),summary=history.summary(),history=history.rows(),
                                                             alerts=history.alert_summary(),alert_history=history.alert_rows(20),token=token))
            if path=='/api/export-alerts':
                out=io.StringIO();writer=csv.writer(out)
                writer.writerow(['id','inicio','fim','duracao_segundos','tipo'])
                for r in history.alert_rows():writer.writerow([r['id'],r['started_at'],r['ended_at'],r['duration'],r['kind']])
                return self.send(200,'\ufeff'+out.getvalue(),'text/csv; charset=utf-8')
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
                elif self.path=='/api/model':
                    body=json.loads(self.rfile.read(min(int(self.headers.get('Content-Length') or 0),1000)) or b'{}')
                    try:engine.set_model(str(body.get('name','')))
                    except ValueError as exc:return self.send(400,{'error':str(exc)})
                elif self.path=='/api/mode':
                    body=json.loads(self.rfile.read(min(int(self.headers.get('Content-Length') or 0),1000)) or b'{}')
                    if not isinstance(body.get('continuous'),bool):return self.send(400,{'error':'Modo inválido.'})
                    engine.set_continuous(body['continuous'])
                elif self.path=='/api/arc-test':
                    body=json.loads(self.rfile.read(min(int(self.headers.get('Content-Length') or 0),1000)) or b'{}')
                    if not isinstance(body.get('on'),bool):return self.send(400,{'error':'Valor inválido.'})
                    engine.set_arc_test(body['on'])
                elif self.path=='/api/gloves':
                    body=json.loads(self.rfile.read(min(int(self.headers.get('Content-Length') or 0),1000)) or b'{}')
                    try:engine.set_gloves(int(body.get('count',0)))
                    except (TypeError,ValueError) as exc:return self.send(400,{'error':str(exc)})
                else:return self.send(404,{'error':'Não encontrado'})
                self.send(200,engine.snapshot())
            except Exception as exc:self.send(500,{'error':str(exc)})

    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    url=f'http://127.0.0.1:{args.port}'
    print(f'Dashboard local: {url}\nHistórico: {args.db}\nLEDs simulados; nenhum equipamento físico conectado.',flush=True)
    print('Máscara: '+(args.mascara_modelo.name if args.mascara_modelo else 'sem classificador (alerta desativado)'),flush=True)
    print('Sirene: '+('neste computador (deixe o volume do Windows no máximo)' if siren.disponivel else (siren.erro or 'desativada; só no navegador')),flush=True)
    print('Arco: '+(args.arco_modelo.name if args.arco_modelo else 'sem modelo (use Simular arco para testar)'),flush=True)
    if args.abrir:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:engine.halt();siren.desligar();server.server_close()


if __name__=='__main__':main()
