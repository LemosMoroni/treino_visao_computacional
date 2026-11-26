"""Regras temporais e histórico de sinais; não aciona equipamento de soldagem."""
from datetime import datetime, timedelta
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3


def now_iso():
    return datetime.now().astimezone().isoformat(timespec='seconds')


class History:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS releases (
                    id INTEGER PRIMARY KEY, started_at TEXT NOT NULL,
                    ended_at TEXT, duration REAL, reason TEXT, model TEXT NOT NULL,
                    evidence TEXT NOT NULL);
                CREATE UNIQUE INDEX IF NOT EXISTS one_active ON releases((1)) WHERE ended_at IS NULL;
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY, started_at TEXT NOT NULL,
                    ended_at TEXT, duration REAL, kind TEXT NOT NULL, evidence TEXT NOT NULL);
            ''')
            db.execute('UPDATE releases SET ended_at=?, reason=? WHERE ended_at IS NULL',
                       (now_iso(), 'Programa reiniciado; duração desconhecida'))
            db.execute('UPDATE alerts SET ended_at=? WHERE ended_at IS NULL', (now_iso(),))

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def begin(self, model, evidence):
        with self.connect() as db:
            return db.execute('INSERT INTO releases(started_at,model,evidence) VALUES(?,?,?)',
                              (now_iso(), model, json.dumps(evidence))).lastrowid

    def end(self, ident, seconds, reason):
        with self.connect() as db:
            db.execute('UPDATE releases SET ended_at=?,duration=?,reason=? WHERE id=? AND ended_at IS NULL',
                       (now_iso(), max(0, seconds), reason, ident))

    def rows(self, limit=100):
        with self.connect() as db:
            if limit is None:
                return [dict(r) for r in db.execute('SELECT * FROM releases ORDER BY id DESC')]
            return [dict(r) for r in db.execute('SELECT * FROM releases ORDER BY id DESC LIMIT ?', (limit,))]

    def alert_begin(self, kind, evidence):
        with self.connect() as db:
            return db.execute('INSERT INTO alerts(started_at,kind,evidence) VALUES(?,?,?)',
                              (now_iso(), kind, json.dumps(evidence))).lastrowid

    def alert_end(self, ident, seconds):
        with self.connect() as db:
            db.execute('UPDATE alerts SET ended_at=?,duration=? WHERE id=? AND ended_at IS NULL',
                       (now_iso(), max(0, seconds), ident))

    def alert_summary(self):
        today = datetime.now().astimezone().date().isoformat()
        with self.connect() as db:
            total = db.execute('SELECT count(*) FROM alerts').fetchone()[0]
            hoje = db.execute('SELECT count(*) FROM alerts WHERE substr(started_at,1,10)=?', (today,)).fetchone()[0]
            last = db.execute('SELECT started_at FROM alerts ORDER BY id DESC LIMIT 1').fetchone()
        return dict(total=total, today=hoje, last=last[0] if last else None)

    def alert_rows(self, limit=None):
        with self.connect() as db:
            sql = 'SELECT id,started_at,ended_at,duration,kind FROM alerts ORDER BY id DESC'
            return [dict(r) for r in (db.execute(sql+' LIMIT ?', (limit,)) if limit else db.execute(sql))]

    def summary(self):
        today = datetime.now().astimezone().date()
        dates = [(today-timedelta(days=i)).isoformat() for i in range(6, -1, -1)]
        with self.connect() as db:
            total = db.execute('SELECT count(*) FROM releases').fetchone()[0]
            grouped = {r[0]:r[1] for r in db.execute('SELECT substr(started_at,1,10),count(*) FROM releases GROUP BY 1')}
            avg = db.execute('SELECT avg(duration) FROM releases WHERE duration IS NOT NULL').fetchone()[0]
        return dict(total=total, today=grouped.get(today.isoformat(), 0), average_seconds=avg or 0,
                    days=[{'date':d,'count':grouped.get(d,0)} for d in dates])


class ReleaseGate:
    def __init__(self, history, model, stable=1.0, grace=1.0, continuous=False):
        self.history, self.model, self.stable, self.grace = history, model, stable, grace
        self.continuous = continuous
        self.since = None
        self.active = None
        self.started = None
        self.lost = None

    def update(self, ready, now, reason, evidence, present=False):
        # Depois de liberado, o sinal fica verde enquanto a pessoa continuar na
        # câmera (present), mesmo que algum EPI deixe de ser detectado. Só termina
        # quando ela some por mais de `grace` segundos (tolera falhas de um quadro).
        # No modo contínuo, a liberação exige os EPIs o tempo todo: se faltar
        # algum por mais de `grace` segundos, o sinal volta a vermelho.
        if self.active is not None:
            if ready if self.continuous else present:
                self.lost = None
                return 'released', 1.0
            if self.lost is None:
                self.lost = now
            if now-self.lost < self.grace:
                return 'released', 1.0
            self.deny(now, reason)
            return 'denied', 0.0
        if not ready:
            self.deny(now, reason)
            return 'denied', 0.0
        if self.since is None:
            self.since = now
        progress = min(1, (now-self.since)/self.stable)
        if progress >= 1 and self.active is None:
            self.active = self.history.begin(self.model, evidence)
            self.started = now
        return ('released' if self.active is not None else 'checking'), progress

    def deny(self, now, reason):
        ident, started = self.active, self.started
        self.active = self.started = self.since = self.lost = None
        if ident is not None:
            self.history.end(ident, now-started, reason)


def iou(a, b):
    x1,y1,x2,y2=max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    area=lambda p:max(0,p[2]-p[0])*max(0,p[3]-p[1])
    return inter/max(1,area(a)+area(b)-inter)


def assess(detections, required_gloves=1):
    people=[d for d in detections if d['class']==1]
    found={'mascara':False,'avental':False,'luvas':False}
    if len(people)!=1:
        return False, ('Posicione uma pessoa na câmera' if not people else 'Mais de uma pessoa: análise suspensa'), found, None
    person=people[0]['box'];x1,y1,x2,y2=person
    height=max(1,y2-y1);gloves=0
    for d in detections:
        if d['class']==1:continue
        a,b,c,e=d['box'];cx,cy=(a+c)/2,(b+e)/2
        if not (x1<=cx<=x2 and y1<=cy<=y2):continue
        relative=(cy-y1)/height
        if d['class']==3 and relative<=.45:found['mascara']=True
        if d['class']==2 and .20<=relative<=.90:found['avental']=True
        if d['class']==0:gloves+=1
    found['luvas']=gloves>=required_gloves
    missing=[k for k,v in found.items() if not v]
    return not missing, ('EPIs associados à pessoa' if not missing else 'Não identificado: '+', '.join(missing)), found, person


def person_mask(detections, person):
    """Caixa da máscara associada à pessoa (mesma regra do assess), a de maior confiança."""
    if person is None:
        return None
    x1, y1, x2, y2 = person
    height = max(1, y2-y1)
    best = None
    for d in detections:
        if d['class'] != 3:
            continue
        a, b, c, e = d['box']
        cx, cy = (a+c)/2, (b+e)/2
        if x1 <= cx <= x2 and y1 <= cy <= y2 and (cy-y1)/height <= .45:
            if best is None or d['confidence'] > best['confidence']:
                best = d
    return best['box'] if best else None


def square_crop(img, box, margin=.35):
    """Recorte quadrado em volta da caixa, igual ao do 09_recortar_mascaras.py."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    side = int(min(max(x2-x1, y2-y1)*(1+2*margin), w, h))
    left = int(min(max(0, (x1+x2)/2-side/2), w-side))
    top = int(min(max(0, (y1+y2)/2-side/2), h-side))
    return img[top:top+side, left:left+side]


class Smooth:
    """Média móvel exponencial de uma probabilidade; esquece o valor se ficar sem medida."""
    def __init__(self, alpha=.5, forget=1.0):
        self.alpha, self.forget = alpha, forget
        self.value = None
        self.at = None

    def update(self, prob, now):
        if prob is None:
            if self.at is not None and now-self.at > self.forget:
                self.value = None
            return self.value
        self.value = prob if self.value is None else self.alpha*prob + (1-self.alpha)*self.value
        self.at = now
        return self.value


class AlertGate:
    """Alerta quando o perigo dura `hold` segundos; encerra após `clear` segundos sem perigo."""
    def __init__(self, history, kind, hold=.5, clear=1.0):
        self.history, self.kind, self.hold, self.clear = history, kind, hold, clear
        self.since = self.ok_since = None
        self.active = self.started = None

    def update(self, danger, now, evidence):
        if danger:
            self.ok_since = None
            if self.since is None:
                self.since = now
            if self.active is None and now-self.since >= self.hold:
                self.active = self.history.alert_begin(self.kind, evidence)
                self.started = now
        else:
            self.since = None
            if self.active is not None:
                if self.ok_since is None:
                    self.ok_since = now
                if now-self.ok_since >= self.clear:
                    self.reset(now)
        return self.active is not None

    def reset(self, now):
        if self.active is not None:
            self.history.alert_end(self.active, now-self.started)
        self.since = self.ok_since = self.active = self.started = None
