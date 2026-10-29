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
            ''')
            db.execute('UPDATE releases SET ended_at=?, reason=? WHERE ended_at IS NULL',
                       (now_iso(), 'Programa reiniciado; duração desconhecida'))

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
    def __init__(self, history, model, stable=3.0):
        self.history, self.model, self.stable = history, model, stable
        self.since = None
        self.active = None
        self.started = None

    def update(self, ready, now, reason, evidence):
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
        self.active = self.started = self.since = None
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
