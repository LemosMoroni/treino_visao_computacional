import tempfile
import unittest
from pathlib import Path
from core import History, ReleaseGate, assess


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.history=History(Path(self.temp.name)/'test.sqlite3')
        self.gate=ReleaseGate(self.history,'teste.pt')

    def tearDown(self):self.temp.cleanup()

    def test_stability_count_and_rearming(self):
        self.assertEqual(self.gate.update(True,0,'ok',{})[0],'checking')
        self.assertEqual(self.gate.update(True,2.9,'ok',{})[0],'checking')
        self.assertEqual(self.history.summary()['total'],0)
        self.assertEqual(self.gate.update(True,3,'ok',{})[0],'released')
        for t in (4,5,6):self.gate.update(True,t,'ok',{})
        self.assertEqual(self.history.summary()['total'],1)
        self.gate.update(False,7,'Máscara não identificada',{})
        self.assertEqual(self.history.rows()[0]['duration'],4)
        self.gate.update(True,8,'ok',{});self.gate.update(True,11,'ok',{})
        self.assertEqual(self.history.summary()['total'],2)

    def test_dropout_resets_confirmation(self):
        self.gate.update(True,0,'ok',{});self.gate.update(False,2,'falha',{})
        self.assertEqual(self.gate.update(True,3,'ok',{})[0],'checking')
        self.assertEqual(self.history.summary()['total'],0)

    def test_restart_does_not_leave_green(self):
        self.gate.update(True,0,'ok',{});self.gate.update(True,3,'ok',{})
        restarted=History(self.history.path)
        self.assertIsNotNone(restarted.rows()[0]['ended_at'])
        self.assertIsNone(restarted.rows()[0]['duration'])

    def test_association_and_multiple_people(self):
        person={'class':1,'box':[0,0,100,200]}
        detections=[person,{'class':3,'box':[20,10,80,50]},
                    {'class':2,'box':[20,60,80,150]}, {'class':0,'box':[5,80,20,110]}]
        self.assertTrue(assess(detections)[0])
        self.assertFalse(assess(detections,required_gloves=2)[0])
        self.assertFalse(assess(detections+[person])[0])
        detections[-1]['box']=[200,80,220,110]
        self.assertFalse(assess(detections)[0])


if __name__=='__main__':unittest.main()
