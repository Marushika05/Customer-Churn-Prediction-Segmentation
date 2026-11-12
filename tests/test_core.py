import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd
from academic.ingestion import clean, read_file, IngestError
from academic.features import build_features, build_snapshots, chronological_split, FEATURES, InsufficientData
from academic.segmentation import RFMSegmenter, quality
from academic.classification import preprocessor, metrics, encoder


def transactions():
    return pd.DataFrame([
        ['1','A',2,'2020-01-01',10,1],
        ['1','BB',1,'2020-01-01',5,1],
        ['2','A',1,'2020-01-11',10,1],
        ['C3','A',-1,'2020-01-12',10,1],
        ['4','A',1,'2020-01-21',10,1],
        ['5','A',1,'2020-01-03',10,2],
    ], columns=['Invoice','StockCode','Quantity','InvoiceDate','Price','Customer ID'])


class CoreTests(unittest.TestCase):
    def test_hand_calculated_rfm(self):
        tx,_ = clean(transactions())
        f = build_features(tx,pd.Timestamp('2020-01-21'))
        self.assertEqual(f.loc[1,'frequency'],2)
        self.assertEqual(f.loc[1,'monetary'],35)
        self.assertEqual(f.loc[1,'recency_days'],10)
        self.assertEqual(f.loc[1,'avg_gap_days'],10)
        self.assertEqual(f.loc[1,'return_rate'],.5)

    def test_future_does_not_change_features(self):
        tx,_=clean(transactions());t=pd.Timestamp('2020-01-21')
        a=build_features(tx,t)
        changed=tx.copy(); changed.loc[changed.invoice_date >= t,'line_total']=1e9
        pd.testing.assert_frame_equal(a,build_features(changed,t))

    def test_labels_and_exact_cutoff_boundary(self):
        tx,_=clean(transactions());t=pd.Timestamp('2020-01-21')
        s=build_snapshots(tx,[t]).set_index('customer_id')
        self.assertEqual(s.loc[1,'churned'],0)
        self.assertEqual(s.loc[2,'churned'],1)
        self.assertEqual(s.loc[1,'frequency'],2)
        # A purchase at T+90 is outside the half-open target window.
        tx.loc[tx.invoice_no == '4','invoice_date']=t+pd.Timedelta(days=90)
        self.assertEqual(build_snapshots(tx,[t]).set_index('customer_id').loc[1,'churned'],1)

    def test_cleaning_invalid_and_anonymous(self):
        raw=transactions(); raw=pd.concat([raw,raw.iloc[[0]]],ignore_index=True)
        raw.loc[len(raw)]=['6','A',0,'2020-01-01',10,1]
        raw.loc[len(raw)]=['7','A',1,'2020-01-01',-1,1]
        raw.loc[len(raw)]=['8','A',1,'2020-01-01',10,None]
        tx,report=clean(raw)
        self.assertEqual(report['dropped']['exact_duplicates'],1)
        self.assertEqual(tx.customer_id.isna().sum(),1)
        self.assertEqual(report['dropped']['invalid_quantity_or_price'],2)
        self.assertEqual(tx.is_cancel.sum(),1)

    def test_blank_nonfinite_fractional(self):
        raw=transactions()
        raw['Quantity']=raw.Quantity.astype(float)
        raw.loc[0,'Invoice']=''
        raw.loc[1,'Quantity']=np.inf
        raw.loc[2,'Quantity']=1.5
        tx,_=clean(raw)
        self.assertEqual(len(tx),3)

    def test_mixed_supported_dates(self):
        raw=transactions();raw.loc[0,'InvoiceDate']='1/1/2020 00:00'
        tx,_=clean(raw)
        self.assertEqual(len(tx),6)

    def test_excel_both_sheets_aliases(self):
        temp_root=Path(__file__).resolve().parent/'temporary'
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as d:
            path=Path(d)/'two.xlsx'
            with pd.ExcelWriter(path) as writer:
                transactions().iloc[:3].to_excel(writer,sheet_name='Year1',index=False)
                transactions().iloc[3:].rename(columns={'Invoice':'InvoiceNo','Price':'UnitPrice','Customer ID':'CustomerID'}).to_excel(writer,sheet_name='Year2',index=False)
            tx,_=clean(read_file(path)); self.assertEqual(len(tx),6)
            self.assertEqual(len(read_file(path,sheets=['Year2'])),3)

    def test_chronological_purging(self):
        rows=[]
        for t in pd.date_range('2020-01-01',periods=12,freq='30D'):
            rows += [{'customer_id':i,'cutoff':t} for i in (1,2)]
        tr,va,te,info=chronological_split(pd.DataFrame(rows))
        self.assertLessEqual(tr.cutoff.max()+pd.Timedelta(days=90),va.cutoff.min())
        self.assertLessEqual(va.cutoff.max()+pd.Timedelta(days=90),te.cutoff.min())

    def test_too_short_fails(self):
        with self.assertRaises(InsufficientData):
            chronological_split(pd.DataFrame({'cutoff':[pd.Timestamp('2020-01-01')]}))

    def test_all_missing_feature_dimensions(self):
        frame=pd.DataFrame(1.,index=range(4),columns=FEATURES)
        frame[['avg_gap_days','gap_std_days','recency_gap_ratio']]=np.nan
        pre=preprocessor().fit(frame)
        self.assertEqual(pre.transform(frame).shape,(4,len(FEATURES)))
        self.assertTrue(np.isfinite(pre.transform(frame)).all())
        medians=pre.named_steps['impute'].statistics_.copy()
        pre.transform(frame.fillna(100000))
        np.testing.assert_array_equal(medians,pre.named_steps['impute'].statistics_)

    def test_degenerate_clustering(self):
        frame=pd.DataFrame({'recency_days':[1.]*20,'frequency':[1.]*20,'monetary':[10.]*20})
        for algorithm in ('agglomerative','dbscan'):
            m=RFMSegmenter(algorithm).fit(frame)
            self.assertIsNone(m.metrics['silhouette'])
            self.assertEqual(len(m.assign(frame)),20)

    def test_noise_and_undefined_metrics(self):
        z=np.zeros((4,3)); self.assertIsNone(quality(z,np.full(4,-1))['davies_bouldin'])
        rng=np.random.default_rng(7)
        x=np.r_[rng.normal(0,.02,(30,3)),rng.normal(3,.02,(30,3))]
        frame=pd.DataFrame(np.exp(x),columns=['recency_days','frequency','monetary'])
        m=RFMSegmenter('dbscan').fit(frame)
        core=m.core.copy()
        extreme=pd.DataFrame([[1e30,1e30,1e30]],columns=frame.columns)
        self.assertEqual(m.assign(extreme)[0],-1)
        np.testing.assert_array_equal(m.core,core)

    def test_all_noise_fit_and_assignment_are_safe(self):
        frame=pd.DataFrame({'recency_days':[1.,2.,3.,4.],
                            'frequency':[1.,2.,3.,4.], 'monetary':[10.,20.,30.,40.]})
        class AllNoise:
            def __init__(self,**kwargs):
                pass
            def fit(self,z):
                self.labels_=np.full(len(z),-1,dtype=int)
                self.core_sample_indices_=np.array([],dtype=int)
                return self
        with patch('academic.segmentation.DBSCAN',AllNoise):
            m=RFMSegmenter('dbscan').fit(frame)
        self.assertIsNone(m.metrics['silhouette'])
        self.assertIsNone(m.metrics['davies_bouldin'])
        self.assertEqual(m.metrics['noise_share'],1.)
        np.testing.assert_array_equal(m.assign(frame),np.full(len(frame),-1))

    def test_cluster_encoding_has_noise_category(self):
        e=encoder(np.array([0,0,1,1]));self.assertEqual(e.transform([[-1]]).sum(),1)

    def test_auc_uses_scores(self):
        y=np.array([0,1,0,1]);p=np.array([.1,.4,.2,.3])
        self.assertEqual(metrics(y,p,.5)['roc_auc'],1.)


if __name__ == '__main__':
    unittest.main()
