"""Checks the actual full-data outputs, when present; no model retraining needed."""
import hashlib
import json
from pathlib import Path
import unittest
import joblib
import numpy as np
import pandas as pd
from academic.features import FEATURES
from academic.classification import metrics, threshold
from academic.classification import preprocessor

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/full'


@unittest.skipUnless((OUT/'models.joblib').exists(), 'Run the full experiment first')
class SavedResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models=joblib.load(OUT/'models.joblib')
        cls.test=pd.read_csv(OUT/'test_observations.csv')
        cls.validation=pd.read_csv(OUT/'validation_observations.csv')
        cls.predictions=pd.read_csv(OUT/'held_out_predictions.csv')
        cls.table=pd.read_csv(OUT/'classification_comparison.csv')
        cls.manifest=json.loads((OUT/'run_manifest.json').read_text())

    def test_full_workbook_and_shared_cohort(self):
        self.assertEqual(self.manifest['raw_rows'],1067371)
        self.assertEqual(len(self.manifest['sheets']),2)
        self.assertFalse(self.manifest['provisional'])
        self.assertEqual(len(self.models),6)
        self.assertEqual(self.table.test_observations_sha256.nunique(),1)
        self.assertEqual(self.table.validation_observations_sha256.nunique(),1)
        np.testing.assert_array_equal(self.test.customer_id,self.predictions.customer_id)
        np.testing.assert_array_equal(self.test.churned,self.predictions.churned)

    def test_frozen_models_reproduce_probabilities_and_metrics(self):
        for key,bundle in self.models.items():
            x=bundle['preprocessor'].transform(self.test[FEATURES])
            if bundle['segmenter'] is not None:
                lab=bundle['segmenter'].assign(self.test)
                np.testing.assert_array_equal(lab,self.predictions[key.split('__')[1]+'_segment'])
                x=np.c_[x,bundle['encoder'].transform(lab.reshape(-1,1))]
            probabilities=bundle['classifier'].predict_proba(x)[:,1]
            np.testing.assert_allclose(probabilities,self.predictions[key],atol=1e-12,rtol=1e-12)
            name='Logistic Regression' if key.startswith('logistic') else 'Random Forest'
            row=self.table[(self.table.model==name)&(self.table.features==key.split('__')[1])].iloc[0]
            for metric,value in metrics(self.test.churned,probabilities,bundle['threshold']).items():
                self.assertAlmostEqual(value,row[metric],places=12)
            x=bundle['preprocessor'].transform(self.validation[FEATURES])
            if bundle['segmenter'] is not None:
                lab=bundle['segmenter'].assign(self.validation)
                x=np.c_[x,bundle['encoder'].transform(lab.reshape(-1,1))]
            pv=bundle['classifier'].predict_proba(x)[:,1]
            self.assertAlmostEqual(threshold(self.validation.churned,pv),bundle['threshold'],places=12)

    def test_chronology_and_no_target_features(self):
        split=self.manifest['split'];h=pd.Timedelta(days=90)
        self.assertLessEqual(pd.Timestamp(split['train_cutoff'])+h,pd.Timestamp(split['validation_cutoff']))
        self.assertLessEqual(pd.Timestamp(split['validation_cutoff'])+h,pd.Timestamp(split['test_cutoff']))
        self.assertLessEqual(pd.Timestamp(split['test_cutoff'])+h,pd.Timestamp(self.manifest['observation_end_exclusive']))
        self.assertTrue(set(FEATURES).isdisjoint({'churned','future_spend','customer_id','cutoff'}))

    def test_preprocessing_fitted_only_on_training(self):
        train=pd.read_csv(OUT/'train_observations.csv')
        expected=preprocessor().fit(train[FEATURES])
        for bundle in self.models.values():
            actual=bundle['preprocessor']
            np.testing.assert_allclose(expected.named_steps['impute'].statistics_,actual.named_steps['impute'].statistics_)
            np.testing.assert_allclose(expected.named_steps['scale'].mean_,actual.named_steps['scale'].mean_)
            if bundle['segmenter'] is not None:
                np.testing.assert_allclose(bundle['segmenter'].scaler.mean_,
                    np.log1p(train[['recency_days','frequency','monetary']]).mean(axis=0))

    def test_original_files_preserved(self):
        manifest_path=ROOT/'legacy_manifest.json'
        if not manifest_path.exists():
            self.skipTest('Original application not included in portable academic archive')
        for relative,sha in json.loads(manifest_path.read_text()).items():
            path=ROOT/'customer_intelligence'/relative
            self.assertTrue(path.exists(),relative)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),sha,relative)


if __name__=='__main__':
    unittest.main()
