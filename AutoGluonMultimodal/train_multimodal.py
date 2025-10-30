"""
AutoGluon MultiModal Model Eğitimi + MLflow Integration
Tabular (price, rating, num_reviews, discount, category) + Text (review_text) verisi kullanır
"""
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from autogluon.multimodal import MultiModalPredictor
import os
import shutil
from datetime import datetime

from data_generator import generate_multimodal_dataset


class AutoGluonMultiModalWrapper(mlflow.pyfunc.PythonModel):
    """
    MLflow için AutoGluon MultiModalPredictor wrapper
    Runtime'da model güncellemesi destekler
    """
    
    def __init__(self):
        self.predictor = None
        self.model_path = None
        
    def load_context(self, context):
        """MLflow model yükleme"""
        self.model_path = context.artifacts["model"]
        self.predictor = MultiModalPredictor.load(self.model_path)
        print(f"Model yüklendi: {self.model_path}")
        
    def predict(self, context, model_input):
        """Tahmin yapma"""
        if self.predictor is None:
            raise ValueError("Model yüklenmemiş!")
        
        predictions = self.predictor.predict(model_input)
        return predictions
    
    def update_model(self, new_data, new_labels, update_path=None):
        """
        Runtime'da model güncelleme (incremental learning)
        """
        if self.predictor is None:
            raise ValueError("Model önce yüklenmelidir!")
        
        print(f"\n{'='*60}")
        print(f"MODEL GÜNCELLEME BAŞLADI - {datetime.now()}")
        print(f"Yeni veri sayısı: {len(new_data)}")
        print(f"{'='*60}\n")
        
        # Yeni veriyi ekleyerek model güncelleme
        new_data['label'] = new_labels
        
        # Güncellenmiş model için yeni path
        if update_path is None:
            update_path = self.model_path + "_updated"
        
        # Continuous learning - mevcut modeli fine-tune et
        try:
            self.predictor = self.predictor.fit(
                new_data,
                time_limit=120,  # 2 dakika
                hyperparameters={
                    'optim.lr': 0.0001,  # Düşük LR ile fine-tune
                    'optim.max_epochs': 5,
                }
            )
            
            # Güncellenmiş modeli kaydet
            self.predictor.save(update_path)
            self.model_path = update_path
            
            print(f"\n✓ Model güncellendi ve kaydedildi: {update_path}")
            return True
            
        except Exception as e:
            print(f"✗ Model güncelleme hatası: {e}")
            return False


def train_and_log_model():
    """
    Ana eğitim fonksiyonu - MLflow ile tam entegrasyon
    """
    
    # MLflow setup
    mlflow.set_experiment("autogluon_multimodal_experiment")
    
    with mlflow.start_run(run_name=f"multimodal_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        
        print(f"\n{'='*60}")
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"{'='*60}\n")
        
        # 1. Veri Hazırlama
        print("📊 Veri seti oluşturuluyor...")
        df = generate_multimodal_dataset(n_samples=500)
        
        # Train-test split
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        
        print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        
        # MLflow'a veri istatistikleri logla
        mlflow.log_param("total_samples", len(df))
        mlflow.log_param("train_samples", len(train_df))
        mlflow.log_param("test_samples", len(test_df))
        mlflow.log_param("num_features", len(df.columns) - 1)
        mlflow.log_param("feature_types", "tabular+text")
        
        # 2. Model Eğitimi
        print("\n🚀 AutoGluon MultiModal Model eğitimi başlıyor...")
        
        label_column = 'label'
        model_path = "./autogluon_multimodal_model"
        
        # Eski model varsa sil
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        
        # MultiModalPredictor oluştur
        predictor = MultiModalPredictor(
            label=label_column,
            problem_type='binary',
            eval_metric='roc_auc',
            path=model_path
        )
        
        # Hyperparameters (güncel API)
        hyperparameters = {
            'optim.lr': 0.0001,
            'optim.max_epochs': 10,
            'env.num_workers': 0,
            'env.num_gpus': 0,  # CPU modunda çalış
        }
        
        mlflow.log_params({f"hp_{k}": v for k, v in hyperparameters.items()})
        
        # Eğitim
        predictor = predictor.fit(
            train_df,
            time_limit=300,  # 5 dakika
            hyperparameters=hyperparameters
        )
        
        print("\n✓ Model eğitimi tamamlandı!")
        
        # 3. Değerlendirme
        print("\n📈 Model değerlendiriliyor...")
        
        # Test predictions
        test_data = test_df.drop(columns=[label_column])
        y_true = test_df[label_column]
        y_pred = predictor.predict(test_data)
        y_pred_proba = predictor.predict_proba(test_data)
        
        # Metrikler
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_proba.iloc[:, 1])
        
        print(f"\n📊 Test Metrikleri:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  ROC AUC:  {auc:.4f}")
        
        # MLflow'a logla
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1_score", f1)
        mlflow.log_metric("test_roc_auc", auc)
        
        # 4. Model kaydetme (MLflow)
        print("\n💾 Model MLflow'a kaydediliyor...")
        
        # Custom wrapper ile kaydet
        conda_env = {
            'channels': ['defaults', 'conda-forge'],
            'dependencies': [
                'python=3.10',
                'pip',
                {
                    'pip': [
                        f'autogluon.multimodal>=1.0.0',
                        f'mlflow>={mlflow.__version__}',
                        'pandas>=2.0.0',
                        'scikit-learn>=1.3.0',
                    ]
                }
            ],
            'name': 'autogluon_env'
        }
        
        artifacts = {"model": model_path}
        
        mlflow.pyfunc.log_model(
            artifact_path="multimodal_model",
            python_model=AutoGluonMultiModalWrapper(),
            artifacts=artifacts,
            conda_env=conda_env
        )
        
        print(f"\n✓ Model MLflow'a kaydedildi!")
        print(f"\n{'='*60}")
        print(f"MLflow UI: mlflow ui")
        print(f"Run ID: {run.info.run_id}")
        print(f"{'='*60}\n")
        
        return run.info.run_id, model_path


if __name__ == "__main__":
    run_id, model_path = train_and_log_model()
    print(f"\n🎉 Başarıyla tamamlandı!")
    print(f"Run ID: {run_id}")
    print(f"Model Path: {model_path}")
