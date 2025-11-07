"""
AutoGluon MultiModal Model Eğitimi + MLflow entegrasyonu
Tabular (price, rating, num_reviews, discount, category) + Text (review_text) verisi kullanır
"""
import os
import shutil
from datetime import datetime

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from autogluon.multimodal import MultiModalPredictor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

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
        return self.predictor.predict(model_input)

    def update_model(self, new_data, new_labels, update_path=None):
        """Runtime'da model güncelleme (incremental/continuous learning)"""
        if self.predictor is None:
            raise ValueError("Model önce yüklenmelidir!")

        print("\n" + "=" * 60)
        print(f"MODEL GÜNCELLEME BAŞLADI - {datetime.now()}")
        print(f"Yeni veri sayısı: {len(new_data)}")
        print("=" * 60 + "\n")

        # Yeni veriyi etiketle
        new_data = new_data.copy()
        new_data["label"] = new_labels

        # Kaydetme yolu
        if update_path is None:
            update_path = (self.model_path or "./autogluon_multimodal_model") + "_updated"

        try:
            # Windows / GPU olmayan makine için CPU'yu zorla
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

            self.predictor = self.predictor.fit(
                new_data,
                time_limit=120,  # 2 dk hızlı güncelleme
                hyperparameters={
                    "optim.lr": 0.0001,
                    "optim.max_epochs": 5,
                    "env.num_workers": 0,
                    "env.num_gpus": -1,
                },
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
    """Ana eğitim fonksiyonu - MLflow ile tam entegrasyon"""

    # Deney adını belirle
    mlflow.set_experiment("autogluon_multimodal_experiment")

    with mlflow.start_run(run_name=f"multimodal_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        print("\n" + "=" * 60)
        print(f"MLflow Run ID: {run.info.run_id}")
        print("=" * 60 + "\n")

        # 1) Veri Hazırlama
        print("📊 Veri seti oluşturuluyor...")
        fast_mode = os.getenv("FAST", "0") == "1"
        n_samples = 120 if fast_mode else 500
        df = generate_multimodal_dataset(n_samples=n_samples)

        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["label"]
        )
        print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

        mlflow.log_param("total_samples", len(df))
        mlflow.log_param("train_samples", len(train_df))
        mlflow.log_param("test_samples", len(test_df))
        mlflow.log_param("num_features", len(df.columns) - 1)
        mlflow.log_param("feature_types", "tabular+text")

        # 2) Model Eğitimi
        print("\n🚀 AutoGluon MultiModal Model eğitimi başlıyor...")
        label_column = "label"
        model_path = "./autogluon_multimodal_model"
        if os.path.exists(model_path):
            shutil.rmtree(model_path)

        predictor = MultiModalPredictor(
            label=label_column,
            problem_type="binary",
            eval_metric="roc_auc",
            path=model_path,
        )

        # CPU zorla
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        hyperparameters = {
            "optim.lr": 0.0001,
            "optim.max_epochs": 10 if not fast_mode else 3,
            "env.num_workers": 0,
            "env.num_gpus": -1,
        }
        mlflow.log_params({f"hp_optim.{k.split('.')[-1]}": v for k, v in hyperparameters.items()})

        time_limit = 30 if fast_mode else 300
        predictor = predictor.fit(
            train_df,
            time_limit=time_limit,
            hyperparameters=hyperparameters,
        )

        print("\n✓ Model eğitimi tamamlandı!")

        # 3) Değerlendirme
        print("\n📈 Model değerlendiriliyor...")
        test_data = test_df.drop(columns=[label_column])
        y_true = test_df[label_column]
        y_pred = predictor.predict(test_data)
        y_pred_proba = predictor.predict_proba(test_data)

        # predict_proba çıktısı DataFrame olabilir, pozitif sınıfı ikinci sütundan al
        if hasattr(y_pred_proba, "iloc"):
            pos_proba = y_pred_proba.iloc[:, 1]
        else:
            pos_proba = np.array(y_pred_proba)[:, 1]

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, pos_proba)

        print("\n📊 Test Metrikleri:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  ROC AUC:  {auc:.4f}")

        mlflow.log_metric("test_accuracy", float(accuracy))
        mlflow.log_metric("test_f1_score", float(f1))
        mlflow.log_metric("test_roc_auc", float(auc))

        # 4) Modeli MLflow'a Kaydet
        print("\n💾 Model MLflow'a kaydediliyor...")
        artifacts = {"model": model_path}
        mlflow.pyfunc.log_model(
            artifact_path="multimodal_model",
            python_model=AutoGluonMultiModalWrapper(),
            artifacts=artifacts,
            pip_requirements=[
                "autogluon.multimodal>=1.0.0",
                f"mlflow>={mlflow.__version__}",
                "pandas>=2.0.0",
                "scikit-learn>=1.3.0",
            ],
        )

        print("\n✓ Model MLflow'a kaydedildi!")
        print("\n" + "=" * 60)
        print("MLflow UI: mlflow ui")
        print(f"Run ID: {run.info.run_id}")
        print("=" * 60 + "\n")

        return run.info.run_id, model_path


if __name__ == "__main__":
    train_and_log_model()
