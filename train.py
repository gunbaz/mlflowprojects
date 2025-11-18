from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import mlflow
import os

# MLflow başlat
mlflow.set_experiment("autogluon_iris")
mlflow.start_run()

# Veri seti yükle
data = TabularDataset("data/iris.csv")

# Hedef sütun
label = "target"

# Eğitim / test ayır
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

# MLflow parametreleri logla
mlflow.log_param("dataset", "iris")
mlflow.log_param("label", label)
mlflow.log_param("train_size", len(train_data))
mlflow.log_param("test_size", len(test_data))
mlflow.log_param("presets", "medium_quality")

# Predictor (model)
predictor = TabularPredictor(
    label=label,
    path="autogluon_models/"
)

print("Model eğitiliyor...")
predictor.fit(train_data, presets="medium_quality")

# Tahmin
test_no_label = test_data.drop(columns=[label])
y_true = test_data[label]

preds = predictor.predict(test_no_label)

# Değerlendirme
results = predictor.evaluate_predictions(
    y_true=y_true,
    y_pred=preds,
    auxiliary_metrics=True
)

print("\n--- MODEL SONUÇLARI ---")
print(results)

# MLflow metrikleri logla
mlflow.log_metric("accuracy", results['accuracy'])
mlflow.log_metric("balanced_accuracy", float(results['balanced_accuracy']))
mlflow.log_metric("mcc", results['mcc'])

# Leaderboard kaydet
leaderboard = predictor.leaderboard(silent=True)
leaderboard.to_csv("leaderboard.csv", index=False)
print("\nLeaderboard kaydedildi → leaderboard.csv")

# MLflow artifact olarak kaydet
mlflow.log_artifact("leaderboard.csv")

# MLflow bitir
mlflow.end_run()

print("\n✅ MLflow loglama tamamlandı!")