import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import os
import requests

def _resolve_tracking_uri() -> str:
    """
    MLFLOW_TRACKING_URI ortam değişkenini okur.
    - Boşsa: yerel dosya deposu (file:./mlruns) kullan.
    - http/https ise: kısa bir health/probe isteğiyle erişilebilirliği test et; erişilemiyorsa file:./mlruns'a düş.
    - file: veya diğer şemalarda: olduğu gibi kullan.
    """
    env_val = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not env_val:
        return "file:./mlruns"
    if env_val.lower().startswith("http"):
        try:
            url = env_val.rstrip("/") + "/api/2.0/mlflow/experiments/list"
            # 2 saniyelik kısa timeout ile dene
            requests.get(url, timeout=2)
            return env_val
        except Exception as e:
            print(f"Uyarı: MLflow sunucusuna bağlanılamadı ({e}). 'file:./mlruns' ile devam ediliyor.")
            return "file:./mlruns"
    return env_val

# 1. MLflow Tracking URI'yi belirle ve ayarla (otomatik fallback ile)
mlflow_tracking_uri = _resolve_tracking_uri()
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Deneyin ismini belirle
mlflow.set_experiment("Odev-Jenkins-Entegrasyonu")

print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

# 2. Basit bir model eğitimi
data = {
    'X1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'X2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    'y':  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)
X = df[['X1', 'X2']]
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. MLflow ile denemeyi başlat
with mlflow.start_run() as run:
    C = 1.5
    solver = 'liblinear'

    # Parametreleri MLflow'a kaydet
    mlflow.log_param("C", C)
    mlflow.log_param("solver", solver)
    print(f"Run ID: {run.info.run_id}")

    # Modeli eğit
    model = LogisticRegression(C=C, solver=solver)
    model.fit(X_train, y_train)

    # Tahmin yap ve metriği hesapla
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    # Metriği MLflow'a kaydet
    mlflow.log_metric("accuracy", accuracy)
    print(f"Accuracy: {accuracy}")

    # Modeli bir "artifact" olarak MLflow'a kaydet
    mlflow.sklearn.log_model(model, "model")

    print("Model ve metrikler MLflow'a başarıyla kaydedildi.")  