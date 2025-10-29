import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import os

# 1. MLflow Sunucusunun adresini ayarla
# ÇOK ÖNEMLİ: Jenkins Docker'da çalıştığı için 'localhost'u görmez.
# Buraya bilgisayarının ağdaki IP adresini yazmalısın.
# IP adresini öğrenmek için komut istemine 'ipconfig' yazabilirsin.
# Örneğin: "http://192.168.1.35:5000" gibi.
mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://192.168.1.35:5000") # IP ADRESİNİ GÜNCELLE
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