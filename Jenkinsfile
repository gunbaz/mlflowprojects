// Scripted Pipeline (Cross-platform)
// Not: Declarative yerine scripted; Linux ajanlarda sh, Windows ajanlarda bat kullanır.

properties([
  parameters([
    string(name: 'MLFLOW_TRACKING_URI', defaultValue: 'http://127.0.0.1:5000', description: 'MLflow Tracking sunucusu URI (örn: http://192.168.1.35:5000)')
  ])
])

// Etiket kısıtı kaldırıldı; mevcut herhangi bir executorda çalışır
node {
  stage('1. Kodu Çek (Checkout)') {
    // Bu job zaten SCM’den tetikleniyorsa checkout scm yeterlidir
    checkout scm
  }

  stage('2. Bağımlılıkları Kur (Install Dependencies)') {
    if (isUnix()) {
      sh '''
        set -e
        (python3 -m pip --version >/dev/null 2>&1 || python -m ensurepip --upgrade || true)
        (python3 -m pip install --upgrade pip || python -m pip install --upgrade pip)
        (python3 -m pip install -r requirements.txt || python -m pip install -r requirements.txt)
      '''
    } else {
      bat '''
        python -m pip install --upgrade pip
        python -m pip install -r requirements.txt
      '''
    }
  }

  stage('3. Modeli Eğit ve MLflowa Kaydet (Train & Track)') {
    withEnv(["MLFLOW_TRACKING_URI=${params.MLFLOW_TRACKING_URI}"]) {
      if (isUnix()) {
        sh '(python3 train.py || python train.py)'
      } else {
        bat 'python train.py'
      }
    }
  }
}