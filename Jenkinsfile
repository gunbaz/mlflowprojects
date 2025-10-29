// Scripted Pipeline (Windows)
// Not: Declarative yerine scripted kullandık; bu sayede 'stages' DSL hatası yaşamazsınız.

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
    bat '''
      python -m pip install --upgrade pip
      python -m pip install -r requirements.txt
    '''
  }

  stage('3. Modeli Eğit ve MLflowa Kaydet (Train & Track)') {
    withEnv(["MLFLOW_TRACKING_URI=${params.MLFLOW_TRACKING_URI}"]) {
      bat 'python train.py'
    }
  }
}