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

  stage('2-3. Python Container içinde Kurulum ve Eğitim') {
    // Docker yüklü ajanlarda, Python ortamını container ile sağlar
    // Not: Jenkins’te Docker kullanılabilmesi için Docker kurulmuş olmalı ve docker-workflow eklentisi olmalı.
    def img = docker.image('python:3.11-slim')
    // MLflow URI’yi container’a geçiriyoruz
    img.pull()
    img.inside("-e MLFLOW_TRACKING_URI=${params.MLFLOW_TRACKING_URI}") {
      if (isUnix()) {
        sh '''
          set -e
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          python train.py
        '''
      } else {
        bat '''
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          python train.py
        '''
      }
    }
  }
}