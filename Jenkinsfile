// Scripted Pipeline (Cross-platform)
// Not: Declarative yerine scripted; Linux ajanlarda sh, Windows ajanlarda bat kullanır.

properties([
  parameters([
    string(name: 'MLFLOW_TRACKING_URI', defaultValue: 'file:./mlruns', description: 'MLflow Tracking URI. Varsayılan: file:./mlruns (yerel dosya). Uzak sunucu kullanacaksanız örn: http://192.168.1.35:5000')
  ])
])

// Etiket kısıtı kaldırıldı; mevcut herhangi bir executorda çalışır
node {
  stage('1. Kodu Çek (Checkout)') {
    // Bu job zaten SCM’den tetikleniyorsa checkout scm yeterlidir
    checkout scm
  }

  stage('2. Ortamı Hazırla (Prepare Environment)') {
    if (isUnix()) {
      sh '''
        set -e
        python3 -m venv .venv || (python3 -m pip install --break-system-packages virtualenv && python3 -m virtualenv .venv)
        . .venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
      '''
    } else {
      bat '''
        python -m venv .venv
        call .venv\\Scripts\\activate
        python -m pip install --upgrade pip
        python -m pip install -r requirements.txt
      '''
    }
  }

  stage('3. Modeli Eğit ve MLflowa Kaydet (Train & Track)') {
    withEnv(["MLFLOW_TRACKING_URI=${params.MLFLOW_TRACKING_URI}"]) {
      if (isUnix()) {
        sh '''
          set -e
          . .venv/bin/activate
          python train.py
        '''
      } else {
        bat '''
          call .venv\\Scripts\\activate
          python train.py
        '''
      }
    }
  }
}