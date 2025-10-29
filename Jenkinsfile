pipeline {
    agent { label 'windows' }

    parameters {
        string(name: 'MLFLOW_TRACKING_URI', defaultValue: 'http://127.0.0.1:5000', description: 'MLflow Tracking sunucusu URI (örn: http://192.168.1.35:5000)')
    }

    environment {
        MLFLOW_TRACKING_URI = "${params.MLFLOW_TRACKING_URI}"
    }

    stages {
        stage('1. Kodu Çek (Checkout)') {
            steps {
                // GitHub reposundan kodu çek
                git 'https://github.com/gunbaz/mlflowprojects.git'
            }
        }
        stage('2. Bağımlılıkları Kur (Install Dependencies)') {
            steps {
                // Windows ajanında Python paketlerini kur
                bat '''
                python -m pip install --upgrade pip
                python -m pip install -r requirements.txt
                '''
            }
        }
        stage('3. Modeli Eğit ve MLflowa Kaydet (Train & Track)') {
            steps {
                // Python scriptini çalıştır (env ile MLFLOW_TRACKING_URI aktarılıyor)
                bat 'python train.py'
            }
        }
    }
}pipeline {
    agent { label 'windows' }

    parameters {
        string(name: 'MLFLOW_TRACKING_URI', defaultValue: 'http://127.0.0.1:5000', description: 'MLflow Tracking sunucusu URI (örn: http://192.168.1.35:5000)')
    }

    environment {
        MLFLOW_TRACKING_URI = "${params.MLFLOW_TRACKING_URI}"
    }

    stages {
        stage('1. Kodu Çek (Checkout)') {
            steps {
                // GitHub reposundan kodu çek
                git 'https://github.com/gunbaz/mlflowprojects.git'
            }
        }
        stage('2. Bağımlılıkları Kur (Install Dependencies)') {
            steps {
                // Windows ajanında Python paketlerini kur
                bat '''
                python -m pip install --upgrade pip
                python -m pip install -r requirements.txt
                '''
            }
        }
        stage('3. Modeli Eğit ve MLflowa Kaydet (Train & Track)') {
            steps {
                // Python scriptini çalıştır (env ile MLFLOW_TRACKING_URI aktarılıyor)
                bat 'python train.py'
            }
        }
    }
}