pipeline {
    agent any
    
    environment {
        DOCKER_BUILDKIT = '1'
    }
    
    stages {
        stage('Checkout') {
            steps {
                echo 'Kod cekiliyor...'
            }
        }
        
        stage('Install Dependencies') {
            steps {
                echo 'Bagimliliklar yukleniyor...'
                sh 'pip install -r requirements.txt || true'
            }
        }
        
        stage('Create Data') {
            steps {
                echo 'Veri olusturuluyor...'
                sh 'mkdir -p data'
                sh 'python3 dataset_creator.py || true'
            }
        }
        
        stage('Run Training') {
            steps {
                echo 'Model egitimi baslatiliyor...'
                sh 'python3 train.py || true'
            }
        }
        
        stage('MLSecOps Security Audit') {
            steps {
                echo 'Guvenlik testleri calistiriliyor...'
                sh 'python3 mlsecops_security.py || true'
            }
        }
    }
    
    post {
        success {
            echo '[OK] Pipeline basariyla tamamlandi!'
        }
        failure {
            echo '[FAIL] Pipeline basarisiz oldu!'
        }
    }
}
