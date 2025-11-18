pipeline {
    agent any
    
    stages {
        stage('Checkout') {
            steps {
                echo 'Kod cekiliyor...'
            }
        }
        
        stage('Install Dependencies') {
            steps {
                echo 'Bagimliliklar yukleniyor...'
                bat 'pip install dvc'
            }
        }
        
        stage('DVC Pull') {
            steps {
                echo 'DVC ile veri cekiliyor...'
                bat 'dvc pull'
            }
        }
        
        stage('Build Docker Image') {
            steps {
                echo 'Docker image olusturuluyor...'
                bat 'docker build -t autogluon-iris .'
            }
        }
        
        stage('Run Training') {
            steps {
                echo 'Model egitimi baslatiliyor...'
                bat 'docker run --rm autogluon-iris'
            }
        }
    }
    
    post {
        success {
            echo 'Pipeline basariyla tamamlandi!'
        }
        failure {
            echo 'Pipeline basarisiz oldu!'
        }
    }
}