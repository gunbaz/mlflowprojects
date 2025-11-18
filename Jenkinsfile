pipeline {
    agent any
    
    stages {
        stage('Checkout') {
            steps {
                echo 'Kod cekiliyor...'
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