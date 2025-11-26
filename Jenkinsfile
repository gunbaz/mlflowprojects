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
                bat 'pip install -r requirements.txt'
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
        
        stage('MLSecOps Security Audit - Iris') {
            steps {
                echo '📊 MLSecOps Guvenlik Denetimi (Iris Dataset)'
                echo 'OWASP ML Top 10 + MITRE ATLAS'
                bat 'python mlsecops_security.py'
            }
        }
        
        stage('LLM Security Testing - Garak + PyRIT') {
            steps {
                echo '🤖 LLM Guvenlik Testleri (GPT-2)'
                echo '🛡️ Garak: Prompt Injection, Jailbreak, Toxicity'
                echo '🔒 PyRIT: PII Detection, Data Privacy, GDPR Compliance'
                bat 'python llm_security/llm_security_test.py'
            }
        }
    }
    
    post {
        success {
            echo '✅ Pipeline basariyla tamamlandi!'
            echo '📊 Iris MLSecOps Security: PASSED'
            echo '🤖 LLM Security (Garak + PyRIT): PASSED'
            echo ''
            echo 'MLflow Experiments:'
            echo '  1. MLSecOps-Security-Audit (Iris)'
            echo '  2. LLM-Security-Garak-PyRIT (GPT-2)'
            echo ''
            echo 'MLflow UI: http://127.0.0.1:5000'
        }
        failure {
            echo '❌ Pipeline basarisiz oldu!'
            echo 'Guvenlik testlerini kontrol edin.'
        }
    }
}