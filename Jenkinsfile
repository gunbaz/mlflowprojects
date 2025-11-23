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
        
        stage('MLSecOps Security Audit') {
            steps {
                echo 'MLSecOps Guvenlik Denetimi Basliyor...'
                echo 'OWASP ML Top 10 + MITRE ATLAS + Garak + PyRIT'
                bat 'python mlsecops_security.py'
            }
        }
        
        stage('NVIDIA Garak LLM Security') {
            steps {
                echo '🛡️ NVIDIA Garak - LLM Guvenlik Taramasi'
                echo 'Prompt Injection, Jailbreak, Toxicity Testleri'
                bat 'python -c "from mlsecops_security import run_garak_security_scan; run_garak_security_scan()"'
            }
        }
        
        stage('PyRIT Data Security') {
            steps {
                echo '🔒 PyRIT - Veri Guvenliği ve Gizlilik Testi'
                echo 'PII Detection, Compliance Kontrolu'
                bat 'python -c "from mlsecops_security import run_pyrit_data_security; run_pyrit_data_security()"'
            }
        }
    }
    
    post {
        success {
            echo '✅ Pipeline basariyla tamamlandi!'
            echo 'MLSecOps guvenlik denetimi tamamlandi.'
            echo '🛡️ Garak LLM Security: PASSED'
            echo '🔒 PyRIT Data Security: PASSED'
            echo 'MLflow UI: http://127.0.0.1:5000'
        }
        failure {
            echo '❌ Pipeline basarisiz oldu!'
            echo 'Guvenlik testlerini kontrol edin.'
        }
    }
}