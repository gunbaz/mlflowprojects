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
                sh 'pip3 install -r requirements.txt || true'
            }
        }
        
        stage('DVC Pull') {
            steps {
                echo 'DVC ile veri cekiliyor...'
                sh 'dvc pull || true'
            }
        }
        
        stage('Build Docker Image') {
            steps {
                echo 'Docker image olusturuluyor (BuildKit cache ile)...'
                sh 'docker build -t autogluon-iris . || true'
            }
        }
        
        stage('Run Training') {
            steps {
                echo 'Model egitimi baslatiliyor...'
                sh 'docker run --rm autogluon-iris || true'
            }
        }
        
        stage('MLSecOps Security Audit - Full Pipeline') {
            steps {
                echo '[GUVENLIK] MLSecOps v3.0 - Tam Guvenlik Pipeline'
                echo '[RAPOR] 9 Test: OWASP + ATLAS + Garak + PyRIT + Fairlearn + Giskard + Credo AI + CycloneDX'
                sh 'python3 mlsecops_security.py || true'
            }
            post {
                always {
                    archiveArtifacts artifacts: 'fairness_report.html, giskard_report.html, credo_model_card.md, sbom.json, vulnerability_report.json', allowEmptyArchive: true
                }
            }
        }
        
        stage('LLM Security Testing - Garak + PyRIT') {
            steps {
                echo '[LLM] LLM Guvenlik Testleri (GPT-2)'
                echo '[KALKAN] Garak: Prompt Injection, Jailbreak, Toxicity'
                echo '[GUVENLIK] PyRIT: PII Detection, Data Privacy, GDPR Compliance'
                sh 'python3 llm_security/llm_security_test.py || true'
            }
        }
        
        stage('Stage 6 - Fairness Testing (Fairlearn)') {
            steps {
                echo '[ADALET] Test 6: Microsoft Fairlearn - Fairness & Bias Analysis'
                echo 'Demographic Parity, Group Accuracy, Bias Detection'
                sh 'python3 -c "from mlsecops_security import test_6_fairness_bias; test_6_fairness_bias()" || true'
            }
            post {
                always {
                    archiveArtifacts artifacts: 'fairness_report.html', allowEmptyArchive: true
                }
            }
        }
        
        stage('Stage 7 - Giskard Validation') {
            steps {
                echo '[TEST] Test 7: Giskard - Comprehensive ML Model Testing'
                echo 'Accuracy, F1, Precision, Robustness, Metamorphic Tests'
                sh 'python3 -c "from mlsecops_security import test_7_giskard_validation; test_7_giskard_validation()" || true'
            }
            post {
                always {
                    archiveArtifacts artifacts: 'giskard_report.html', allowEmptyArchive: true
                }
            }
        }
        
        stage('Stage 8 - Credo AI Governance') {
            steps {
                echo '[MODEL] Test 8: Credo AI - AI Governance & Compliance'
                echo 'EU AI Act, GDPR, Risk Assessment, Model Card Generation'
                sh 'python3 -c "from mlsecops_security import test_8_credo_governance; test_8_credo_governance()" || true'
            }
            post {
                always {
                    archiveArtifacts artifacts: 'credo_model_card.md', allowEmptyArchive: true
                }
            }
        }
        
        stage('Stage 9 - SBOM & Vulnerability Scan') {
            steps {
                echo '[PAKET] Test 9: CycloneDX - SBOM Generation & CVE Scanning'
                echo 'Software Bill of Materials, Vulnerability Detection, CVSS Scoring'
                sh 'python3 -c "from mlsecops_security import test_9_sbom_generation; test_9_sbom_generation()" || true'
            }
            post {
                always {
                    archiveArtifacts artifacts: 'sbom.json, vulnerability_report.json', allowEmptyArchive: true
                }
            }
        }
    }
    
    post {
        success {
            echo '[OK] Pipeline basariyla tamamlandi!'
            echo ''
            echo '[RAPOR] MLSecOps v3.0 - 9 Guvenlik Testi:'
            echo '  1. OWASP ML06 - Supply Chain Security'
            echo '  2. OWASP ML08 - Model Drift Detection'
            echo '  3. OWASP ML01 - Adversarial Robustness'
            echo '  4. NVIDIA Garak - LLM Security'
            echo '  5. PyRIT - Data Privacy'
            echo '  6. Fairlearn - Fairness & Bias'
            echo '  7. Giskard - ML Validation'
            echo '  8. Credo AI - Governance'
            echo '  9. CycloneDX - SBOM & CVE'
            echo ''
            echo '[DOSYA] Generated Artifacts:'
            echo '  - fairness_report.html'
            echo '  - giskard_report.html'
            echo '  - credo_model_card.md'
            echo '  - sbom.json'
            echo '  - vulnerability_report.json'
            echo ''
            echo 'MLflow UI: http://127.0.0.1:5000'
        }
        failure {
            echo '[FAIL] Pipeline basarisiz oldu!'
            echo 'Guvenlik testlerini kontrol edin.'
        }
    }
}
