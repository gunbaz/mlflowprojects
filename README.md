# AutoGluon Iris - MLSecOps Security Pipeline 🛡️

AutoGluon ile Iris veri seti üzerinde makine öğrenimi modeli eğitimi ve **kapsamlı güvenlik testleri** içeren MLSecOps pipeline projesi.

## 🔒 Güvenlik Özellikleri

Bu proje, modern ML güvenlik standartlarını uygular:

### NVIDIA Garak - LLM Güvenlik Tarayıcı
- ✅ Prompt Injection saldırı testi
- ✅ Jailbreak denemesi simülasyonu
- ✅ Toxicity ve bias tespiti
- ✅ Hallucination kontrolü
- ✅ PII sızıntısı testi

### PyRIT - Veri Güvenliği
- ✅ PII Detection (Presidio)
- ✅ Sensitive data risk analizi
- ✅ GDPR/KVKK compliance kontrolü
- ✅ Veri gizliliği testleri

### OWASP ML Top 10 + MITRE ATLAS
- ✅ **ML06**: AI Supply Chain Attacks (Bandit, Safety)
- ✅ **ML08**: Model Skewing / Drift Detection
- ✅ **ML01**: Input Manipulation / Adversarial Testing
- ✅ **ML09**: Data Poisoning / Privacy

## 🚀 Hızlı Başlangıç

### Gereksinimler
- Python 3.12+
- Docker Desktop
- Jenkins
- DVC (Data Version Control)

### Kurulum

1. **Bağımlılıkları yükle:**
```bash
pip install -r requirements.txt
```

2. **Veriyi çek (DVC):**
```bash
dvc pull
```

3. **Model eğitimi:**
```bash
python train.py
```

4. **Güvenlik testleri:**
```bash
python mlsecops_security.py
```

5. **MLflow UI:**
```bash
python -m mlflow ui
# http://127.0.0.1:5000
```

## 🐳 Docker Kullanımı

```bash
# Image oluştur
docker build -t autogluon-iris .

# Container çalıştır
docker run --rm autogluon-iris
```

## 🔧 Jenkins Pipeline

Jenkins'te build etmek için:

1. **Docker Desktop'ı aç**
2. Jenkins'te yeni pipeline oluştur
3. Bu repository'yi bağla
4. Pipeline'ı çalıştır

Pipeline aşamaları:
- ✅ Checkout
- ✅ Install Dependencies
- ✅ DVC Pull
- ✅ Build Docker Image
- ✅ Run Training
- ✅ MLSecOps Security Audit
- ✅ NVIDIA Garak LLM Security
- ✅ PyRIT Data Security

## 📊 MLflow Sonuçları

Tüm güvenlik testleri ve model metrikleri MLflow'a otomatik loglanır:

### Garak Metrikleri
- `garak_vulnerabilities`: Tespit edilen güvenlik açıkları
- `prompt_injection_risk`: Prompt injection risk skoru
- `toxicity_score`: Toxicity seviyesi
- `jailbreak_attempts`: Jailbreak denemesi sayısı

### PyRIT Metrikleri
- `pii_detected`: Tespit edilen PII sayısı
- `sensitive_data_risk`: Veri gizliliği risk skoru
- `compliance_score`: GDPR/KVKK uyumluluk skoru

### Model Metrikleri
- `accuracy`: Model doğruluğu
- `balanced_accuracy`: Dengeli doğruluk
- `mcc`: Matthews Correlation Coefficient
- `robustness_score`: Adversarial sağlamlık skoru

## 📁 Proje Yapısı

```
autogloun_iris/
├── data/                    # Veri seti (DVC ile yönetilir)
├── autogluon_models/        # Eğitilmiş modeller
├── mlruns/                  # MLflow çalıştırmaları
├── train.py                 # Model eğitim scripti
├── mlsecops_security.py     # Güvenlik test pipeline'ı
├── Jenkinsfile              # Jenkins pipeline tanımı
├── Dockerfile               # Docker image tanımı
├── requirements.txt         # Python bağımlılıkları
└── README.md                # Bu dosya
```

## 🛡️ Güvenlik Framework'leri

- **OWASP ML Top 10**: ML sistemleri için güvenlik standartları
- **MITRE ATLAS**: AI/ML saldırı taktikleri ve teknikleri
- **NVIDIA Garak**: LLM güvenlik tarayıcı
- **Microsoft PyRIT**: Veri güvenliği ve gizlilik

## 📝 Lisans

MIT License

## 👨‍💻 Geliştirici

MLSecOps Pipeline v2.0 - Garak + PyRIT + OWASP + ATLAS
