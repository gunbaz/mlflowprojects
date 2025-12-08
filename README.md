# AutoGluon Iris - MLSecOps Security Pipeline v3.0 🛡️

AutoGluon ile Iris veri seti üzerinde makine öğrenimi modeli eğitimi ve **kapsamlı 9 güvenlik testi** içeren MLSecOps pipeline projesi.

## 🔒 Güvenlik Özellikleri

Bu proje, modern ML güvenlik standartlarını uygular ve **9 farklı güvenlik testi** içerir:

### Test 1-5: Temel Güvenlik Testleri

| Test | Framework | Açıklama |
|------|-----------|----------|
| T1 | OWASP ML06 | Supply Chain Security (Bandit, Safety) |
| T2 | OWASP ML08 | Model Drift Detection |
| T3 | OWASP ML01 | Adversarial Robustness Testing |
| T4 | NVIDIA Garak | LLM Security (Prompt Injection, Jailbreak) |
| T5 | PyRIT | Data Privacy & PII Detection |

### Test 6-9: Yeni Eklenen Testler ✨

| Test | Framework | Açıklama |
|------|-----------|----------|
| T6 | **Microsoft Fairlearn** | Fairness & Bias Analysis |
| T7 | **Giskard** | ML Model Validation (Accuracy, F1, Robustness) |
| T8 | **Credo AI** | AI Governance & EU AI Act Compliance |
| T9 | **CycloneDX** | SBOM Generation & CVE Vulnerability Scan |

## 📊 Oluşturulan Raporlar

Pipeline çalıştırıldığında aşağıdaki raporlar otomatik oluşturulur:

| Dosya | Açıklama |
|-------|----------|
| `fairness_report.html` | Fairlearn adalet ve önyargı raporu |
| `giskard_report.html` | Giskard ML validasyon raporu |
| `credo_model_card.md` | AI Model Card (EU AI Act uyumlu) |
| `sbom.json` | CycloneDX Software Bill of Materials |
| `vulnerability_report.json` | CVE güvenlik açığı raporu |

## 🚀 Hızlı Başlangıç

### Gereksinimler
- Python 3.12+
- Docker Desktop
- Jenkins
- DVC (Data Version Control)

### Kurulum

```bash
# 1. Bağımlılıkları yükle
pip install -r requirements.txt

# 2. Veriyi çek (DVC)
dvc pull

# 3. Model eğitimi
python train.py

# 4. Güvenlik testleri (9 test)
python mlsecops_security.py

# 5. MLflow UI
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

Pipeline aşamaları:

| Stage | Açıklama |
|-------|----------|
| Checkout | Kod çekme |
| Install Dependencies | Bağımlılık kurulumu |
| DVC Pull | Veri çekme |
| Build Docker Image | Docker image oluşturma |
| Run Training | Model eğitimi |
| MLSecOps Security Audit | 9 güvenlik testi (tam pipeline) |
| LLM Security Testing | Garak + PyRIT |
| **Stage 6 - Fairness Testing** | Fairlearn bias analizi |
| **Stage 7 - Giskard Validation** | ML model doğrulama |
| **Stage 8 - Credo AI Governance** | AI yönetişim değerlendirmesi |
| **Stage 9 - SBOM & Vulnerability** | SBOM + CVE taraması |

## 📊 MLflow Metrikleri

Tüm testler MLflow'a otomatik loglanır:

### Fairlearn (T6)
- `T6_Fairness_Score` - Adalet skoru
- `T6_Demographic_Parity_Diff` - Demografik parite farkı
- `T6_Group_A_Accuracy`, `T6_Group_B_Accuracy` - Grup bazlı doğruluk

### Giskard (T7)
- `T7_Giskard_Pass_Rate` - Test geçme oranı
- `T7_Giskard_Tests_Passed`, `T7_Giskard_Tests_Failed`

### Credo AI (T8)
- `T8_Credo_Overall_Risk` - Genel risk skoru
- `T8_Credo_Fairness_Risk`, `T8_Credo_Privacy_Risk`, `T8_Credo_Transparency_Risk`

### CycloneDX (T9)
- `T9_SBOM_Components` - Toplam bileşen sayısı
- `T9_SBOM_Vulnerabilities` - Güvenlik açığı sayısı
- `T9_Critical_Vulns`, `T9_High_Vulns`, `T9_Medium_Vulns`

## 📁 Proje Yapısı

```
autogloun_iris/
├── data/                    # Veri seti (DVC ile yönetilir)
├── autogluon_models/        # Eğitilmiş modeller
├── mlruns/                  # MLflow çalıştırmaları
├── llm_security/            # LLM güvenlik testleri
├── train.py                 # Model eğitim scripti
├── mlsecops_security.py     # 9 güvenlik testi pipeline'ı
├── Jenkinsfile              # Jenkins pipeline (11 stage)
├── Dockerfile               # Docker image (Grype dahil)
├── requirements.txt         # Python bağımlılıkları
├── fairness_report.html     # Fairlearn raporu
├── giskard_report.html      # Giskard raporu
├── credo_model_card.md      # AI Model Card
├── sbom.json                # CycloneDX SBOM
├── vulnerability_report.json # CVE raporu
└── README.md                # Bu dosya
```

## 🛡️ Güvenlik Framework'leri

| Framework | Amaç |
|-----------|------|
| OWASP ML Top 10 | ML sistemleri için güvenlik standartları |
| MITRE ATLAS | AI/ML saldırı taktikleri ve teknikleri |
| NVIDIA Garak | LLM güvenlik tarayıcı |
| Microsoft PyRIT | Veri güvenliği ve gizlilik |
| **Microsoft Fairlearn** | Model adalet ve önyargı testi |
| **Giskard** | ML model validasyonu |
| **Credo AI** | AI yönetişim ve uyumluluk |
| **CycloneDX** | SBOM ve güvenlik açığı taraması |

## 📝 Lisans

MIT License

## 👨‍💻 Geliştirici

MLSecOps Pipeline v3.0 - 9 Güvenlik Testi
OWASP + ATLAS + Garak + PyRIT + Fairlearn + Giskard + Credo AI + CycloneDX
