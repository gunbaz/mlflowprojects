# GitHub'a Yükleme Rehberi 🚀

## Adım Adım GitHub Upload

### 1. Git Repository Başlat (Eğer yoksa)

```bash
cd c:\Users\Monster\OneDrive\Desktop\autogloun_iris
git init
```

### 2. .gitignore Kontrol

`.gitignore` dosyasının şu içerikte olduğundan emin ol:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/

# AutoGluon
autogluon_models/

# MLflow
mlruns/

# DVC
.dvc/cache

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
```

### 3. Dosyaları Stage'e Ekle

```bash
git add .
git commit -m "feat: Add NVIDIA Garak and PyRIT security integration

- Added NVIDIA Garak LLM security scanner
- Added PyRIT data security and privacy testing
- Updated Jenkins pipeline with Garak and PyRIT stages
- Enhanced MLflow logging with security metrics
- Updated README with comprehensive documentation"
```

### 4. GitHub Repository Oluştur

1. GitHub.com'a git
2. "New repository" butonuna tıkla
3. Repository adı: `autogluon-iris-mlsecops`
4. Description: "AutoGluon ML pipeline with NVIDIA Garak and PyRIT security testing"
5. Public/Private seç
6. "Create repository" tıkla

### 5. Remote Ekle ve Push

```bash
# Remote ekle (GitHub'dan aldığın URL ile değiştir)
git remote add origin https://github.com/KULLANICI_ADIN/autogluon-iris-mlsecops.git

# Ana branch'i main olarak ayarla
git branch -M main

# Push et
git push -u origin main
```

## Jenkins'te Build Etme 🔧

### Ön Hazırlık

1. **Docker Desktop'ı Aç**
   - Docker Desktop uygulamasını başlat
   - Docker'ın çalıştığından emin ol

2. **Jenkins'i Başlat**
   - Jenkins'i tarayıcıda aç: `http://localhost:8080`

### Jenkins Pipeline Oluşturma

1. **New Item**
   - "New Item" tıkla
   - İsim: `AutoGluon-MLSecOps-Pipeline`
   - Type: "Pipeline" seç
   - OK tıkla

2. **Pipeline Configuration**
   - **Pipeline** bölümüne git
   - **Definition**: "Pipeline script from SCM" seç
   - **SCM**: "Git" seç
   - **Repository URL**: GitHub repository URL'ini yapıştır
   - **Branch**: `*/main`
   - **Script Path**: `Jenkinsfile`

3. **Save ve Build**
   - "Save" tıkla
   - "Build Now" tıkla

### Pipeline Aşamaları

Pipeline şu aşamalardan geçecek:

1. ✅ **Checkout** - Kod çekiliyor
2. ✅ **Install Dependencies** - Bağımlılıklar yükleniyor
3. ✅ **DVC Pull** - Veri çekiliyor
4. ✅ **Build Docker Image** - Docker image oluşturuluyor
5. ✅ **Run Training** - Model eğitiliyor
6. ✅ **MLSecOps Security Audit** - Tüm güvenlik testleri
7. ✅ **NVIDIA Garak LLM Security** - Garak taraması
8. ✅ **PyRIT Data Security** - PyRIT testleri

### MLflow Sonuçlarını Görüntüleme

Build tamamlandıktan sonra:

```bash
cd c:\Users\Monster\OneDrive\Desktop\autogloun_iris
python -m mlflow ui
```

Tarayıcıda: `http://127.0.0.1:5000`

## Beklenen Sonuçlar 📊

### MLflow'da Göreceğin Metrikler

**Garak Metrikleri:**
- `garak_vulnerabilities`: 0-4 arası
- `prompt_injection_risk`: 0.0-1.0
- `toxicity_score`: 0.0-1.0
- `jailbreak_attempts`: 0-4

**PyRIT Metrikleri:**
- `pii_detected`: 0 (Iris dataset'inde PII yok)
- `sensitive_data_risk`: 0.0
- `compliance_score`: 1.0

**Model Metrikleri:**
- `accuracy`: ~0.95+
- `balanced_accuracy`: ~0.95+
- `robustness_score`: ~0.70+

## Sorun Giderme 🔍

### Docker Hatası
```bash
# Docker servisini başlat
# Docker Desktop'ı aç ve bekle
```

### DVC Hatası
```bash
# DVC remote'u yapılandır
dvc remote add -d storage .dvc/cache
```

### Bağımlılık Hatası
```bash
# Sanal ortam oluştur ve bağımlılıkları yükle
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Başarı! 🎉

Tüm adımlar tamamlandığında:
- ✅ Kod GitHub'da
- ✅ Jenkins pipeline çalışıyor
- ✅ Garak ve PyRIT testleri aktif
- ✅ MLflow'da sonuçlar görünüyor

Ödevin hazır! 🚀
