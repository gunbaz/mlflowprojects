# MLflow + Jenkins

## Context
MLflow'a Jenkins entegre etmek ve jenkinste göstermek  
Kaynak: [Jenkins Tutorial](https://www.datacamp.com/tutorial/jenkins-tutorial)

## Hızlı Başlangıç (Lokal)

- Bağımlılıklar (opsiyonel — mevcut ortamda varsa atlayın):
  - `MLflowJenkins/requirements.txt`
- Eğitimi çalıştır:
  - Windows PowerShell:
    ```powershell
    Set-Location c:\Users\pc\mlflow0\mlflowprojects\MLflowJenkins
    python -u train.py
    ```
  - Linux/macOS:
    ```bash
    cd ./MLflowJenkins
    python3 -u train.py
    ```
- MLflow UI (opsiyonel):
  - Proje klasöründe çalıştırın ve tarayıcıdan açın: http://127.0.0.1:5004
    ```bash
    mlflow ui --port 5004 --host 127.0.0.1
    ```

Not: `MLFLOW_TRACKING_URI` tanımlı değilse script otomatik olarak `file:./mlruns` (proje içi) deposunu kullanır.

## Jenkins Pipeline

- Pipeline dosyası: kök dizindeki `Jenkinsfile` (scripted pipeline).
- Parametreler:
  - `MLFLOW_TRACKING_URI` (varsayılan: `file:./mlruns`). Uzak bir MLflow sunucusuna yazmak için örn. `http://<host>:5000` verin.
- Aşamalar:
  1. Checkout
  2. Sanal ortam oluşturma ve bağımlılık kurulumu (`MLflowJenkins/requirements.txt`)
  3. `MLFLOW_TRACKING_URI` ile `MLflowJenkins/train.py` çalıştırma
- Windows ajanlarda `bat`, Linux ajanlarda `sh` kullanılır. Ek ayara gerek yoktur.

## Jenkins'te Proje Kurulumu

### Ön Gereksinimler
1. **Jenkins Kurulumu**
   - Windows: [jenkins.io/download](https://www.jenkins.io/download/) üzerinden Windows installer (`.msi`) ile kurun.
   - Linux: Paket yöneticisi veya WAR dosyası ile kurun.
   - Varsayılan Jenkins UI: http://localhost:8080
   
2. **Gerekli Jenkins Eklentileri**
   - Git Plugin (repo'yu çekmek için)
   - Pipeline Plugin (Jenkinsfile çalıştırmak için)
   - Pipeline: Declarative/Scripted (genellikle varsayılan gelir)

3. **Sistem Gereksinimleri**
   - Python 3.x kurulu ve PATH'te olmalı
   - Git kurulu ve PATH'te olmalı

### Adım Adım Jenkins Job Oluşturma

#### 1. Yeni Pipeline Job Oluştur
- Jenkins ana sayfasında → **"New Item"** / **"Yeni Öğe"**
- Job adı girin: `MLflow-Jenkins-Integration` (veya istediğiniz ad)
- Tip olarak **"Pipeline"** seçin → **"OK"** / **"Tamam"**

#### 2. Job Yapılandırması

**General Sekmesi:**
- ✅ **"This project is parameterized"** / **"Bu proje parametrelidir"** işaretleyin
- **Add Parameter** → **"String Parameter"**
  - Name: `MLFLOW_TRACKING_URI`
  - Default Value: `file:./mlruns`
  - Description: `MLflow Tracking URI (varsayılan: yerel dosya deposu)`

**Pipeline Sekmesi:**
- **Definition:** `Pipeline script from SCM` seçin
- **SCM:** `Git` seçin
- **Repository URL:** `https://github.com/gunbaz/mlflowprojects.git` (kendi repo URL'iniz)
- **Credentials:** Gerekiyorsa GitHub credentials ekleyin (public repo ise gerekmez)
- **Branch Specifier:** `*/main` (veya kullandığınız branch)
- **Script Path:** `Jenkinsfile` (repo kök dizininde)

**Kaydı tamamlayın:** "Save" / "Kaydet"

#### 3. Pipeline'ı Çalıştır
- Job sayfasında → **"Build with Parameters"** / **"Parametrelerle Oluştur"**
- `MLFLOW_TRACKING_URI` değerini kontrol edin (varsayılan `file:./mlruns` genellikle yeterli)
- **"Build"** / **"Oluştur"** butonuna tıklayın

#### 4. Build Loglarını İncele
- Build numarasına tıklayın (örn. `#1`)
- **"Console Output"** / **"Konsol Çıktısı"** → Pipeline aşamalarını ve Python çıktılarını görün
- Başarılı build sonunda: `Model ve metrikler MLflow'a başarıyla kaydedildi.` mesajını göreceksiniz

#### 5. MLflow Sonuçlarını Kontrol Et
- Jenkins workspace'inde MLflow run'ları oluştu:
  - Windows: `C:\ProgramData\Jenkins\.jenkins\workspace\<job-name>\MLflowJenkins\mlruns\`
  - Linux: `/var/lib/jenkins/workspace/<job-name>/MLflowJenkins/mlruns/`
- Jenkins makinesinde MLflow UI açarak run'ları görüntüleyebilirsiniz:
  ```bash
  cd /path/to/jenkins/workspace/<job-name>/MLflowJenkins
  mlflow ui --port 5004 --host 0.0.0.0
  ```

### Jenkins'te Uzak MLflow Tracking Server Kullanma

Eğer merkezi bir MLflow tracking server'ınız varsa:
1. Job → **"Configure"** / **"Yapılandır"**
2. **"Build with Parameters"** → `MLFLOW_TRACKING_URI` parametresine sunucu URL'sini girin:
   - Örnek: `http://192.168.1.100:5000`
3. Build çalıştır → Metrikler uzak sunucuya yazılır

### Troubleshooting (Sorun Giderme)

**Hata: `python: command not found`**
- Jenkins node'unda Python yüklü ve PATH'te olmalı
- Jenkins → **Manage Jenkins** → **Global Tool Configuration** → Python ekleyin
- Veya Jenkinsfile'da tam path kullanın: `/usr/bin/python3` veya `C:\Python311\python.exe`

**Hata: `No module named 'mlflow'`**
- Pipeline venv oluşturma aşamasında hata olabilir
- Manuel test: Jenkins workspace'inde `python -m venv .venv && .venv/Scripts/activate && pip install -r MLflowJenkins/requirements.txt`

**Hata: Workspace'e yazma izni yok**
- Windows: Jenkins service'in kullanıcı hesabına workspace klasörü üzerinde yazma izni verin
- Linux: Jenkins kullanıcısının (`jenkins`) workspace dizinine yazabildiğinden emin olun

**Pipeline Timeout**
- Jenkinsfile'a timeout ekleyin:
  ```groovy
  options {
    timeout(time: 10, unit: 'MINUTES')
  }
  ```

## Dosyalar
- `MLflowJenkins/train.py`: Basit Logistic Regression eğitimi; metrik ve modeli MLflow'a loglar.
- `MLflowJenkins/requirements.txt`: Gerekli Python paketleri.
- `Jenkinsfile`: CI aşamaları (scripted pipeline).

## Çıktılar
- Varsayılan izleme konumu: `MLflowJenkins/mlruns/`
- Örnek metrik: `accuracy`
- Model artifact: `mlruns/<exp_id>/<run_id>/artifacts/model/`

## Örnek Jenkins Build Akışı
```
Stage 1: Checkout ✅
  → Repo çekiliyor: https://github.com/gunbaz/mlflowprojects.git

Stage 2: Prepare Environment ✅
  → Virtual environment oluşturuluyor
  → pip install -r MLflowJenkins/requirements.txt

Stage 3: Train & Track ✅
  → MLFLOW_TRACKING_URI=file:./mlruns
  → python MLflowJenkins/train.py
  → MLflow Tracking URI: file:./mlruns
  → Run ID: 98cdc8f7224240f49090d239cc6c86eb
  → Accuracy: 1.0
  → Model ve metrikler MLflow'a başarıyla kaydedildi.

Build SUCCESS ✅
```
