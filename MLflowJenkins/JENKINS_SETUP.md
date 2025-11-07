# Jenkins Kurulum ve Yapılandırma Kılavuzu

Bu doküman, MLflow + Jenkins entegrasyonunu Windows'ta sıfırdan kurmak için gerekli tüm adımları içerir.

## 1. Jenkins Kurulumu (Windows)

### Adım 1.1: Jenkins'i İndirin ve Kurun
1. Tarayıcıda [jenkins.io/download](https://www.jenkins.io/download/)'a gidin
2. **Windows** sekmesini seçin
3. **LTS (Long-Term Support)** `.msi` installer'ı indirin
4. İndirilen `.msi` dosyasını çalıştırın
5. Kurulum sihirbazını takip edin:
   - Installation Directory: Varsayılan (`C:\Program Files\Jenkins`) veya istediğiniz yol
   - Service Account: **Local System Account** (varsayılan) veya özel kullanıcı
   - Port: `8080` (varsayılan)
6. Kurulum tamamlandığında Jenkins service otomatik başlar

### Adım 1.2: İlk Kurulum Sihirbazı
1. Tarayıcıda `http://localhost:8080` adresini açın
2. **Unlock Jenkins** ekranı:
   - Gösterilen dosya yolundan (`C:\ProgramData\Jenkins\.jenkins\secrets\initialAdminPassword`) şifreyi kopyalayın
   - Şifreyi girin ve **Continue**
3. **Customize Jenkins** ekranı:
   - **Install suggested plugins** seçin (önerilen)
   - Eklentiler yüklenirken bekleyin (2-5 dakika)
4. **Create First Admin User** ekranı:
   - Admin kullanıcı bilgilerini girin (veya **Skip and continue as admin**)
5. **Instance Configuration**:
   - Jenkins URL: `http://localhost:8080/` (varsayılan)
   - **Save and Finish**

Jenkins artık hazır! 🎉

## 2. Gerekli Sistem Araçlarını Doğrulama

### Python Kontrol
```powershell
python --version
# Çıktı: Python 3.11.0 (veya daha yüksek)
```

Eğer Python kurulu değilse:
- [python.org/downloads](https://www.python.org/downloads/) üzerinden indirin
- Kurulum sırasında **"Add Python to PATH"** seçeneğini işaretleyin

### Git Kontrol
```powershell
git --version
# Çıktı: git version 2.x.x
```

Eğer Git kurulu değilse:
- [git-scm.com/download/win](https://git-scm.com/download/win) üzerinden indirin
- Kurulum sırasında varsayılan seçeneklerle devam edin

## 3. Jenkins'te Pipeline Job Oluşturma

### Adım 3.1: Yeni Item (Job) Oluştur
1. Jenkins ana sayfasında sol menüden **"New Item"** tıklayın
2. Enter an item name: `MLflow-Jenkins-Integration`
3. Tip seçin: **Pipeline**
4. **OK** butonuna tıklayın

### Adım 3.2: Job Yapılandırması

#### General Sekmesi
1. ✅ **"This project is parameterized"** kutusunu işaretleyin
2. **Add Parameter** → **String Parameter**:
   - **Name:** `MLFLOW_TRACKING_URI`
   - **Default Value:** `file:./mlruns`
   - **Description:** `MLflow tracking URI. Varsayılan: yerel dosya deposu. Uzak sunucu için örn: http://192.168.1.100:5000`

#### Build Triggers (Opsiyonel)
- ✅ **Poll SCM** (periyodik kontrol için)
  - Schedule: `H/5 * * * *` (her 5 dakikada bir Git repo'yu kontrol et)
- ✅ **GitHub hook trigger for GITScm polling** (webhook için)

#### Pipeline Sekmesi
1. **Definition:** `Pipeline script from SCM` seçin
2. **SCM:** `Git` seçin
3. **Repository URL:** 
   ```
   https://github.com/gunbaz/mlflowprojects.git
   ```
   (kendi repo URL'inizi buraya yazın)
4. **Credentials:** 
   - Public repo için **"- none -"** seçin
   - Private repo için **Add** → **Jenkins** → Username + Password/Token ekleyin
5. **Branches to build:**
   - Branch Specifier: `*/main` (veya `*/master`)
6. **Script Path:** `Jenkinsfile`
7. **Lightweight checkout** ✅ işaretleyin (opsiyonel, daha hızlı)

### Adım 3.3: Kaydet
- En alttaki **Save** butonuna tıklayın

## 4. İlk Build'i Çalıştırma

### Adım 4.1: Build Tetikleme
1. Job sayfasında sol menüden **"Build with Parameters"** tıklayın
2. Parametre değerlerini kontrol edin:
   - `MLFLOW_TRACKING_URI`: `file:./mlruns` (varsayılan yeterli)
3. **Build** butonuna tıklayın

### Adım 4.2: Build İlerlemesini İzleme
1. Sol alt köşedeki **Build History** bölümünden build numarasını (örn. `#1`) tıklayın
2. **Console Output** seçeneğine tıklayın
3. Gerçek zamanlı logları izleyin:
   ```
   [Pipeline] stage (1. Kodu Çek)
   [Pipeline] { (1. Kodu Çek)
   [Pipeline] checkout
   Cloning repository https://github.com/gunbaz/mlflowprojects.git
   ...
   [Pipeline] stage (2. Ortamı Hazırla)
   ...
   [Pipeline] stage (3. Modeli Eğit ve MLflowa Kaydet)
   MLflow Tracking URI: file:./mlruns
   Run ID: 98cdc8f7224240f49090d239cc6c86eb
   Accuracy: 1.0
   Model ve metrikler MLflow'a başarıyla kaydedildi.
   [Pipeline] End of Pipeline
   Finished: SUCCESS
   ```

### Adım 4.3: MLflow Sonuçlarını Görüntüleme

**Jenkins Workspace'de MLflow UI Başlatma:**
```powershell
# Jenkins workspace yolunu bulun (Console Output'tan bakın veya):
cd C:\ProgramData\Jenkins\.jenkins\workspace\MLflow-Jenkins-Integration\MLflowJenkins
mlflow ui --port 5004 --host 127.0.0.1
```

Tarayıcıda: http://127.0.0.1:5004

## 5. Uzak MLflow Tracking Server Kullanımı (Opsiyonel)

### Senaryo: Merkezi MLflow sunucunuz var

1. MLflow tracking server'ı başlatın (ayrı bir makine/VM'de):
   ```bash
   mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
   ```

2. Jenkins job'ı yeniden çalıştırın:
   - **Build with Parameters**
   - `MLFLOW_TRACKING_URI`: `http://<sunucu-ip>:5000` (örn. `http://192.168.1.100:5000`)
   - **Build**

3. MLflow UI'da sonuçlar uzak sunucuda görünecektir.

## 6. Troubleshooting (Yaygın Sorunlar)

### Problem 1: `python: command not found`
**Sebep:** Jenkins service Python'ı bulamıyor.

**Çözüm 1 - Sistem PATH'ini Güncelle:**
1. **Windows Sistem Özellikleri** → **Ortam Değişkenleri**
2. **System variables** → `Path` → **Edit**
3. Python yolunu ekleyin: `C:\Users\pc\AppData\Local\Programs\Python\Python311\`
4. Jenkins service'i yeniden başlatın:
   ```powershell
   Restart-Service Jenkins
   ```

**Çözüm 2 - Jenkinsfile'da Tam Path Kullan:**
```groovy
bat '''
  C:\\Users\\pc\\AppData\\Local\\Programs\\Python\\Python311\\python.exe -m venv .venv
  ...
'''
```

### Problem 2: `No module named 'mlflow'`
**Sebep:** Virtual environment düzgün oluşmadı veya activate olmadı.

**Çözüm:**
Manuel test yapın (PowerShell):
```powershell
cd C:\ProgramData\Jenkins\.jenkins\workspace\MLflow-Jenkins-Integration
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r MLflowJenkins\requirements.txt
python MLflowJenkins\train.py
```

Eğer bu çalışıyorsa Jenkinsfile'daki komutları kontrol edin.

### Problem 3: `Permission denied` (workspace yazma hatası)
**Sebep:** Jenkins service kullanıcısının workspace'e yazma izni yok.

**Çözüm:**
1. **Services** → **Jenkins** → **Properties** → **Log On** sekmesi
2. **This account** seçin ve yönetici hesabı girin
3. Service'i yeniden başlatın

### Problem 4: Git credentials hatası (Private Repo)
**Sebep:** Jenkins'in private repo'ya erişim yetkisi yok.

**Çözüm:**
1. GitHub'da **Personal Access Token** oluşturun:
   - Settings → Developer settings → Personal access tokens → Generate new token
   - Scope: `repo` (tüm repo yetkileri)
2. Jenkins'te credential ekleyin:
   - **Manage Jenkins** → **Manage Credentials**
   - **(global)** → **Add Credentials**
   - Kind: **Username with password**
   - Username: GitHub kullanıcı adınız
   - Password: Token'ı yapıştırın
3. Job'da bu credential'ı seçin

### Problem 5: Pipeline çok yavaş çalışıyor
**Çözüm:**
- Jenkinsfile'da `pip install --no-cache-dir` kullanın
- Jenkins node'una daha fazla RAM/CPU ayırın
- Git shallow clone kullanın: Pipeline → **Additional Behaviours** → **Shallow clone**

## 7. Best Practices (En İyi Pratikler)

### 7.1. Pipeline Optimizasyonu
- **Cache pip packages:** Shared library veya custom plugin kullanarak pip cache'i koruyun
- **Parallel stages:** Bağımsız aşamaları paralel çalıştırın
- **Lightweight checkout:** Gereksiz Git history indirmekten kaçının

### 7.2. Güvenlik
- ✅ Jenkins'i sadece güvenilir ağda açın veya reverse proxy (nginx) ile koruyun
- ✅ Admin hesabına güçlü şifre koyun
- ✅ Credentials'ları Jenkins Credentials Store'da saklayın, Jenkinsfile'da plaintext yazmayın

### 7.3. İzleme ve Bildirim
- Email/Slack bildirimleri ekleyin (Jenkins Email Extension Plugin)
- Build metrikleri için **BlueOcean** plugin'ini kurun (modern UI)

## 8. İleri Seviye: Multi-Branch Pipeline

Birden fazla branch'i otomatik test etmek için:

1. **New Item** → **Multibranch Pipeline**
2. **Branch Sources** → **Git** → Repo URL
3. **Build Configuration** → Script Path: `Jenkinsfile`
4. **Scan Multibranch Pipeline Triggers** → Periyodik tarama etkinleştir

Her push'ta otomatik build tetiklenir ve sonuçlar branch bazında görüntülenir.

## 9. Kaynak ve Dokümantasyon

- Jenkins Resmi Dokümantasyon: [jenkins.io/doc](https://www.jenkins.io/doc/)
- Pipeline Syntax: [jenkins.io/doc/book/pipeline/syntax](https://www.jenkins.io/doc/book/pipeline/syntax/)
- MLflow Documentation: [mlflow.org/docs/latest](https://mlflow.org/docs/latest/)
- Datacamp Jenkins Tutorial: [datacamp.com/tutorial/jenkins-tutorial](https://www.datacamp.com/tutorial/jenkins-tutorial)

---

## Sonuç

Bu kılavuzu tamamladıysanız:
- ✅ Jenkins kurulu ve çalışıyor
- ✅ MLflow entegrasyonlu bir pipeline job'ınız var
- ✅ Her commit'te model eğitimi ve metrik logging otomatik yapılıyor
- ✅ MLflow UI'da sonuçları görüntüleyebiliyorsunuz

Başarılar! 🚀
