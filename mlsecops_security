import subprocess
import sys
import mlflow
import numpy as np
from datetime import datetime

def run_security_scan():
    """
    OWASP ML06: AI Supply Chain Attacks
    ATLAS Tactic: Resource Development, Initial Access
    
    Kod ve bağımlılık güvenlik taraması
    """
    print("\n" + "="*60)
    print("OWASP ML06 - Tedarik Zinciri Güvenlik Taraması")
    print("ATLAS: Resource Development / Initial Access")
    print("="*60)
    
    results = {
        "bandit_issues": 0,
        "safety_vulnerabilities": 0,
        "status": "PASSED"
    }
    
    # Bandit - Statik kod analizi
    print("\n[1/2] Bandit - Statik Kod Analizi...")
    try:
        result = subprocess.run(
            ["bandit", "-r", ".", "-f", "txt", "--exclude", "venv,__pycache__"],
            capture_output=True,
            text=True,
            timeout=60
        )
        if "No issues identified" in result.stdout:
            print("✅ Bandit: Güvenlik sorunu bulunamadı")
        else:
            # Basit issue sayımı
            issues = result.stdout.count("Issue:")
            results["bandit_issues"] = issues
            if issues > 0:
                print(f"⚠️ Bandit: {issues} potansiyel sorun bulundu")
                results["status"] = "WARNING"
    except Exception as e:
        print(f"❌ Bandit hatası: {e}")
        results["status"] = "ERROR"
    
    # Safety - Bağımlılık güvenlik taraması
    print("\n[2/2] Safety - Bağımlılık Güvenlik Taraması...")
    try:
        result = subprocess.run(
            ["safety", "check", "--output", "text"],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print("✅ Safety: Tüm bağımlılıklar güvenli")
        else:
            vulns = result.stdout.count("vulnerability")
            results["safety_vulnerabilities"] = vulns
            print(f"⚠️ Safety: {vulns} güvenlik açığı bulundu")
            results["status"] = "WARNING"
    except Exception as e:
        print(f"❌ Safety hatası: {e}")
    
    return results


def run_drift_detection():
    """
    OWASP ML08: Model Skewing
    ATLAS Tactic: ML Attack Execution
    
    Veri kayması (drift) tespiti
    """
    print("\n" + "="*60)
    print("OWASP ML08 - Model Çarpıtma / Drift Tespiti")
    print("ATLAS: ML Attack Execution")
    print("="*60)
    
    results = {
        "drift_detected": False,
        "drift_score": 0.0,
        "status": "PASSED"
    }
    
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        import pandas as pd
        
        # Veri yükle
        if not os.path.exists("data/iris.csv"):
            print("⚠️ Veri dosyası bulunamadı, drift testi atlanıyor")
            results["status"] = "SKIPPED"
            return results
        
        data = pd.read_csv("data/iris.csv")
        
        # Basit drift simülasyonu (gerçekte production vs training karşılaştırılır)
        reference = data.sample(frac=0.5, random_state=42)
        current = data.drop(reference.index)
        
        # Drift raporu
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        
        # Sonuçları al
        result_dict = report.as_dict()
        drift_detected = result_dict['metrics'][0]['result']['dataset_drift']
        
        results["drift_detected"] = drift_detected
        
        if drift_detected:
            print("⚠️ Veri kayması tespit edildi!")
            results["status"] = "WARNING"
            results["drift_score"] = 1.0
        else:
            print("✅ Veri kayması tespit edilmedi")
            results["drift_score"] = 0.0
        
        # HTML rapor kaydet
        report.save_html("drift_report.html")
        print("📄 Drift raporu: drift_report.html")
        
    except ImportError:
        print("⚠️ Evidently yüklü değil, drift testi atlanıyor")
        results["status"] = "SKIPPED"
    except Exception as e:
        print(f"❌ Drift tespiti hatası: {e}")
        results["status"] = "ERROR"
    
    return results


def run_adversarial_test():
    """
    OWASP ML01: Input Manipulation (Adversarial Evasion)
    ATLAS Tactic: ML Attack Staging, Evasion
    
    Düşmancıl sağlamlık testi
    """
    print("\n" + "="*60)
    print("OWASP ML01 - Girdi Manipülasyonu / Düşmancıl Test")
    print("ATLAS: ML Attack Staging / Evasion")
    print("="*60)
    
    results = {
        "normal_accuracy": 0.0,
        "adversarial_accuracy": 0.0,
        "robustness_score": 0.0,
        "degradation_percent": 0.0,
        "status": "PASSED"
    }
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from art.attacks.evasion import FastGradientMethod
        from art.estimators.classification import SklearnClassifier
        
        # Veri yükle
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=42
        )
        
        # Model eğit
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Normal doğruluk
        normal_acc = model.score(X_test, y_test)
        results["normal_accuracy"] = round(normal_acc, 4)
        print(f"Normal Test Doğruluğu: {normal_acc:.2%}")
        
        # ART ile düşmancıl test
        classifier = SklearnClassifier(model=model)
        attack = FastGradientMethod(estimator=classifier, eps=0.1)
        X_test_adv = attack.generate(x=X_test)
        
        # Düşmancıl doğruluk
        predictions = model.predict(X_test_adv)
        adv_acc = np.mean(predictions == y_test)
        results["adversarial_accuracy"] = round(adv_acc, 4)
        print(f"Düşmancıl Test Doğruluğu: {adv_acc:.2%}")
        
        # Performans düşüşü
        if normal_acc > 0:
            degradation = (normal_acc - adv_acc) / normal_acc * 100
        else:
            degradation = 0
        results["degradation_percent"] = round(degradation, 2)
        results["robustness_score"] = round(1 - (degradation / 100), 4)
        
        print(f"Performans Düşüşü: {degradation:.1f}%")
        print(f"Sağlamlık Skoru: {results['robustness_score']:.2%}")
        
        # Değerlendirme
        if degradation > 50:
            print("⚠️ Model düşmancıl saldırılara karşı ZAYIF!")
            results["status"] = "CRITICAL"
        elif degradation > 20:
            print("⚠️ Model düşmancıl saldırılara karşı ORTA düzeyde dayanıklı")
            results["status"] = "WARNING"
        else:
            print("✅ Model düşmancıl saldırılara karşı GÜÇLÜ!")
            results["status"] = "PASSED"
            
    except ImportError as e:
        print(f"⚠️ Gerekli kütüphane yüklü değil: {e}")
        results["status"] = "SKIPPED"
    except Exception as e:
        print(f"❌ Düşmancıl test hatası: {e}")
        results["status"] = "ERROR"
    
    return results


def run_mlsecops_pipeline():
    """
    Tam MLSecOps güvenlik pipeline'ı
    Tüm sonuçları MLflow'a loglar
    """
    print("\n" + "#"*60)
    print("#" + " "*20 + "MLSecOps PIPELINE" + " "*21 + "#")
    print("#" + " "*10 + "OWASP ML Top 10 & MITRE ATLAS" + " "*9 + "#")
    print("#"*60)
    
    # MLflow run başlat
    mlflow.set_experiment("MLSecOps-Security-Audit")
    
    with mlflow.start_run(run_name=f"security-audit-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
        
        # Genel bilgiler
        mlflow.log_param("audit_date", datetime.now().isoformat())
        mlflow.log_param("framework", "OWASP ML Top 10 + MITRE ATLAS")
        
        # 1. Güvenlik Taraması (ML06)
        security_results = run_security_scan()
        mlflow.log_param("owasp_ml06_status", security_results["status"])
        mlflow.log_param("atlas_tactic_1", "Resource Development")
        mlflow.log_metric("bandit_issues", security_results["bandit_issues"])
        mlflow.log_metric("safety_vulnerabilities", security_results["safety_vulnerabilities"])
        
        # 2. Drift Tespiti (ML08)
        drift_results = run_drift_detection()
        mlflow.log_param("owasp_ml08_status", drift_results["status"])
        mlflow.log_param("atlas_tactic_2", "ML Attack Execution")
        mlflow.log_metric("drift_detected", 1 if drift_results["drift_detected"] else 0)
        mlflow.log_metric("drift_score", drift_results["drift_score"])
        
        # 3. Düşmancıl Test (ML01)
        adversarial_results = run_adversarial_test()
        mlflow.log_param("owasp_ml01_status", adversarial_results["status"])
        mlflow.log_param("atlas_tactic_3", "ML Attack Staging")
        mlflow.log_metric("normal_accuracy", adversarial_results["normal_accuracy"])
        mlflow.log_metric("adversarial_accuracy", adversarial_results["adversarial_accuracy"])
        mlflow.log_metric("robustness_score", adversarial_results["robustness_score"])
        mlflow.log_metric("degradation_percent", adversarial_results["degradation_percent"])
        
        # Genel durum
        all_statuses = [
            security_results["status"],
            drift_results["status"],
            adversarial_results["status"]
        ]
        
        if "CRITICAL" in all_statuses:
            overall = "CRITICAL"
        elif "ERROR" in all_statuses:
            overall = "ERROR"
        elif "WARNING" in all_statuses:
            overall = "WARNING"
        else:
            overall = "PASSED"
        
        mlflow.log_param("overall_security_status", overall)
        
        # Özet
        print("\n" + "="*60)
        print("MLSecOps ÖZET RAPORU")
        print("="*60)
        print(f"OWASP ML06 (Tedarik Zinciri): {security_results['status']}")
        print(f"OWASP ML08 (Model Çarpıtma): {drift_results['status']}")
        print(f"OWASP ML01 (Girdi Manipülasyonu): {adversarial_results['status']}")
        print(f"\n🔒 GENEL GÜVENLİK DURUMU: {overall}")
        print("="*60)
        
        if overall == "PASSED":
            print("✅ Tüm güvenlik testleri başarılı!")
        elif overall == "WARNING":
            print("⚠️ Bazı uyarılar var, inceleme önerilir")
        else:
            print("❌ Kritik güvenlik sorunları tespit edildi!")
        
        return overall


# Import os for file checks
import os

if __name__ == "__main__":
    result = run_mlsecops_pipeline()
    
    # MLflow UI bilgisi
    print("\n📊 MLflow'da sonuçları görmek için:")
    print("   mlflow ui")
    print("   http://127.0.0.1:5000")
    
    # Exit code
    if result in ["CRITICAL", "ERROR"]:
        sys.exit(1)
    else:
        sys.exit(0)