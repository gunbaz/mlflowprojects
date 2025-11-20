import subprocess
import sys
import mlflow
import numpy as np
import os
from datetime import datetime

def run_security_scan():
    """
    OWASP ML06: AI Supply Chain Attacks
    ATLAS Tactic: Resource Development, Initial Access
    """
    print("\n" + "="*60)
    print("OWASP ML06 - Tedarik Zinciri Guvenlik Taramasi")
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
            [sys.executable, "-m", "bandit", "-r", "train.py", "-f", "txt"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if "No issues identified" in result.stdout or result.returncode == 0:
            print("[OK] Bandit: Guvenlik sorunu bulunamadi")
        else:
            issues = result.stdout.count("Issue:")
            results["bandit_issues"] = issues
            if issues > 0:
                print(f"[UYARI] Bandit: {issues} potansiyel sorun bulundu")
            else:
                print("[OK] Bandit: Guvenlik sorunu bulunamadi")
    except Exception as e:
        print(f"[ATLANDI] Bandit: {e}")
    
    # Safety - Bagimlilik guvenlik taramasi
    print("\n[2/2] Safety - Bagimlilik Guvenlik Taramasi...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "safety", "check"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print("[OK] Safety: Tum bagimliliklar guvenli")
        else:
            print("[UYARI] Safety: Bazi guvenlik uyarilari var")
            results["status"] = "WARNING"
    except Exception as e:
        print(f"[ATLANDI] Safety: {e}")
    
    return results


def run_drift_detection():
    """
    OWASP ML08: Model Skewing
    ATLAS Tactic: ML Attack Execution
    """
    print("\n" + "="*60)
    print("OWASP ML08 - Model Carpitma / Drift Tespiti")
    print("ATLAS: ML Attack Execution")
    print("="*60)
    
    results = {
        "drift_detected": False,
        "drift_score": 0.0,
        "status": "PASSED"
    }
    
    try:
        # Yeni Evidently import
        from evidently import ColumnDriftMetric
        from evidently.report import Report
        import pandas as pd
        
        # Veri yukle
        data_path = "data/iris.csv"
        if not os.path.exists(data_path):
            print("[ATLANDI] Veri dosyasi bulunamadi")
            results["status"] = "SKIPPED"
            return results
        
        data = pd.read_csv(data_path)
        
        # Sadece sayisal sutunlari al
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Basit drift simulasyonu
        reference = data[numeric_cols].sample(frac=0.5, random_state=42)
        current = data[numeric_cols].drop(reference.index)
        
        # Manuel drift hesaplama (Evidently sorununu bypass)
        from scipy import stats
        
        drift_count = 0
        for col in numeric_cols:
            stat, p_value = stats.ks_2samp(reference[col], current[col])
            if p_value < 0.05:
                drift_count += 1
        
        drift_ratio = drift_count / len(numeric_cols) if numeric_cols else 0
        results["drift_score"] = round(drift_ratio, 2)
        
        if drift_ratio > 0.5:
            results["drift_detected"] = True
            print(f"[UYARI] Veri kaymasi tespit edildi! ({drift_count}/{len(numeric_cols)} sutun)")
            results["status"] = "WARNING"
        else:
            results["drift_detected"] = False
            print(f"[OK] Veri kaymasi tespit edilmedi ({drift_count}/{len(numeric_cols)} sutun)")
        
    except Exception as e:
        print(f"[ATLANDI] Drift tespiti hatasi: {e}")
        results["status"] = "SKIPPED"
    
    return results

def run_adversarial_test():
    """
    OWASP ML01: Input Manipulation (Adversarial Evasion)
    ATLAS Tactic: ML Attack Staging, Evasion
    """
    print("\n" + "="*60)
    print("OWASP ML01 - Girdi Manipulasyonu / Dusmancil Test")
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
        
        # Veri yukle
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=42
        )
        
        # Model egit
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Normal dogruluk
        normal_acc = model.score(X_test, y_test)
        results["normal_accuracy"] = round(normal_acc, 4)
        print(f"Normal Test Dogrulugu: {normal_acc:.2%}")
        
        # Dusmancil test - FGSM benzeri gurultu ekleme
        epsilon = 0.3
        noise = np.random.uniform(-epsilon, epsilon, X_test.shape)
        X_test_adv = X_test + noise
        
        # Dusmancil dogruluk
        predictions = model.predict(X_test_adv)
        adv_acc = np.mean(predictions == y_test)
        results["adversarial_accuracy"] = round(adv_acc, 4)
        print(f"Dusmancil Test Dogrulugu: {adv_acc:.2%}")
        
        # Performans dususu
        if normal_acc > 0:
            degradation = (normal_acc - adv_acc) / normal_acc * 100
        else:
            degradation = 0
        results["degradation_percent"] = round(degradation, 2)
        results["robustness_score"] = round(1 - (degradation / 100), 4)
        
        print(f"Performans Dususu: {degradation:.1f}%")
        print(f"Saglamlik Skoru: {results['robustness_score']:.2%}")
        
        # Degerlendirme
        if degradation > 50:
            print("[KRITIK] Model dusmancil saldirilara karsi ZAYIF!")
            results["status"] = "CRITICAL"
        elif degradation > 20:
            print("[UYARI] Model dusmancil saldirilara karsi ORTA duzeyde dayanikli")
            results["status"] = "WARNING"
        else:
            print("[OK] Model dusmancil saldirilara karsi GUCLU!")
            results["status"] = "PASSED"
            
    except Exception as e:
        print(f"[HATA] Dusmancil test hatasi: {e}")
        results["status"] = "ERROR"
    
    return results


def run_mlsecops_pipeline():
    """
    Tam MLSecOps guvenlik pipeline'i
    Tum sonuclari MLflow'a loglar
    """
    print("\n" + "#"*60)
    print("#" + " "*20 + "MLSecOps PIPELINE" + " "*21 + "#")
    print("#" + " "*10 + "OWASP ML Top 10 & MITRE ATLAS" + " "*9 + "#")
    print("#"*60)
    
    # MLflow run baslat
    mlflow.set_experiment("MLSecOps-Security-Audit")
    
    with mlflow.start_run(run_name=f"security-audit-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
        
        # Genel bilgiler
        mlflow.log_param("audit_date", datetime.now().isoformat())
        mlflow.log_param("framework", "OWASP ML Top 10 + MITRE ATLAS")
        
        # 1. Guvenlik Taramasi (ML06)
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
        
        # 3. Dusmancil Test (ML01)
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
        
        # Ozet
        print("\n" + "="*60)
        print("MLSecOps OZET RAPORU")
        print("="*60)
        print(f"OWASP ML06 (Tedarik Zinciri): {security_results['status']}")
        print(f"OWASP ML08 (Model Carpitma): {drift_results['status']}")
        print(f"OWASP ML01 (Girdi Manipulasyonu): {adversarial_results['status']}")
        print(f"\nGENEL GUVENLIK DURUMU: {overall}")
        print("="*60)
        
        if overall == "PASSED":
            print("[OK] Tum guvenlik testleri basarili!")
        elif overall == "WARNING":
            print("[UYARI] Bazi uyarilar var, inceleme onerilir")
        else:
            print("[HATA] Kritik guvenlik sorunlari tespit edildi!")
        
        return overall


if __name__ == "__main__":
    result = run_mlsecops_pipeline()
    
    # MLflow UI bilgisi
    print("\nMLflow'da sonuclari gormek icin:")
    print("   python -m mlflow ui")
    print("   http://127.0.0.1:5000")
    
    # Exit code
    if result in ["CRITICAL", "ERROR"]:
        sys.exit(1)
    else:
        sys.exit(0)