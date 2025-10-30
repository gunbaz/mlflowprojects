"""
Runtime'da Model Güncelleme Demo
Yeni veri geldiğinde modeli canlıda güncelleyen örnek
"""
import mlflow
import pandas as pd
import numpy as np
from datetime import datetime
from data_generator import generate_multimodal_dataset


def simulate_runtime_update(run_id):
    """
    Runtime'da model güncelleme simülasyonu
    """
    
    print(f"\n{'='*70}")
    print(f"RUNTIME MODEL UPDATE DEMO")
    print(f"{'='*70}\n")
    
    # 1. Orijinal modeli yükle
    print("📦 Orijinal model yükleniyor...")
    model_uri = f"runs:/{run_id}/multimodal_model"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    
    print("✓ Model yüklendi!\n")
    
    # 2. Test verisi ile tahmin yap
    print("🔮 İlk tahminler yapılıyor...")
    test_data = generate_multimodal_dataset(n_samples=50)
    test_features = test_data.drop(columns=['label'])
    test_labels = test_data['label']
    
    initial_predictions = loaded_model.predict(test_features)
    initial_accuracy = (initial_predictions == test_labels).mean()
    
    print(f"İlk test accuracy: {initial_accuracy:.4f}\n")
    
    # 3. Yeni veri simüle et (örneğin stream'den gelen)
    print("📨 Yeni veri akışı simüle ediliyor...")
    print("(Gerçek senaryoda: Kafka, API, database'den gelebilir)\n")
    
    new_data = generate_multimodal_dataset(n_samples=100)
    new_features = new_data.drop(columns=['label'])
    new_labels = new_data['label']
    
    print(f"Yeni veri sayısı: {len(new_data)}")
    
    # 4. Model güncelleme
    print("\n🔄 Model runtime'da güncelleniyor...")
    print("(Incremental/Continuous Learning)\n")
    
    # Not: Gerçek runtime update için model wrapper'ın update_model metodunu kullan
    # Bu demo amaçlı basitleştirilmiş versiyondur
    
    success = loaded_model._model_impl.python_model.update_model(
        new_data=new_data,
        new_labels=new_labels,
        update_path="./autogluon_multimodal_model_updated"
    )
    
    if success:
        print("\n✓ Model güncelleme başarılı!")
        
        # 5. Güncellenmiş model ile tekrar tahmin
        print("\n🔮 Güncellenmiş model ile tahminler yapılıyor...")
        
        updated_predictions = loaded_model.predict(test_features)
        updated_accuracy = (updated_predictions == test_labels).mean()
        
        print(f"\nGüncellenmiş test accuracy: {updated_accuracy:.4f}")
        print(f"Accuracy değişimi: {updated_accuracy - initial_accuracy:+.4f}")
        
        # 6. MLflow'a güncellemeyi logla
        with mlflow.start_run(run_name=f"runtime_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("original_run_id", run_id)
            mlflow.log_param("update_data_size", len(new_data))
            mlflow.log_metric("initial_accuracy", initial_accuracy)
            mlflow.log_metric("updated_accuracy", updated_accuracy)
            mlflow.log_metric("accuracy_improvement", updated_accuracy - initial_accuracy)
            
            print("\n✓ Güncelleme MLflow'a loglandı!")
    
    else:
        print("\n✗ Model güncelleme başarısız!")
    
    print(f"\n{'='*70}")
    print("DEMO TAMAMLANDI")
    print(f"{'='*70}\n")


def main():
    """
    Ana fonksiyon - kullanıcıdan run_id alarak demo çalıştırır
    """
    
    print("\n" + "="*70)
    print("AutoGluon MultiModal + MLflow Runtime Update Demo")
    print("="*70 + "\n")
    
    # En son run'ı bul
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("autogluon_multimodal_experiment")
    
    if experiment:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if runs:
            latest_run_id = runs[0].info.run_id
            print(f"En son run bulundu: {latest_run_id}")
            print(f"Run name: {runs[0].data.tags.get('mlflow.runName', 'N/A')}\n")
            
            response = input("Bu run ile devam edilsin mi? (y/n): ")
            
            if response.lower() == 'y':
                simulate_runtime_update(latest_run_id)
            else:
                custom_run_id = input("Run ID girin: ")
                simulate_runtime_update(custom_run_id)
        else:
            print("❌ Hiç run bulunamadı!")
            print("Önce train_multimodal.py çalıştırın.\n")
    else:
        print("❌ Experiment bulunamadı!")
        print("Önce train_multimodal.py çalıştırın.\n")


if __name__ == "__main__":
    main()
