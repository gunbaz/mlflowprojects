"""
Auto çalıştırılan runtime update demo:
- Son run'ı otomatik bulur
- Kullanıcı onayı istemeden simulate_runtime_update çağırır
"""
import mlflow
from runtime_update_demo import simulate_runtime_update

def main():
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("autogluon_multimodal_experiment")
    if not exp:
        print("Experiment bulunamadı. Önce train_multimodal.py çalıştırın.")
        return
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        print("Hiç run yok. Önce train_multimodal.py çalıştırın.")
        return
    run_id = runs[0].info.run_id
    print(f"Son run: {run_id}")
    simulate_runtime_update(run_id)

if __name__ == "__main__":
    main()
