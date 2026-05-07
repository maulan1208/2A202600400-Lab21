import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

EVAL_THRESHOLD = 0.70

def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huấn luyện mô hình và ghi nhận kết quả vào MLflow.

    Tham số:
        params: dict chứa các siêu tham số cho RandomForestClassifier
        data_path: đường dẫn đến file dữ liệu huấn luyện
        eval_path: đường dẫn đến file dữ liệu đánh giá

    Trả về:
        accuracy (float): độ chính xác trên tập đánh giá
    """

    # --- TODO 1.5.1: Đọc dữ liệu ---
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # --- TODO 1.5.2: Tách đặc trưng (X) và nhãn (y) ---
    X_train = df_train.drop("target", axis=1)
    y_train = df_train["target"]
    
    X_eval = df_eval.drop("target", axis=1)
    y_eval = df_eval["target"]

    # --- TODO 1.5.3: Bắt đầu MLflow run ---
    with mlflow.start_run():
        
        # --- TODO 1.5.4: Ghi nhận siêu tham số ---
        mlflow.log_params(params)

        # --- TODO 1.5.5: Khởi tạo và huấn luyện mô hình ---
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # --- TODO 1.5.6: Dự đoán và tính toán metrics ---
        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        # --- TODO 1.5.7: Ghi nhận metrics vào MLflow ---
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # --- TODO 1.5.8: Log mô hình vào MLflow artifact ---
        mlflow.sklearn.log_model(model, "model")

        # --- TODO 1.5.9: In kết quả ra màn hình ---
        print("-" * 30)
        print(f"Training with params: {params}")
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")
        print("-" * 30)

        # --- TODO 1.5.10: Lưu metrics ra file outputs/metrics.json ---
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump({"accuracy": acc, "f1_score": f1}, f, indent=4)

        # --- TODO 1.5.11: Lưu mô hình ra file models/model.pkl ---
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

        # --- TODO 1.5.12: Trả về accuracy ---
        return acc


if __name__ == "__main__":
    # Đọc siêu tham số từ params.yaml và gọi hàm train()
    if not os.path.exists("params.yaml"):
        print("Error: params.yaml not found!")
    else:
        with open("params.yaml") as f:
            params = yaml.safe_load(f)
        train(params)