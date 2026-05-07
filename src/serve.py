from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os

app = FastAPI()

GCS_BUCKET = os.environ["GCS_BUCKET"]
GCS_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def download_model():
    """
    Tai file model.pkl tu GCS ve may khi server khoi dong.

    Ham nay duoc goi mot lan khi module duoc import. Su dung
    GOOGLE_APPLICATION_CREDENTIALS de xac thuc (duoc dat trong systemd service).
    """
    print(f"Đang tải model từ bucket {GCS_BUCKET}...")
    try:
        # TODO 2.6.1 -> 2.6.4
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(GCS_MODEL_KEY)
        
        # Tạo thư mục models nếu chưa có
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        blob.download_to_filename(MODEL_PATH)
        print(f"--- Tải model thành công: {MODEL_PATH} ---")
    except Exception as e:
        print(f"Lỗi khi tải model: {e}")
        # Nếu lỗi tải từ Cloud, kiểm tra xem có file local cũ không
        if not os.path.exists(MODEL_PATH):
             raise e


download_model()
model = joblib.load(MODEL_PATH)

class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """
    Endpoint kiem tra suc khoe server.
    GitHub Actions goi endpoint nay sau khi deploy de xac nhan server dang chay.

    Tra ve: {"status": "ok"}
    """
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luan chinh.

    Dau vao : JSON {"features": [f1, f2, ..., f12]}
    Dau ra  : JSON {"prediction": <0|1|2>, "label": <"thap"|"trung_binh"|"cao">}

    Thu tu 12 dac trung (khop voi thu tu trong FEATURE_NAMES cua test):
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
        pH, sulphates, alcohol, wine_type
    """
    if len(req.features) != 12:
        raise HTTPException(
            status_code=400, 
            detail=f"Cần chính xác 12 đặc trưng. Bạn đang gửi {len(req.features)}."
        )

    prediction = int(model.predict([req.features])[0])

    labels = {0: "thấp", 1: "trung_bình", 2: "cao"}
    
    return {
        "prediction": prediction,
        "label": labels.get(prediction, "không xác định")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
