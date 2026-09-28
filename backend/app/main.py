from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from app.config import get_settings
from app.inference import BreedPredictor
from app.schemas import BatchResponse, ImageResult

settings = get_settings()
predictor = BreedPredictor(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    predictor.load()
    yield


app = FastAPI(title="Livestock Breed ID API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/predict", response_model=BatchResponse)
async def predict_breeds(files: list[UploadFile] = File(...)) -> BatchResponse:
    if not files:
        raise HTTPException(400, "Upload at least one image.")
    if len(files) > settings.max_files:
        raise HTTPException(400, f"Upload no more than {settings.max_files} images.")

    max_size = settings.max_file_size_mb * 1024 * 1024
    results: list[ImageResult] = []
    for file in files:
        filename = file.filename or "unnamed-image"
        try:
            if file.content_type and not file.content_type.startswith("image/"):
                raise ValueError("Only image uploads are supported.")
            contents = await file.read()
            if len(contents) > max_size:
                raise ValueError(f"Image exceeds the {settings.max_file_size_mb} MB limit.")
            image = Image.open(BytesIO(contents)).convert("RGB")
            predictions, crop = predictor.predict(image)
            results.append(ImageResult(filename=filename, success=True, predictions=predictions, animal_detected=crop.found, detection_confidence=crop.confidence))
        except (UnidentifiedImageError, ValueError) as exc:
            results.append(ImageResult(filename=filename, success=False, error=str(exc)))
        except Exception:
            results.append(ImageResult(filename=filename, success=False, error="Unable to process this image."))
        finally:
            await file.close()
    return BatchResponse(results=results)