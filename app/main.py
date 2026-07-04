#!/usr/bin/env python
"""FastAPI entry point for the Voice Authentication service.

Provides a minimal health‑check endpoint. Additional routes will be added later.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import numpy as np

from app.embedding import generate_embedding, generate_embedding_from_waveform
from constants import DEFAULT_THRESHOLD

from fastapi.middleware.cors import CORSMiddleware
import vad_processor
from app import customer_manager

app = FastAPI(title="Voice Authentication API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Ensure required directories exist
BASE_DIR = Path(__file__).resolve().parent.parent
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
SPEAKER_DB_DIR = BASE_DIR / "database" / "speakers"
SPEAKER_DB_DIR.mkdir(parents=True, exist_ok=True)

speaker_embeddings_cache: dict[str, np.ndarray] = {}

def reload_cache():
    """Helper to reload all speaker embeddings into the in-memory cache."""
    global speaker_embeddings_cache
    new_cache = {}
    for npy_path in SPEAKER_DB_DIR.glob("*.npy"):
        new_cache[npy_path.stem] = np.load(npy_path)
    speaker_embeddings_cache.clear()
    speaker_embeddings_cache.update(new_cache)

@app.on_event("startup")
async def startup_event():
    reload_cache()

@app.get("/", response_class=JSONResponse)
async def root() -> dict:
    """Root health‑check endpoint.

    Returns a simple JSON payload confirming that the API is running.
    """
    return {"message": "Voice Authentication API Running"}

@app.post("/enroll", response_class=JSONResponse)
async def enroll(
    name: str = Form(...),
    audio: UploadFile = File(...)
) -> dict:
    """Enroll a new speaker.

    Saves the uploaded audio temporarily, generates an embedding, stores it,
    and cleans up the temporary file. Returns JSON with status and details.
    """
    temp_path = TEMP_DIR / audio.filename
    try:
        # Save uploaded file to temporary location
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        # Generate embedding
        embedding = generate_embedding(str(temp_path))
        # Save embedding to database
        save_path = SPEAKER_DB_DIR / f"{name}.npy"
        np.save(save_path, embedding)
        # Update in-memory cache
        speaker_embeddings_cache[name] = embedding
        # Cleanup temporary file
        temp_path.unlink(missing_ok=True)
        return {
            "status": "success",
            "speaker": name,
            "embedding_dimension": embedding.shape[0]
        }
    except Exception as exc:
        import traceback
        with open("error.log", "a") as f:
            f.write(f"Error in /enroll: {exc}\\n{traceback.format_exc()}\\n")
        # Ensure temp file is removed on error
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(exc))

# Verification endpoint
@app.post("/verify", response_class=JSONResponse)
async def verify(
    name: str = Form(...),
    audio: UploadFile = File(...)
) -> dict:
    """Verify a speaker by comparing an uploaded audio embedding against the enrolled embedding."""
    # Save uploaded audio to temporary file
    temp_path = TEMP_DIR / audio.filename
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        # Generate embedding from the uploaded audio
        embedding = generate_embedding(str(temp_path))
        # Load enrolled speaker embedding
        enrolled_path = SPEAKER_DB_DIR / f"{name}.npy"
        if not enrolled_path.is_file():
            raise HTTPException(status_code=404, detail=f"Speaker '{name}' not found")
        enrolled_emb = np.load(enrolled_path)
        # Cosine similarity
        norm_a = np.linalg.norm(embedding)
        norm_b = np.linalg.norm(enrolled_emb)
        similarity = float(np.dot(embedding, enrolled_emb) / (norm_a * norm_b)) if norm_a and norm_b else 0.0
        threshold = DEFAULT_THRESHOLD
        verified = similarity >= threshold
        return {
            "speaker": name,
            "similarity": round(similarity * 100, 2),
            "threshold": threshold * 100,
            "verified": verified,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        temp_path.unlink(missing_ok=True)
# Identification endpoint
@app.post("/identify", response_class=JSONResponse)
async def identify(
    audio: UploadFile = File(...)
) -> dict:
    """Identify the most similar enrolled speaker."""
    temp_path = TEMP_DIR / audio.filename
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        # Generate embedding from uploaded audio
        embedding = generate_embedding(str(temp_path))
        if not speaker_embeddings_cache:
            raise HTTPException(status_code=404, detail="No enrolled speakers found")
        # Compute similarities
        best_name = "Unknown"
        best_sim = 0.0
        for name, emb in speaker_embeddings_cache.items():
            norm_a = np.linalg.norm(embedding)
            norm_b = np.linalg.norm(emb)
            sim = float(np.dot(embedding, emb) / (norm_a * norm_b)) if norm_a and norm_b else 0.0
            if sim > best_sim:
                best_sim = sim
                best_name = name
        threshold = DEFAULT_THRESHOLD
        identified = best_sim >= threshold
        result = {
            "predicted_speaker": best_name if identified else "Unknown",
            "similarity": round(best_sim * 100, 2),
            "identified": identified,
        }
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        temp_path.unlink(missing_ok=True)

# Authenticate endpoint (Orchestration)
@app.post("/authenticate")
async def authenticate(audio: UploadFile = File(...)):
    print("STEP 1", flush=True)
    temp_path = TEMP_DIR / audio.filename
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
    print("STEP 2 - File Saved", flush=True)
    waveform, sr = vad_processor.load_audio(temp_path)
    print("STEP 3 - Audio Loaded", flush=True)
    speech_waveform = vad_processor.remove_silence(waveform, sr)
    print("STEP 4 - Silence Removed", flush=True)
    duration_seconds = len(speech_waveform) / sr
    print(f"Speech Duration: {duration_seconds:.2f} seconds", flush=True)

    print("STEP 5 - Starting embedding generation", flush=True)
    try:
        embedding = generate_embedding_from_waveform(speech_waveform, sr)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Embedding Error: {e}", flush=True)
        raise

    print("STEP 6 - Embedding generated", flush=True)
    print(f"Embedding shape: {embedding.shape}", flush=True)

    return {
        "status": "embedding_generated",
        "dimension": int(embedding.shape[0])
    }
