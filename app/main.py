#!/usr/bin/env python
"""FastAPI entry point for the Voice Authentication service.

Provides a minimal health‑check endpoint and endpoints for enrolling,
verifying, identifying, and authenticating speakers.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import numpy as np
import time
from datetime import datetime
import soundfile as sf
from app import models
from app.embedding import generate_embedding, generate_embedding_from_waveform
from constants import DEFAULT_THRESHOLD

from fastapi.middleware.cors import CORSMiddleware
import vad_processor
from app import customer_manager

# Database Integration
from sqlalchemy.orm import Session
from app.database import Base, engine, get_db
from app import crud
from app.audio_adapter import pcm_to_wav, pcm_bytes_to_numpy

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

@app.on_event("startup")
async def startup_event():
    import sys
    from sqlalchemy import text
    try:
        print("[DB] Connecting to PostgreSQL...", flush=True)
        # Create extension if not exists
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        print("[DB] pgvector extension verified.", flush=True)
        
        print("[DB] Creating database tables...", flush=True)
        # Automatically create database tables during FastAPI startup
        Base.metadata.create_all(bind=engine)
        print("[DB] Database initialization completed successfully.", flush=True)
    except Exception as exc:
        import traceback
        print("=" * 80, flush=True)
        print("DATABASE INITIALIZATION FAILED", flush=True)
        traceback.print_exc()
        print("=" * 80, flush=True)
        sys.exit(1)

@app.get("/", response_class=JSONResponse)
async def root() -> dict:
    """Root health‑check endpoint.

    Returns a simple JSON payload confirming that the API is running.
    """
    return {"message": "Voice Authentication API Running"}

@app.get("/customers", response_class=JSONResponse)
async def list_customers(db: Session = Depends(get_db)) -> dict:
    """List all enrolled customers from the database.

    Fetches each customer along with their total voice embedding count.
    """
    try:
        results = crud.get_all_customers(db)
        
        customer_list = []
        for customer, embedding_count in results:
            customer_list.append({
                "customer_id": str(customer.customer_id),
                "customer_name": customer.customer_name,
                "customer_reference": customer.customer_reference,
                "mobile_number": customer.mobile_number,
                "status": customer.status,
                "embedding_count": embedding_count,
                "created_at": customer.created_at.isoformat() if customer.created_at else None,
                "updated_at": customer.updated_at.isoformat() if customer.updated_at else None
            })
            
        return {
            "status": "success",
            "count": len(customer_list),
            "customers": customer_list
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while fetching customers: {str(e)}"
        )


@app.post("/enroll", response_class=JSONResponse)
async def enroll(
    name: str = Form(...),
    audio: UploadFile = File(...),
    db: Session = Depends(get_db)
) -> dict:
    """Enroll a new speaker.

    Saves the uploaded audio temporarily, generates an embedding, stores it
    into PostgreSQL database, and cleans up the temporary file.
    """
    total_start = time.perf_counter()
    temp_path = TEMP_DIR / audio.filename
    try:
        # Save uploaded file to temporary location
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
            
        # Get sample rate and audio duration
        audio_np, sr = sf.read(str(temp_path))
        audio_duration = len(audio_np) / sr
        
        # Generate embedding
        embedding = generate_embedding(str(temp_path))
        
        # Create Customer if not exists
        customer = crud.get_customer_by_name(db, name)
        if not customer:
            customer = crud.create_customer(db, name)
            
        # Save embedding into PostgreSQL
        crud.save_embedding(
            db=db,
            customer_id=customer.customer_id,
            embedding=embedding,
            sample_rate=sr,
            audio_duration=audio_duration
        )
        
        processing_time_ms = (time.perf_counter() - total_start) * 1000.0
        
        # Cleanup temporary file
        temp_path.unlink(missing_ok=True)
        
        return {
            "status": "success",
            "customer_id": str(customer.customer_id),
            "customer_name": customer.customer_name,
            "embedding_saved": True,
            "embedding_dimension": int(embedding.shape[0]),
            "sample_rate": int(sr),
            "audio_duration": round(audio_duration, 2),
            "processing_time_ms": round(processing_time_ms, 2),
            "message": "Speaker enrolled successfully."
        }
    except Exception as exc:
        import traceback
        with open("error.log", "a") as f:
            f.write(f"Error in /enroll: {exc}\n{traceback.format_exc()}\n")
        # Ensure temp file is removed on error
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/enroll/pcm", response_class=JSONResponse)
async def enroll_pcm(
    customer_name: str,
    audio_data: bytes = File(...),
    db: Session = Depends(get_db)
) -> dict:
    """Enroll raw PCM audio under an existing customer by name.
    
    Used to add additional voice embeddings (e.g., mobile phone audio) to an
    already-enrolled customer so they can be recognized from multiple call types.
    """
    total_start = time.perf_counter()
    try:
        # Convert raw 8kHz 16-bit PCM bytes to numpy waveform
        waveform, sr = pcm_bytes_to_numpy(audio_data, sample_rate=8000)
        duration_seconds = len(waveform) / sr

        if duration_seconds < 1.0:
            raise HTTPException(status_code=400, detail="Audio too short for enrollment (< 1 second)")

        # Run VAD to extract only active speech
        import vad_processor
        speech_waveform = vad_processor.filter_speech(waveform, sr)
        if speech_waveform is None or len(speech_waveform) < sr * 0.5:
            speech_waveform = waveform  # fall back to raw if VAD strips too much

        # Generate embedding
        embedding = generate_embedding_from_waveform(speech_waveform, sr)

        # Look up existing customer — do NOT create new
        customer = crud.get_customer_by_name(db, customer_name)
        if not customer:
            raise HTTPException(status_code=404, detail=f"Customer '{customer_name}' not found. Enroll them first via /enroll.")

        # Save new embedding under the existing customer
        crud.save_embedding(
            db=db,
            customer_id=customer.customer_id,
            embedding=embedding,
            sample_rate=sr,
            audio_duration=duration_seconds
        )

        processing_time_ms = (time.perf_counter() - total_start) * 1000.0
        print(f"[Enroll PCM] Added embedding for existing customer '{customer_name}' — duration: {duration_seconds:.2f}s", flush=True)

        return {
            "status": "success",
            "customer_id": str(customer.customer_id),
            "customer_name": customer.customer_name,
            "embedding_saved": True,
            "audio_duration": round(duration_seconds, 2),
            "processing_time_ms": round(processing_time_ms, 2),
            "message": f"Additional voice embedding enrolled for '{customer_name}' successfully."
        }
    except HTTPException:
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/verify", response_class=JSONResponse)
async def verify(
    name: str = Form(...),
    audio: UploadFile = File(...),
    db: Session = Depends(get_db)
) -> dict:
    """Verify a speaker by comparing an uploaded audio embedding against the enrolled embedding."""
    # Save uploaded audio to temporary file
    temp_path = TEMP_DIR / audio.filename
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        # Generate embedding from the uploaded audio
        embedding = generate_embedding(str(temp_path))
        
        # Load enrolled speaker embeddings from DB
        customer = crud.get_customer_by_name(db, name)
        if not customer or not customer.voice_embeddings:
            raise HTTPException(status_code=404, detail=f"Speaker '{name}' not found")
            
        best_sim = 0.0
        for ve in customer.voice_embeddings:
            emb = np.array(ve.embedding)
            norm_a = np.linalg.norm(embedding)
            norm_b = np.linalg.norm(emb)
            sim = float(np.dot(embedding, emb) / (norm_a * norm_b)) if norm_a and norm_b else 0.0
            if sim > best_sim:
                best_sim = sim
                
        threshold = DEFAULT_THRESHOLD
        verified = best_sim >= threshold
        return {
            "speaker": name,
            "similarity": round(best_sim * 100, 2),
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
    audio: UploadFile = File(...),
    db: Session = Depends(get_db)
) -> dict:
    """Identify the most similar enrolled speaker."""
    temp_path = TEMP_DIR / audio.filename
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        # Generate embedding from uploaded audio
        embedding = generate_embedding(str(temp_path))
        
        # Fetch all embeddings from PostgreSQL
        all_embeddings = crud.get_all_embeddings(db)
        if not all_embeddings:
            raise HTTPException(status_code=404, detail="No enrolled speakers found")
            
        best_name = "Unknown"
        best_sim = 0.0
        for row in all_embeddings:
            emb = np.array(row.embedding)
            norm_a = np.linalg.norm(embedding)
            norm_b = np.linalg.norm(emb)
            sim = float(np.dot(embedding, emb) / (norm_a * norm_b)) if norm_a and norm_b else 0.0
            if sim > best_sim:
                best_sim = sim
                best_name = row.customer_name
                
        threshold = DEFAULT_THRESHOLD
        identified = best_sim >= threshold
        return {
            "predicted_speaker": best_name if identified else "Unknown",
            "similarity": round(best_sim * 100, 2),
            "identified": identified,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        temp_path.unlink(missing_ok=True)

# Authenticate endpoint (Orchestration)
@app.post("/authenticate", response_class=JSONResponse)
async def authenticate(
    audio: UploadFile = File(...),
    db: Session = Depends(get_db)
) -> dict:
    """Automatic Customer Voice Identity Service endpoint."""
    total_start = time.perf_counter()
    temp_path = TEMP_DIR / audio.filename

    try:
        # Save the uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
            
        # Fetch all embeddings from PostgreSQL
        all_embeddings = crud.get_all_embeddings(db)
        
        # 1. Check if we have any enrolled speakers at all
        if not all_embeddings:
            processing_time_ms = (time.perf_counter() - total_start) * 1000.0
            return {
                "status": "warning",
                "existing_customer": False,
                "authentication_result": "NO_REGISTERED_SPEAKERS",
                "similarity": 0.0,
                "processing_time_ms": round(processing_time_ms, 2),
                "audio_duration": 0.0,
                "message": "No enrolled speakers found in the system."
            }

        # 2. VAD & Audio Loading
        waveform, sr = vad_processor.load_audio(temp_path)
        speech_waveform = vad_processor.remove_silence(waveform, sr)
        
        duration_seconds = len(speech_waveform) / sr
        if duration_seconds < 1.5:
            processing_time_ms = (time.perf_counter() - total_start) * 1000.0
            return {
                "status": "error",
                "authentication_result": "INSUFFICIENT_AUDIO",
                "required_duration_seconds": 1.5,
                "received_duration_seconds": round(duration_seconds, 2),
                "processing_time_ms": round(processing_time_ms, 2),
                "message": "Need at least 1.5 seconds of clear speech."
            }

        # 3. Generate the embedding
        embedding = generate_embedding_from_waveform(speech_waveform, sr)        # 4. Cosine similarity search
        best_name = None
        best_customer_id = None
        best_sim = 0.0
        
        for row in all_embeddings:
            emb = np.array(row.embedding)
            norm_a = np.linalg.norm(embedding)
            norm_b = np.linalg.norm(emb)
            sim = (
                float(np.dot(embedding, emb) / (norm_a * norm_b))
                if norm_a and norm_b
                else 0.0
            )
            if sim > best_sim:
                best_sim = sim
                best_customer_id = row.customer_id
                best_name = row.customer_name
                
        threshold = DEFAULT_THRESHOLD
        
        # Determine authentication result
        authenticated = best_customer_id is not None and best_sim >= threshold
        
        processing_time_ms = (time.perf_counter() - total_start) * 1000.0
        current_timestamp = datetime.utcnow().isoformat()

        if authenticated:
            # Save authentication log into PostgreSQL
            crud.save_authentication_log(
                db=db,
                customer_id=best_customer_id,
                similarity=best_sim,
                threshold=threshold,
                authenticated=True,
                processing_time_ms=processing_time_ms,
                audio_duration=duration_seconds
            )
            
            # Fetch matched customer to get matched_embedding_count
            matched_customer = crud.get_customer_by_id(db, best_customer_id)
            matched_embedding_count = len(matched_customer.voice_embeddings) if matched_customer else 0

            return {
                "status": "success",
                "existing_customer": True,
                "customer_id": str(best_customer_id),
                "customer_name": best_name,
                "authentication_result": "AUTHENTICATED",
                "similarity": round(best_sim * 100, 2),
                "threshold": threshold,
                "processing_time_ms": round(processing_time_ms, 2),
                "audio_duration": round(duration_seconds, 2),
                "matched_embedding_count": matched_embedding_count,
                "model": "ECAPA-TDNN",
                "timestamp": current_timestamp,
                "message": "Speaker authenticated successfully."
            }
        else:
            # Voice similarity was below threshold.
            # Do NOT create a fake CUSTOMER_XXXXXX profile when enrolled profiles exist,
            # so low-quality turns never pollute the database or steal identity matches.
            return {
                "status": "unauthenticated",
                "existing_customer": False,
                "customer_id": str(best_customer_id) if best_customer_id else None,
                "customer_name": best_name if best_name else "Guest",
                "authentication_result": "UNAUTHENTICATED",
                "similarity": round(best_sim * 100, 2),
                "threshold": threshold,
                "processing_time_ms": round(processing_time_ms, 2),
                "audio_duration": round(duration_seconds, 2),
                "message": f"Voice similarity ({round(best_sim * 100, 2)}%) was below threshold ({int(threshold * 100)}%)."
            }

    except Exception as exc:
        processing_time_ms = (time.perf_counter() - total_start) * 1000.0
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "authentication_result": "INTERNAL_SERVER_ERROR",
            "processing_time_ms": round(processing_time_ms, 2),
            "message": str(exc)
        }
    finally:
        temp_path.unlink(missing_ok=True)

# PCM Telephony endpoint
@app.post("/authenticate/pcm", response_class=JSONResponse)
async def authenticate_pcm(
    audio: UploadFile = File(..., description="Raw PCM file: 16-bit signed little-endian, 8kHz mono"),
    db: Session = Depends(get_db)
) -> dict:
    """
    Telephony-optimised authentication endpoint.

    Accepts raw PCM audio (16-bit signed little-endian, 8 kHz, mono) from a
    telephony platform, converts it to WAV via the Audio Adapter layer, then
    runs the same VAD + embedding + similarity pipeline as /authenticate.

    The /authenticate endpoint is NOT touched by this endpoint.
    """
    total_start = time.perf_counter()
    # Use a .wav suffix so vad_processor/soundfile can read it correctly
    temp_pcm_path = TEMP_DIR / (audio.filename + ".pcm.tmp")
    temp_wav_path = TEMP_DIR / (audio.filename + ".wav")

    try:
        # 1. Read the raw PCM bytes from the upload
        pcm_bytes = await audio.read()
        if not pcm_bytes:
            raise HTTPException(status_code=400, detail="Uploaded PCM file is empty.")

        # 2. Convert PCM -> WAV via Audio Adapter
        pcm_to_wav(pcm_bytes, temp_wav_path, sample_rate=8000, channels=1, sample_width=2)

        # 3. Fetch all embeddings from PostgreSQL
        all_embeddings = crud.get_all_embeddings(db)

        if not all_embeddings:
            processing_time_ms = (time.perf_counter() - total_start) * 1000.0
            return {
                "status": "warning",
                "existing_customer": False,
                "authentication_result": "NO_REGISTERED_SPEAKERS",
                "similarity": 0.0,
                "processing_time_ms": round(processing_time_ms, 2),
                "audio_duration": 0.0,
                "audio_format": "PCM_8KHZ",
                "message": "No enrolled speakers found in the system."
            }

        # 4. VAD & Audio Loading (reuse existing vad_processor pipeline)
        waveform, sr = vad_processor.load_audio(temp_wav_path)
        speech_waveform = vad_processor.remove_silence(waveform, sr)

        duration_seconds = len(speech_waveform) / sr
        if duration_seconds < 1.5:
            processing_time_ms = (time.perf_counter() - total_start) * 1000.0
            return {
                "status": "error",
                "authentication_result": "INSUFFICIENT_AUDIO",
                "required_duration_seconds": 1.5,
                "received_duration_seconds": round(duration_seconds, 2),
                "processing_time_ms": round(processing_time_ms, 2),
                "audio_format": "PCM_8KHZ",
                "message": "Need at least 1.5 seconds of clear speech."
            }

        # 5. Generate embedding
        embedding = generate_embedding_from_waveform(speech_waveform, sr)

        # 6. Cosine similarity search
        best_name = None
        best_customer_id = None
        best_sim = 0.0

        for row in all_embeddings:
            emb = np.array(row.embedding)
            norm_a = np.linalg.norm(embedding)
            norm_b = np.linalg.norm(emb)
            sim = (
                float(np.dot(embedding, emb) / (norm_a * norm_b))
                if norm_a and norm_b
                else 0.0
            )
            if sim > best_sim:
                best_sim = sim
                best_customer_id = row.customer_id
                best_name = row.customer_name
        threshold = DEFAULT_THRESHOLD
        authenticated = best_customer_id is not None and best_sim >= threshold
        processing_time_ms = (time.perf_counter() - total_start) * 1000.0
        current_timestamp = datetime.utcnow().isoformat()

        print(f"[Voice Auth] Best match: {best_name} ({best_customer_id}) with similarity {best_sim:.4f} (Threshold: {threshold})")

        if authenticated:
            # Save authentication log
            crud.save_authentication_log(
                db=db,
                customer_id=best_customer_id,
                similarity=best_sim,
                threshold=threshold,
                authenticated=True,
                processing_time_ms=processing_time_ms,
                audio_duration=duration_seconds
            )
            matched_customer = crud.get_customer_by_id(db, best_customer_id)
            matched_embedding_count = len(matched_customer.voice_embeddings) if matched_customer else 0

            # Auto-learning: save this audio as an additional embedding so the model
            # improves over time and learns the caller's mobile/codec voice variations.
            # Only save if we have fewer than 10 embeddings (avoid unbounded growth).
            if matched_embedding_count < 10:
                try:
                    crud.save_embedding(
                        db=db,
                        customer_id=best_customer_id,
                        embedding=embedding,
                        sample_rate=sr,
                        audio_duration=duration_seconds
                    )
                    print(f"[Voice Auth] Auto-learned new embedding for '{best_name}' (now {matched_embedding_count + 1} embeddings)", flush=True)
                except Exception as learn_err:
                    print(f"[Voice Auth] Auto-learn save failed (non-fatal): {learn_err}", flush=True)

            return {
                "status": "success",
                "existing_customer": True,
                "customer_id": str(best_customer_id),
                "customer_name": best_name,
                "authentication_result": "AUTHENTICATED",
                "similarity": round(best_sim * 100, 2),
                "threshold": threshold,
                "processing_time_ms": round(processing_time_ms, 2),
                "audio_duration": round(duration_seconds, 2),
                "matched_embedding_count": matched_embedding_count,
                "model": "ECAPA-TDNN",
                "audio_format": "PCM_8KHZ",
                "timestamp": current_timestamp,
                "message": "Speaker authenticated successfully."
            }
        else:
            # Voice not recognized — do NOT auto-create fake CUSTOMER_XXXXXX profiles.
            # Return UNRECOGNIZED so orchestration treats the caller as an unknown guest.
            print(f"[Voice Auth] No match above threshold {threshold:.2f} — returning UNRECOGNIZED (best similarity: {best_sim:.4f})", flush=True)
            return {
                "status": "success",
                "existing_customer": False,
                "new_customer_created": False,
                "customer_id": None,
                "customer_name": None,
                "authentication_result": "UNRECOGNIZED",
                "similarity": round(best_sim * 100, 2),
                "threshold": threshold,
                "processing_time_ms": round(processing_time_ms, 2),
                "audio_duration": round(duration_seconds, 2),
                "model": "ECAPA-TDNN",
                "audio_format": "PCM_8KHZ",
                "timestamp": current_timestamp,
                "message": "Voice not recognized. No match above threshold."
            }

    except HTTPException:
        raise
    except Exception as exc:
        processing_time_ms = (time.perf_counter() - total_start) * 1000.0
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "authentication_result": "INTERNAL_SERVER_ERROR",
            "processing_time_ms": round(processing_time_ms, 2),
            "audio_format": "PCM_8KHZ",
            "message": str(exc)
        }
    finally:
        temp_pcm_path.unlink(missing_ok=True)
        temp_wav_path.unlink(missing_ok=True)
