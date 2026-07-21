"""
=========================================================
 VOICE AUTHENTICATION - INTEGRATION GUIDE FOR SNAPSERVE.AI
=========================================================

 Voice Auth API:  https://<RAILWAY_DEPLOYED_URL>
 Endpoint:       POST /authenticate/pcm
 Audio Format:   Raw PCM, 8kHz, 16-bit, Mono
 Min Duration:   3 seconds of speech
 Response Time:  ~100ms (on 24 vCPU Railway)

=========================================================
 HOW TO INTEGRATE WITH SNAPSERVE.AI PIPELINE
=========================================================

 Pipeline Flow:
   Customer Call → Vobiz → Sarvam (STT) → Voice Auth API → Gemini LLM → Cartesia (TTS) → Customer

 The Voice Auth step runs in parallel with STT.
 It identifies WHO is speaking while STT converts WHAT they're saying.

=========================================================
"""

import requests
import io
import time


# =========================================================
# CONFIGURATION - FULLY AUTOMATIC (No code changes needed!)
# =========================================================
# The manager just sets these Environment Variables in Railway dashboard:
#   VOICE_AUTH_URL = https://your-app.railway.app
#
# The code below picks them up automatically.
# =========================================================
import os

VOICE_AUTH_BASE = os.getenv("VOICE_AUTH_URL")
if not VOICE_AUTH_BASE:
    raise RuntimeError("Set VOICE_AUTH_URL environment variable in Railway (e.g., https://your-app.railway.app)")
VOICE_AUTH_API_URL = VOICE_AUTH_BASE + "/authenticate/pcm"
BUFFER_SECONDS = int(os.getenv("VOICE_AUTH_BUFFER_SECONDS", "3"))


# =========================================================
# FUNCTION 1: Authenticate caller from PCM audio bytes
# =========================================================
def authenticate_caller(pcm_audio_bytes: bytes) -> dict:
    """
    Send raw PCM audio (8kHz, 16-bit, mono) to Voice Auth API.
    
    Args:
        pcm_audio_bytes: Raw PCM audio bytes from the live call
        
    Returns:
        dict with keys:
            - action: "GRANTED" (known customer) or "NEW_CUSTOMER"
            - customer_id: int
            - customer_name: str
            - similarity: float (0.0 to 1.0)
            - is_new: bool
    """
    try:
        start = time.time()
        
        response = requests.post(
            VOICE_AUTH_API_URL,
            files={"audio": ("call_audio.pcm", pcm_audio_bytes, "audio/octet-stream")},
            timeout=10  # 10 second timeout
        )
        
        latency_ms = int((time.time() - start) * 1000)
        
        if response.status_code == 200:
            result = response.json()
            result["latency_ms"] = latency_ms
            return result
        else:
            # API error - treat as unknown caller
            return {
                "action": "ERROR",
                "customer_name": "Guest",
                "customer_id": None,
                "similarity": 0.0,
                "is_new": True,
                "error": response.text,
                "latency_ms": latency_ms
            }
            
    except requests.exceptions.Timeout:
        return {
            "action": "TIMEOUT",
            "customer_name": "Guest",
            "customer_id": None,
            "is_new": True,
            "error": "Voice Auth API timed out"
        }
    except Exception as e:
        return {
            "action": "ERROR",
            "customer_name": "Guest",
            "customer_id": None,
            "is_new": True,
            "error": str(e)
        }


# =========================================================
# FUNCTION 2: Build LLM prompt with caller identity
# =========================================================
def build_llm_prompt(auth_result: dict, base_prompt: str) -> str:
    """
    Inject caller identity into the Gemini LLM prompt.
    
    Args:
        auth_result: Response from authenticate_caller()
        base_prompt: Your existing Gemini system prompt
        
    Returns:
        Updated prompt with caller identity
    """
    if auth_result["action"] == "GRANTED":
        identity = f"""
CALLER IDENTITY (from Voice Authentication):
- Name: {auth_result['customer_name']}
- Customer ID: {auth_result['customer_id']}
- Voice Match Confidence: {auth_result['similarity'] * 100:.0f}%
- Status: Returning Customer

INSTRUCTION: Greet the caller by name. You already know who they are.
"""
    else:
        identity = """
CALLER IDENTITY (from Voice Authentication):
- Status: New / Unknown Caller
- Voice Match: No match found in database

INSTRUCTION: This is a new caller. Ask for their name politely.
"""
    return base_prompt + "\n" + identity


# =========================================================
# EXAMPLE: How to use in SnapServe.ai pipeline
# =========================================================
if __name__ == "__main__":
    
    # ----- STEP 1: Simulate capturing audio from Vobiz -----
    # In production, replace this with actual Vobiz audio stream
    # Vobiz sends raw PCM 8kHz audio via WebSocket/SIP
    
    print("=" * 60)
    print("SnapServe.ai + Voice Authentication Integration Test")
    print("=" * 60)
    
    # Example: Read a test PCM file (replace with live Vobiz stream)
    try:
        with open("test_voice_1_deep.pcm", "rb") as f:
            pcm_audio = f.read()
        print(f"[OK] Loaded test audio: {len(pcm_audio)} bytes")
    except FileNotFoundError:
        print("[!] No test file found. In production, audio comes from Vobiz.")
        pcm_audio = None
    
    if pcm_audio:
        # ----- STEP 2: Authenticate the caller -----
        print("\n[...] Sending audio to Voice Auth API...")
        result = authenticate_caller(pcm_audio)
        print(f"[OK] Response in {result.get('latency_ms', '?')} ms:")
        print(f"     Action:    {result['action']}")
        print(f"     Customer:  {result['customer_name']}")
        print(f"     Match:     {result.get('similarity', 0) * 100:.0f}%")
        
        # ----- STEP 3: Build Gemini prompt with identity -----
        base_prompt = "You are a helpful SnapServe.ai voice assistant."
        final_prompt = build_llm_prompt(result, base_prompt)
        print(f"\n[OK] Gemini Prompt Built:")
        print(final_prompt)
        
        # ----- STEP 4: Send to Gemini 2.5 Flash Lite -----
        # In production:
        # response = gemini_client.generate(prompt=final_prompt, user_text=stt_text)
        
        # ----- STEP 5: Send Gemini response to Cartesia TTS -----
        # In production:
        # audio_response = cartesia_client.speak(response.text)
        # vobiz.play_audio(audio_response)
        
        print("\n[OK] Integration test complete!")
