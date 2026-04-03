import os
import base64
import tempfile
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── IMAGE HANDLER ────────────────────────────────────────────────────────────
def extract_text_from_image(image_bytes: bytes) -> dict:
    """
    Takes image bytes, sends to Groq vision model, extracts math problem text.
    Returns dict with extracted_text and confidence.
    """
    # Encode image to base64
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": """Extract the math problem from this image exactly as written.
Return ONLY the math problem text, nothing else.
If there are multiple problems, extract all of them.
Preserve all mathematical notation, numbers, and symbols accurately.
If the image is unclear or not a math problem, say: UNCLEAR_IMAGE"""
                        }
                    ]
                }
            ],
            temperature=0.0
        )

        extracted = response.choices[0].message.content.strip()

        # Determine confidence based on response
        if "UNCLEAR_IMAGE" in extracted:
            return {
                "extracted_text": "",
                "confidence": 0.0,
                "success": False,
                "error": "Image is unclear or does not contain a math problem."
            }

        # Estimate confidence based on length and content
        confidence = 0.9 if len(extracted) > 10 else 0.5

        return {
            "extracted_text": extracted,
            "confidence": confidence,
            "success": True,
            "error": ""
        }

    except Exception as e:
        return {
            "extracted_text": "",
            "confidence": 0.0,
            "success": False,
            "error": str(e)
        }


# ── AUDIO HANDLER ────────────────────────────────────────────────────────────
def extract_text_from_audio(audio_bytes: bytes, file_extension: str = "wav") -> dict:
    """
    Takes audio bytes, transcribes using Groq Whisper API.
    Returns dict with transcript and confidence.
    """
    try:
        # Write audio bytes to a temp file
        with tempfile.NamedTemporaryFile(
            suffix=f".{file_extension}", delete=False
        ) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Send to Groq Whisper
        with open(tmp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                response_format="verbose_json"  # gives us more detail
            )

        os.unlink(tmp_path)  # clean up temp file

        transcript = transcription.text.strip()

        if not transcript:
            return {
                "transcript": "",
                "confidence": 0.0,
                "success": False,
                "error": "No speech detected in audio."
            }

        # Post-process math phrases
        transcript = fix_math_phrases(transcript)

        return {
            "transcript": transcript,
            "confidence": 0.9,
            "success": True,
            "error": ""
        }

    except Exception as e:
        return {
            "transcript": "",
            "confidence": 0.0,
            "success": False,
            "error": str(e)
        }
    finally:
        # Ensure temp file cleanup
        try:
            if 'tmp_path' in locals():
                os.unlink(tmp_path)
        except Exception:
            pass


# ── MATH PHRASE FIXER ────────────────────────────────────────────────────────
def fix_math_phrases(text: str) -> str:
    """
    Converts spoken math phrases to proper notation.
    """
    replacements = {
        "square root of":     "sqrt(",
        "square root":        "sqrt(",
        "raised to the power": "^",
        "raised to":          "^",
        "to the power of":    "^",
        "to the power":       "^",
        "divided by":         "/",
        "multiplied by":      "*",
        "times":              "*",
        "plus":               "+",
        "minus":              "-",
        "equals":             "=",
        "equal to":           "=",
        "pi":                 "π",
        "infinity":           "∞",
        "alpha":              "α",
        "beta":               "β",
        "theta":              "θ",
        "delta":              "Δ",
        "sigma":              "Σ",
        "integral of":        "∫",
        "log base":           "log_",
        "natural log of":     "ln(",
        "absolute value of":  "|",
    }

    result = text.lower()
    for phrase, symbol in replacements.items():
        result = result.replace(phrase, symbol)

    return result

        print(f"  Input : {phrase}")
        print(f"  Output: {fixed}")
        print()

    print("Input handlers ready.")
    print("Image and audio processing will be tested via the UI in Step 5.")
