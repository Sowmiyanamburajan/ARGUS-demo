# src/chatbot.py
import sqlite3, os, uuid, datetime, json
from textblob import TextBlob
from .audio_emotion import map_audio_features_to_emotion
from .voice_stress import voice_stress_score
from .emotion_manipulation import flag_emotion_manipulation
from pathlib import Path
import numpy as np
import librosa

# Optional OpenAI usage (for human-like followups & summaries)
openai = None
OPENAI_API_KEY = os.environ.get("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")  # make sure to set your env var

if OPENAI_API_KEY:
    try:
        import openai as _openai
        _openai.api_key = OPENAI_API_KEY
        openai = _openai
    except Exception:
        openai = None

DB = "outputs/chat_sessions.db"
os.makedirs("outputs", exist_ok=True)

ADVICE = {
    "happy": ["Keep enjoying your day! 😊", "Share your happiness with someone you trust.", "Consider journaling about what made you happy today."],
    "sad": ["Talk to a friend or family member.", "Consider a short walk or breathing exercises.", "Remember that it's okay to feel sad sometimes.", "Try listening to uplifting music or watching something funny."],
    "angry": ["Take deep breaths and count to 10.", "Step away for a few minutes to cool down.", "Try physical exercise to release tension.", "Consider what specifically triggered this feeling."],
    "fear": ["Grounding techniques can help - try the 5-4-3-2-1 method.", "Talk to someone you trust if possible.", "Focus on your breathing and remind yourself you're safe.", "Consider what you can control vs. what you cannot."],
    "surprised": ["Take a moment to process this new information.", "Consider whether this surprise is positive or concerning.", "Talk to someone about what surprised you."],
    "neutral": ["Take some rest and check in with yourself.", "Consider mindfulness or a short break.", "Reflect on your current needs and wants."],
    "ended": ["Chat ended. Take care! 👋"]
}

# --- DB helper ---
def _conn():
    conn = sqlite3.connect(DB, check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS answers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT, answer_text TEXT, audio_path TEXT,
        text_emotion TEXT, audio_emotion TEXT, combined_emotion TEXT, ts TEXT
    )""")
    conn.commit()
    return conn

# --- Enhanced Text emotion detection ---
def detect_text_emotion(text):
    if not text or text.strip() == "":
        return "neutral"
    
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    t = text.lower()
    
    # Enhanced keyword detection for more accurate emotion classification
    sad_keywords = ["sad", "depress", "upset", "down", "unhappy", "miserable", "gloomy", "melancholy", "heartbroken", "devastated", "disappointed", "frustrated"]
    angry_keywords = ["angry", "mad", "furious", "rage", "irritated", "annoyed", "frustrated", "outraged", "livid", "fuming", "hostile", "aggressive"]
    fear_keywords = ["scared", "afraid", "fearful", "anxious", "worried", "nervous", "terrified", "panic", "alarmed", "concerned", "apprehensive", "uneasy"]
    happy_keywords = ["happy", "joy", "excited", "thrilled", "delighted", "ecstatic", "cheerful", "pleased", "content", "satisfied", "elated", "blissful"]
    surprise_keywords = ["surprised", "shocked", "amazed", "astonished", "stunned", "bewildered", "confused", "perplexed", "startled"]
    
    # Check for specific emotion keywords first
    if any(k in t for k in sad_keywords): 
        return "sad"
    if any(k in t for k in angry_keywords): 
        return "angry"
    if any(k in t for k in fear_keywords): 
        return "fear"
    if any(k in t for k in happy_keywords): 
        return "happy"
    if any(k in t for k in surprise_keywords): 
        return "surprised"
    
    # Use polarity and subjectivity for more nuanced detection
    if polarity >= 0.6: 
        return "happy"
    if polarity <= -0.6: 
        return "sad"
    if polarity >= 0.3 and subjectivity > 0.5: 
        return "happy"
    if polarity <= -0.3 and subjectivity > 0.5: 
        return "sad"
    if polarity <= -0.4 and subjectivity < 0.3: 
        return "angry"
    if subjectivity > 0.7 and abs(polarity) < 0.2: 
        return "surprised"
    
    return "neutral"

# --- Voice Stress Analysis ---
def analyze_voice_stress(audio_path):
    """Analyze voice stress indicators for deception detection"""
    try:
        if not audio_path or not os.path.exists(audio_path):
            return {"stress_score": 0.0, "indicators": [], "deception_prob": 0.0}
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Basic voice stress indicators
        stress_indicators = []
        stress_score = 0.0
        
        # 1. Pitch variation analysis
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) > 10:
            pitch_std = np.std(pitch_values)
            pitch_mean = np.mean(pitch_values)
            pitch_cv = pitch_std / pitch_mean if pitch_mean > 0 else 0
            
            # High pitch variation often indicates stress
            if pitch_cv > 0.3:
                stress_indicators.append("high_pitch_variation")
                stress_score += 0.3
        
        # 2. Speaking rate analysis
        # Count syllables (simplified approach)
        words = len(y) / (sr * 0.1)  # Rough word count based on duration
        speaking_rate = words / (len(y) / sr) if len(y) > 0 else 0
        
        if speaking_rate > 3.0:  # Very fast speech
            stress_indicators.append("rapid_speech")
            stress_score += 0.2
        elif speaking_rate < 1.0:  # Very slow speech
            stress_indicators.append("slow_speech")
            stress_score += 0.2
        
        # 3. Voice tremor analysis
        # Analyze micro-tremors in the voice
        frame_length = 2048
        hop_length = 512
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        frame_energy = np.sum(frames**2, axis=0)
        
        # Calculate energy variation
        energy_std = np.std(frame_energy)
        energy_mean = np.mean(frame_energy)
        energy_cv = energy_std / energy_mean if energy_mean > 0 else 0
        
        if energy_cv > 0.5:  # High energy variation indicates tremor
            stress_indicators.append("voice_tremor")
            stress_score += 0.25
        
        # 4. Pause analysis
        # Detect unnatural pauses
        silence_threshold = 0.01 * np.max(frame_energy)
        silence_frames = frame_energy < silence_threshold
        silence_ratio = np.sum(silence_frames) / len(silence_frames)
        
        if silence_ratio > 0.3:  # Too many pauses
            stress_indicators.append("excessive_pauses")
            stress_score += 0.15
        
        # 5. Spectral analysis
        # Stressed voices often have different spectral characteristics
        stft = librosa.stft(y)
        magnitude = np.abs(stft)
        
        # High-frequency energy analysis
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        high_freq_mask = freqs > 4000
        high_freq_energy = np.sum(magnitude[high_freq_mask, :])
        total_energy = np.sum(magnitude)
        high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        
        if high_freq_ratio > 0.3:  # High frequency content
            stress_indicators.append("high_frequency_stress")
            stress_score += 0.1
        
        # Calculate deception probability
        deception_prob = min(0.95, max(0.05, stress_score))
        
        return {
            "stress_score": float(stress_score),
            "indicators": stress_indicators,
            "deception_prob": float(deception_prob),
            "pitch_variation": float(pitch_cv) if len(pitch_values) > 10 else 0.0,
            "speaking_rate": float(speaking_rate),
            "voice_tremor": float(energy_cv),
            "pause_ratio": float(silence_ratio)
        }
        
    except Exception as e:
        print(f"Error in voice stress analysis: {e}")
        return {"stress_score": 0.0, "indicators": [], "deception_prob": 0.0}

# --- Emotion Manipulation Detection ---
def detect_emotion_manipulation(text_emotion, audio_emotion, voice_stress_data):
    """Detect if emotions appear artificially forced or manipulated"""
    try:
        manipulation_indicators = []
        manipulation_score = 0.0
        
        # 1. Cross-modal emotion inconsistency
        if text_emotion and audio_emotion and text_emotion != audio_emotion:
            # Check if the difference is significant
            emotion_pairs = [
                ("happy", "sad"), ("happy", "angry"), ("happy", "fear"),
                ("sad", "happy"), ("sad", "angry"),
                ("angry", "happy"), ("angry", "sad"),
                ("fear", "happy"), ("fear", "sad")
            ]
            
            if (text_emotion, audio_emotion) in emotion_pairs:
                manipulation_indicators.append("emotion_inconsistency")
                manipulation_score += 0.4
        
        # 2. Voice stress vs emotion mismatch
        if voice_stress_data and voice_stress_data.get("deception_prob", 0) > 0.7:
            if text_emotion in ["happy", "excited"]:
                manipulation_indicators.append("stressed_positive_emotion")
                manipulation_score += 0.3
        
        # 3. Unnatural emotion intensity
        if text_emotion in ["happy", "excited", "thrilled"]:
            # Check if the text shows excessive positivity
            if "extremely" in text_emotion or "incredibly" in text_emotion:
                manipulation_indicators.append("excessive_positivity")
                manipulation_score += 0.2
        
        # 4. Voice characteristics vs emotion
        if voice_stress_data:
            pitch_var = voice_stress_data.get("pitch_variation", 0)
            speaking_rate = voice_stress_data.get("speaking_rate", 0)
            
            # High pitch variation with calm emotions might indicate manipulation
            if pitch_var > 0.4 and text_emotion in ["neutral", "calm"]:
                manipulation_indicators.append("unnatural_voice_calm")
                manipulation_score += 0.2
            
            # Very fast speech with sad emotions
            if speaking_rate > 2.5 and text_emotion in ["sad", "depressed"]:
                manipulation_indicators.append("rapid_sad_speech")
                manipulation_score += 0.2
        
        # 5. Temporal analysis (if we had multiple samples)
        # This would require storing previous emotions for comparison
        
        manipulation_prob = min(0.95, max(0.05, manipulation_score))
        
        return {
            "manipulation_prob": float(manipulation_prob),
            "indicators": manipulation_indicators,
            "is_manipulated": manipulation_prob > 0.6
        }
        
    except Exception as e:
        print(f"Error in emotion manipulation detection: {e}")
        return {"manipulation_prob": 0.0, "indicators": [], "is_manipulated": False}

def fuse_emotions(text_em, audio_em):
    if audio_em and audio_em == text_em: return text_em
    if audio_em and text_em == "neutral": return audio_em
    if audio_em and text_em != "neutral" and audio_em != "neutral":
        if audio_em in ["angry","fear"] or text_em in ["angry","fear"]:
            return audio_em if audio_em in ["angry","fear"] else text_em
    return text_em

# --- Enhanced follow-up generation ---
def generate_followup(combined_emotion, recent_text=None):
    if openai:
        prompt = f"You are a supportive AI assistant specializing in sentiment analysis and emotional support. The user's detected emotion is: {combined_emotion}. Their recent message was: '{recent_text}'\n\nGive one empathetic, personalized follow-up question to better understand the user's situation. Keep it conversational and supportive (max 80 characters)."
        try:
            resp = openai.Completion.create(model="gpt-3.5-turbo", prompt=prompt, max_tokens=60)
            return resp.choices[0].text.strip()
        except Exception:
            pass
    
    # Enhanced follow-up questions based on emotion
    followups = {
        "sad": [
            "I'm sorry to hear that. Would you like to tell me what happened?",
            "That sounds really difficult. What's been weighing on your mind?",
            "I can sense you're going through a tough time. What's making you feel this way?"
        ],
        "angry": [
            "That sounds really frustrating. What specifically made you feel angry?",
            "I can tell you're upset. What triggered this feeling?",
            "That must be really aggravating. Can you tell me more about what happened?"
        ],
        "fear": [
            "I understand you're feeling worried. What's concerning you most right now?",
            "That sounds scary. What's making you feel anxious?",
            "I can sense some anxiety. What's on your mind that's causing worry?"
        ],
        "happy": [
            "That's wonderful! What made you feel so good today?",
            "I love hearing that! What brought you this joy?",
            "That's fantastic! Tell me more about what's making you happy!"
        ],
        "surprised": [
            "Wow, that sounds unexpected! What happened that surprised you?",
            "That must have been quite a shock! Can you tell me more?",
            "I can tell that caught you off guard. What was it that surprised you?"
        ],
        "neutral": [
            "How are you feeling about everything right now?",
            "What's on your mind today?",
            "Would you like to share what you're thinking about?"
        ]
    }
    
    import random
    return random.choice(followups.get(combined_emotion, ["How are you feeling right now?"]))

# --- Main chat receiver ---
def chat_receive_dynamic(session_id, answer_text=None, audio_path=None, audio_emotion=None):
    # Check for end-chat keywords
    END_CHAT_KEYWORDS = ["thanks", "thank you", "bye", "goodbye", "exit", "stop"]
    if answer_text and any(word in answer_text.lower() for word in END_CHAT_KEYWORDS):
        ts = datetime.datetime.utcnow().isoformat()
        conn = _conn()
        conn.execute("""INSERT INTO answers (session_id, answer_text, audio_path, text_emotion, audio_emotion, combined_emotion, ts)
                        VALUES (?,?,?,?,?,?,?)""", (session_id, answer_text, audio_path, None, None, "ended", ts))
        conn.commit(); conn.close()
        return {
            "session_id": session_id,
            "combined_emotion": "ended",
            "next_question": "Thank you for chatting! Take care 🙂",
            "advice": []
        }

    # Normal processing
    text_em = detect_text_emotion(answer_text) if answer_text is not None else None
    audio_em = None
    if audio_path:
        if audio_emotion and isinstance(audio_emotion, dict):
            audio_em = audio_emotion.get("emotion")
        else:
            af = map_audio_features_to_emotion(audio_path)
            audio_em = af.get("emotion")
    combined = fuse_emotions(text_em, audio_em) if text_em or audio_em else "neutral"

    # Voice stress analysis for fraud detection
    voice_stress_data = None
    if audio_path:
        voice_stress_data = analyze_voice_stress(audio_path)
    
    # Emotion manipulation detection
    manipulation_data = detect_emotion_manipulation(text_em, audio_em, voice_stress_data)

    # Store to DB
    ts = datetime.datetime.utcnow().isoformat()
    conn = _conn()
    conn.execute("""INSERT INTO answers (session_id, answer_text, audio_path, text_emotion, audio_emotion, combined_emotion, ts)
                    VALUES (?,?,?,?,?,?,?)""", (session_id, answer_text, audio_path, text_em, audio_em, combined, ts))
    conn.commit(); conn.close()

    # Generate contextual follow-up based on analysis
    next_q = generate_contextual_followup(combined, text_em, audio_em, voice_stress_data, manipulation_data, recent_text=(answer_text or ""))
    
    # Enhanced return with comprehensive analysis
    result = {
        "session_id": session_id,
        "combined_emotion": combined,
        "next_question": next_q,
        "advice": ADVICE.get(combined, []),
        "sentiment": combined,
        "text_emotion": text_em,
        "audio_emotion": audio_em,
        "confidence": "high" if text_em and audio_em and text_em == audio_em else "medium",
        "voice_stress": voice_stress_data,
        "emotion_manipulation": manipulation_data
    }
    
    # Add behavioral insights based on analysis
    if voice_stress_data and voice_stress_data.get("deception_prob", 0) > 0.7:
        result["behavior_insights"] = "I notice some stress indicators in your voice. This might indicate you're feeling pressured or anxious. Would you like to talk about what's causing this stress?"
    elif manipulation_data and manipulation_data.get("is_manipulated", False):
        result["behavior_insights"] = "I sense there might be some inconsistency between what you're saying and how you're feeling. It's okay to be honest about your true emotions."
    elif combined in ["angry", "fear"]:
        result["behavior_insights"] = "I notice you might be experiencing some stress. Consider taking a moment to breathe deeply."
    elif combined == "sad":
        result["behavior_insights"] = "It's completely normal to feel down sometimes. Remember that these feelings are temporary."
    elif combined == "happy":
        result["behavior_insights"] = "It's wonderful to see you feeling positive! This is a great time to practice gratitude."
    
    return result

# --- Enhanced contextual follow-up generation ---
def generate_contextual_followup(combined_emotion, text_emotion, audio_emotion, voice_stress_data, manipulation_data, recent_text=""):
    """Generate follow-up questions based on comprehensive analysis"""
    
    # Check for manipulation or stress indicators
    if manipulation_data and manipulation_data.get("is_manipulated", False):
        return "I sense there might be some mixed feelings here. Would you like to share what you're really feeling right now?"
    
    if voice_stress_data and voice_stress_data.get("deception_prob", 0) > 0.7:
        return "I notice some tension in your voice. What's making you feel stressed or anxious right now?"
    
    # Use existing follow-up generation for normal cases
    return generate_followup(combined_emotion, recent_text)

# --- Chat history fetcher ---
def get_chat_history(session_id):
    conn = _conn()
    cur = conn.cursor()
    cur.execute("""SELECT answer_text, audio_path, text_emotion, audio_emotion, combined_emotion, ts
                   FROM answers WHERE session_id=? ORDER BY id ASC""", (session_id,))
    rows = cur.fetchall(); conn.close()
    out = []
    for r in rows:
        out.append({
            "answer_text": r[0],
            "audio_path": r[1],
            "text_emotion": r[2],
            "audio_emotion": r[3],
            "combined_emotion": r[4],
            "ts": r[5]
        })
    return out





# import sqlite3, os, uuid, datetime
# from textblob import TextBlob
# from .audio_emotion import map_audio_features_to_emotion

# DB = "outputs/chat_sessions.db"
# os.makedirs("outputs", exist_ok=True)

# ADVICE = {
#     "happy": ["Keep enjoying your day!", "Share your happiness with someone!"],
#     "sad": ["Consider talking to a close friend.", "Go for a short walk to refresh."],
#     "angry": ["Take deep breaths.", "Step away for a few minutes."],
#     "fear": ["Try grounding techniques.", "Speak with someone you trust."],
#     "neutral": ["Keep observing your feelings.", "Maybe relax for a while."]
# }

# def _conn():
#     conn = sqlite3.connect(DB, check_same_thread=False)
#     conn.execute("""CREATE TABLE IF NOT EXISTS answers (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         session_id TEXT, answer_text TEXT, audio_path TEXT,
#         text_emotion TEXT, audio_emotion TEXT, combined_emotion TEXT, ts TEXT
#     )""")
#     conn.commit()
#     return conn

# def detect_text_emotion(text):
#     if not text or text.strip()=="": return "neutral"
#     blob = TextBlob(text)
#     polarity = blob.sentiment.polarity
#     if polarity >= 0.4: return "happy"
#     if polarity <= -0.35: return "angry"
#     t = text.lower()
#     if any(k in t for k in ["sad","depress","upset","down","unhappy"]): return "sad"
#     if any(k in t for k in ["scared","afraid","fearful","anxious","worried"]): return "fear"
#     return "neutral" if abs(polarity)<0.2 else ("happy" if polarity>0 else "sad")

# def fuse_emotions(text_em, audio_em):
#     if audio_em and audio_em == text_em: return text_em
#     if audio_em and text_em=="neutral": return audio_em
#     if audio_em and text_em!="neutral" and audio_em!="neutral":
#         if audio_em in ["angry","fear"] or text_em in ["angry","fear"]:
#             return audio_em if audio_em in ["angry","fear"] else text_em
#     return text_em

# def chat_receive_dynamic(session_id, answer_text=None, audio_path=None):
#     text_em = detect_text_emotion(answer_text)
#     audio_em = None
#     if audio_path:
#         audio_features = map_audio_features_to_emotion(audio_path)
#         audio_em = audio_features.get("emotion")
#     combined_em = fuse_emotions(text_em, audio_em)

#     # Store answer
#     ts = datetime.datetime.utcnow().isoformat()
#     conn = _conn()
#     conn.execute("""INSERT INTO answers
#         (session_id, answer_text, audio_path, text_emotion, audio_emotion, combined_emotion, ts)
#         VALUES (?,?,?,?,?,?,?)""",
#         (session_id, answer_text, audio_path, text_em, audio_em, combined_em, ts))
#     conn.commit(); conn.close()

#     # AI-like dynamic questions
#     next_question = {
#         "sad": "I see you're feeling sad. Can you tell me why?",
#         "angry": "It seems you're upset. What made you feel angry?",
#         "fear": "I sense some anxiety. What is worrying you?",
#         "happy": "You're feeling good! What made your day nice?",
#         "neutral": "Would you like to share more about how you're feeling?"
#     }.get(combined_em, "Can you tell me more about it?")

#     return {
#         "combined_emotion": combined_em,
#         "next_question": next_question,
#         "advice": ADVICE.get(combined_em)
#     }

# def get_chat_history(session_id):
#     conn = _conn()
#     cur = conn.cursor()
#     cur.execute("""SELECT answer_text, audio_path, text_emotion, audio_emotion, combined_emotion, ts
#                    FROM answers WHERE session_id=? ORDER BY id ASC""", (session_id,))
#     rows = cur.fetchall(); conn.close()
#     return [{"text": r[0], "audio": r[1], "text_emotion": r[2], "audio_emotion": r[3], "combined": r[4], "ts": r[5]} for r in rows]

