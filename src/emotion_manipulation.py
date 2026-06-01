# src/emotion_manipulation.py
# Enhanced emotion manipulation detection with advanced heuristics
import cv2
import numpy as np
import librosa
from scipy import stats
import json

def analyze_facial_emotion_intensity(image_path):
    """Analyze facial emotion intensity using computer vision"""
    try:
        import face_recognition
        from PIL import Image
        
        # Load image
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        
        if not face_locations:
            return None
            
        # Extract face region
        top, right, bottom, left = face_locations[0]
        face_image = image[top:bottom, left:right]
        
        # Convert to grayscale for analysis
        gray_face = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        
        # Analyze facial features intensity
        # Eye region analysis
        eye_region = gray_face[int(0.1*gray_face.shape[0]):int(0.4*gray_face.shape[0]), :]
        eye_intensity = np.mean(eye_region)
        
        # Mouth region analysis
        mouth_region = gray_face[int(0.6*gray_face.shape[0]):, :]
        mouth_intensity = np.mean(mouth_region)
        
        # Calculate intensity variance (artificial emotions tend to be more uniform)
        intensity_variance = np.var(gray_face)
        
        return {
            'eye_intensity': float(eye_intensity),
            'mouth_intensity': float(mouth_intensity),
            'intensity_variance': float(intensity_variance),
            'overall_intensity': float(np.mean(gray_face))
        }
    except Exception as e:
        print(f"Error analyzing facial emotion: {e}")
        return None

def analyze_audio_emotion_patterns(audio_path):
    """Analyze audio emotion patterns for manipulation detection"""
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Extract prosodic features
        pitch = librosa.yin(y, fmin=50, fmax=400)
        pitch = pitch[~np.isnan(pitch)]
        
        # Calculate pitch statistics
        pitch_mean = np.mean(pitch) if len(pitch) > 0 else 0
        pitch_std = np.std(pitch) if len(pitch) > 0 else 0
        pitch_range = np.max(pitch) - np.min(pitch) if len(pitch) > 0 else 0
        
        # Extract energy features
        energy = librosa.feature.rms(y=y)[0]
        energy_mean = np.mean(energy)
        energy_std = np.std(energy)
        
        # Calculate tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Analyze spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        return {
            'pitch_mean': float(pitch_mean),
            'pitch_std': float(pitch_std),
            'pitch_range': float(pitch_range),
            'energy_mean': float(energy_mean),
            'energy_std': float(energy_std),
            'tempo': float(tempo),
            'spectral_centroid_mean': float(np.mean(spectral_centroids)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff))
        }
    except Exception as e:
        print(f"Error analyzing audio emotion: {e}")
        return None

def detect_artificial_emotion_patterns(facial_data, audio_data):
    """Detect patterns that suggest artificially forced emotions"""
    manipulation_indicators = []
    confidence = 0.0
    
    if facial_data:
        # Check for unnatural intensity patterns
        if facial_data['intensity_variance'] < 100:  # Too uniform
            manipulation_indicators.append('unnatural_facial_uniformity')
            confidence += 0.3
            
        # Check for extreme intensity values
        if facial_data['overall_intensity'] > 200 or facial_data['overall_intensity'] < 50:
            manipulation_indicators.append('extreme_facial_intensity')
            confidence += 0.2
    
    if audio_data:
        # Check for unnatural pitch patterns
        if audio_data['pitch_std'] < 10:  # Too monotone
            manipulation_indicators.append('unnatural_pitch_monotony')
            confidence += 0.3
            
        # Check for extreme energy variations
        if audio_data['energy_std'] > audio_data['energy_mean'] * 0.8:
            manipulation_indicators.append('extreme_energy_variation')
            confidence += 0.2
            
        # Check for unnatural tempo
        if audio_data['tempo'] < 60 or audio_data['tempo'] > 200:
            manipulation_indicators.append('unnatural_tempo')
            confidence += 0.1
    
    return {
        'manipulation_indicators': manipulation_indicators,
        'confidence': min(confidence, 1.0),
        'is_manipulated': len(manipulation_indicators) > 0
    }

def flag_emotion_manipulation(text_emotion, audio_emotion, video_emotion, 
                           audio_path=None, image_path=None):
    """Enhanced emotion manipulation detection with multi-modal analysis"""
    
    # Basic emotion consistency check
    basic_result = basic_emotion_consistency_check(text_emotion, audio_emotion, video_emotion)
    
    # Advanced pattern analysis if paths provided
    advanced_result = None
    if audio_path and image_path:
        facial_data = analyze_facial_emotion_intensity(image_path)
        audio_data = analyze_audio_emotion_patterns(audio_path)
        advanced_result = detect_artificial_emotion_patterns(facial_data, audio_data)
    
    # Combine results
    final_confidence = basic_result.get('confidence', 0.0)
    if advanced_result:
        final_confidence = max(final_confidence, advanced_result['confidence'])
    
    manipulation_indicators = basic_result.get('indicators', [])
    if advanced_result:
        manipulation_indicators.extend(advanced_result['manipulation_indicators'])
    
    return {
        'manipulated': basic_result.get('manipulated', False) or (advanced_result and advanced_result['is_manipulated']),
        'confidence': final_confidence,
        'indicators': list(set(manipulation_indicators)),
        'basic_analysis': basic_result,
        'advanced_analysis': advanced_result
    }

def basic_emotion_consistency_check(text_emotion, audio_emotion, video_emotion):
    """Basic emotion consistency check between modalities"""
    if text_emotion is None and audio_emotion is None and video_emotion is None:
        return {'manipulated': False, 'confidence': 0.0, 'indicators': [], 'reason': 'no_signals'}
    
    indicators = []
    confidence = 0.0
    
    # Define emotion opposites and inconsistencies
    opposites = {('happy','sad'),('happy','angry'),('sad','happy'),('angry','happy'),('fear','happy')}
    strong_emotions = ['angry', 'fear', 'disgust']
    neutral_emotions = ['neutral', 'calm']
    
    # Check video-audio mismatch
    if video_emotion and audio_emotion:
        if (video_emotion, audio_emotion) in opposites:
            indicators.append('video_audio_emotion_mismatch')
            confidence += 0.4
        elif video_emotion != audio_emotion:
            indicators.append('video_audio_emotion_difference')
            confidence += 0.2
    
    # Check text-audio mismatch
    if text_emotion and audio_emotion:
        if audio_emotion in strong_emotions and text_emotion in neutral_emotions:
            indicators.append('audio_stress_vs_neutral_text')
            confidence += 0.3
        elif (text_emotion, audio_emotion) in opposites:
            indicators.append('text_audio_emotion_mismatch')
            confidence += 0.3
    
    # Check text-video mismatch
    if text_emotion and video_emotion:
        if (text_emotion, video_emotion) in opposites:
            indicators.append('text_video_emotion_mismatch')
            confidence += 0.3
    
    return {
        'manipulated': len(indicators) > 0,
        'confidence': min(confidence, 1.0),
        'indicators': indicators,
        'reason': 'consistency_check'
    }
