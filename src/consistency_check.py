# src/consistency_check.py
# Consistency Check Across Modalities - compare text with media
import re
import json
import numpy as np
from pathlib import Path
import cv2
import librosa
from textblob import TextBlob
import face_recognition
from .audio_emotion import analyze_audio_emotion, transcribe_audio
from .emotion_manipulation import flag_emotion_manipulation

class ModalityConsistencyChecker:
    def __init__(self):
        self.emotion_keywords = {
            'happy': ['happy', 'joy', 'excited', 'cheerful', 'delighted', 'pleased'],
            'sad': ['sad', 'depressed', 'gloomy', 'melancholy', 'sorrowful', 'unhappy'],
            'angry': ['angry', 'mad', 'furious', 'rage', 'irritated', 'annoyed'],
            'fear': ['fear', 'afraid', 'scared', 'terrified', 'anxious', 'worried'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'startled'],
            'disgust': ['disgusted', 'revolted', 'sickened', 'repulsed', 'nauseated']
        }
        
    def extract_text_emotion(self, text):
        """Extract emotion from text using keyword analysis and sentiment"""
        if not text or not text.strip():
            return None
            
        text_lower = text.lower()
        
        # Count emotion keywords
        emotion_scores = {}
        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotion_scores[emotion] = score
        
        # Get sentiment from TextBlob
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity  # -1 to 1
        
        # Map sentiment to emotion
        if sentiment > 0.3:
            emotion_scores['happy'] = emotion_scores.get('happy', 0) + 1
        elif sentiment < -0.3:
            emotion_scores['sad'] = emotion_scores.get('sad', 0) + 1
        
        # Return dominant emotion
        if emotion_scores:
            return max(emotion_scores, key=emotion_scores.get)
        return 'neutral'
    
    def extract_text_content_claims(self, text):
        """Extract factual claims and assertions from text"""
        if not text:
            return []
            
        # Simple claim extraction using patterns
        claims = []
        
        # Look for factual statements
        factual_patterns = [
            r'(?:I|we|they|he|she|it)\s+(?:am|is|are|was|were)\s+([^.!?]+)',
            r'(?:this|that|it)\s+(?:is|was)\s+([^.!?]+)',
            r'(?:I|we|they|he|she)\s+(?:have|had|has)\s+([^.!?]+)',
            r'(?:I|we|they|he|she)\s+(?:did|do|does)\s+([^.!?]+)'
        ]
        
        for pattern in factual_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            claims.extend([match.strip() for match in matches])
        
        # Look for temporal claims
        temporal_patterns = [
            r'(?:yesterday|today|tomorrow|last|next)\s+([^.!?]+)',
            r'(?:in|on|at)\s+(\d{4}|\d{1,2}/\d{1,2}/\d{2,4})\s+([^.!?]+)'
        ]
        
        for pattern in temporal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            claims.extend([' '.join(match).strip() if isinstance(match, tuple) else match.strip() 
                          for match in matches])
        
        return claims[:10]  # Limit to top 10 claims
    
    def analyze_image_content(self, image_path):
        """Analyze image content for consistency with text"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return None
                
            # Face detection and emotion analysis
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image)
            
            # Basic content analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Detect if image appears to be a screenshot or document
            is_screenshot = self._detect_screenshot(image)
            is_document = self._detect_document(image)
            
            return {
                'face_count': len(face_locations),
                'brightness': float(brightness),
                'contrast': float(contrast),
                'is_screenshot': is_screenshot,
                'is_document': is_document,
                'image_type': self._classify_image_type(image)
            }
        except Exception as e:
            print(f"Error analyzing image content: {e}")
            return None
    
    def analyze_audio_content(self, audio_path):
        """Analyze audio content for consistency with text"""
        try:
            # Transcribe audio
            transcript = transcribe_audio(audio_path)
            
            # Analyze emotion
            emotion_result = analyze_audio_emotion(audio_path)
            
            # Extract audio features
            y, sr = librosa.load(audio_path, sr=16000)
            duration = len(y) / sr
            
            # Analyze prosodic features
            pitch = librosa.yin(y, fmin=50, fmax=400)
            pitch = pitch[~np.isnan(pitch)]
            
            return {
                'transcript': transcript,
                'emotion': emotion_result.get('emotion') if emotion_result else None,
                'duration': float(duration),
                'pitch_mean': float(np.mean(pitch)) if len(pitch) > 0 else 0,
                'pitch_std': float(np.std(pitch)) if len(pitch) > 0 else 0,
                'speech_rate': len(transcript.split()) / duration if duration > 0 else 0
            }
        except Exception as e:
            print(f"Error analyzing audio content: {e}")
            return None
    
    def check_text_media_consistency(self, text, media_path, media_type):
        """Check consistency between text and media"""
        results = {
            'consistent': True,
            'confidence': 1.0,
            'issues': [],
            'details': {}
        }
        
        # Extract text information
        text_emotion = self.extract_text_emotion(text)
        text_claims = self.extract_text_content_claims(text)
        
        results['details']['text_emotion'] = text_emotion
        results['details']['text_claims'] = text_claims
        
        if media_type == 'image':
            media_analysis = self.analyze_image_content(media_path)
            if media_analysis:
                # Check emotion consistency
                if text_emotion and text_emotion != 'neutral':
                    # This is a simplified check - in practice, you'd need more sophisticated emotion detection
                    results['details']['media_analysis'] = media_analysis
                
                # Check content consistency
                if media_analysis.get('is_document') and any('photo' in claim.lower() or 'image' in claim.lower() 
                                                           for claim in text_claims):
                    results['issues'].append('text_claims_photo_but_document')
                    results['confidence'] *= 0.7
                
                if media_analysis.get('is_screenshot') and any('live' in claim.lower() or 'real' in claim.lower() 
                                                             for claim in text_claims):
                    results['issues'].append('text_claims_live_but_screenshot')
                    results['confidence'] *= 0.6
        
        elif media_type == 'audio':
            media_analysis = self.analyze_audio_content(media_path)
            if media_analysis:
                audio_emotion = media_analysis.get('emotion')
                transcript = media_analysis.get('transcript', '')
                
                results['details']['media_analysis'] = media_analysis
                
                # Check emotion consistency
                if text_emotion and audio_emotion and text_emotion != audio_emotion:
                    results['issues'].append('emotion_mismatch')
                    results['confidence'] *= 0.5
                
                # Check transcript consistency with text
                if transcript and text:
                    similarity = self._calculate_text_similarity(text, transcript)
                    if similarity < 0.3:
                        results['issues'].append('transcript_text_mismatch')
                        results['confidence'] *= 0.4
        
        # Determine overall consistency
        if results['issues']:
            results['consistent'] = False
            results['confidence'] = max(0.0, results['confidence'])
        
        return results
    
    def _detect_screenshot(self, image):
        """Detect if image appears to be a screenshot"""
        # Look for common screenshot indicators
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Check for uniform borders (common in screenshots)
        border_thickness = 5
        top_border = gray[:border_thickness, :]
        bottom_border = gray[-border_thickness:, :]
        left_border = gray[:, :border_thickness]
        right_border = gray[:, -border_thickness:]
        
        borders = [top_border, bottom_border, left_border, right_border]
        uniform_borders = sum(1 for border in borders if np.std(border) < 10)
        
        return uniform_borders >= 2
    
    def _detect_document(self, image):
        """Detect if image appears to be a document"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Look for text-like patterns
        # This is a simplified approach - in practice, you'd use OCR
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count rectangular contours (typical of text)
        rectangular_contours = 0
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if 0.1 < aspect_ratio < 10:  # Reasonable aspect ratio for text
                    rectangular_contours += 1
        
        return rectangular_contours > 20  # Threshold for document-like content
    
    def _classify_image_type(self, image):
        """Classify image type (photo, screenshot, document, etc.)"""
        if self._detect_screenshot(image):
            return 'screenshot'
        elif self._detect_document(image):
            return 'document'
        else:
            return 'photo'
    
    def _calculate_text_similarity(self, text1, text2):
        """Calculate similarity between two texts"""
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

def check_modality_consistency(text, media_path, media_type):
    """Main function to check consistency across modalities"""
    checker = ModalityConsistencyChecker()
    
    # Basic text-media consistency check
    consistency_result = checker.check_text_media_consistency(text, media_path, media_type)
    
    # Enhanced emotion manipulation check if we have both text and media
    emotion_result = None
    if text and media_path:
        # Extract emotions from different modalities
        text_emotion = checker.extract_text_emotion(text)
        
        if media_type == 'audio':
            audio_analysis = checker.analyze_audio_content(media_path)
            audio_emotion = audio_analysis.get('emotion') if audio_analysis else None
            emotion_result = flag_emotion_manipulation(
                text_emotion, audio_emotion, None, 
                media_path if media_type == 'audio' else None,
                media_path if media_type == 'image' else None
            )
        elif media_type == 'image':
            # For images, we'd need more sophisticated emotion detection
            emotion_result = {'manipulated': False, 'confidence': 0.0}
    
    return {
        'consistency_check': consistency_result,
        'emotion_manipulation': emotion_result,
        'overall_consistent': consistency_result.get('consistent', False) and 
                            (not emotion_result or not emotion_result.get('manipulated', False)),
        'confidence_score': min(
            consistency_result.get('confidence', 1.0),
            1.0 - emotion_result.get('confidence', 0.0) if emotion_result else 1.0
        )
    }
