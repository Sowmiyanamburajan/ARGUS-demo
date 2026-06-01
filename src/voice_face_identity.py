# src/voice_face_identity.py
# Cross-Check Voice & Face Identity verification system
import cv2
import numpy as np
import librosa
from pathlib import Path
import face_recognition
import speech_recognition as sr
from scipy.spatial.distance import cosine
import json
import os

class VoiceFaceIdentityChecker:
    def __init__(self):
        self.known_voices = {}  # Store voice embeddings
        self.known_faces = {}   # Store face embeddings
        
    def extract_voice_embedding(self, audio_path):
        """Extract voice embedding from audio file"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Extract spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            
            # Combine features
            features = np.concatenate([
                np.mean(mfccs, axis=1),
                np.mean(spectral_centroids),
                np.mean(spectral_rolloff),
                np.mean(zero_crossing_rate)
            ])
            
            return features
        except Exception as e:
            print(f"Error extracting voice embedding: {e}")
            return None
    
    def extract_face_embedding(self, image_path):
        """Extract face embedding from image"""
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Find face locations
            face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                return None
                
            # Get face encodings
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            if not face_encodings:
                return None
                
            return face_encodings[0]  # Return first face encoding
        except Exception as e:
            print(f"Error extracting face embedding: {e}")
            return None
    
    def register_identity(self, person_id, audio_path, image_path):
        """Register a person's voice and face"""
        voice_embedding = self.extract_voice_embedding(audio_path)
        face_embedding = self.extract_face_embedding(image_path)
        
        if voice_embedding is not None and face_embedding is not None:
            self.known_voices[person_id] = voice_embedding
            self.known_faces[person_id] = face_embedding
            return True
        return False
    
    def verify_voice_face_match(self, audio_path, image_path, threshold=0.6):
        """Verify if voice and face belong to the same person"""
        voice_embedding = self.extract_voice_embedding(audio_path)
        face_embedding = self.extract_face_embedding(image_path)
        
        if voice_embedding is None or face_embedding is None:
            return {
                'match': False,
                'confidence': 0.0,
                'reason': 'insufficient_data'
            }
        
        # Find best matching voice
        best_voice_match = None
        best_voice_score = 0.0
        
        for person_id, known_voice in self.known_voices.items():
            # Calculate cosine similarity
            similarity = 1 - cosine(voice_embedding, known_voice)
            if similarity > best_voice_score:
                best_voice_score = similarity
                best_voice_match = person_id
        
        # Find best matching face
        best_face_match = None
        best_face_score = 0.0
        
        for person_id, known_face in self.known_faces.items():
            # Calculate face distance (lower is better)
            distance = face_recognition.face_distance([known_face], face_embedding)[0]
            similarity = 1 - distance  # Convert distance to similarity
            if similarity > best_face_score:
                best_face_score = similarity
                best_face_match = person_id
        
        # Check if voice and face match the same person
        voice_face_match = (best_voice_match == best_face_match and 
                           best_voice_match is not None and 
                           best_face_match is not None)
        
        # Calculate overall confidence
        overall_confidence = (best_voice_score + best_face_score) / 2
        
        return {
            'match': voice_face_match and overall_confidence > threshold,
            'confidence': overall_confidence,
            'voice_match': best_voice_match,
            'face_match': best_face_match,
            'voice_score': best_voice_score,
            'face_score': best_face_score,
            'reason': 'success' if voice_face_match else 'mismatch'
        }
    
    def cross_verify_with_known_identities(self, audio_path, image_path):
        """Cross-verify against known identities in database"""
        results = []
        
        for person_id in self.known_voices.keys():
            if person_id in self.known_faces:
                voice_similarity = 1 - cosine(
                    self.extract_voice_embedding(audio_path), 
                    self.known_voices[person_id]
                )
                face_similarity = 1 - face_recognition.face_distance(
                    [self.known_faces[person_id]], 
                    self.extract_face_embedding(image_path)
                )[0]
                
                results.append({
                    'person_id': person_id,
                    'voice_similarity': voice_similarity,
                    'face_similarity': face_similarity,
                    'combined_score': (voice_similarity + face_similarity) / 2
                })
        
        return sorted(results, key=lambda x: x['combined_score'], reverse=True)

def check_voice_face_identity(audio_path, image_path, known_identities=None):
    """Main function to check voice-face identity consistency"""
    checker = VoiceFaceIdentityChecker()
    
    # Load known identities if provided
    if known_identities:
        for person_id, data in known_identities.items():
            if 'audio' in data and 'image' in data:
                checker.register_identity(person_id, data['audio'], data['image'])
    
    # Perform verification
    result = checker.verify_voice_face_match(audio_path, image_path)
    
    # Add cross-verification results
    cross_verify_results = checker.cross_verify_with_known_identities(audio_path, image_path)
    result['cross_verification'] = cross_verify_results[:3]  # Top 3 matches
    
    return result
