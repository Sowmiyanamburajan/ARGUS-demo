# src/behavioral.py
# Enhanced Behavioral Biometrics for Authenticity Detection
import sqlite3, os, datetime, json, numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import statistics
from scipy import stats
from scipy.spatial.distance import euclidean
import librosa
import cv2

def _conn():
    c = sqlite3.connect(DB, check_same_thread=False)
    c.execute("CREATE TABLE IF NOT EXISTS profiles (id TEXT PRIMARY KEY, user TEXT, typing_stats TEXT, mouse_stats TEXT, created_at TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS samples (id INTEGER PRIMARY KEY AUTOINCREMENT, profile_id TEXT, typing_stats TEXT, mouse_stats TEXT, ts TEXT)")
    c.commit()
    return c

def save_profile(profile_id, user, typing_stats, mouse_stats):
    conn = _conn()
    conn.execute("INSERT OR REPLACE INTO profiles (id,user,typing_stats,mouse_stats,created_at) VALUES (?,?,?,?,?)", 
                 (profile_id, user, json.dumps(typing_stats), json.dumps(mouse_stats), datetime.datetime.utcnow().isoformat()))
    conn.commit(); conn.close()
    return True

def save_sample(profile_id, typing_stats, mouse_stats):
    conn = _conn()
    conn.execute("INSERT INTO samples (profile_id,typing_stats,mouse_stats,ts) VALUES (?,?,?,?)", 
                 (profile_id, json.dumps(typing_stats), json.dumps(mouse_stats), datetime.datetime.utcnow().isoformat()))
    conn.commit(); conn.close()
    return True

class BehavioralBiometricsAnalyzer:
    def __init__(self):
        self.db_path = "outputs/behavior.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize database with enhanced schema"""
        os.makedirs("outputs", exist_ok=True)
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                id TEXT PRIMARY KEY, 
                user TEXT, 
                typing_stats TEXT, 
                mouse_stats TEXT, 
                voice_stats TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                profile_id TEXT, 
                typing_stats TEXT, 
                mouse_stats TEXT, 
                voice_stats TEXT,
                ts TEXT,
                authenticity_score REAL
            )
        """)
        conn.commit()
        conn.close()
    
    def analyze_typing_patterns(self, typing_stats: Dict) -> Dict:
        """Analyze typing patterns for authenticity indicators"""
        if not typing_stats:
            return {'authentic': False, 'confidence': 0.0, 'reason': 'no_data'}
        
        # Extract key metrics
        inter_key_intervals = typing_stats.get('intervals', [])
        key_press_durations = typing_stats.get('durations', [])
        typing_rhythm = typing_stats.get('rhythm', [])
        
        if not inter_key_intervals:
            return {'authentic': False, 'confidence': 0.0, 'reason': 'insufficient_data'}
        
        # Calculate statistical features
        features = {
            'mean_iki': np.mean(inter_key_intervals),
            'std_iki': np.std(inter_key_intervals),
            'median_iki': np.median(inter_key_intervals),
            'cv_iki': np.std(inter_key_intervals) / np.mean(inter_key_intervals) if np.mean(inter_key_intervals) > 0 else 0,
            'skewness_iki': stats.skew(inter_key_intervals),
            'kurtosis_iki': stats.kurtosis(inter_key_intervals)
        }
        
        # Add duration features if available
        if key_press_durations:
            features.update({
                'mean_duration': np.mean(key_press_durations),
                'std_duration': np.std(key_press_durations),
                'cv_duration': np.std(key_press_durations) / np.mean(key_press_durations) if np.mean(key_press_durations) > 0 else 0
            })
        
        # Authenticity indicators
        authenticity_indicators = []
        confidence = 1.0
        
        # Check for human-like variability
        if features['cv_iki'] > 0.3:  # High coefficient of variation indicates human typing
            authenticity_indicators.append('natural_variability')
            confidence += 0.2
        else:
            confidence -= 0.3
        
        # Check for realistic timing patterns
        if 50 < features['mean_iki'] < 500:  # Realistic inter-key interval
            authenticity_indicators.append('realistic_timing')
            confidence += 0.1
        else:
            confidence -= 0.2
        
        # Check for natural rhythm patterns
        if typing_rhythm and len(typing_rhythm) > 5:
            rhythm_variance = np.var(typing_rhythm)
            if 0.1 < rhythm_variance < 2.0:  # Natural rhythm variation
                authenticity_indicators.append('natural_rhythm')
                confidence += 0.15
        
        # Check for bot-like patterns (too regular)
        if features['std_iki'] < 10:  # Very low variation suggests automation
            authenticity_indicators.append('suspicious_regularity')
            confidence -= 0.4
        
        return {
            'authentic': confidence > 0.5,
            'confidence': max(0.0, min(1.0, confidence)),
            'features': features,
            'indicators': authenticity_indicators,
            'reason': 'analysis_complete'
        }
    
    def analyze_mouse_patterns(self, mouse_stats: Dict) -> Dict:
        """Analyze mouse movement patterns for authenticity"""
        if not mouse_stats:
            return {'authentic': False, 'confidence': 0.0, 'reason': 'no_data'}
        
        movements = mouse_stats.get('movements', [])
        clicks = mouse_stats.get('clicks', [])
        
        if not movements:
            return {'authentic': False, 'confidence': 0.0, 'reason': 'insufficient_data'}
        
        # Calculate movement features
        if len(movements) > 1:
            distances = []
            angles = []
            velocities = []
            
            for i in range(1, len(movements)):
                prev_x, prev_y, prev_t = movements[i-1]
                curr_x, curr_y, curr_t = movements[i]
                
                dist = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                time_diff = curr_t - prev_t
                
                if time_diff > 0:
                    velocity = dist / time_diff
                    distances.append(dist)
                    velocities.append(velocity)
                    
                    if i > 1:
                        # Calculate angle between consecutive movements
                        prev_prev_x, prev_prev_y, _ = movements[i-2]
                        angle = np.arctan2(curr_y - prev_y, curr_x - prev_x) - np.arctan2(prev_y - prev_prev_y, prev_x - prev_prev_x)
                        angles.append(angle)
            
            features = {
                'mean_distance': np.mean(distances),
                'std_distance': np.std(distances),
                'mean_velocity': np.mean(velocities),
                'std_velocity': np.std(velocities),
                'angle_variance': np.var(angles) if angles else 0
            }
        else:
            features = {'mean_distance': 0, 'std_distance': 0, 'mean_velocity': 0, 'std_velocity': 0, 'angle_variance': 0}
        
        # Authenticity analysis
        authenticity_indicators = []
        confidence = 1.0
        
        # Check for natural movement patterns
        if features['std_velocity'] > 10:  # Natural velocity variation
            authenticity_indicators.append('natural_velocity_variation')
            confidence += 0.2
        
        # Check for realistic movement distances
        if 5 < features['mean_distance'] < 200:  # Realistic movement range
            authenticity_indicators.append('realistic_movement_range')
            confidence += 0.1
        
        # Check for human-like angle variation
        if features['angle_variance'] > 0.5:  # Natural direction changes
            authenticity_indicators.append('natural_direction_changes')
            confidence += 0.15
        
        # Check for bot-like patterns
        if features['std_velocity'] < 1:  # Too regular velocity
            authenticity_indicators.append('suspicious_regularity')
            confidence -= 0.3
        
        return {
            'authentic': confidence > 0.5,
            'confidence': max(0.0, min(1.0, confidence)),
            'features': features,
            'indicators': authenticity_indicators,
            'reason': 'analysis_complete'
        }
    
    def analyze_voice_biometrics(self, audio_path: str) -> Dict:
        """Analyze voice biometrics for authenticity"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Extract voice features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            
            # Calculate prosodic features
            pitch = librosa.yin(y, fmin=50, fmax=400)
            pitch = pitch[~np.isnan(pitch)]
            
            features = {
                'mfcc_mean': np.mean(mfccs),
                'mfcc_std': np.std(mfccs),
                'spectral_centroid_mean': np.mean(spectral_centroids),
                'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
                'pitch_mean': np.mean(pitch) if len(pitch) > 0 else 0,
                'pitch_std': np.std(pitch) if len(pitch) > 0 else 0,
                'pitch_range': np.max(pitch) - np.min(pitch) if len(pitch) > 0 else 0
            }
            
            # Authenticity indicators
            authenticity_indicators = []
            confidence = 1.0
            
            # Check for natural pitch variation
            if features['pitch_std'] > 20:  # Natural pitch variation
                authenticity_indicators.append('natural_pitch_variation')
                confidence += 0.2
            
            # Check for realistic pitch range
            if 80 < features['pitch_mean'] < 300:  # Human voice range
                authenticity_indicators.append('realistic_pitch_range')
                confidence += 0.1
            
            # Check for natural spectral characteristics
            if 0.1 < features['zero_crossing_rate_mean'] < 0.3:  # Natural speech
                authenticity_indicators.append('natural_speech_characteristics')
                confidence += 0.15
            
            return {
                'authentic': confidence > 0.5,
                'confidence': max(0.0, min(1.0, confidence)),
                'features': features,
                'indicators': authenticity_indicators,
                'reason': 'analysis_complete'
            }
        except Exception as e:
            return {'authentic': False, 'confidence': 0.0, 'reason': f'error: {str(e)}'}
    
    def comprehensive_authenticity_check(self, profile_id: str, typing_stats: Dict = None, 
                                       mouse_stats: Dict = None, audio_path: str = None) -> Dict:
        """Perform comprehensive authenticity check across all available modalities"""
        results = {
            'overall_authentic': True,
            'overall_confidence': 1.0,
            'modality_results': {},
            'indicators': [],
            'recommendations': []
        }
        
        # Analyze typing patterns
        if typing_stats:
            typing_result = self.analyze_typing_patterns(typing_stats)
            results['modality_results']['typing'] = typing_result
            if not typing_result['authentic']:
                results['overall_authentic'] = False
                results['indicators'].extend(typing_result['indicators'])
        
        # Analyze mouse patterns
        if mouse_stats:
            mouse_result = self.analyze_mouse_patterns(mouse_stats)
            results['modality_results']['mouse'] = mouse_result
            if not mouse_result['authentic']:
                results['overall_authentic'] = False
                results['indicators'].extend(mouse_result['indicators'])
        
        # Analyze voice patterns
        if audio_path and os.path.exists(audio_path):
            voice_result = self.analyze_voice_biometrics(audio_path)
            results['modality_results']['voice'] = voice_result
            if not voice_result['authentic']:
                results['overall_authentic'] = False
                results['indicators'].extend(voice_result['indicators'])
        
        # Calculate overall confidence
        modality_confidences = [result['confidence'] for result in results['modality_results'].values()]
        if modality_confidences:
            results['overall_confidence'] = np.mean(modality_confidences)
        
        # Generate recommendations
        if not results['overall_authentic']:
            results['recommendations'].append('Consider additional verification methods')
            if 'suspicious_regularity' in results['indicators']:
                results['recommendations'].append('Detected automated behavior patterns')
        
        return results

# Legacy functions for backward compatibility
DB = "outputs/behavior.db"
os.makedirs("outputs", exist_ok=True)

def _conn():
    c = sqlite3.connect(DB, check_same_thread=False)
    c.execute("CREATE TABLE IF NOT EXISTS profiles (id TEXT PRIMARY KEY, user TEXT, typing_stats TEXT, mouse_stats TEXT, created_at TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS samples (id INTEGER PRIMARY KEY AUTOINCREMENT, profile_id TEXT, typing_stats TEXT, mouse_stats TEXT, ts TEXT)")
    c.commit()
    return c

def save_profile(profile_id, user, typing_stats, mouse_stats):
    conn = _conn()
    conn.execute("INSERT OR REPLACE INTO profiles (id,user,typing_stats,mouse_stats,created_at) VALUES (?,?,?,?,?)", 
                 (profile_id, user, json.dumps(typing_stats), json.dumps(mouse_stats), datetime.datetime.utcnow().isoformat()))
    conn.commit(); conn.close()
    return True

def save_sample(profile_id, typing_stats, mouse_stats):
    conn = _conn()
    conn.execute("INSERT INTO samples (profile_id,typing_stats,mouse_stats,ts) VALUES (?,?,?,?)", 
                 (profile_id, json.dumps(typing_stats), json.dumps(mouse_stats), datetime.datetime.utcnow().isoformat()))
    conn.commit(); conn.close()
    return True

def compare_to_profile(profile_id, typing_stats, mouse_stats):
    """Enhanced comparison with behavioral biometrics analysis"""
    analyzer = BehavioralBiometricsAnalyzer()
    
    # Get comprehensive analysis
    comprehensive_result = analyzer.comprehensive_authenticity_check(
        profile_id, typing_stats, mouse_stats
    )
    
    # Legacy format for backward compatibility
    return {
        'match': comprehensive_result['overall_authentic'],
        'confidence': comprehensive_result['overall_confidence'],
        'detailed_analysis': comprehensive_result
    }
