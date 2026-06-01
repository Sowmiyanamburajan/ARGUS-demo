# src/explainable_output.py
# Explainable Output system with heatmaps and waveform anomaly detection
import cv2
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont
import io
import base64

class ExplainableOutputGenerator:
    def __init__(self):
        self.heatmap_colormap = 'hot'
        self.anomaly_threshold = 0.7
        
    def generate_image_heatmap(self, image_path, detection_results, output_path=None):
        """Generate enhanced explainable heatmap for image analysis"""
        try:
            # Load original image
            image = cv2.imread(image_path)
            if image is None:
                return None
                
            # Create multiple heatmap layers for different analysis aspects
            height, width = image.shape[:2]
            
            # Layer 1: Face detection heatmap
            face_heatmap = np.zeros((height, width), dtype=np.float32)
            if 'faces' in detection_results and detection_results['faces']:
                for face in detection_results['faces']:
                    if 'bbox' in face:
                        x, y, w, h = face['bbox']
                        confidence = face.get('confidence', 0.5)
                        face_heatmap[y:y+h, x:x+w] += confidence
                    elif 'crop' in face:
                        # Estimate face region
                        center_x, center_y = width//2, height//2
                        bbox_size = min(width, height) // 4
                        x1 = max(0, center_x - bbox_size//2)
                        y1 = max(0, center_y - bbox_size//2)
                        x2 = min(width, center_x + bbox_size//2)
                        y2 = min(height, center_y + bbox_size//2)
                        face_heatmap[y1:y2, x1:x2] += 0.7
            
            # Layer 2: Edge anomaly heatmap
            edge_heatmap = self._generate_edge_anomaly_heatmap(image)
            
            # Layer 3: Frequency domain heatmap
            freq_heatmap = self._generate_frequency_heatmap(image)
            
            # Layer 4: Color consistency heatmap
            color_heatmap = self._generate_color_consistency_heatmap(image)
            
            # Combine all heatmaps
            combined_heatmap = (
                face_heatmap * 0.3 +
                edge_heatmap * 0.25 +
                freq_heatmap * 0.25 +
                color_heatmap * 0.2
            )
            
            # Normalize combined heatmap
            if combined_heatmap.max() > 0:
                combined_heatmap = combined_heatmap / combined_heatmap.max()
            
            # Create detailed visualizations
            visualizations = self._create_detailed_visualizations(
                image, combined_heatmap, detection_results
            )
            
            # Save main heatmap
            if output_path:
                cv2.imwrite(output_path, visualizations['main_heatmap'])
            
            return {
                'heatmap_image': output_path,
                'heatmap_data': combined_heatmap.tolist(),
                'max_intensity': float(combined_heatmap.max()),
                'anomaly_regions': self._identify_anomaly_regions(combined_heatmap),
                'detailed_analysis': {
                    'face_regions': face_heatmap.tolist(),
                    'edge_anomalies': edge_heatmap.tolist(),
                    'frequency_artifacts': freq_heatmap.tolist(),
                    'color_inconsistencies': color_heatmap.tolist()
                },
                'visualizations': visualizations,
                'confidence_scores': self._calculate_confidence_scores(combined_heatmap)
            }
            
        except Exception as e:
            print(f"Error generating image heatmap: {e}")
            return None
    
    def generate_audio_waveform_analysis(self, audio_path, detection_results, output_path=None):
        """Generate explainable waveform analysis for audio"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Extract features for analysis
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            zero_crossings = librosa.feature.zero_crossing_rate(y)[0]
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Create time axis
            times = librosa.frames_to_time(np.arange(len(spectral_centroids)), sr=sr)
            
            # Detect anomalies in audio features
            anomalies = self._detect_audio_anomalies(y, sr, spectral_centroids, zero_crossings)
            
            # Generate waveform visualization
            fig, axes = plt.subplots(4, 1, figsize=(12, 10))
            
            # Original waveform
            axes[0].plot(np.linspace(0, len(y)/sr, len(y)), y, alpha=0.7, color='blue')
            axes[0].set_title('Audio Waveform')
            axes[0].set_ylabel('Amplitude')
            
            # Spectral centroid
            axes[1].plot(times, spectral_centroids, color='green')
            axes[1].set_title('Spectral Centroid (Brightness)')
            axes[1].set_ylabel('Frequency (Hz)')
            
            # Zero crossing rate
            axes[2].plot(times, zero_crossings, color='red')
            axes[2].set_title('Zero Crossing Rate (Voiced/Unvoiced)')
            axes[2].set_ylabel('Rate')
            
            # MFCC features (first 3 components)
            mfcc_times = librosa.frames_to_time(np.arange(mfccs.shape[1]), sr=sr)
            for i in range(min(3, mfccs.shape[0])):
                axes[3].plot(mfcc_times, mfccs[i], label=f'MFCC {i+1}', alpha=0.7)
            axes[3].set_title('MFCC Features')
            axes[3].set_ylabel('Coefficient')
            axes[3].legend()
            
            # Highlight anomalies
            for ax in axes:
                for anomaly in anomalies:
                    ax.axvspan(anomaly['start_time'], anomaly['end_time'], 
                             alpha=0.3, color='red', label='Anomaly' if anomaly == anomalies[0] else "")
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
            
            return {
                'waveform_analysis': output_path,
                'anomalies': anomalies,
                'tempo': float(tempo),
                'spectral_features': {
                    'centroid_mean': float(np.mean(spectral_centroids)),
                    'centroid_std': float(np.std(spectral_centroids)),
                    'zcr_mean': float(np.mean(zero_crossings)),
                    'zcr_std': float(np.std(zero_crossings))
                }
            }
            
        except Exception as e:
            print(f"Error generating audio waveform analysis: {e}")
            return None
    
    def generate_video_temporal_analysis(self, video_path, detection_results, output_path=None):
        """Generate explainable temporal analysis for video"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Extract frame-level features
            frame_features = []
            timestamps = []
            
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Analyze frame
                features = self._analyze_video_frame(frame)
                features['frame_idx'] = frame_idx
                features['timestamp'] = frame_idx / fps
                
                frame_features.append(features)
                timestamps.append(frame_idx / fps)
                
                frame_idx += 1
                
                # Limit analysis to prevent memory issues
                if frame_idx > 1000:
                    break
            
            cap.release()
            
            # Generate temporal analysis visualization
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            
            # Brightness over time
            brightness_values = [f['brightness'] for f in frame_features]
            axes[0].plot(timestamps, brightness_values, color='blue')
            axes[0].set_title('Brightness Over Time')
            axes[0].set_ylabel('Brightness')
            
            # Motion over time
            motion_values = [f['motion'] for f in frame_features]
            axes[1].plot(timestamps, motion_values, color='green')
            axes[1].set_title('Motion Over Time')
            axes[1].set_ylabel('Motion Intensity')
            
            # Face detection confidence over time
            face_confidences = [f.get('face_confidence', 0) for f in frame_features]
            axes[2].plot(timestamps, face_confidences, color='red')
            axes[2].set_title('Face Detection Confidence Over Time')
            axes[2].set_ylabel('Confidence')
            axes[2].set_xlabel('Time (seconds)')
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
            
            return {
                'temporal_analysis': output_path,
                'frame_features': frame_features,
                'temporal_anomalies': self._detect_temporal_anomalies(frame_features),
                'video_stats': {
                    'fps': fps,
                    'frame_count': frame_count,
                    'duration': frame_count / fps if fps > 0 else 0
                }
            }
            
        except Exception as e:
            print(f"Error generating video temporal analysis: {e}")
            return None
    
    def _add_annotations(self, image, detection_results):
        """Add text annotations to image"""
        try:
            # Convert to PIL for text rendering
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image_pil)
            
            # Add detection summary
            summary_text = f"Detections: {len(detection_results.get('faces', []))}"
            if 'confidence' in detection_results:
                summary_text += f" | Confidence: {detection_results['confidence']:.2f}"
            
            # Try to use a font, fallback to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # Draw text with background
            bbox = draw.textbbox((10, 10), summary_text, font=font)
            draw.rectangle(bbox, fill=(0, 0, 0, 128))
            draw.text((10, 10), summary_text, fill=(255, 255, 255), font=font)
            
            # Convert back to OpenCV format
            return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error adding annotations: {e}")
            return image
    
    def _identify_anomaly_regions(self, heatmap):
        """Identify regions with high anomaly scores"""
        threshold = np.percentile(heatmap, 90)  # Top 10% of values
        anomaly_mask = heatmap > threshold
        
        # Find connected components
        contours, _ = cv2.findContours(
            anomaly_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        anomaly_regions = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                anomaly_regions.append({
                    'bbox': [x, y, w, h],
                    'area': cv2.contourArea(contour),
                    'intensity': float(np.mean(heatmap[y:y+h, x:x+w]))
                })
        
        return anomaly_regions
    
    def _generate_edge_anomaly_heatmap(self, image):
        """Generate heatmap highlighting edge anomalies"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply multiple edge detection methods
        edges_canny = cv2.Canny(gray, 50, 150)
        edges_sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        edges_laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Combine edge information
        edge_combined = (
            edges_canny.astype(np.float32) / 255.0 +
            np.abs(edges_sobel) / np.max(np.abs(edges_sobel)) +
            np.abs(edges_laplacian) / np.max(np.abs(edges_laplacian))
        ) / 3.0
        
        # Detect anomalies (unusual edge patterns)
        edge_anomaly = np.abs(edge_combined - np.mean(edge_combined))
        return edge_anomaly / np.max(edge_anomaly) if np.max(edge_anomaly) > 0 else edge_anomaly
    
    def _generate_frequency_heatmap(self, image):
        """Generate heatmap highlighting frequency domain artifacts"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Detect high-frequency artifacts
        height, width = magnitude_spectrum.shape
        center_y, center_x = height // 2, width // 2
        
        # Create frequency mask
        y, x = np.ogrid[:height, :width]
        mask = (x - center_x)**2 + (y - center_y)**2 > (min(height, width) // 4)**2
        
        # Extract high-frequency components
        high_freq = magnitude_spectrum * mask
        high_freq_normalized = high_freq / np.max(high_freq) if np.max(high_freq) > 0 else high_freq
        
        # Convert back to spatial domain
        freq_heatmap = np.fft.ifft2(np.fft.ifftshift(high_freq_normalized))
        return np.abs(freq_heatmap)
    
    def _generate_color_consistency_heatmap(self, image):
        """Generate heatmap highlighting color inconsistencies"""
        # Convert to different color spaces
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate color variance in local regions
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        
        # Calculate local color variance
        l_var = cv2.filter2D(lab[:,:,0].astype(np.float32), -1, kernel)
        a_var = cv2.filter2D(lab[:,:,1].astype(np.float32), -1, kernel)
        b_var = cv2.filter2D(lab[:,:,2].astype(np.float32), -1, kernel)
        
        # Combine color variances
        color_variance = (l_var + a_var + b_var) / 3.0
        
        # Normalize
        return color_variance / np.max(color_variance) if np.max(color_variance) > 0 else color_variance
    
    def _create_detailed_visualizations(self, image, heatmap, detection_results):
        """Create detailed visualizations for explainable output"""
        visualizations = {}
        
        # Main heatmap overlay
        heatmap_colored = plt.cm.get_cmap(self.heatmap_colormap)(heatmap)
        heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
        alpha = 0.6
        main_heatmap = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
        visualizations['main_heatmap'] = main_heatmap
        
        # Edge detection visualization
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        visualizations['edge_detection'] = edge_vis
        
        # Frequency domain visualization
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        freq_vis = ((magnitude_spectrum - np.min(magnitude_spectrum)) / 
                   (np.max(magnitude_spectrum) - np.min(magnitude_spectrum)) * 255).astype(np.uint8)
        freq_vis = cv2.cvtColor(freq_vis, cv2.COLOR_GRAY2BGR)
        visualizations['frequency_domain'] = freq_vis
        
        # Anomaly regions highlighted
        anomaly_mask = heatmap > self.anomaly_threshold
        anomaly_vis = image.copy()
        anomaly_vis[anomaly_mask] = [0, 0, 255]  # Red for anomalies
        visualizations['anomaly_regions'] = anomaly_vis
        
        return visualizations
    
    def _calculate_confidence_scores(self, heatmap):
        """Calculate confidence scores for different regions"""
        scores = {
            'overall_confidence': float(np.mean(heatmap)),
            'max_anomaly_score': float(np.max(heatmap)),
            'anomaly_coverage': float(np.sum(heatmap > self.anomaly_threshold) / heatmap.size),
            'spatial_distribution': float(np.std(heatmap))
        }
        return scores
    
    def _detect_audio_anomalies(self, y, sr, spectral_centroids, zero_crossings):
        """Detect anomalies in audio features"""
        anomalies = []
        
        # Detect sudden changes in spectral centroid
        centroid_diff = np.abs(np.diff(spectral_centroids))
        centroid_threshold = np.mean(centroid_diff) + 2 * np.std(centroid_diff)
        anomaly_frames = np.where(centroid_diff > centroid_threshold)[0]
        
        for frame in anomaly_frames:
            start_time = librosa.frames_to_time(frame, sr=sr)
            end_time = librosa.frames_to_time(frame + 1, sr=sr)
            anomalies.append({
                'start_time': start_time,
                'end_time': end_time,
                'type': 'spectral_anomaly',
                'severity': float(centroid_diff[frame] / centroid_threshold)
            })
        
        return anomalies
    
    def _analyze_video_frame(self, frame):
        """Analyze a single video frame for features"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate brightness
        brightness = np.mean(gray)
        
        # Calculate motion (simplified - would need previous frame for real motion)
        motion = 0.0  # Placeholder
        
        # Face detection (simplified)
        face_confidence = 0.0  # Placeholder
        
        return {
            'brightness': float(brightness),
            'motion': float(motion),
            'face_confidence': float(face_confidence)
        }
    
    def _detect_temporal_anomalies(self, frame_features):
        """Detect temporal anomalies in video"""
        anomalies = []
        
        # Detect sudden brightness changes
        brightness_values = [f['brightness'] for f in frame_features]
        if len(brightness_values) > 1:
            brightness_diff = np.abs(np.diff(brightness_values))
            threshold = np.mean(brightness_diff) + 2 * np.std(brightness_diff)
            
            for i, diff in enumerate(brightness_diff):
                if diff > threshold:
                    anomalies.append({
                        'frame_idx': i,
                        'type': 'brightness_anomaly',
                        'severity': float(diff / threshold)
                    })
        
        return anomalies

def generate_explainable_output(file_path, file_type, detection_results, output_dir):
    """Main function to generate explainable output for any file type"""
    generator = ExplainableOutputGenerator()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results = {
        'file_path': file_path,
        'file_type': file_type,
        'explanations': {}
    }
    
    if file_type == 'image':
        heatmap_result = generator.generate_image_heatmap(
            file_path, detection_results, 
            str(output_path / f"heatmap_{Path(file_path).stem}.png")
        )
        if heatmap_result:
            results['explanations']['heatmap'] = heatmap_result
    
    elif file_type == 'audio':
        waveform_result = generator.generate_audio_waveform_analysis(
            file_path, detection_results,
            str(output_path / f"waveform_{Path(file_path).stem}.png")
        )
        if waveform_result:
            results['explanations']['waveform'] = waveform_result
    
    elif file_type == 'video':
        temporal_result = generator.generate_video_temporal_analysis(
            file_path, detection_results,
            str(output_path / f"temporal_{Path(file_path).stem}.png")
        )
        if temporal_result:
            results['explanations']['temporal'] = temporal_result
    
    return results
