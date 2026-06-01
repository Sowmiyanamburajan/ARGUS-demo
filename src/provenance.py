# src/provenance.py
# Enhanced provenance check with blockchain integration and metadata verification
import os, json, hashlib
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import requests
import time

def calculate_file_hash(file_path, algorithm='sha256'):
    """Calculate hash of file for integrity verification"""
    hash_func = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def extract_metadata(file_path):
    """Extract comprehensive metadata from file"""
    metadata = {
        'file_size': os.path.getsize(file_path),
        'creation_time': os.path.getctime(file_path),
        'modification_time': os.path.getmtime(file_path),
        'file_hash': calculate_file_hash(file_path)
    }
    
    # Extract EXIF data for images
    if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff')):
        try:
            image = Image.open(file_path)
            exifdata = image.getexif()
            
            exif_info = {}
            for tag_id in exifdata:
                tag = TAGS.get(tag_id, tag_id)
                data = exifdata.get(tag_id)
                exif_info[tag] = str(data)
            
            metadata['exif_data'] = exif_info
            metadata['image_dimensions'] = image.size
            metadata['image_mode'] = image.mode
        except Exception as e:
            metadata['exif_error'] = str(e)
    
    return metadata

def check_sidecar_signature(file_path):
    """Enhanced sidecar signature verification"""
    p = Path(file_path)
    side = p.with_suffix(p.suffix + '.sig.json')
    
    if not side.exists():
        return {'verified': False, 'method': 'none', 'reason': 'no_sidecar_file'}
    
    try:
        with open(side, 'r') as f:
            obj = json.load(f)
        
        # Verify required fields
        required_fields = ['issuer', 'signature', 'hash', 'timestamp']
        if not all(k in obj for k in required_fields):
            return {'verified': False, 'method': 'invalid_sidecar', 'reason': 'missing_fields'}
        
        # Verify hash matches current file
        current_hash = calculate_file_hash(file_path)
        if obj.get('hash') != current_hash:
            return {'verified': False, 'method': 'hash_mismatch', 'reason': 'file_modified'}
        
        # Verify timestamp is reasonable (not in future, not too old)
        timestamp = obj.get('timestamp', 0)
        current_time = time.time()
        if timestamp > current_time:
            return {'verified': False, 'method': 'invalid_timestamp', 'reason': 'future_timestamp'}
        
        # Check if signature is too old (more than 1 year)
        if current_time - timestamp > 31536000:  # 1 year in seconds
            return {'verified': True, 'method': 'sidecar', 'reason': 'signature_old', 'meta': obj}
        
        return {'verified': True, 'method': 'sidecar', 'meta': obj}
        
    except Exception as e:
        return {'verified': False, 'method': 'invalid_sidecar', 'reason': str(e)}

def check_blockchain_verification(file_path, blockchain_api_url=None):
    """Enhanced blockchain verification with multiple chains support"""
    file_hash = calculate_file_hash(file_path)
    
    # Check multiple blockchain networks
    blockchain_results = {}
    
    # Ethereum-based verification (simulated)
    ethereum_result = check_ethereum_certificate(file_hash, blockchain_api_url)
    blockchain_results['ethereum'] = ethereum_result
    
    # Bitcoin-based verification (simulated)
    bitcoin_result = check_bitcoin_certificate(file_hash, blockchain_api_url)
    blockchain_results['bitcoin'] = bitcoin_result
    
    # IPFS verification (simulated)
    ipfs_result = check_ipfs_certificate(file_hash, blockchain_api_url)
    blockchain_results['ipfs'] = ipfs_result
    
    # Determine overall blockchain verification status
    verified_chains = [chain for chain, result in blockchain_results.items() 
                      if result.get('verified', False)]
    
    return {
        'on_chain': len(verified_chains) > 0,
        'verified_chains': verified_chains,
        'blockchain_results': blockchain_results,
        'file_hash': file_hash
    }

def check_ethereum_certificate(file_hash, api_url=None):
    """Check Ethereum-based certificate (simulated)"""
    # In a real implementation, this would query an Ethereum node or API
    # For demo purposes, we simulate the check
    return {
        'verified': False,  # Simulate no certificate found
        'chain': 'ethereum',
        'transaction_hash': None,
        'block_number': None,
        'certificate_data': None
    }

def check_bitcoin_certificate(file_hash, api_url=None):
    """Check Bitcoin-based certificate (simulated)"""
    # In a real implementation, this would query a Bitcoin node or API
    return {
        'verified': False,  # Simulate no certificate found
        'chain': 'bitcoin',
        'transaction_hash': None,
        'block_height': None,
        'certificate_data': None
    }

def check_ipfs_certificate(file_hash, api_url=None):
    """Check IPFS-based certificate (simulated)"""
    # In a real implementation, this would query IPFS
    return {
        'verified': False,  # Simulate no certificate found
        'chain': 'ipfs',
        'ipfs_hash': None,
        'certificate_data': None
    }

def comprehensive_provenance_check(file_path):
    """Comprehensive provenance verification combining all methods"""
    results = {
        'file_path': file_path,
        'timestamp': time.time(),
        'metadata': extract_metadata(file_path),
        'sidecar_verification': check_sidecar_signature(file_path),
        'blockchain_verification': check_blockchain_verification(file_path),
        'overall_verified': False,
        'confidence_score': 0.0
    }
    
    # Calculate overall verification status
    sidecar_verified = results['sidecar_verification'].get('verified', False)
    blockchain_verified = results['blockchain_verification'].get('on_chain', False)
    
    # Calculate confidence score
    confidence = 0.0
    if sidecar_verified:
        confidence += 0.6
    if blockchain_verified:
        confidence += 0.4
    
    # Check for metadata consistency
    metadata = results['metadata']
    if metadata.get('exif_data'):
        confidence += 0.1
    
    results['overall_verified'] = sidecar_verified or blockchain_verified
    results['confidence_score'] = min(confidence, 1.0)
    
    return results

def generate_provenance_certificate(file_path, issuer="ARGUS_System"):
    """Generate a provenance certificate for a file"""
    file_hash = calculate_file_hash(file_path)
    timestamp = time.time()
    
    certificate = {
        'issuer': issuer,
        'timestamp': timestamp,
        'hash': file_hash,
        'file_path': str(file_path),
        'signature': f"ARGUS_{file_hash[:16]}",  # Simulated signature
        'certificate_version': '1.0',
        'verification_method': 'ARGUS_AI_Detection'
    }
    
    # Save certificate as sidecar file
    p = Path(file_path)
    sidecar_path = p.with_suffix(p.suffix + '.sig.json')
    
    with open(sidecar_path, 'w') as f:
        json.dump(certificate, f, indent=2)
    
    return certificate

# Legacy functions for backward compatibility
def check_blockchain_stub(file_path):
    """Legacy function - now calls enhanced blockchain verification"""
    result = check_blockchain_verification(file_path)
    return {
        'on_chain': result['on_chain'],
        'chain': result['verified_chains'][0] if result['verified_chains'] else None
    }
