import pytesseract, cv2, os
from PIL import Image, ExifTags
from datetime import datetime
from pathlib import Path

# Directory where OCR files will be saved
OCR_DIR = Path("ocr_texts")
OCR_DIR.mkdir(parents=True, exist_ok=True)

def detect_fake_document(img_path):
    """
    Analyze document for text tampering, metadata tampering, AI probability,
    save OCR text to a file, and produce explanation + summary.
    """
    result = {
        'score': 0.0,
        'ai_prob': 0.0,
        'text_tamper': False,
        'metadata_tamper': False,
        'ocr_text': '',
        'explanation': '',
        'summary': '',
        'metadata': {}
    }

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        result['explanation'] = 'Invalid document image.'
        result['summary'] = 'Document could not be analyzed.'
        return result

    # Convert to grayscale and OCR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    result['ocr_text'] = text[:2000]

    # Save OCR text to file
    filename_stem = Path(img_path).stem
    ocr_file_path = OCR_DIR / f"{filename_stem}_ocr.txt"
    with open(ocr_file_path, "w", encoding="utf-8") as f:
        f.write(text)

    # Text tampering detection (repeated lines)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    repeated_lines = len(lines) - len(set(lines))
    result['text_tamper'] = repeated_lines > len(lines) * 0.3 if lines else False

    # Metadata tampering detection
    metadata_tamper = False
    metadata = {}
    try:
        pil_img = Image.open(img_path)
        exif_data = pil_img._getexif()
        if exif_data:
            metadata = {ExifTags.TAGS.get(k): v for k, v in exif_data.items() if ExifTags.TAGS.get(k)}
            if any("Adobe" in str(v) for v in metadata.values()):
                metadata_tamper = True

            # Optional: detect very recent modification
            if os.path.exists(img_path):
                mod_time = os.path.getmtime(img_path)
                if datetime.now().timestamp() - mod_time < 600:
                    metadata_tamper = True
    except:
        pass
    result['metadata'] = metadata
    result['metadata_tamper'] = metadata_tamper

    # AI probability calculation
    ai_prob = 0
    if result['text_tamper']:
        ai_prob += 50  # text tampering weight
    if metadata_tamper:
        ai_prob += 30  # metadata tampering weight

    # Clamp AI probability between 0 and 99
    ai_prob = min(ai_prob, 99)
    result['ai_prob'] = round(ai_prob, 2)
    result['score'] = round(100 - ai_prob, 2)

    # Explanation
    explanation = []
    if result['text_tamper']:
        explanation.append("Text tampering detected: repeated patterns in OCR.")
    if metadata_tamper:
        explanation.append("Metadata tampering detected: suspicious EXIF or recent file modification.")
    if not explanation:
        explanation.append("Document appears authentic with consistent text and metadata.")
    result['explanation'] = " ".join(explanation)

    # Summary
    summary = []
    if result['text_tamper']:
        summary.append("Potential text tampering.")
    if metadata_tamper:
        summary.append("Metadata inconsistency detected.")
    if not summary:
        summary.append("Document is likely authentic.")
    result['summary'] = " ".join(summary)

    # Add OCR file path to result for download
    result['ocr_file_path'] = str(ocr_file_path)

    return result




# current
# import torch, cv2, pytesseract, os, json
# import numpy as np
# from PIL import Image, ExifTags
# from torchvision import transforms
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from src.fake_document_detection.model_loader import load_docauth_model

# MODEL_PATH = "models/docauth_cnn.pth"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # 1️⃣ Load the CNN model
# model = load_docauth_model(MODEL_PATH)
# model.to(DEVICE).eval()

# # 2️⃣ Preprocess for CNN input
# def preprocess_image(img_path):
#     img = Image.open(img_path).convert("RGB")
#     transform = transforms.Compose([
#         transforms.Resize((224,224)),
#         transforms.ToTensor()
#     ])
#     return transform(img).unsqueeze(0).to(DEVICE), np.array(img)

# # 3️⃣ OCR Consistency Check
# def check_text_consistency(img_path):
#     try:
#         text = pytesseract.image_to_string(Image.open(img_path))
#         lines = [l.strip() for l in text.splitlines() if l.strip()]
#         repeated = len(lines) - len(set(lines))
#         suspicious = repeated > len(lines)*0.3
#         return {"ocr_text": text, "text_tamper": suspicious}
#     except Exception as e:
#         return {"ocr_error": str(e)}

# # 4️⃣ Metadata Check
# def check_metadata(img_path):
#     try:
#         img = Image.open(img_path)
#         exif = {ExifTags.TAGS.get(k): v for k, v in img._getexif().items()} if img._getexif() else {}
#         suspicious = any("Adobe" in str(v) for v in exif.values())
#         return {"metadata": exif, "metadata_tamper": suspicious}
#     except Exception as e:
#         return {"metadata_error": str(e)}

# # 5️⃣ GradCAM Visualization
# def generate_heatmap(img_path):
#     try:
#         input_tensor, img_np = preprocess_image(img_path)
#         target_layer = model.layer4[-1] if hasattr(model, "layer4") else list(model.children())[-1]
#         cam = GradCAM(model=model, target_layers=[target_layer])
#         grayscale_cam = cam(input_tensor=input_tensor)[0, :]
#         visualization = show_cam_on_image(img_np.astype(np.float32)/255.0, grayscale_cam, use_rgb=True)
#         out_path = os.path.join("outputs", os.path.basename(img_path).split('.')[0] + "_doc_heatmap.jpg")
#         Image.fromarray(visualization).save(out_path)
#         return out_path
#     except Exception as e:
#         return str(e)

# # 6️⃣ Main Detection Function
# def detect_fake_document(file_path):
#     try:
#         # Model Prediction
#         tensor, _ = preprocess_image(file_path)
#         with torch.no_grad():
#             pred = torch.sigmoid(model(tensor)).item()
#         score = round(float(pred), 3)

#         # Heatmap
#         heatmap_path = generate_heatmap(file_path)

#         # Additional Checks
#         ocr_check = check_text_consistency(file_path)
#         meta_check = check_metadata(file_path)

#         explanation = (
#             f"AI model score: {score:.2f}. "
#             f"Text tampering: {'Yes' if ocr_check.get('text_tamper') else 'No'}. "
#             f"Metadata tampering: {'Yes' if meta_check.get('metadata_tamper') else 'No'}."
#         )

#         return {
#             "type": "document",
#             "ai_prob": score,
#             "score": score,
#             "heatmap": heatmap_path,
#             "ocr_text": ocr_check.get("ocr_text", ""),
#             "text_tamper": ocr_check.get("text_tamper", False),
#             "metadata": meta_check.get("metadata", {}),
#             "metadata_tamper": meta_check.get("metadata_tamper", False),
#             "explanation": explanation
#         }
#     except Exception as e:
#         return {"error": str(e)}
