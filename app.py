from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import base64
import os
import sys

app = Flask(__name__)
# Allow specific origins or all (*)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global CORS header injection to be absolutely sure
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    return response

# Load model
try:
    print("Loading YOLOv8 model...", file=sys.stderr)
    model = YOLO('best.pt')
    print("Model loaded successfully!", file=sys.stderr)
except Exception as e:
    print(f"Error loading model: {e}", file=sys.stderr)
    model = None

@app.route('/api/detect', methods=['POST'])
def detect():
    # Note: No need to handle OPTIONS here, Flask protocols it automatically or CORS handles it
    
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data received'}), 400
            
        image_data = data.get('image', '')
        
        if not image_data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400
        
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return jsonify({'success': False, 'error': f'Invalid image data: {str(e)}'}), 400
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Run inference
        results = model(image, conf=0.2, iou=0.45, verbose=False)
        
        # Process results
        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id] if class_id < len(model.names) else 'caries'
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                })
        
        # Calculate statistics
        statistics = {
            'totalDetections': len(detections),
            'averageConfidence': 0,
            'maxConfidence': 0
        }
        
        if len(detections) > 0:
            confidences = [d['confidence'] for d in detections]
            statistics['averageConfidence'] = sum(confidences) / len(confidences)
            statistics['maxConfidence'] = max(confidences)
        
        return jsonify({
            'success': True,
            'imageSize': {'width': image.width, 'height': image.height},
            'detections': detections,
            'statistics': statistics
        })
        
    except Exception as e:
        print(f"Error during detection: {e}", file=sys.stderr)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
