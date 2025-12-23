from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
CORS(app)

# Load model
print("Loading YOLOv8 model...")
model = YOLO('best.pt')
print("Model loaded successfully!")

@app.route('/api/detect', methods=['POST', 'OPTIONS'])
def detect():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        image_data = data.get('image', '')
        
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
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
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
