# 🦷 Dental Caries Detection API

Flask API for dental caries detection using YOLOv8.

## Deployment on Render.com

### Quick Deploy
1. Connect this repository to Render.com
2. Create a new Web Service
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `gunicorn app:app`
5. Deploy!

## API Endpoint

`POST /api/detect`

### Request
```json
{
  "image": "data:image/jpeg;base64,..."
}
```

### Response
```json
{
  "success": true,
  "detections": [...],
  "statistics": {...}
}
```

## Local Testing

```bash
pip install -r requirements.txt
python app.py
```

## Features
- YOLOv8 dental caries detection
- CORS enabled
- Health check endpoint
- Production-ready with Gunicorn
