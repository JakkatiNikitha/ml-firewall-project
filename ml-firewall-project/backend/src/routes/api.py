from flask import Blueprint, request, jsonify
from services.detection_service import DetectionService

api = Blueprint('api', __name__)

@api.route('/detect', methods=['POST'])
def detect_attack():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    try:
        results = DetectionService.detect(data)
        return jsonify(results), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200