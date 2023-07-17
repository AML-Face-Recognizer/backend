from flask import Flask, jsonify, request, render_template
from infer import FaceRecognizer

import numpy as np

app = Flask(__name__)

recognizer = FaceRecognizer()

@app.route('/face-detection/api/v1.0/people', methods=['POST'])
def add_person():
    if not request.json or not 'face_photo' in request.json or not 'name' in request.json:
        return jsonify({'status':400})
    
    request.json['face_photo'] = np.array(request.json['face_photo'], dtype='uint8')
        
    person_id = recognizer.add_person(**request.json)
        
    return jsonify({'person_id': person_id, 'status': 201})

@app.route('/face-detection/api/v1.0/people', methods=['GET'])
def get_people():
    return jsonify({'people_list':recognizer._id2name, 'status':200})

@app.route('/face-detection/api/v1.0/people/<int:person_id>', methods=['POST'])
def add_photo(person_id):
    if not request.json or not 'face_photo' in request.json:
        return jsonify({'status':400})
    
    request.json['face_photo'] = np.array(request.json['face_photo'], dtype='uint8')
        
    is_successful = recognizer.add_photo(**request.json, person_id=person_id)
        
    if (not is_successful): 
        # Face is not found on the image
        return jsonify({'status':400})
        
    return jsonify({'status': 204})


@app.route('/face-detection/api/v1.0/recognize', methods=['POST'])
def recognize():
    if not request.json or not 'image' in request.json:
        return jsonify({'status':400})
    
    request.json['image'] = np.array(request.json['image'], dtype='uint8')
        
    result = recognizer.recognize(**request.json)
    
    if result is None:
        # No faces in the system
        return jsonify({'status':400})
    
    data = [{'face_aligned':x[0].tolist(), 'name':x[1], 'id':int(x[2]), 'confidence':float(x[3])} for x in result]
        
    return jsonify({'status': 200, 'body' : data})


if __name__ == '__main__':
    app.run(debug=True)