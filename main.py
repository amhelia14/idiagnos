from flask import Flask, request, jsonify
import pickle
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
import os
import sys
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

firebase_credentials_str = os.getenv("FIREBASE_CREDENTIALS").replace(r'\n', '\\n')

if not firebase_credentials_str:
    print("ERROR: FIREBASE_CREDENTIALS environment variable not found!")
    sys.exit(1)

try:
    firebase_credentials = json.loads(firebase_credentials_str)
    cred = credentials.Certificate(firebase_credentials)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firestore connected successfully!")
except Exception as e:
    print(f"ERROR: Failed to initialize Firestore - {e}")
    sys.exit(1)

try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    
    with open('label_mapping.pkl', 'rb') as f:
        label_mapping = pickle.load(f)
    
    le_y = encoders['diagnosis']
    print("Model, encoders, and label mapping loaded successfully!")
except Exception as e:
    print(f"ERROR: Failed to load model files - {e}")
    sys.exit(1)

@app.route('/')
def home():
    return "Glaucoma Diagnosis API is running!"

@app.route('/test_firestore', methods=['GET'])
def test_firestore():
    if not firebase_initialized:
        return jsonify({"error": "Firestore not initialized"}), 500
    try:
        test_ref = db.collection("test_collection").document("test_doc")
        test_ref.set({"message": "Firestore connection successful!"})
        doc = test_ref.get()
        return jsonify({"success": True, "data": doc.to_dict()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/register_patient', methods=['POST'])
def register_patient():
    if not firebase_initialized:
        return jsonify({"error": "Firestore not initialized"}), 500
    data = request.json
    user_id = data.get('userId')
    patient_name = data.get('patientName')
    if not user_id:
        return jsonify({"error": "userId is required"}), 400
    try:
        patient_ref = db.collection('patients').document()
        patient_id = patient_ref.id
        patient_data = {
            'userId': user_id,
            'patientName': patient_name,
            'registrationDate': firestore.SERVER_TIMESTAMP
        }
        patient_ref.set(patient_data)
        return jsonify({"success": True, "message": "Patient registered successfully", "patientId": patient_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/diagnose', methods=['POST'])
def diagnose():
    if not firebase_initialized:
        return jsonify({"error": "Firestore not initialized"}), 500
    data = request.json
    user_id = data.get('userId')
    patient_id = data.get('patientId')
    symptoms = data.get('symptoms')
    if not all([user_id, patient_id, symptoms]):
        return jsonify({"error": "userId, patientId, and symptoms are required"}), 400
    if len(symptoms) != 20:
        return jsonify({"error": "Symptoms data must contain exactly 20 values (0 atau 1)"}), 400
    try:
        diagnosis_ref = db.collection('Diagnosis').document()
        diagnosis_id = diagnosis_ref.id
        input_data = np.array(symptoms).reshape(1, -1)
        prediction = model.predict(input_data)
        predicted_label = le_y.inverse_transform(prediction)[0]
        diagnosis_data = {
            'userId': user_id,
            'Patient_ID': patient_id,
            'diagnosisDate': firestore.SERVER_TIMESTAMP,
            'gejala': symptoms,
            'diagnosis_result': predicted_label
        }
        diagnosis_ref.set(diagnosis_data)
        gc.collect()
        return jsonify({"success": True, "diagnosisId": diagnosis_id, "result": predicted_label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_diagnosis_history', methods=['GET'])
def get_diagnosis_history():
    if not firebase_initialized:
        return jsonify({"error": "Firestore not initialized"}), 500
    user_id = request.args.get('userId')
    patient_id = request.args.get('patientId')
    if not user_id:
        return jsonify({"error": "userId is required"}), 400
    try:
        query = db.collection('Diagnosis').where('userId', '==', user_id)
        if patient_id:
            query = query.where('Patient_ID', '==', patient_id)
        diagnoses = []
        for doc in query.stream():
            diagnosis_data = doc.to_dict()
            diagnosis_data['id'] = doc.id
            diagnoses.append(diagnosis_data)
        return jsonify({"success": True, "diagnoses": diagnoses})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/test_prediction', methods=['POST'])
def test_prediction():
    data = request.json
    symptoms = data.get('symptoms')
    if not symptoms:
        return jsonify({"error": "symptoms are required"}), 400
    if len(symptoms) != 20:
        return jsonify({"error": "Symptoms data must contain exactly 20 values (0 atau 1)"}), 400
    try:
        input_data = np.array(symptoms).reshape(1, -1)
        prediction = model.predict(input_data)
        predicted_label = le_y.inverse_transform(prediction)[0]
        probabilities = model.predict_proba(input_data)[0]
        proba_dict = {le_y.inverse_transform([i])[0]: float(proba) for i, proba in enumerate(probabilities)}
        return jsonify({"success": True, "result": predicted_label, "probabilities": proba_dict})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/symptom_names', methods=['GET'])
def get_symptom_names():
    symptom_names = [f"gejala_{i+1}" for i in range(20)]
    return jsonify({"success": True, "symptom_names": symptom_names})

if __name__ == '__main__':
    app.run(debug=True)
