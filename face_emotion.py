from deepface import DeepFace

def detect_emotion(frame):
    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )
        emotion = result[0]['dominant_emotion']
        confidence = result[0]['emotion'][emotion]
        return emotion, confidence
    except:
        return "Neutral", 0.0
