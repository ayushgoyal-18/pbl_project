import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image

# ── App setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER']      = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}

# ── Model config
IMG_SIZE      = (224, 224)
MODEL_PATH    = 'medicine_cnn_best.h5'
CLASS_INDICES = {'fake': 0, 'real': 1}

# ── Lazy load
model         = None
preprocess_fn = tf.keras.applications.mobilenet_v2.preprocess_input


def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file '{MODEL_PATH}' not found. "
                f"Please download medicine_cnn_best.h5 from Google Drive "
                f"and place it next to app.py."
            )
        print("Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded ✔")
    return model


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_image(image_path):
    """
    Basic checks before running CNN.
    Returns (is_valid, error_message).
    """
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception:
        return False, "Could not open image. Please upload a valid JPG or PNG file."

    # Reject tiny images
    if img.width < 100 or img.height < 100:
        return False, "Image is too small. Please upload a clear medicine photo."

    # Reject extremely wide or tall images (e.g. ID cards, screenshots, banners)
    aspect_ratio = max(img.width, img.height) / min(img.width, img.height)
    if aspect_ratio > 4.0:
        return False, "Image shape looks unusual. Please upload a standard medicine photo."

    return True, "OK"


def predict_image(image_path):
    """Run validation then CNN inference. Returns result dict."""

    # Step 1 — basic image sanity check
    valid, msg = validate_image(image_path)
    if not valid:
        raise ValueError(msg)

    # Step 2 — run model
    m         = get_model()
    img       = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_fn(img_array)

    prob      = float(m.predict(img_array, verbose=0)[0][0])
    real_prob = prob
    fake_prob = 1.0 - prob
    predicted  = 'real' if real_prob >= 0.5 else 'fake'
    confidence = real_prob if predicted == 'real' else fake_prob

    # Step 3 — reject low confidence (likely a non-medicine image)
    if confidence < 0.70:
        raise ValueError(
            f"Low confidence ({confidence * 100:.1f}%). "
            "This may not be a medicine image. "
            "Please upload a clear photo of a medicine strip, bottle, or packet."
        )

    return {
        'predicted_class': predicted,
        'confidence_pct' : round(confidence * 100, 2),
        'real_prob_pct'  : round(real_prob   * 100, 2),
        'fake_prob_pct'  : round(fake_prob   * 100, 2),
        'is_real'        : predicted == 'real'
    }


# ── Routes

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Use JPG, PNG, BMP or WEBP.'}), 400

    filename  = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(save_path)

    try:
        result = predict_image(save_path)
        return jsonify(result)

    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 500

    except ValueError as e:
        # Validation failures — wrong image type, low confidence etc.
        return jsonify({'error': str(e)}), 422

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    finally:
        # Clean up uploaded file after prediction
        if os.path.exists(save_path):
            os.remove(save_path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)