import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, flash

app = Flask(__name__)
app.secret_key = 'supersecretkey'
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Standard resistor color code mapping for 3-band resistors
RESISTOR_COLOR_CODE = {
    'black': {'digit': 0, 'multiplier': 1},
    'brown': {'digit': 1, 'multiplier': 10},
    'red':   {'digit': 2, 'multiplier': 100},
    'orange':{'digit': 3, 'multiplier': 1000},
    'yellow':{'digit': 4, 'multiplier': 10000},
    'green': {'digit': 5, 'multiplier': 100000},
    'blue':  {'digit': 6, 'multiplier': 1000000},
    'violet':{'digit': 7, 'multiplier': 10000000},
    'grey':  {'digit': 8, 'multiplier': 100000000},
    'white': {'digit': 9, 'multiplier': 1000000000},
}

# HSV thresholds for common resistor colors.
# These are approximate and should be tuned with real data.
HSV_RANGES = {
    'black':   ([0, 0, 0],        [180, 255, 50]),
    'brown':   ([10, 50, 50],      [20, 255, 200]),
    'red':     ([0, 70, 50],       [10, 255, 255]),
    'orange':  ([10, 100, 50],     [25, 255, 255]),
    'yellow':  ([25, 100, 100],    [35, 255, 255]),
    'green':   ([35, 50, 50],      [85, 255, 255]),
    'blue':    ([85, 50, 50],      [130, 255, 255]),
    'violet':  ([130, 50, 50],     [160, 255, 255]),
    'grey':    ([0, 0, 50],        [180, 50, 200]),
    'white':   ([0, 0, 200],       [180, 30, 255]),
}

def preprocess_image(image):
    """
    Preprocess the image:
    - Crop or extract ROI if needed.
    - Resize for consistency.
    - Convert to HSV.
    """
    # (Optional) Crop image to the region of interest if known
    # image = image[y1:y2, x1:x2]

    # Resize the image for consistency (optional)
    image = cv2.resize(image, (400, 300))
    
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return image, hsv_image

def segment_colors(hsv_image):
    """
    Use HSV thresholds to create binary masks for each color.
    """
    color_masks = {}
    for color, (lower, upper) in HSV_RANGES.items():
        lower_np = np.array(lower, dtype="uint8")
        upper_np = np.array(upper, dtype="uint8")
        mask = cv2.inRange(hsv_image, lower_np, upper_np)
        color_masks[color] = mask
    return color_masks

def find_resistor_bands(image):
    """
    Detect resistor bands using a combination of color segmentation
    and contour detection. This function:
    1. Preprocesses the image.
    2. Applies color segmentation to create masks.
    3. Uses contours on the masks to detect bands.
    4. Sorts the bands from left-to-right.
    """
    # Step 1: Preprocess image and convert to HSV
    processed_image, hsv_image = preprocess_image(image)
    
    # Step 2: Create color masks for each resistor color
    color_masks = segment_colors(hsv_image)
    
    # Step 3: Detect contours and filter them based on size and aspect ratio
    band_candidates = []
    for color, mask in color_masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = h / float(w)
            if w > 10 and h > 20 and aspect_ratio > 2:
                band_candidates.append((x, y, w, h, color))
    
    # Step 4: Sort candidates by x-coordinate (left to right)
    band_candidates = sorted(band_candidates, key=lambda item: item[0])
    
    # Take the first 3 candidates (if available)
    detected_bands = [item[4] for item in band_candidates[:3]]
    return detected_bands

def compute_resistance(bands):
    """
    Compute the resistor value using the standard 3-band approach.
    """
    if len(bands) < 3:
        return None, "Not enough bands detected."
    try:
        digit1 = RESISTOR_COLOR_CODE[bands[0]]['digit']
        digit2 = RESISTOR_COLOR_CODE[bands[1]]['digit']
        multiplier = RESISTOR_COLOR_CODE[bands[2]]['multiplier']
    except KeyError:
        return None, "Error mapping colors."
    resistance = ((digit1 * 10) + digit2) * multiplier
    return resistance, None

def process_resistor_image(image_path):
    """
    Complete processing pipeline:
    - Read image.
    - Preprocess.
    - Detect resistor bands.
    - Compute resistance.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None, "Error reading the image."
    
    detected_bands = find_resistor_bands(image)
    if len(detected_bands) < 3:
        return None, "Could not detect 3 distinct color bands. Try a clearer image."
    
    resistance, err = compute_resistance(detected_bands)
    if err:
        return None, err
    return {'bands': detected_bands, 'resistance': resistance}, None

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None
    if request.method == 'POST':
        if 'resistor_image' not in request.files:
            flash('No file part.')
            return redirect(request.url)
        file = request.files['resistor_image']
        if file.filename == '':
            flash('No selected file.')
            return redirect(request.url)
        if file:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            result, error = process_resistor_image(file_path)
    return render_template('index.html', result=result, error=error)

if __name__ == '__main__':
    app.run(debug=True)
