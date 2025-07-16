import matplotlib.pyplot as plt
import cv2
import os
import easyocr
from pylab import rcParams
import nltk
import dlib
import re
import numpy as np
from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image
import requests

# 🔸 Download NLTK punkt tokenizer if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 🔸 Explicitly download 'punkt_tab' as it might be required by newer NLTK versions
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# 🔸 Initialize Flask
app = Flask(__name__)

# 🔸 EasyOCR reader - Now supports both English and Bengali
en_bn_reader = easyocr.Reader(['en', 'bn'])

# 🔸 DLIB model download and setup
DLIB_MODEL_PATH = "./shape_predictor_68_face_landmarks.dat"

def download_dlib_model(path):
    url = "https://raw.githubusercontent.com/AKSHAYUBHAT/TensorFace/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat"
    print("Attempting to download shape_predictor_68_face_landmarks.dat...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download of dlib model complete.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download dlib model file: {e}")
        raise

if not os.path.exists(DLIB_MODEL_PATH):
    download_dlib_model(DLIB_MODEL_PATH)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DLIB_MODEL_PATH)

MONTH_NAME = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']


def EASY_month_matching(sent):
    sent = sent.strip()
    date_pattern = re.compile(r'\b(\d{1,2})\s*(' + '|'.join(MONTH_NAME) + r')\s*(\d{4})\b', re.IGNORECASE)
    match = date_pattern.search(sent)
    if match:
        day, month, year = match.groups()
        return f"{day} {month} {year}"
    return None


def EASY_NID_matching(sent):
    nid_pattern = re.compile(r'\b(?:\d{10}|\d{17}|(?:\d{3}\s?\d{3}\s?\d{3}\s?\d{1})|(?:\d{3}\s?\d{4}\s?\d{3}\s?\d{4}\s?\d{3}\s?\d{2}))\b')
    match = nid_pattern.search(sent)
    if match:
        nid = match.group(0).replace(' ', '').replace('-', '')
        if len(nid) == 10:
            return f"{nid[0:3]} {nid[3:6]} {nid[6:9]} {nid[9]}"
        elif len(nid) == 17:
            return f"{nid[0:3]} {nid[3:7]} {nid[7:11]} {nid[11:15]} {nid[15:17]}"
        return nid
    return None


def make_dataset(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode image from URL.")
    if img.size == 0:
        raise ValueError("Decoded image is empty.")

    img = cv2.resize(img, (640, 480))
    imgOriginal = img.copy()

    faces = detector(imgOriginal)
    cropped_img_bytes = None

    if faces:
        for face in faces:
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()

            # Broaden the crop to definitely include the text area to the right of the face
            cropped = imgOriginal[max(0, y1-50):min(imgOriginal.shape[0], y2+200),
                                  max(0, x2):min(imgOriginal.shape[1], x2 + 350)]

            if cropped.size == 0:
                print("Warning: Cropped image is empty after face-based cropping. Trying next face or full image.")
                continue

            is_success, buffer = cv2.imencode(".png", cropped)
            if is_success:
                cropped_img_bytes = BytesIO(buffer)
                break
        
    if cropped_img_bytes is None:
        print("No suitable face detected or unable to crop effectively. Using full image for OCR.")
        is_success, buffer = cv2.imencode(".png", imgOriginal)
        if is_success:
            cropped_img_bytes = BytesIO(buffer)
        else:
            raise ValueError("Unable to encode full image for OCR.")

    return cropped_img_bytes


def Extract_NID_INFO_EASY_OCR(img_bytes):
    info_dict = {
        "Name": "N/A",
        "Name_bn": "N/A",
        "NID_no": "N/A",
        "Father's Name": "N/A",
        "Mother's Name": "N/A",
        "Address": "N/A",
        "Blood Group": "N/A"
    }

    result_detailed = en_bn_reader.readtext(img_bytes.getvalue(), detail=1)

    print("\n--- RAW OCR RESULTS (Text and Bounding Boxes) ---")
    for (bbox, text, prob) in result_detailed:
        print(f"BBox: {bbox}, Text: '{text}', Confidence: {prob:.2f}")
    print("--------------------------------------------------\n")

    text_lines_with_coords = []
    for (bbox, text, prob) in result_detailed:
        if text.strip():
            text_lines_with_coords.append({
                "text": text.strip(),
                "lower_text": text.strip().lower(),
                "y_min": bbox[0][1],
                "y_max": bbox[2][1],
                "x_min": bbox[0][0],
                "x_max": bbox[1][0],
                "height": bbox[2][1] - bbox[0][1]
            })

    # Sort by Y then X for reading order
    text_lines_with_coords.sort(key=lambda x: (x["y_min"], x["x_min"]))

    all_text_combined = " ".join([item["text"] for item in text_lines_with_coords])

    nid_match = EASY_NID_matching(all_text_combined)
    if nid_match:
        # Remove spaces for the final NID_no format
        info_dict["NID_no"] = nid_match.replace(" ", "")

    dob_match = EASY_month_matching(all_text_combined)
    if dob_match:
        info_dict["Date of Birth"] = dob_match

    # Define keywords for field extraction
    father_keywords = ["পিতা", "পিতাঃ", "পিতার নাম", "father", "father's name", "মো:", "md."]
    mother_keywords = ["মাতা", "মাতাঃ", "মাতার নাম", "mother", "mother's name", "মোসাঃ", "mosa."]
    address_keywords = [
        "ঠিকানা", "বর্তমান ঠিকানা", "স্থায়ী ঠিকানা", "address",
        "বাসা", "গ্রাম", "পোস্ট", "থানা", "জেলা", "উপজেলা", "সিটি কর্পোরেশন",
        "house", "village", "post office", "upazila", "district", "city corporation",
        "road", "sector", "block", "গোপালগঞ্জ", "সদর", "৮১০০"
    ]
    blood_group_keywords = ["রক্তের গ্রুপ", "রক্তের", "blood group", "blood"]
    blood_group_pattern = re.compile(r'\b(A|B|AB|O)\s*([+-]|positive|negative)\b', re.IGNORECASE)

    # Track used lines
    used_line_indices = set()

    # Name extraction: Look for both Bengali and English names
    for i, line in enumerate(text_lines_with_coords):
        if i in used_line_indices:
            continue

        # Bengali name: Starts with title, contains Bengali chars
        if info_dict["Name_bn"] == "N/A" and \
           re.match(r'^(?:মোঃ|শ্রী|প্রো|Mr|Ms|Mrs|Md)\.?.?\s*[\u0980-\u09FF\s]+$', line["text"]) and \
           len(line["text"]) > 5:
            info_dict["Name_bn"] = line["text"]
            used_line_indices.add(i)
            continue

        # English name: Uppercase letters, appears near the top, not too short
        if info_dict["Name"] == "N/A" and \
           re.match(r'^[A-Z][A-Z\s.]+$', line["text"]) and \
           len(line["text"]) > 5 and \
           len([c for c in line["text"] if c.isupper()]) / len(line["text"]) > 0.7:
            info_dict["Name"] = line["text"]
            used_line_indices.add(i)
            continue

    # Extract other fields
    for i, line in enumerate(text_lines_with_coords):
        if i in used_line_indices:
            continue

        line_text = line["text"]
        line_lower_text = line["lower_text"]

        # Father's Name
        if info_dict["Father's Name"] == "N/A":
            for kw in father_keywords:
                if kw in line_lower_text:
                    start_val_idx = line_lower_text.find(kw) + len(kw)
                    potential_value_on_same_line = line_text[start_val_idx:].strip()

                    if potential_value_on_same_line and len(potential_value_on_same_line) > 1 and not re.search(r'^\d+$', potential_value_on_same_line):
                        info_dict["Father's Name"] = potential_value_on_same_line
                        used_line_indices.add(i)
                        break
                    else:
                        collected_name_parts = []
                        if potential_value_on_same_line:
                            collected_name_parts.append(potential_value_on_same_line)

                        j = i + 1
                        while j < len(text_lines_with_coords):
                            next_line = text_lines_with_coords[j]
                            if (next_line["y_min"] - line["y_max"]) < 20 and \
                               abs(next_line["x_min"] - line["x_min"]) < 50:
                                if not any(k in next_line["lower_text"] for k in mother_keywords + address_keywords + blood_group_keywords) and \
                                   not re.search(r'^\d+$', next_line["text"]):
                                    collected_name_parts.append(next_line["text"])
                                    used_line_indices.add(j)
                                    j += 1
                                else:
                                    break
                            else:
                                break
                        if collected_name_parts:
                            info_dict["Father's Name"] = " ".join(collected_name_parts).strip()
                            used_line_indices.add(i)
                        break

        # Mother's Name
        if info_dict["Mother's Name"] == "N/A":
            for kw in mother_keywords:
                if kw in line_lower_text:
                    start_val_idx = line_lower_text.find(kw) + len(kw)
                    potential_value_on_same_line = line_text[start_val_idx:].strip()

                    if potential_value_on_same_line and len(potential_value_on_same_line) > 1 and not re.search(r'^\d+$', potential_value_on_same_line):
                        info_dict["Mother's Name"] = potential_value_on_same_line
                        used_line_indices.add(i)
                        break
                    else:
                        collected_name_parts = []
                        if potential_value_on_same_line:
                            collected_name_parts.append(potential_value_on_same_line)

                        j = i + 1
                        while j < len(text_lines_with_coords):
                            next_line = text_lines_with_coords[j]
                            if (next_line["y_min"] - line["y_max"]) < 20 and \
                               abs(next_line["x_min"] - line["x_min"]) < 50:
                                if not any(k in next_line["lower_text"] for k in father_keywords + address_keywords + blood_group_keywords) and \
                                   not re.search(r'^\d+$', next_line["text"]):
                                    collected_name_parts.append(next_line["text"])
                                    used_line_indices.add(j)
                                    j += 1
                                else:
                                    break
                            else:
                                break
                        if collected_name_parts:
                            info_dict["Mother's Name"] = " ".join(collected_name_parts).strip()
                            used_line_indices.add(i)
                        break

        # Blood Group
        if info_dict["Blood Group"] == "N/A":
            bg_match = blood_group_pattern.search(line_text)
            if bg_match:
                info_dict["Blood Group"] = bg_match.group(0).upper()
                used_line_indices.add(i)
            else:
                for kw in blood_group_keywords:
                    if kw in line_lower_text:
                        remaining_text = line_text.split(kw, 1)[-1].strip()
                        bg_match_after_kw = blood_group_pattern.search(remaining_text)
                        if bg_match_after_kw:
                            info_dict["Blood Group"] = bg_match_after_kw.group(0).upper()
                            used_line_indices.add(i)
                            break
                        elif i + 1 < len(text_lines_with_coords):
                            next_line = text_lines_with_coords[i+1]
                            if (next_line["y_min"] - line["y_max"]) < 20:
                                bg_match_next_line = blood_group_pattern.search(next_line["text"])
                                if bg_match_next_line:
                                    info_dict["Blood Group"] = bg_match_next_line.group(0).upper()
                                    used_line_indices.add(i)
                                    used_line_indices.add(i+1)
                                    break
                if info_dict["Blood Group"] != "N/A":
                    continue

        # Address
        if info_dict["Address"] == "N/A":
            for kw in address_keywords:
                if kw in line_lower_text:
                    collected_address_parts = []
                    start_val_idx = line_lower_text.find(kw) + len(kw)
                    current_address_line_segment = line_text[start_val_idx:].strip()

                    if current_address_line_segment:
                        collected_address_parts.append(current_address_line_segment)
                    
                    current_address_line_y_max = line["y_max"]
                    
                    j = i + 1
                    while j < len(text_lines_with_coords):
                        next_line = text_lines_with_coords[j]
                        is_another_major_field_keyword = any(mk in next_line["lower_text"] for mk in father_keywords + mother_keywords + blood_group_keywords + ["Date of Birth", "NID", "Name"])
                        
                        if (next_line["y_min"] - current_address_line_y_max) < 40 and \
                           not is_another_major_field_keyword and \
                           len(next_line["text"]) > 1 and \
                           not re.search(r'[^a-zA-Z\s\u0980-\u09FF\d,\.-]', next_line["text"]):
                            collected_address_parts.append(next_line["text"])
                            used_line_indices.add(j)
                            current_address_line_y_max = next_line["y_max"]
                            j += 1
                        else:
                            break

                    if collected_address_parts:
                        info_dict["Address"] = " ".join(collected_address_parts).strip()
                        used_line_indices.add(i)
                        break

    return info_dict


@app.route('/ocr', methods=['GET'])
def ocr_endpoint():
    image_url = request.args.get('url')
    if not image_url:
        return jsonify({"error": "Missing 'url' parameter"}), 400

    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image_bytes = BytesIO(response.content)

        cropped_image_stream = make_dataset(image_bytes.getvalue())
        nid_info = Extract_NID_INFO_EASY_OCR(cropped_image_stream)

        return jsonify(nid_info)

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch image from URL: {e}"}), 400
    except ValueError as e:
        return jsonify({"error": f"Image processing error: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)