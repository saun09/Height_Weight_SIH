import sys
import os

# --- Fix Python path so imports work ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "Height-Detection"))
sys.path.append(os.path.join(BASE_DIR, "face-to-bmi-vit", "scripts"))

from Body_Detection import measure_height_from_webcam
from capture_face import capture_face_from_webcam
from demo import predict_bmi


def main():
    print("---- Step 1: Measuring Height ----")
    height_cm = measure_height_from_webcam()
    if height_cm is None:
        print("❌ Could not measure height.")
        return
    print(f"✅ Height detected: {height_cm:.2f} cm")

    print("\n---- Step 2: Capturing Face ----")
    face_path = capture_face_from_webcam()
    if face_path is None:
        print("❌ Face capture failed.")
        return
    print(f"✅ Face captured at {face_path}")

    print("\n---- Step 3: Predicting BMI ----")
    weight_path = os.path.join(BASE_DIR, "face-to-bmi-vit", "weights", "aug_epoch_7.pt")
    bmi_value = predict_bmi(face_path, weight_path)
    if bmi_value is None:
        print("❌ BMI prediction failed.")
        return
    print(f"✅ Predicted BMI: {bmi_value:.2f}")

    print("\n---- Step 4: Estimating Weight ----")
    height_m = height_cm / 100.0
    weight_kg = round(bmi_value * (height_m ** 2), 2)
    print(f"✅ Estimated Weight: {weight_kg:.2f} kg")

    print("\n---- Final Results ----")
    print(f"Height : {height_cm:.2f} cm")
    print(f"BMI    : {bmi_value:.2f}")
    print(f"Weight : {weight_kg:.2f} kg")


if __name__ == "__main__":
    main()
