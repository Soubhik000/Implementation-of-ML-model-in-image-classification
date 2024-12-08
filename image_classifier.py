import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt

model = MobileNetV2(weights="imagenet")
animal_categories = [
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "panda", "rabbit"
]
tree_categories = [
    "pine_tree", "oak_tree", "maple_tree", "willow_tree", "cedar_tree", "palm_tree"
]
def classify_image(img_array):
    """Classify an image using the pre-trained model."""
    img_resized = cv2.resize(img_array, (224, 224))
    if img_resized is None or img_resized.shape != (224, 224, 3):
        print("Error: Image resizing failed.")
        return []
    img_preprocessed = preprocess_input(img_resized)
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    predictions = model.predict(img_batch)
    decoded_preds = decode_predictions(predictions, top=3)[0]  
    return decoded_preds
def select_file():
    """Open a file dialog to select an image and classify it."""
    root = Tk()
    root.deiconify()  
    
    file_path = filedialog.askopenfilename(
        title="Choose a file", 
        filetypes=[("Image files", "*.jpg *.jpeg *.png")],  
        initialdir="."  
    )
    root.withdraw()  
    if file_path: 
        img = cv2.imread(file_path)
        if img is None:
            print("Error: Unable to load the image.")
            return
        print(f"Image loaded: {file_path.split('/')[-1]} with shape {img.shape}")

        results = classify_image(img)
        if results: 
            print(f"Classification Results for {file_path.split('/')[-1]}:")
            classified_as_animal_or_tree = False  
            for label, description, score in results:
                
                if any(animal in description.lower() for animal in animal_categories):
                    print(f"The image is likely a {description}.")
                    classified_as_animal_or_tree = True 
                elif any(tree in description.lower() for tree in tree_categories):
                    print(f"The image is likely a {description}.")
                    classified_as_animal_or_tree = True
            if not classified_as_animal_or_tree:
                print("The image is not recognized as an animal or tree.")
        else:
            print("No results returned.")
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            plt.title(f"Selected Image: {file_path.split('/')[-1]}")
            plt.axis("off")
            plt.show()
        except Exception as e:
            print(f"Error displaying image: {e}")
        input("Press Enter to close the image window.")
    else:
        print("No file selected.")
if __name__ == "__main__":
    select_file()
