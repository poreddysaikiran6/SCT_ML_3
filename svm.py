import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# ✅ Settings
DATADIR = "train"  # Folder containing images
CATEGORIES = ["cat", "dog"]
IMG_SIZE = 64  # Resize all images to 64x64

data = []
labels = []

# ✅ Step 1–3: Load 500 cats & 500 dogs, convert to grayscale, resize, flatten
for category in CATEGORIES:
    path = os.path.join(DATADIR)
    label = 0 if category == "cat" else 1
    count = 0
    for file in os.listdir(path):
        if file.startswith(category):
            try:
                img_path = os.path.join(path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                data.append(img.flatten())
                labels.append(label)
                count += 1
                if count >= 500:
                    break
            except Exception as e:
                continue

print("✅ Loaded", len(data), "images.")

# ✅ Step 4: Prepare data
X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Step 5: Train SVM model
model = SVC(kernel='linear')  # Try 'rbf' or 'poly' if needed
model.fit(X_train, y_train)

# ✅ Step 6: Predict and Evaluate
y_pred = model.predict(X_test)

print("\n🔍 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# ✅ Step 7 (Optional): Visualize 5 predictions
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    img = X_test[i].reshape(IMG_SIZE, IMG_SIZE)
    true_label = "Cat" if y_test[i] == 0 else "Dog"
    pred_label = "Cat" if y_pred[i] == 0 else "Dog"
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"True: {true_label}\nPred: {pred_label}")
    axes[i].axis('off')
plt.suptitle("🔍 Sample Predictions")
plt.tight_layout()
plt.show()
