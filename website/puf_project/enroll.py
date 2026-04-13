import cv2
import pickle
import os

# ORB detector
orb = cv2.ORB_create()

database = {}

dataset_path = "dataset"

for file in os.listdir(dataset_path):
    if file.endswith(".jpg"):
        path = os.path.join(dataset_path, file)

        img = cv2.imread(path, 0)

        keypoints, descriptors = orb.detectAndCompute(img, None)

        database[file] = descriptors

        print(f"[ENROLLED] {file}")

# save database
with open("database.pkl", "wb") as f:
    pickle.dump(database, f)

print("\nDatabase saved!")