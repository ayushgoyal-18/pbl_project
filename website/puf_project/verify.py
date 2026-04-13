import cv2
import pickle

# load database
with open("database.pkl", "rb") as f:
    database = pickle.load(f)

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# test image
test_img_path = "dataset/puf1.jpg"  # change this to test

img = cv2.imread(test_img_path, 0)

kp, des = orb.detectAndCompute(img, None)

best_match = None
best_score = 0

for name, db_des in database.items():
    matches = bf.match(des, db_des)

    score = len(matches)

    print(f"{name} -> Matches: {score}")

    if score > best_score:
        best_score = score
        best_match = name

print("\nBest Match:", best_match)

# threshold decision
if best_score > 30:
    print("✅ GENUINE")
else:
    print("❌ FAKE")