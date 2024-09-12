#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt

imageA = cv2.imread('images/upsight_table.png', cv2.IMREAD_UNCHANGED)
orb = cv2.ORB_create(nfeatures=1500)
keypoints_orb, descriptors = orb.detectAndCompute(imageA, None)
image_with_keypoints = cv2.drawKeypoints(imageA, keypoints_orb, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure(figsize=(10,10))
plt.imshow(image_with_keypoints)
# %%
