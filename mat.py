from PIL import Image
import numpy as np
np.set_printoptions(threshold=np.inf)
# Open the image and convert to grayscale
img = Image.open('white.jpg').convert('L')

# Convert the image data to a numpy array
img_array = np.array(img)

# Find the indices where the pixel value is 0 (black)
global indices
indi np.where(img_array == 0)

