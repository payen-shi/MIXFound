from PIL import Image
import numpy as np

# Import FLAIR
from flair import FLAIRModel

# Set model
model = FLAIRModel(from_checkpoint=True)

# Load image and set target categories 
# (if the repo is not cloned, download the image and change the path!)

image = np.array(Image.open("/home/ps/data/jinjingzhu/VisionFM/FLAIR/RET032OS.jpg"))
# text = ["normal", "healthy", "macular edema", "diabetic retinopathy", "glaucoma", "macular hole",
#         "lesion", "lesion in the macula"]
text = ["high myopia","glaucoma"]

# Forward FLAIR model to compute similarities
feature, probs, logits = model(image, text)
print(feature.shape)

print("Image-Text similarities:")
print(logits.round(3)) # [[-0.32  -2.782  3.164  4.388  5.919  6.639  6.579 10.478]]
print("Probabilities:")
print(probs.round(3))  # [[0.      0.     0.001  0.002  0.01   0.02   0.019  0.948]]