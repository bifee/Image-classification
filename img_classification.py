import os
from skimage import color
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pylab as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

train_dir = 'database/train'
val_dir = 'database/val'
categories = ['cat', 'dog']
train_images = []
val_images = []
train_classification = []
val_classification = []


def load_data(dir, images, classification):
    for classification_idx, category in enumerate(categories):
        for file in os.listdir(os.path.join(dir, category)):
            img_path = os.path.join(dir, category, file)
            img = imread(img_path)
            img = color.rgb2gray(img)
            img = resize(img, (150, 150))
            images.append(img.flatten())
            classification.append(classification_idx)
    images = np.asarray(images)
    classification = np.asarray(classification)
    return images, classification


def plot_misclassified_images(images, true_labels, predicted_labels):
    for i in range(len(images)):
        if true_labels[i] != predicted_labels[i]:
            resized_img = np.reshape(images[i], (150, 150))
            plt.imshow(resized_img, cmap='gray')
            plt.title(f'True: {categories[true_labels[i]]}, Predicted: {categories[predicted_labels[i]]}')
            plt.axis('off')
            plt.show()

# Load images as npArrays
load_data(train_dir, train_images, train_classification)
load_data(val_dir, val_images, val_classification)

# Create Decision Tree
tree_classifier = DecisionTreeClassifier()
# Train Decision Tree
tree_classifier.fit(train_images, train_classification)
# Test Decision Tree
val_predictions = tree_classifier.predict(val_images)
# Show Accuracy
accuracy = accuracy_score(val_classification, val_predictions)
print(f'Accuracy: {accuracy}')
# Show misclassified images if necessary
# plot_misclassified_images(val_images, val_classification, val_predictions)


