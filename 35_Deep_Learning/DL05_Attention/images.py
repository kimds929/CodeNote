import os
import numpy as np
import matplotlib.pyplot as plt


path = 'images/'


def load(file):
    image = (plt.imread(path + file) / 255).astype('float32')
    digits = np.array(list(map(int, file[:-4])))
    label = np.zeros(shape=10)
    for digit in digits:
        label[digit] = 1
    return image, label


def plot(image, label):
    plt.imshow(image)
    plt.title(str(label))
    plt.axis('off')
    plt.show()
    
    
def load_all():
    files = os.listdir(path)
    np.random.shuffle(files)

    images = []
    labels = []

    for file in files:
        image, label = load(file)
        images.append(image)
        labels.append(label)

    images = np.array(images, copy=False)
    labels = np.array(labels, copy=False)
    
    return images, labels
