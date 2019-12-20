import numpy as np
from matplotlib import pyplot as plt
import sys
from copy import deepcopy

np.set_printoptions(threshold=sys.maxsize)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray)

# add random noise to the images
def add_sp_noise(image):
    # check type of [pic]
    if not _is_numpy_image(image):
        raise TypeError('img should be numpy array. Got {}'.format(type(image)))

    amount_min = 0.0115
    amount_max = 0.0175
    out = deepcopy(image)
    num_salt = np.random.randint(np.ceil(amount_min * image.size), np.ceil(amount_max * image.size))
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    for i in range(int(num_salt)):
            out[coords[0][i]][coords[1][i]] = np.random.random_sample()/float(2)
    return out

def generate_noisy_images(images):
    images = [add_sp_noise(image) for image in images]
    return images

def generate_more_noisy_images(images, number_of_repetitions):
    print(images.shape)
    x = []
    for i in range(number_of_repetitions):
        x.append(generate_noisy_images(images))
    x = np.array(x)
    return np.reshape(x, (number_of_repetitions * len(images), 40, 60))


if __name__ == "__main__":

    number_of_repetitions = 10

    img_arrayA = np.load('data/training/class_a.npy')
    img_arrayB = np.load('data/training/class_b.npy')
    img_arrayC = np.load('data/classification/field.npy')

    print(img_arrayC.shape)

    img_arrayA_noisy_many = generate_more_noisy_images(img_arrayA, number_of_repetitions)
    print("img_arrayA_noisy shape: ", img_arrayA_noisy_many.shape)
    np.save('data/training/many_noisy_class_a.npy', img_arrayA_noisy_many)

    img_arrayB_noisy_many = generate_more_noisy_images(img_arrayB, number_of_repetitions)
    print("img_arrayB_noisy shape: ", img_arrayB_noisy_many.shape)
    np.save('data/training/many_noisy_class_b.npy', img_arrayB_noisy_many)

    print("Dataset A, mean: ", np.mean(img_arrayA), "std: ", np.std(img_arrayA))
    print("Dataset B, mean: ", np.mean(img_arrayB), "std: ", np.std(img_arrayB))

    print("Dataset A noisy, mean: ", np.mean(img_arrayA_noisy_many), "std: ", np.std(img_arrayA_noisy_many))
    print("Dataset B noisy, mean: ", np.mean(img_arrayB_noisy_many), "std: ", np.std(img_arrayB_noisy_many))
    print("Dataset C, mean: ", np.mean(img_arrayC), "std: ", np.std(img_arrayC))

    for i in range(10):
         #see the noisy images
         plt.imshow(img_arrayA_noisy_many[i], cmap='gray')
         plt.savefig('images\\noisyfigA'+ str(i) + '.png')

         plt.imshow(img_arrayB_noisy_many[i], cmap='gray')
         plt.savefig('images\\noisyfigB'+ str(i) + '.png')

