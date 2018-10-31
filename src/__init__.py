from keras.datasets import mnist
from itertools import chain
import matplotlib.pyplot as plt


# load (downloaded if needed) the MNIST dataset
def convert_Unary_image(img):
    result = []
    for row in img:
        new_row = []
        for elem in row:
            if elem != 0:
                elem = 1
            new_row.append(elem)
        result.append(new_row)
    return result


(X_train, y_train), (X_test, y_test) = mnist.load_data()
# plot 4 images as gray scale
trained_img_0 = X_train[0]
# print(len(X_train[0]))
# print(y_train[0])
# print(X_train[0])
images = X_train
x = images.reshape((images.shape[0], images.shape[1] * images.shape[2]))
print(x[0], "\n\n\n")
c = convert_Unary_image(x[0])
print(c)
