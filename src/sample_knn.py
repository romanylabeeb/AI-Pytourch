import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import time
from keras.datasets import mnist
import torch


class simple_knn():
    "a simple kNN with L2 distance"

    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1):
        dists = self.compute_distances(X)
        # print("computed distances")

        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            k_closest_y = []
            labels = self.y_train[np.argsort(dists[i, :])].flatten()
            # find k nearest lables
            k_closest_y = labels[:k]

            # out of these k nearest lables which one is most common
            # for 5NN [1, 1, 1, 2, 3] returns 1
            # break ties by selecting smaller label
            # for 5NN [1, 2, 1, 2, 3] return 1 even though 1 and 2 appeared twice.
            c = Counter(k_closest_y)
            y_pred[i] = c.most_common(1)[0][0]

        return (y_pred)

    def compute_distances(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]

        dot_pro = np.dot(X, self.X_train.T)
        sum_square_test = np.square(X).sum(axis=1)
        sum_square_train = np.square(self.X_train).sum(axis=1)
        dists = np.sqrt(-2 * dot_pro + sum_square_train + np.matrix(sum_square_test).T)

        return (dists)


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train[:100]
y_train = y_train[:100]
X_test = X_test[:100]
y_test = y_test[:100]
print(X_train.shape, y_train.shape, X_test.shape)
# runs for 10 seconds
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
num_classes = len(classes)
samples = 8
# predict labels for batch_size number of test images at a time.
batch_size = 10
# k = 3
k = 1
classifier = simple_knn()
classifier.train(X_train, y_train)

# runs for 13 minutes
predictions = []

for i in range(int(len(X_test)/(2*batch_size))):
    # predicts from i * batch_size to (i+1) * batch_size
    print("Computing batch " + str(i+1) + "/" + str(int(len(X_test)/batch_size)) + "...")
    tic = time.time()
    predts = classifier.predict(X_test[i * batch_size:(i+1) * batch_size], k)
    toc = time.time()
    predictions = predictions + list(predts)
#     print("Len of predictions: " + str(len(predictions)))
    print("Completed this batch in " + str(toc-tic) + " Secs.")

print("Completed predicting the test data.")
# for y, cls in enumerate(classes):
#     idxs = np.nonzero([i == y for i in y_train])
#     idxs = np.random.choice(idxs[0], samples, replace=False)
#     for i, idx in enumerate(idxs):
#         plt_idx = i * num_classes + y + 1
#         plt.subplot(samples, num_classes, plt_idx)
#         plt.imshow(X_train[idx].reshape((28, 28)))
#         plt.axis("off")
#         if i == 0:
#             plt.title(cls)
#
# plt.show()
