import matplotlib.pyplot as plt
import random


targets: dict[int, str] = {
    0: "t-shirt/top",
    1: "trouser",
    2: "pullover",
    3: "dress",
    4: "coat",
    5: "sandal",
    6: "shirt",
    7: "sneaker",
    8: "bag",
    9: "ankle boot",
}


def load_mnist(path, kind="train"):
    import os
    import gzip
    import numpy as np

    labels_path = os.path.join(path, "%s-labels-idx1-ubyte.gz" % kind)
    images_path = os.path.join(path, "%s-images-idx3-ubyte.gz" % kind)

    with gzip.open(labels_path, "rb") as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, "rb") as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(
            len(labels), 784
        )

    return images, labels


def main():
    x_train, y_train = load_mnist("fashion-mnist/data/fashion/", kind="train")
    x_test, y_test = load_mnist("fashion-mnist/data/fashion", kind="t10k")

    i = random.randint(0, x_train.shape[0])
    plt.imshow(x_train[i].reshape(28, 28), cmap="gray")
    plt.title(f"This is a {targets.get(y_train[i])}")
    plt.show()


if __name__ == "__main__":
    main()
