import datetime
from enum import Enum
import time
from sklearn.linear_model import LogisticRegression, SGDClassifier
import pickle
import os

CACHE_DIR = "cache"

TARGETS: dict[int, str] = {
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


class Models(Enum):
    SGDC = 1
    LR = 2


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


def create_or_load_model(
    selected_model: Models, force_train: bool = False
) -> LogisticRegression | SGDClassifier:
    file_suffix: str | None = None
    if selected_model == Models.SGDC:
        file_suffix = "SGDC"
    elif selected_model == Models.LR:
        file_suffix = "LR"

    if file_suffix is None:
        raise Exception

    filePath = os.path.join(CACHE_DIR, f"fashion_mnist_{file_suffix}")
    if not force_train and os.path.isfile(filePath):
        return pickle.load(open(filePath, "rb"))

    x_train, y_train = load_mnist("fashion-mnist/data/fashion/", kind="train")

    x_train = x_train / 255

    model: SGDClassifier | LogisticRegression | None = None
    if selected_model == Models.SGDC:
        model = SGDClassifier()
    elif selected_model == Models.LR:
        model = LogisticRegression(multi_class="ovr", max_iter=10**10)

    if model is None:
        raise Exception

    model.fit(x_train, y_train)

    pickle.dump(model, open(filePath, "wb"))

    return model


def main():
    x_test, y_test = load_mnist("fashion-mnist/data/fashion", kind="t10k")
    x_test = x_test / 255

    start = time.perf_counter()
    model = create_or_load_model(Models.LR, force_train=True)
    end = time.perf_counter()

    y_predicted = model.predict(x_test)

    success_log: list[bool] = []
    for predicted, expected in zip(y_predicted, y_test):
        if predicted == expected:
            success_log.append(True)
        else:
            success_log.append(False)

    print(
        f"LogisticRegression model's training took {datetime.timedelta(seconds=(end - start))}"
    )
    print(
        f"LogisticRegression model's success rate: {success_log.count(True)  / len(success_log)}"
    )

    start = time.perf_counter()
    model = create_or_load_model(Models.SGDC, force_train=True)
    end = time.perf_counter()

    y_predicted = model.predict(x_test)

    success_log: list[bool] = []
    for predicted, expected in zip(y_predicted, y_test):
        if predicted == expected:
            success_log.append(True)
        else:
            success_log.append(False)

    print(
        f"SGDClassifier model's training took {datetime.timedelta(seconds=(end - start))}"
    )
    print(
        f"SGDClassifier model's success rate: {success_log.count(True)  / len(success_log)}"
    )


if __name__ == "__main__":
    main()
