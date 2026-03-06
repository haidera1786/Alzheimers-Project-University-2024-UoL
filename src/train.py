from data_loader import create_data_generators
from cnn_model import build_cnn_model
from resnet_model import build_resnet_model


TRAIN_DIR = "data/train"
VAL_DIR = "data/val"


def train_cnn():

    train_gen, val_gen = create_data_generators(TRAIN_DIR, VAL_DIR)

    model = build_cnn_model()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20
    )

    model.save("models/cnn_model.h5")

    return history


def train_resnet():

    train_gen, val_gen = create_data_generators(TRAIN_DIR, VAL_DIR)

    model = build_resnet_model()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20
    )

    model.save("models/resnet_model.h5")

    return history


if __name__ == "__main__":

    print("Training CNN Model...")
    train_cnn()

    print("Training ResNet Model...")
    train_resnet()
