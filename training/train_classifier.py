"""Fine-tune MobileNetV2 for livestock breed classification."""

import argparse
import json
from pathlib import Path

import tensorflow as tf


def build_model(class_count: int) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )

    # Freeze the pretrained MobileNetV2 layers
    base.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))

    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    outputs = tf.keras.layers.Dense(
        class_count,
        activation="softmax"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MobileNetV2 for livestock breed classification."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Folder containing one subfolder per breed.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("backend/models/breed_classifier.keras"),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    print(f"Loading dataset from: {args.data_dir}")

    # Load 80% of images for training
    train = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(224, 224),
        batch_size=args.batch_size,
    )

    # Load 20% of images for validation
    validation = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(224, 224),
        batch_size=args.batch_size,
        class_names=train.class_names,
    )

    class_names = train.class_names

    print("\nDetected breeds:")
    for index, name in enumerate(class_names):
        print(f"{index}: {name}")

    print(f"\nTotal breeds: {len(class_names)}")

    # Data augmentation
    augmenter = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.1),
        ]
    )

    train = train.map(
        lambda images, labels: (
            augmenter(images, training=True),
            labels,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    train = train.prefetch(tf.data.AUTOTUNE)
    validation = validation.prefetch(tf.data.AUTOTUNE)

    # Build model
    model = build_model(len(class_names))

    print("\nStarting training...\n")

    # Train model
    model.fit(
        train,
        validation_data=validation,
        epochs=args.epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=3,
                restore_best_weights=True,
            )
        ],
    )

    # Create output folder
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Save trained model
    model.save(args.output)

    # Save breed names
    labels_path = args.output.with_name("breed_labels.json")

    labels_path.write_text(
        json.dumps(class_names, indent=2),
        encoding="utf-8",
    )

    print("\nTraining completed successfully!")
    print(f"Saved model: {args.output}")
    print(f"Saved labels: {labels_path}")


if __name__ == "__main__":
    main()