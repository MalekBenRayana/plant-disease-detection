
import  random
import  os
import  json
import  numpy as np
import  matplotlib.pyplot as plt
import  matplotlib.image as mpimg
from  zipfile import ZipFile
from  PIL import Image

import tensorflow  as tf
from tensorflow.keras.preprocessing.image  import ImageDataGenerator
from tensorflow.keras  import layers, models

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

def load_kaggle_credentials(kaggle_json_path) :
    with open(kaggle_json_path, 'r' ) as f:
        return  json.load(f)

def setup_kaggle_api(kaggle_credentials) :
    os.environ['KAGGLE_USERNAME'] =  kaggle_credentials["username"]
    os.environ['KAGGLE_KEY'] =  kaggle_credentials["key"]

def main() :
    # Paths and Configuration
    BASE_DIR =  'plantvillage dataset/color'  # Adjust if your dataset structure is different
    IMAGE_SIZE =  224
    BATCH_SIZE =  32
    EPOCHS =  5


    # Verify dataset directories
    print("Dataset directories and sample files:" )
    dataset_root =  "plantvillage dataset"
    for subdir in  ['segmented', 'color', 'grayscale']:
        path =  os.path.join(dataset_root, subdir)
        if os.path.exists(path) :
            print(f"\n{subdir} ({len(os.listdir(path))} files):")
            sample = os.listdir(path)[:5]
            print(sample)
        else:
            print(f"\n{subdir} directory does not exist.")

    # Example Image Display (Optional)
    sample_image_path = os.path.join(BASE_DIR, 'Apple___Cedar_apple_rust', 
                                     '025b2b9a-0ec4-4132-96ac-7f2832d0db4a___FREC_C.Rust 3655.JPG')
    if os.path.exists(sample_image_path):
        img = mpimg.imread(sample_image_path)
        print(f"\nSample Image Shape: {img.shape}")
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    else:
        print(f"\nSample image '{sample_image_path}' not found.")

    # Image Data Generators
    data_gen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2  # 20% for validation
    )

    # Train Generator
    train_generator = data_gen.flow_from_directory(
        BASE_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        subset='training',
        class_mode='categorical'
    )

    
    validation_generator = data_gen.flow_from_directory(
        BASE_DIR,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        subset='validation',
        class_mode='categorical'
    )

    # Model Definition
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        layers.MaxPooling2D(2, 2),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(train_generator.num_classes, activation='softmax')
    ])

    # Model Summary
    model.summary()

    # Compile the Model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Training the Model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE
    )

    # Model Evaluation
    print("\nEvaluating model on validation data...")
    val_loss, val_accuracy = model.evaluate(validation_generator, steps=validation_generator.samples // BATCH_SIZE)
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

    # Plot Training & Validation Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()

    # Plot Training & Validation Loss
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()

    # Save Class Indices
    class_indices = {v: k for k, v in train_generator.class_indices.items()}
    with open('class_indices.json', 'w') as f:
        json.dump(class_indices, f)
    print("\nClass indices saved to 'class_indices.json'.")

    # Save the Model
    model_save_path = 'plant_disease_prediction_model.h5'
    model.save(model_save_path)
    print(f"Model saved to '{model_save_path}'.")

if __name__ == "__main__":
    main()
