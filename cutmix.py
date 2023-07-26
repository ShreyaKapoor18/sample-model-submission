#%%
import torch
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from matplotlib import pyplot, image

#%%
def cutmix_data_augmentation(images, labels, alpha=1.0):
    """
    Applies CutMix data augmentation to a batch of images and their corresponding labels.

    Args:
        images (Tensor): Batch of input images of shape (batch_size, C, H, W).
        labels (Tensor): Batch of corresponding labels of shape (batch_size,).
        alpha (float): Hyperparameter controlling the strength of CutMix. Default is 1.0.

    Returns:
        (Tensor, Tensor): Mixed images and their corresponding labels.
    """
    batch_size, _, height, width = images.shape

    # Randomly choose a sample from the batch to mix with
    idx = torch.randperm(batch_size)
    img_a = images.copy()
    img_b = images[idx]

    # Generate random bounding box
    lam = np.random.beta(alpha, alpha)
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = np.int(width * cut_ratio)
    cut_h = np.int(height * cut_ratio)
    cx = np.random.randint(width)
    cy = np.random.randint(height)

    # Apply CutMix to images
    images[:, :, max(0, cy - cut_h // 2):min(height, cy + cut_h // 2), 
                  max(0, cx - cut_w // 2):min(width, cx + cut_w // 2)] = img_b[:, :, max(0, cy - cut_h // 2):min(height, cy + cut_h // 2), 
                                                                                   max(0, cx - cut_w // 2):min(width, cx + cut_w // 2)]
    
    # Calculate new labels using the mix ratio
    labels_a = labels
    labels_b = labels[idx]
    new_labels = lam * labels_a + (1.0 - lam) * labels_b

    return images, new_labels

#%%

def create_efficientnet_b0(num_classes):
    """
    Create a custom EfficientNet-B0 model with a given number of output classes.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        A TensorFlow Keras model instance of EfficientNet-B0.
    """
    # Load the pre-trained EfficientNet-B0 model (without the top classification layers)
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

    # Add custom classification layers on top
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)  # You can customize the number of units in this layer
    outputs = Dense(num_classes, activation='softmax')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=outputs)

    return model

#%%
# Example usage:
# Create an EfficientNet-B0 model with 10 output classes
num_classes = 10
efficientnet_b0_model = create_efficientnet_b0(num_classes)

# Compile the model with an optimizer, loss function, and metrics
efficientnet_b0_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Now you can use the model for training or evaluation on your dataset
#%%
import brainscore
neural_data = brainscore.get_assembly(name="dicarlo.MajajHong2015.public")
stimulus_set = neural_data.attrs['stimulus_set']

X = []
for i in range(len(stimulus_set)):
    stimulus_path = stimulus_set.get_stimulus(stimulus_set['stimulus_id'][i])
    img = image.imread(stimulus_path)
    X.append(img)

# %%
