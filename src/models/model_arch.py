import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_model(
    input_dim: int,
    num_classes: int,
    units_1=128,
    units_2=64,
    dropout_rate=0.3,
    l2_reg=0.001,
    learning_rate=0.001,
) -> tf.keras.Model:
    """
    Constructs a Multi-Layer Perceptron (MLP) for classification.

    Args:
        input_dim (int): Number of input features.
        num_classes (int): Number of output classes.
        units_1 (int): Neurons in the first hidden layer.
        units_2 (int): Neurons in the second hidden layer.
        dropout_rate (float): Dropout probability for regularization.
        l2_reg (float): L2 regularization factor.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        model (tf.keras.Model): Compiled Keras model.
    """

    # Define Input Layer
    inputs = layers.Input(shape=(input_dim,), name="Input_Layer")

    # Hidden Layer 1: Dense + Batch Norm + ReLU + Dropout
    x = layers.Dense(
        units_1, kernel_regularizer=regularizers.l2(l2_reg), name="Hidden_Layer_1"
    )(inputs)
    x = layers.BatchNormalization(name="Batch_Norm_1")(x)
    x = layers.Activation("relu", name="ReLU_1")(x)
    x = layers.Dropout(dropout_rate, name="Dropout_1")(x)

    # Hidden Layer 2: Dense + Batch Norm + ReLU + Dropout
    x = layers.Dense(
        units_2, kernel_regularizer=regularizers.l2(l2_reg), name="Hidden_Layer_2"
    )(x)
    x = layers.BatchNormalization(name="Batch_Norm_2")(x)
    x = layers.Activation("relu", name="ReLU_2")(x)
    x = layers.Dropout(dropout_rate, name="Dropout_2")(x)

    # Output Layer: Softmax for Multi-Class Classification
    outputs = layers.Dense(num_classes, activation="softmax", name="Output_Layer")(x)

    # Instantiate Model
    model = models.Model(
        inputs=inputs, outputs=outputs, name="Crash_Severity_Classifier"
    )

    # Compile Model
    # We use Adam optimizer and SparseCategoricalCrossentropy (since y is encoded as integers 0,1,2,3)
    # If y was one-hot, we would use CategoricalCrossentropy
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
