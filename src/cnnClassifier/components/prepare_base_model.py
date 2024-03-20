import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    """
    A class to prepare and update a base model for TensorFlow.

    Attributes:
        config (PrepareBaseModelConfig): An instance of PrepareBaseModelConfig containing configuration parameters.
    """

    def __init__(self, config: PrepareBaseModelConfig):
        """
        Initializes a PrepareBaseModel instance.

        Args:
            config (PrepareBaseModelConfig): Configuration parameters for preparing the base model.
        """
        self.config = config

    def get_base_model(self):
        """
        Retrieves the base model specified in the configuration and saves it.
        """
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        Prepares a full model based on the provided parameters.

        Args:
            model (tf.keras.Model): The base model to build upon.
            classes (int): The number of output classes.
            freeze_all (bool): If True, freezes all layers of the model.
            freeze_till (int or None): If provided, freezes layers up to this index (exclusive).
            learning_rate (float): The learning rate for the optimizer.

        Returns:
            tf.keras.Model: The prepared full model.
        """
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:freeze_till]:
                model.trainable = False
            
        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation='softmax'
        )(flatten_in)
        
        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )
        full_model.summary()
        return full_model
    
    def update_base_model(self):
        """
        Updates the base model by preparing a full model and saving it.
        """
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )
        self.save_model(path=self.config.updated_base_mode_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Saves the provided TensorFlow model to the specified path.

        Args:
            path (Path): The file path to save the model.
            model (tf.keras.Model): The TensorFlow model to be saved.
        """
        model.save(path)
