import tensorflow as tf
from keras import layers

from typing import Tuple

from ..utility_constants import BOARD_SIZE

NUM_ACTIONS = BOARD_SIZE[1]

class ActorCritic(tf.keras.Model):
    """Combined actor-critic network"""
    def __init__(
            self,
            num_hidden_units: int):
        """Initialize."""
        super().__init__()

        self.common = layers.Dense(num_hidden_units, activation=layers.LeakyReLU(alpha=0.3))
        self.actor = layers.Dense(NUM_ACTIONS)
        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        return self.actor(x), self.critic(x)
