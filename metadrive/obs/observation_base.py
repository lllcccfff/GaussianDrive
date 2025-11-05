from abc import ABC
from copy import deepcopy

import gymnasium as gym
import numpy as np

from metadrive.utils.logger import get_logger

logger = get_logger()


class BaseObservation(ABC):
    """
    BaseObservation Class. Observation should implement all abstracted methods
    """

    INITIALIZED = False

    def __init__(self, config):
        # assert not engine_initialized(), "Observations can not be created after initializing the simulation"
        if not self.INITIALIZED:
            self.INITIALIZED = True
            self.config = deepcopy(config)
            self.current_observation = None

    @property
    def observation_space(self):
        raise NotImplementedError

    def observe(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        pass

    def destroy(self):
        """
        Clear allocated memory
        """
        pass
        # Config.clear_nested_dict(self.config)
        # self.config = None


class DummyObservation(BaseObservation):
    """
    Fake Observation class, can be used as placeholder
    """

    def __init__(self, config=None):
        super(DummyObservation, self).__init__(config)
        logger.warning("You are using DummyObservation which doesn't collect information from the environment.")

    @property
    def observation_space(self):
        return gym.spaces.Box(-0.0, 1.0, shape=(1,), dtype=np.float32)

    def observe(self, *args, **kwargs):
        return np.array([0])


class DefaultObservation(BaseObservation):
    """
    Default Observation class that returns None
    """

    def __init__(self, config=None):
        super(DefaultObservation, self).__init__(config or {})

    @property
    def observation_space(self):
        return gym.spaces.Box(-0.0, 1.0, shape=(1,), dtype=np.float32)

    def observe(self, *args, **kwargs):
        return None
