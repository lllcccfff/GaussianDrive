import os

from metadrive.envs import (
    ScenarioEnv, BaseEnv
)
from metadrive.utils.registry import get_metadrive_class

MetaDrive_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
