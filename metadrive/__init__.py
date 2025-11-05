import os

from metadrive.envs import BaseEnv, ScenarioEnv
from metadrive.utils.registry import get_metadrive_class

MetaDrive_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

from rich.console import Console

CONSOLE = Console()
