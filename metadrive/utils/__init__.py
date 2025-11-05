from metadrive.utils.config import Config, merge_config, merge_config_with_unknown_keys
from metadrive.utils.coordinates_shift import metadrive_vector, panda_vector
from metadrive.utils.math import Vector, clip, distance_greater, norm, safe_clip, safe_clip_for_small_array
from metadrive.utils.random_utils import get_np_random, random_string
from metadrive.utils.registry import get_metadrive_class
from metadrive.utils.utils import (
    concat_step_infos,
    import_pygame,
    is_mac,
    is_win,
    merge_dicts,
    recursive_equal,
    setup_logger,
    time_me,
)
