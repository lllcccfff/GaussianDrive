import inspect
from typing import Dict, Any

import gymnasium as gym

from metadrive.obs.observation_base import BaseObservation


class AssemblyObservation(BaseObservation):
    """
    Compose multiple observers and return a dict of their observations.

    Config format example (under actor_config.observer_config):
      {
        'gaussian': {
            'observer_class': GaussianObservation,
            'clip_rgb': False,
            'stack_size': 3,
          },
        'navigation': {
            'observer_class': NavigationObservation,
            'navigating_type': 'destination_following',
          }
      }

    AssemblyObservation.reset(...) forwards only the kwargs accepted by each
    sub-observer's reset signature, so you can pass a superset of inputs
    (e.g., controller, render_fn, camera_params, trajdata_map, init_state,
    state, collector, seed, etc.) without worrying about per-observer kwargs.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config or {})
        # name -> instance
        self._observers: Dict[str, BaseObservation] = {}
        self._obs_cls = {}
        for name, sub_cfg in (config or {}).items():
            # Support both 'observer_class' and a common misspelling 'obsever_class'
            cls = sub_cfg.get("observer_class")
            if cls is None:
                raise ValueError(f"observer_class missing for sub-observer '{name}'")
            sub_cfg.pop("observer_class", None)
            self._observers[name] = cls(sub_cfg)
            self._obs_cls[name] = cls

    def reset(self, **kwargs):
        for obs in self._observers.values():
            obs.reset(**kwargs)

    @property
    def observation_space(self):
        spaces = {}
        for name, obs in self._observers.items():
            sub_space = obs.observation_space
            # Wrap plain dicts of spaces into gym.spaces.Dict
            if isinstance(sub_space, dict):
                spaces[name] = gym.spaces.Dict(sub_space)
            else:
                spaces[name] = sub_space
        return gym.spaces.Dict(spaces)

    def observe(self):
        ret = {}
        for name, obs in self._observers.items():
            ret[name] = obs.observe()
        return ret

    def destroy(self):
        super().destroy()
        for obs in self._observers.values():
            obs.destroy()
        self._observers.clear()
