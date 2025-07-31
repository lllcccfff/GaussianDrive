import copy
from metadrive.engine.engine_utils import get_global_config
from metadrive.constants import DEFAULT_AGENT

from gymnasium.spaces import Space

from metadrive.base_class.randomizable import Randomizable


class BaseManager(Randomizable):
    """
    Managers should be created and registered after launching BaseEngine
    """
    PRIORITY = 10  # the engine will call managers according to the priority

    def __init__(self):
        from metadrive.engine.engine_utils import get_engine, engine_initialized
        assert engine_initialized(), "You should not create manager before the initialization of BaseEngine"
        # self.engine = get_engine()
        Randomizable.__init__(self, get_engine().global_random_seed)
        self.spawned_objects = {}
        self._object_policies = {}

    @property
    def episode_step(self):
        """
        Return how many steps are taken from env.reset() to current step
        Returns:

        """
        return self.engine.episode_step

    def before_step(self, *args, **kwargs) -> dict:
        """
        Usually used to set actions for all elements with their policies
        """
        return dict()

    def step(self, *args, **kwargs):
        pass

    def after_step(self, *args, **kwargs) -> dict:
        """
        Update state for this manager after system advancing dt
        """
        return dict()

    def before_reset(self):
        """
        Update episode level config to this manager and clean element or detach element
        """
        self.clear_objects([object_id for object_id in self.spawned_objects.keys()])
        self.spawned_objects = {}

    def reset(self):
        """
        Generate objects according to some pre-defined rules
        """
        pass

    def after_reset(self):
        """
        Usually used to record information after all managers called reset(),
        Since reset() of managers may influence each other
        """
        pass

    def destroy(self):
        """
        Destroy manager
        """
        # self.engine = None
        super(BaseManager, self).destroy()
        self.clear_objects(list(self.spawned_objects.keys()), force_destroy=True)
        self.spawned_objects = None

    def spawn_object(self, object_class, **kwargs):
        """
        Spawn one objects
        """
        object = object_class(**kwargs)
        self.spawned_objects[object.id] = object
        return object

    def clear_object(self, object_id):
        policy = self._object_policies.pop(object_id)
        policy.destroy()
        obj = self.spawned_objects.pop(object_id)
        obj.destroy()
        return obj

    def add_policy(self, object_id, policy_class, *policy_args, **policy_kwargs):
        policy = policy_class(*policy_args, **policy_kwargs)
        self._object_policies[object_id] = policy
        return policy

    def get_policy(self, object_id):
        return self._object_policies[object_id]

    def has_policy(self, object_id, policy_cls=None):
        return object_id in self._object_policies

    def get_state(self):
        """This function will be called by RecordManager to collect manager state, usually some mappings"""
        return {"spawned_objects": {name: v.class_name for name, v in self.spawned_objects.items()}}

    def set_state(self, state: dict, old_name_to_current=None):
        """
        A basic function for restoring spawned objects mapping
        """
        assert self.episode_step == 0, "This func can only be called after env.reset() without any env.step() called"
        if old_name_to_current is None:
            old_name_to_current = {key: key for key in state.keys()}
        spawned_objects = state["spawned_objects"]
        ret = {}
        for name, class_name in spawned_objects.items():
            current_name = old_name_to_current[name]
            name_obj = self.engine.get_objects([current_name])
            assert current_name in name_obj and name_obj[current_name
                                                         ].class_name == class_name, "Can not restore mappings!"
            ret[current_name] = name_obj[current_name]
        self.spawned_objects = ret

    @property
    def class_name(self):
        return self.__class__.__name__

    @property
    def engine(self):
        from metadrive.engine.engine_utils import get_engine
        return get_engine()

    def get_metadata(self):
        """
        This function will store the metadata of each manager before the episode start, usually, we put some raw real
        world data in it, so that we won't lose information
        """
        assert self.episode_step == 0, "This func can only be called after env.reset() without any env.step() called"
        return {}

    @property
    def global_config(self):
        return get_global_config()
