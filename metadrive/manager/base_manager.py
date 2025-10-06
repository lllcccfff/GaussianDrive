import copy
from metadrive.constants import DEFAULT_AGENT

from gymnasium.spaces import Space

from metadrive.base_class.randomizable import Randomizable


class BaseManager(Randomizable):
    """
    Managers should be created and registered after launching BaseEngine
    """
    PRIORITY = 10  # the engine will call managers according to the priority

    def __init__(self):
        Randomizable.__init__(self, None)
        self.spawned_objects = {}

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
        self.clear_all_objects()

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
        self.clear_all_objects()

    def spawn_object(self, object_class, **kwargs):
        """
        Spawn one objects
        """
        object = object_class(**kwargs)
        self.spawned_objects[object.id] = object
        return object

    def clear_object(self, object_id):
        obj = self.spawned_objects.pop(object_id)
        obj.destroy()
        return obj

    def clear_all_objects(self):
        id_list = list(self.spawned_objects.keys())
         
        for obj_id in id_list:
            self.clear_object(obj_id)
        self.spawned_objects = {}

    def get_metadata(self):
        """
        This function will store the metadata of each manager before the episode start, usually, we put some raw real
        world data in it, so that we won't lose information
        """
        assert self.episode_step == 0, "This func can only be called after env.reset() without any env.step() called"
        return {}
