import logging

from metadrive.policy.base_policy import BasePolicy
from metadrive.scenario.parse_object_state import parse_object_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReplayTrafficParticipantPolicy(BasePolicy):
    """
       Replay policy from Real data. For adding new policy, overwrite get_trajectory_info()
       This policy is designed for Waymo Policy by default
       """
    DEBUG_MARK_COLOR = (3, 140, 252, 255)

    def __init__(self, control_object, track, random_seed=None):
        super(ReplayTrafficParticipantPolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        self.start_index = 0
        self._velocity_local_frame = False
        self.traj_info = self.parse_track_infos(track)
        
        self.control_object.set_kinematic(True)

    @property
    def is_current_step_valid(self):
        return self.traj_info[self.episode_step] is not None

    def parse_track_infos(self, track):
        return track

    def act(self, *args, **kwargs):

        info = self.traj_info[index]

        # Before step
        # Warning by LQY: Don't call before step here! Before step should be called by manager
        # action = self.traj_info[int(self.episode_step)].get("action", None)
        # self.control_object.before_step(action)

        if not bool(info["valid"]):
            return None  # Return None action so the base vehicle will not overwrite the steering & throttle

        if "throttle_brake" in info:
            if hasattr(self.control_object, "set_throttle_brake"):
                self.control_object.set_throttle_brake(float(info["throttle_brake"].item()))
        if "steering" in info:
            if hasattr(self.control_object, "set_steering"):
                self.control_object.set_steering(float(info["steering"].item()))

        if "transform" in info:
            self.control_object.set_transform(info["transform"])
        else:
            self.control_object.set_position(info["position"])
            self.control_object.set_heading_theta(info["heading"])

        self.control_object.set_velocity(info["velocity"])
        self.control_object.set_angular_velocity(info["angular_velocity"])

        return None  # Return None action so the base vehicle will not overwrite the steering & throttle