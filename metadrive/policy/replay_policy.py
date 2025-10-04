import logging

from metadrive.policy.base_policy import BasePolicy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReplayTrafficParticipantPolicy(BasePolicy):
    """
       Replay policy from Real data. For adding new policy, overwrite get_trajectory_info()
       This policy is designed for Waymo Policy by default
       """
        
    def reset(self, object, seed, tracking, **kwargs):
        super().reset(object, seed, tracking, **kwargs)
        self.controller.set_kinematic(True)

        frame_list = sorted(self.trajectory.keys())
        self.terminate_frame = frame_list[-1]
        
    def act(self, *args, **kwargs):

        info = self.trajectory[self.step_manager.current_frame]

        if not bool(info["valid"]):
            return None  # Return None action so the base vehicle will not overwrite the steering & throttle

        return info
    
    @property
    def is_arrive(self):
        return self.step_manager < self.terminate_frame
