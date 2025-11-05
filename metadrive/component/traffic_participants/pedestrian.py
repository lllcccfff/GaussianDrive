from metadrive.component.traffic_participants.base_traffic_participant import BaseTrafficParticipant
from metadrive.constants import MetaDriveType


class Pedestrian(BaseTrafficParticipant):
    MASS = 70  # kg
    TYPE_NAME = MetaDriveType.PEDESTRIAN
