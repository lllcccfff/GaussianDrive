from metadrive.component.traffic_participants.base_traffic_participant import BaseTrafficParticipant
from metadrive.constants import MetaDriveType


class Cyclist(BaseTrafficParticipant):
    MASS = 80  # kg
    TYPE_NAME = MetaDriveType.CYCLIST
