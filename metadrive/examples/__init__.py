from metadrive.constants import TerminationState


def get_terminal_state(info):
    if info[TerminationState.CRASH_VEHICLE]:
        state = "Crash Vehicle"
    elif info[TerminationState.OUT_OF_ROAD]:
        state = "Out of Road"
    elif info[TerminationState.SUCCESS]:
        state = "Success"
    else:
        state = "Max Step"
    return state
