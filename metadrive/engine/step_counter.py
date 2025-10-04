class StepCounter:
    def __init__(self):
        pass

    def reset(self, frame_range):
        self.begin_frame = frame_range[0]
        self.end_frame = frame_range[1] 
        self.eposide_step = 0
    
    def step(self):
        self.eposide_step += 1

        assert self.current_frame > self.end_frame, "Counter exceed."
    
    @property
    def current_frame(self):
        return self.begin_frame + self.eposide_step