class StepCounter:
    def __init__(self, step_size, ):
        self.step_size = step_size

    def reset(self, timestamp_range, **kwargs):
        self.begin_timestamp = timestamp_range[0]
        self.end_timestamp = timestamp_range[1]
        self.eposide_step = 0
    
    def step(self):
        self.eposide_step += 1
        # assert self.current_timestamp > self.end_timestamp, "Counter exceed."
    
    @property
    def current_timestamp(self):
        return self.begin_timestamp + self.eposide_step * self.step_size