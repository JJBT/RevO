class StopAtStep:
    def __init__(self, last_step):
        self.last_step = last_step

    def __call__(self, state):
        if state.step < self.last_step:
            return False
        else:
            return True


class NoStopping:
    def __call__(self, state):
        return False
