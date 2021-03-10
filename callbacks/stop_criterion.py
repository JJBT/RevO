class StopAtStep:
    def __init__(self, last_step):
        self.last_step = last_step

    def __call__(self, state):
        if isinstance(state, int):
            state_step = state
        else:
            state_step = state.step

        if state_step < self.last_step:
            return False
        else:
            return True


class NoStopping:
    def __call__(self, state):
        return False
