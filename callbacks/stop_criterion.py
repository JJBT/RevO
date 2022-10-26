class StopAtStep:
    def __init__(self, last_step):
        self.last_step = last_step

    def __call__(self, state):
        state_step = state if isinstance(state, int) else state.step
        return state_step >= self.last_step


class NoStopping:
    def __call__(self, state):
        return False
