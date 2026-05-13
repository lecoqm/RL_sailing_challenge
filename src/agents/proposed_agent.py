try:
    from evaluator.base_agent import BaseAgent
except Exception:
    from agents.base_agent import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.np_random = np.random.default_rng(0)
        self.model = ...

    def act(self, observation):
        image, vector = make_cnn_input(observation)
        logits = self.model(image, vector)
        action = argmax(logits)
        action = safety_shield(observation, action)
        return int(action)

    def reset(self):
        pass

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)