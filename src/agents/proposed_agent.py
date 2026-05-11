from evaluator.base_agent import BaseAgent

class MyAgent(BaseAgent):
    def act(self, observation):
        # observation = [x, y, vx, vy, wx, wy, ...wind_field..., ...world_map...]
        # Return an integer between 0 and 8 (direction or stay)
        ...

    def reset(self):
        # Called at the start of each episode
        ...

    def seed(self, seed=None):
        # Set random seed for reproducibility
        ...