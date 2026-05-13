class CNNActorCritic(nn.Module):
    def __init__(self, n_actions=9):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.vector_net = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
        )

        self.shared = nn.Sequential(
            nn.Linear(64 * 8 * 8 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1)

    def forward(self, image, vector):
        z_img = self.cnn(image)
        z_vec = self.vector_net(vector)
        z = torch.cat([z_img, z_vec], dim=1)
        z = self.shared(z)
        logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)
        return logits, value
    

def shaped_reward(prev_obs, obs, env_reward, terminated, truncated, info):
    prev_pos = prev_obs[0:2]
    pos = obs[0:2]

    prev_dist = np.linalg.norm(GOAL - prev_pos)
    dist = np.linalg.norm(GOAL - pos)

    progress = prev_dist - dist
    vertical_progress = pos[1] - prev_pos[1]

    reward = 0.0
    reward += 0.4 * progress
    reward += 0.05 * vertical_progress
    reward -= 0.01

    if info.get("is_stuck", False):
        reward -= 30.0

    if env_reward > 0:
        reward += 100.0

    return float(reward)