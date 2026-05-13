import numpy as np

GRID = 128
CNN_GRID = 32
BLOCK = GRID // CNN_GRID

GOAL = np.array([64, 127], dtype=np.float32)


def block_mean(x):
    return x.reshape(CNN_GRID, BLOCK, CNN_GRID, BLOCK).mean(axis=(1, 3))


def block_max(x):
    return x.reshape(CNN_GRID, BLOCK, CNN_GRID, BLOCK).max(axis=(1, 3))


def make_cnn_input(observation):
    pos = observation[0:2].astype(np.float32)
    vel = observation[2:4].astype(np.float32)
    local_wind = observation[4:6].astype(np.float32)

    wind_flat_size = GRID * GRID * 2
    wind_field = observation[6:6 + wind_flat_size].reshape(GRID, GRID, 2)
    world = observation[6 + wind_flat_size:].reshape(GRID, GRID)

    wind_x = block_mean(wind_field[:, :, 0]) / 10.0
    wind_y = block_mean(wind_field[:, :, 1]) / 10.0
    obstacle = block_max(world)

    boat = np.zeros((CNN_GRID, CNN_GRID), dtype=np.float32)
    bx = int(np.clip(pos[0] / BLOCK, 0, CNN_GRID - 1))
    by = int(np.clip(pos[1] / BLOCK, 0, CNN_GRID - 1))
    boat[by, bx] = 1.0

    goal = np.zeros((CNN_GRID, CNN_GRID), dtype=np.float32)
    gx = int(np.clip(GOAL[0] / BLOCK, 0, CNN_GRID - 1))
    gy = int(np.clip(GOAL[1] / BLOCK, 0, CNN_GRID - 1))
    goal[gy, gx] = 1.0

    risk = obstacle.copy()
    for _ in range(2):
        padded = np.pad(risk, 1)
        risk = np.maximum.reduce([
            padded[1:-1, 1:-1],
            padded[:-2, 1:-1],
            padded[2:, 1:-1],
            padded[1:-1, :-2],
            padded[1:-1, 2:],
        ])

    image = np.stack([wind_x, wind_y, obstacle, boat, goal, risk], axis=0)

    vector = np.array([
        pos[0] / 127.0,
        pos[1] / 127.0,
        vel[0] / 8.0,
        vel[1] / 8.0,
        local_wind[0] / 10.0,
        local_wind[1] / 10.0,
        np.linalg.norm(GOAL - pos) / 180.0,
    ], dtype=np.float32)

    return image.astype(np.float32), vector