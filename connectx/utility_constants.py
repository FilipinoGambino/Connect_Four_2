import numpy as np
from scipy.signal import convolve2d

BOARD_SIZE = (6,7)
IN_A_ROW = 4
GAME_STATUS = dict(
    NO_WINNING_MOVE=1,
    UNDEFENDED_POSITION=2,
    ACTIVE_PLAYER_WINS=3,
)

horizontal_kernel = np.ones([1, IN_A_ROW], dtype=np.uint8)
vertical_kernel = np.transpose(horizontal_kernel)
diag1_kernel = np.eye(IN_A_ROW, dtype=np.uint8)
diag2_kernel = np.fliplr(diag1_kernel)

VICTORY_KERNELS = [
    horizontal_kernel,
    vertical_kernel,
    diag1_kernel,
    diag2_kernel,
]

