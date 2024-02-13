import numpy as np

BOARD_SIZE = (6,7)
IN_A_ROW = 4

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