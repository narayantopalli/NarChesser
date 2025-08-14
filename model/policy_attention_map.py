import numpy as np


def get_attention_map() -> np.array:
    attention_map = np.zeros(shape=(64, 64))
    for i in range(64):
        rank = i // 8
        file = i % 8
        # rook moves
        for r in range(8):
            attention_map[i][8 * r + file] = 1
        for f in range(8):
            attention_map[i][8 * rank + f] = 1

        # bishop moves
        for f in range(8):
            r = (f - file) + rank
            if r in range(8):
                attention_map[i][8 * r + f] = 1
            r = -(f - file) + rank
            if r in range(8):
                attention_map[i][8 * r + f] = 1

        # knight moves
        knight = [2, 2, 1, -1, -2, -2, -1, 1]
        for d in range(8):
            r = rank + knight[d]
            f = file + knight[d - 2]
            if r in range(8) and f in range(8):
                attention_map[i][8 * r + f] = 1

        attention_map[i][8 * rank + file] = -1
    return attention_map
