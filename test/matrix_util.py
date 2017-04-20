import numpy as np


def get_fi_stream(filepath):
    return open(filepath, mode='rt').readlines()


def get_block(all_lines_list):
    block_lists = []
    global curr_row_num
    curr_row_num = 0
    while curr_row_num < len(all_lines_list):
        atom_num = int(all_lines_list[curr_row_num])
        curr_row_num += 2
        block_list = all_lines_list[curr_row_num: curr_row_num + atom_num]
        curr_row_num += atom_num
        block_lists.append(block_list)

    return block_lists


def char(block_list):
    char = []
    for item in block_list:
        f = item.split()
        if str(f[0]) == 'Zn':
            char.append(30)
        elif str(f[0]) == 'O':
            char.append(8)
        elif str(f[0]) == 'H':
            char.append(1)
    char = np.array(char)
    return char


def matrix(block_list):
    coord = coor(block_list)
    charge = char(block_list)
    n = len(block_list)
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                C[i][j] = 0.5 * charge[i] ** 2.4
            else:
                r1 = coord[i]
                r2 = coord[j]
                r12 = r1 - r2
                C[i][j] = charge[i] * charge[j] / np.sqrt(np.dot(r12, r12))
    print(C)


def coor(block_list):
    coord = []
    for item in block_list:
        f = item.split()
        c1 = f[1:]
        e = [float(x) for x in c1]
        coord.append(e)
    coord = np.array(coord)
    return coord


if __name__ == '__main__':
    block_lists = get_block(get_fi_stream('dump.xyz'))
    for block_list in block_lists:
        matrix(block_list)
