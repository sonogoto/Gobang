#!/usr/bin/env python3

import numpy as np


# 1. up flip(same as down flip)
def up_flip(x):
    total_row = x.shape[0]
    total_column = x.shape[1]
    y = np.zeros_like(x)
    for row_id in range(total_row//2):
        for col_id in range(total_column):
            y[row_id, col_id] = x[total_row-1-row_id, col_id]
            y[total_row-1-row_id, col_id] = x[row_id, col_id]
    return y


# 2. right flip(same as left flip)
def right_flip(x):
    total_row = x.shape[0]
    total_column = x.shape[1]
    y = np.zeros_like(x)
    for col_id in range(total_column//2):
        for row_id in range(total_row):
            y[row_id, col_id] = x[row_id, total_column-1-col_id]
            y[row_id, total_column-1-col_id] = x[row_id, col_id]
    return y


# 3. up-right flip(same as down-left flip)
def up_right_flip(x):
    total_row = x.shape[0]
    total_column = x.shape[1]
    y = np.zeros_like(x)
    assert total_row == total_column
    for row_id in range(total_row):
        for col_id in range(row_id):
            y[row_id, col_id] = x[col_id, row_id]
            y[col_id, row_id] = x[row_id, col_id]
    return y


# 4. down-right flip(same as up-left flip)
def down_right_flip(x):
    total_row = x.shape[0]
    total_column = x.shape[1]
    y = np.zeros_like(x)
    assert total_row == total_column
    for row_id in range(total_row):
        for col_id in range(total_row-1-row_id):
            y[row_id, col_id] = x[total_column-1-col_id, total_row-1-row_id]
            y[total_column-1-col_id, total_row-1-row_id] = x[row_id, col_id]
    return y


# 5. 90 degree rotating
def rotate_90(x):
    return up_flip(down_right_flip(x))


# 6. 180 degree rotating
def rotate_180(x):
    return right_flip(up_flip(x))


# 7. 270 degree rotating
def rotate_270(x):
    return right_flip(down_right_flip(x))


# 8. identical
def identical(x):
    return x


NUM_OF_ROTATION = 8
            
INVERSION = [up_flip, 
             right_flip, 
             up_right_flip, 
             down_right_flip, 
             rotate_270, 
             rotate_180, 
             rotate_90, 
             identical]

ROTATION = [up_flip, 
            right_flip, 
            up_right_flip, 
            down_right_flip, 
            rotate_90, 
            rotate_180, 
            rotate_270, 
            identical]
