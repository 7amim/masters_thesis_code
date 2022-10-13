import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import math
from helpers import featureExtractor
from helpers import timefunc as tf

def drop_index_from_window(window, index, stics=True):
    window.x = window.x.drop(index)
    window.y = window.y.drop(index)
    window.ts_utc = window.ts_utc.drop(index)
    if stics:
        window.geofence_type = window.geofence_type.drop(index)
        window.room = window.room.drop(index)
        window.confidence = window.confidence.drop(index)
    return window

''' Uses a hard threshold to filter points based on step-distance '''
def filter_points_by_distance(window):
    x, y, ts, ang = [], [], [], []
    prev_x, prev_y, prev_ts, prev_ang = 0, 0, window.ts_utc.iloc[0], 0
    # angles = featureExtractor.get_avg_angle(window)
    current_index = window.x.index[0]
    for idx, curr_x, curr_y, curr_ts, in zip(window.x.index, window.x, window.y, window.ts_utc):
        if math.hypot(curr_x - prev_x, curr_y - prev_y) > DIST_THRESH:
            window = drop_index_from_window(window, idx)
        prev_x = curr_x
        prev_y = curr_y
        prev_ts = curr_ts
    return window

def filter_points_by_confidence(window):
    previous_confidence = window.confidence.iloc[0]
    for idx, current_confidence in zip(window.confidence.index, window.confidence):
        if current_confidence < 0.5*previous_confidence or current_confidence <= 0.4:
            window = drop_index_from_window(window, idx)
        previous_confidence = current_confidence
    return window

''' Uses a cyclic window 5 points to calculate the mean speed of the participant. If the speed 
exceeds a threshold value (N times the mean speed), then it is an outlier that should be removed
'''
def filter_points_by_mean_speed(window, stics=True):
    SIZE = 6
    THRESHOLD = 4
    filtered_x, filtered_y, filtered_ts = [], [], []
    # print(window)
    prev_x, prev_y, prev_ts = window.x.iloc[SIZE-1], window.y.iloc[SIZE-1], window.ts_utc.iloc[SIZE-1]
    moving_window = [list(window.x)[:SIZE-1], list(window.y)[:SIZE-1], list(window.ts_utc)[:SIZE-1]]
    for idx, x, y, ts_utc in zip(window.x[SIZE:].index, window.x[SIZE:], window.y[SIZE:], window.ts_utc[SIZE:]):
        if prev_ts != ts_utc:
            if len(moving_window[0]) < 5:
                moving_window[0].append(x)
                moving_window[1].append(y)
                moving_window[2].append(ts_utc)
            # print('MOVING WINDOW:', moving_window[0], moving_window[1], moving_window[2])
            avg_speed_in_window = featureExtractor.calculate_average_speed_in_moving_window(moving_window, stics=stics)
            speed_tuple = (x - prev_x, y - prev_y, tf.utc_to_seconds(ts_utc, stics=stics) - tf.utc_to_seconds(prev_ts, stics=stics))
            speed = math.sqrt((speed_tuple[0]/(speed_tuple[2]+1))**2 + (speed_tuple[1] / (speed_tuple[2]+1))**2) 

            # print(avg_speed_in_window, speed)
            if speed / THRESHOLD > avg_speed_in_window:
                # print("SPEED TUPLE ", speed_tuple)
                # print(avg_speed_in_window, speed)
                # print('MOVING WINDOW:', moving_window[0], moving_window[1], moving_window[2])
                # print("DROPPING POINT")
                window = drop_index_from_window(window, idx, stics=stics)

            for sublist in moving_window:
                del sublist[0]
            prev_x = x
            prev_y = y 
            prev_ts = ts_utc
    return window