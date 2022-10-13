import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import operator
import numpy as np
import traja
import math
import pandas as pd

from scipy.stats import gmean, hmean
from helpers import timefunc as tf

def calculate_average_speed_in_moving_window(moving_window, stics=True):
    dx_list = [j-i for i, j in zip(moving_window[0][:-1], moving_window[0][1:])]
    dy_list = [j-i for i, j in zip(moving_window[1][:-1], moving_window[1][1:])]
    dt_list = [tf.utc_to_seconds(j, stics=stics)-tf.utc_to_seconds(i, stics=stics) for i, j in zip(moving_window[2][:-1], moving_window[2][1:])]
    speed = []

    for dx, dy, dt in zip(dx_list, dy_list, dt_list):
        speed.append(np.sqrt(dx**2 + dy**2) / (dt+1))
    return np.mean(speed)

'''------------------------- ACTIVITY LEVELS ---------------------------- '''

def get_time_spent_walking(window, threshold = 0.15):
    # print(window.ts_utc)
    start_row = window.rows[0]
    prev_ts = window.ts_utc.iloc[0]
    prev_x = window.y.iloc[0]
    prev_y = window.y.iloc[0]
    total_time = 0
    for curr_x, curr_y, curr_ts in zip(window.x, window.y, window.ts_utc):
        if math.hypot(curr_x - prev_x, curr_y - prev_y) > threshold:
            total_time += (tf.utc_to_seconds(curr_ts) - tf.utc_to_seconds(prev_ts))
            # Replace previous values only if new values are different
        prev_x = curr_x
        prev_y = curr_y
        prev_ts = curr_ts
    return total_time

def get_total_distance(window, threshold = 0.15):
    total_distance = 0
    prev_x = window.x.iloc[0]
    prev_y = window.y.iloc[0]
    prev_ts = window.ts_utc.iloc[0]

    for curr_x, curr_y, in zip(window.x, window.y):
        distance = math.hypot(curr_x - prev_x, curr_y - prev_y)
        if math.hypot(curr_x - prev_x, curr_y - prev_y) > threshold:
            total_distance += distance
        prev_x = curr_x
        prev_y = curr_y

    return total_distance

'''Dwell time'''

def get_stationary_time(window, threshold=0.15):
    dt = tf.utc_to_seconds(window.ts_utc.iloc[-1]) - tf.utc_to_seconds(window.ts_utc.iloc[0])
    return dt - get_time_spent_walking(window, threshold=threshold)

def get_total_time_in_geolocation(dict_of_geofences, window):
    try: 
        duration = 0
        rooms = window.room.dropna()
        prev_ts = window.ts_utc.iloc[0]
        start_ts = window.ts_utc.iloc[0]
        prev_room = rooms.iloc[0]
        start_index = 0
        curr_index = 0
        if type(prev_room) is not str:
            start_index = 1
            prev_room = rooms.iloc[1]

        for ts, room in zip(window.ts_utc.iloc[start_index:], rooms.iloc[start_index:]):
            if room != prev_room or curr_index == len(rooms)-1:
                duration = tf.utc_to_seconds(prev_ts) - tf.utc_to_seconds(start_ts)
                dict_of_geofences = update_duration(dict_of_geofences, prev_room, duration)
                start_ts = ts
                duration = 0
            prev_room = room
            prev_ts = ts
            curr_index += 1
        return dict_of_geofences
    except IndexError:
        return dict.fromkeys(dict_of_geofences, -1)
        pass

def update_duration(dict_of_geofences, room, duration):
    for key in dict_of_geofences:
        if key in room:
            dict_of_geofences[key] += duration
            dict_of_geofences[key + " #"] += 1
    return dict_of_geofences

'''-------------------------- TRAJECTORIES ------------------------------ '''

''' 1. Measures of central tendency of points '''

def mean_center(window):
    return np.mean(window.x), np.mean(window.y)

'''Minimizes the sum of the absolute deviations from itself to the other 
n points of the distribution. '''
def median_center(window):
    iterate = True
    # initial guess
    max_error = 0.0000001

    x_start, y_start = np.median(window.x), np.median(window.y)

    x = window.x.to_numpy()
    y = window.y.to_numpy()
    
    iters = 0
    while iterate and iters < 10:
        distance = (((x - x_start)**2) + ((y - y_start)**2))**0.5
        denominator = sum( 1 / distance)
        new_x = (sum(x / distance)) / denominator
        new_y = (sum(y / distance)) / denominator
    
        iterate = (abs(new_x - x_start) > max_error) or (abs(new_y - y_start) > max_error)
        x_start = new_x
        y_start = new_y
        iters += 1

    return(new_x, new_y)

def normalize_window_values(window):
    x, y = window.x.to_numpy(), window.y.to_numpy()
    y_min = np.min(y)

    if np.sign(y_min) == -1:
        window.y = pd.Series([i - y_min for i in window.y], name='y')
    window.y = window.y.replace(0, 1)
    # Harmonic mean only applies to positive values
    return window

''' 2. Measures of dispersion '''

def standard_deviation(window):
    return np.std(window.x), np.std(window.y)

'''Spatial equivalent to standard deviation; Std of distance of each point
from the mean center. Provides a singular statistic.'''
def standard_distance(window):
    x_c, y_c = mean_center(window)
    standard_distance = 0
    n = 1 / len(window.x)
    for x, y in zip(window.x, window.y):
        standard_distance += n * (x - x_c)**2 + (y - y_c)**2
    return np.sqrt(standard_distance)

'''Meausures the anistropy, or skew of a set of points towards a particular direction'''
def standard_deviational_ellipse(window):
    x_c, y_c = mean_center(window)

    # Y-axis is rotated clockwise through an angle theta
    theta = 0
    sum_x, sum_y, sum_xy, sum_xyy = 0, 0, 0, 0

    for x, y in zip(window.x, window.y):
        sum_x += (x - x_c)**2
        sum_y += (y - y_c)**2
        sum_xy += (x - x_c) * (y - y_c)
        sum_xyy += (x - x_c) * (y - y_c)
    
    denominator = 2 * sum_xy
    numerator = sum_x - sum_y + np.sqrt((sum_x - sum_y)**2 + 4*sum_xyy**2)
    theta = np.arctan(numerator)/denominator

    # Two standard deviations calculated for transposed X and Y
    s_x_numerator = 0
    s_y_numerator = 0
    for x, y in zip(window.x, window.y):
        s_x_numerator += ((x - x_c)*np.cos(theta) - (y - y_c)*np.sin(theta))**2
        s_y_numerator += ((x - x_c)*np.sin(theta) - (y - y_c)*np.cos(theta))**2

    # Subtract 2 to produce an unbiased estimate of SDE since there are two constants
    # from which the distance along the axis is measured (i.e. the means of each)
    s_x = np.sqrt(2 * s_x_numerator / (window.x.size - 2))
    s_y = np.sqrt(2 * s_y_numerator / (window.y.size - 2))

    # Get the x and y axis of the ellipse as well as its area
    return 2*s_x, 2*s_y, np.pi * s_x * s_y

'''The mean associated with the mean of the logarithms and mean of the inverse.
These differ from mean center when the distribution is directionally skewed'''
def geometric_mean(window):
    x, y = window.x.to_numpy(), window.y.to_numpy()
    y_min = np.min(y)
    y = [i - y_min for i in y]
    return gmean(x), gmean(y)

def harmonic_mean(window):
    x, y = window.x.to_numpy(), window.y.to_numpy()

    x_min = np.min(x)
    y_min = np.min(y)

    # Harmonic mean only applies to positive values
    x = [i - x_min for i in x]
    y = [i - y_min for i in y]

    return hmean(x), hmean(y)

'''3. Velocity and Angle of the trajectory'''

def get_avg_velocity(window, threshold=0.15):
    prev_x = window.x.iloc[0]
    prev_y = window.y.iloc[0]
    prev_ts = window.ts_utc.iloc[0]

    speed = []
    dx = 0
    dy = 0
    dt = 0
    for curr_x, curr_y, curr_ts, in zip(window.x[1:], window.y[1:], window.ts_utc[1:]):
        if math.hypot(curr_x - prev_x, curr_y - prev_y) > threshold:
            dt = tf.utc_to_seconds(curr_ts) - tf.utc_to_seconds(prev_ts)
            dx = ((curr_x - prev_x) / (dt + 1))
            dy = ((curr_y - prev_y) / (dt + 1))
            speed.append(np.sqrt(dx**2 + dy**2))
        
        prev_x = curr_x
        prev_y = curr_y
        prev_ts = curr_ts
 
    min_speed = np.min(speed)
    avg_speed = np.mean(speed)
    max_speed = np.max(speed)
    std_speed = np.std(speed)

    return min_speed, avg_speed, max_speed, std_speed

MIN_SPEED = 0
LOW_SPEED = 0.25
MID_SPEED = 0.5
HIGH_SPEED = 0.75
MAX_SPEED = 1
def get_time_spent_in_speed_range(window, threshold=0.15):
    
    speed_range = {MIN_SPEED: 0, LOW_SPEED: 0, MID_SPEED: 0, 
                   HIGH_SPEED: 0, MAX_SPEED: 0}

    prev_x = window.x.iloc[0]
    prev_y = window.y.iloc[0]
    prev_ts = window.ts_utc.iloc[0]

    dx = 0
    dy = 0
    dt = 0

    for curr_x, curr_y, curr_ts, in zip(window.x[1:], window.y[1:], window.ts_utc[1:]):
        if math.hypot(curr_x - prev_x, curr_y - prev_y) > threshold:
            dt = tf.utc_to_seconds(curr_ts) - tf.utc_to_seconds(prev_ts)
            dx += (curr_x - prev_x) / (dt + 1)
            dy += (curr_y - prev_y) / (dt + 1)
            speed = np.sqrt(dx**2 + dy**2)
            if speed < LOW_SPEED:
                speed_range[MIN_SPEED] += dt
            elif speed < MID_SPEED:
                speed_range[LOW_SPEED] += dt
            elif speed < HIGH_SPEED:
                speed_range[MID_SPEED] += dt
            elif speed < MAX_SPEED:
                speed_range[HIGH_SPEED] += dt
            elif speed >= MAX_SPEED:
                speed_range[MAX_SPEED] += dt   
        prev_x = curr_x
        prev_y = curr_y
        prev_ts = curr_ts

    return speed_range

def get_total_angle(window):
    df = pd.concat([window.x, window.y, window.ts_utc], axis=1)
    trj = traja.TrajaDataFrame(df)
    angles = traja.trajectory.calc_angle(trj)

    return np.sum(angles.dropna())

    # return sum(angles.dropna()) / len(angles)

'''Steps 
1. Instantiate a circle of radius, R at first point in the list
2. Iterate through sequence of points until we find a point
outside of the circle (last point to intersect with the circle)
3. Calculate the distance the distance between first point and
last point in the circle
4. Do so until points run out
'''
def fractalD(window, scale1, scale2):
    x = list(window.x)
    y = list(window.y)
    d1 = estimate_distance(x, y, scale1)
    d2 = estimate_distance(x, y, scale2)
    # print("ESTIMATE 1 ",  d1, "ESTIMATE 2 ", d2)
    # print((np.log(scale1) - np.log(scale2)))
    D = (np.log(d1) - np.log(d2)) / (np.log(scale1) - np.log(scale2))
    if np.isnan(D) or np.isinf(D): D = -1
    return 1 - D

def estimate_distance(x_, y_, radius):
    distance = 0
    sphere = (x_[0], y_[0], radius)
    point = (x_[0], y_[0])
    prev_point = point
    idx = 1
    while idx < len(x_):
        x = x_[idx]
        y = y_[idx]
    # for x, y in zip(x_[1:], y_[1:]):
        point = (x, y)
        # print('CURRENT POINT', point)
        # print('CURRENT SPHERE: ', sphere)
        if not point_in_sphere(point, sphere):
            idx = idx - 1
            # print("POINT NOT IN SPHERE")
            sphere = find_new_centre(sphere, point, sphere[2])
            # print("NEW SPHERE ", sphere, '\n')
            distance += radius
        prev_point = point
        idx = idx + 1
    distance += np.sqrt((point[0] - sphere[0])**2 + (point[1] - sphere[1])**2)
    # print(distance)
    return distance

def point_in_sphere(point, sphere):
    # print((point[0]- sphere[0])**2 + (point[1] - sphere[1])**2)
    distance = np.sqrt((point[0] - sphere[0])**2 + (point[1] - sphere[1])**2)
    if distance <= sphere[2]:
        return True
    return False

pm = np.array([+1, -1])

def find_new_centre(p1, p2, radius):

    v = (p2[0] - p1[0], p2[1] - p1[1])

    norm = np.sqrt(v[0]**2 + v[1]**2)
    u = (v[0] / norm, v[1] / norm)
    
    new_point = (p1[0] + u[0] * radius, p1[1] + u[1] * radius)
    return (new_point[0], new_point[1], radius)