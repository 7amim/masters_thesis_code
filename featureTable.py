import copy
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import locationProfile as pf
import traja

from pathlib import Path
from sklearn.model_selection import train_test_split
from pathlib import Path

from helpers import featureExtractor


class FeatureTable(object):
    columns =  ['ID', 'tot_dist_walking_0_15', 'tot_dist_walking_0_30', 'time_spent_walking_0_15', 'time_spent_walking_0_30', 
                'time_spent_stationary_0_15', 'time_spent_stationary_0_30', 'angles', 
                'min_speed_0_15', 'max_speed_0_15', 'avg_speed_0_15', 'std_speed_0_15', 'min_speed_0_30', 'max_speed_0_30', 
                'avg_speed_0_30', 'std_speed_0_30', 'speed_min_0_15', 'speed_low_0_15', 'speed_mid_0_15', 'speed_high_0_15',
                'speed_max_0_15', 'speed_min_0_30', 'speed_low_0_30', 'speed_mid_0_30', 'speed_high_0_30', 'speed_max_0_30', 
                'fractal_dimension', 'median_center_x', 'median_center_y', 'std_x', 'std_y', 'standard_distance', 'sde_length',  
                'sde_width', 'sde_area', 'geo_mean_x', 'geo_mean_y', 'path_length']

    def __init__(self, lpc=None):
        self.lpc = lpc
        
        self.list_of_geofences, self.dict_of_geofences, self.collapsed_geofences, self.collapsed_geofences2 = self.read_list_of_geofences()
        self.geofence_features = self.create_feature_table_geofences()
        geo_spatial_features  = {feature: [] for feature in self.columns}

        self.feature_list = {**geo_spatial_features, **self.collapsed_geofences2}

    def set_lpc(self, lpc):
        self.lpc = lpc

    def extract_all_features(self, profile_id, windows=None):

        if windows == None:
            profile = self.lpc.get_single_profile(profile_id)
            windows = list(profile.windows.values())

        for window in windows:
            window = featureExtractor.normalize_window_values(window)
            if window.x.size > 0:
                # micellaneous
                self.feature_list['ID'].append(window.directory)
                # Location-Time
                self.feature_list['tot_dist_walking_0_15'].append(featureExtractor.get_total_distance(window, threshold=0.15))
                self.feature_list['tot_dist_walking_0_30'].append(featureExtractor.get_total_distance(window, threshold=0.30))
                self.feature_list['time_spent_walking_0_15'].append(featureExtractor.get_time_spent_walking(window, threshold=0.15))
                self.feature_list['time_spent_walking_0_30'].append(featureExtractor.get_stationary_time(window, threshold=0.30))
                self.feature_list['time_spent_stationary_0_15'].append(featureExtractor.get_stationary_time(window, threshold=0.15))
                self.feature_list['time_spent_stationary_0_30'].append(featureExtractor.get_stationary_time(window, threshold=0.30))
                self.feature_list['angles'].append(featureExtractor.get_total_angle(window))
                # Average speed calculations
                avg_velocity = featureExtractor.get_avg_velocity(window, threshold=0.15)
                self.feature_list['min_speed_0_15'].append(avg_velocity[0])
                self.feature_list['max_speed_0_15'].append(avg_velocity[1])
                self.feature_list['avg_speed_0_15'].append(avg_velocity[2])
                self.feature_list['std_speed_0_15'].append(avg_velocity[3])
                avg_velocity = featureExtractor.get_avg_velocity(window, threshold=0.30)
                self.feature_list['min_speed_0_30'].append(avg_velocity[0])
                self.feature_list['avg_speed_0_30'].append(avg_velocity[1])
                self.feature_list['max_speed_0_30'].append(avg_velocity[2])
                self.feature_list['std_speed_0_30'].append(avg_velocity[3])
                # Speed Ranges
                speed_range = featureExtractor.get_time_spent_in_speed_range(window, threshold = 0.15)
                self.feature_list['speed_min_0_15'].append(speed_range[featureExtractor.MIN_SPEED])
                self.feature_list['speed_low_0_15'].append(speed_range[featureExtractor.LOW_SPEED])
                self.feature_list['speed_mid_0_15'].append(speed_range[featureExtractor.MID_SPEED])
                self.feature_list['speed_high_0_15'].append(speed_range[featureExtractor.HIGH_SPEED])
                self.feature_list['speed_max_0_15'].append(speed_range[featureExtractor.MAX_SPEED])
                speed_range = featureExtractor.get_time_spent_in_speed_range(window, threshold = 0.30)
                self.feature_list['speed_min_0_30'].append(speed_range[featureExtractor.MIN_SPEED])
                self.feature_list['speed_low_0_30'].append(speed_range[featureExtractor.LOW_SPEED])
                self.feature_list['speed_mid_0_30'].append(speed_range[featureExtractor.MID_SPEED])
                self.feature_list['speed_high_0_30'].append(speed_range[featureExtractor.HIGH_SPEED])
                self.feature_list['speed_max_0_30'].append(speed_range[featureExtractor.MAX_SPEED])
                # Fractal D
                self.feature_list['fractal_dimension'].append(featureExtractor.fractalD(window, 0.5, 1))
                # Object-Location
                median_center = featureExtractor.median_center(window)
                self.feature_list['median_center_x'].append(median_center[0])
                self.feature_list['median_center_y'].append(median_center[1])
                self.feature_list['std_x'].append(featureExtractor.standard_deviation(window)[0])
                self.feature_list['std_y'].append(featureExtractor.standard_deviation(window)[1])
                self.feature_list['standard_distance'].append(featureExtractor.standard_distance(window))
                sde = featureExtractor.standard_deviational_ellipse(window)
                self.feature_list['sde_length'].append(sde[0])
                self.feature_list['sde_width'].append(sde[1])
                self.feature_list['sde_area'].append(sde[2])
                geo_mean = featureExtractor.geometric_mean(window)
                self.feature_list['geo_mean_x'].append(geo_mean[0])
                self.feature_list['geo_mean_y'].append(geo_mean[1])
                self.feature_list['path_length'].append(len(window.x))
                # labels
                # labels.append(label)  
                extracted_geofence_features = featureExtractor.get_total_time_in_geolocation(self.dict_of_geofences, window)
                self.update_geofence_features(extracted_geofence_features)
                self.update_collapsed_geofence_dict()
        feature_table = pd.DataFrame(self.feature_list)
        return feature_table

    def read_list_of_geofences(self):
        
        LIST_OF_GEOFENCES_FILENAME = "/mnt/c/Users/Tamim Faruk/OneDrive/Documents/Academics/4B/FYDP/STICS/floorplans/ListOfGeofences2.txt"
        with open(LIST_OF_GEOFENCES_FILENAME) as f:
            list_of_geofences = f.readlines()

        list_of_geofences = [line.strip() for line in list_of_geofences]
        expanded_list = list_of_geofences
        
        expanded_list = [line.split(',')[0] for line in list_of_geofences]
        collapsed_list = set([line.split(',')[1].lstrip() for line in list_of_geofences])
        
        number_of_times_in_geofence = []
        for geofence in expanded_list:
            key = geofence + " #"
            number_of_times_in_geofence.append(key)
        
        geofences = expanded_list + number_of_times_in_geofence

        dict_of_geofences = {geofence: 0 for geofence in geofences}
        
        collapsed_geofences = {collapsed_geofence: 0 for collapsed_geofence in collapsed_list}
        collapsed_geofences2 = {collapsed_geofence: [] for collapsed_geofence in collapsed_list}

        return list_of_geofences, dict_of_geofences, collapsed_geofences, collapsed_geofences2

    def create_feature_table_geofences(self):
        keys = list(self.dict_of_geofences.keys())
        return {key: [] for key in keys}

    def update_geofence_features(self, extracted_geofence_features):
        for k, v in extracted_geofence_features.items():
            self.dict_of_geofences[k] = v

    def update_collapsed_geofence_dict(self):
        for line in self.list_of_geofences:
            collapsed_label = line.split(',')[1].lstrip()
            geofence = line.split(',')[0].strip()
            # print("COLLAPSED: ", collapsed_label, "GEOFENCE: ", geofence, "TIME: ", self.dict_of_geofences[geofence])
            self.collapsed_geofences[collapsed_label] += self.dict_of_geofences[geofence]
            self.dict_of_geofences[geofence] = 0
        
        for k, v in self.collapsed_geofences.items():
            self.feature_list[k].append(v)
            self.collapsed_geofences[k] = 0