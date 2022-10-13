import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
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

# helper functions
from helpers import animation 
from helpers import featureExtractor
from helpers import featureTable
from helpers import windowFilter 
from helpers import timefunc as tf
from helpers import plot_usf_data
from mpl_toolkits.axes_grid1 import make_axes_locatable
# matplotlib.use('Agg')

# Constants
X_LIM = 900 # for STICS data plots
Y_LIM = 1300 # for STICS data plots
DIST_THRESH = 1 # for filtering STICS data
MAP = '/mnt/c/Users/FarukT/Desktop/STICS/floorplans/TRI_SDU.png'
MAP = '/mnt/c/Users/Tamim Faruk/OneDrive/Documents/Academics/4B/FYDP/STICS/floorplans/TRI_SDU.png'

''' Parses a CSV file to retrieve all profile ids and associated location data
and stores in a list of unique locationProfile objects. The list is indexed
by the profile_id. Gives access to tools to plot and animate the profiles.'''
class LocationProfileCollector:
	def __init__(self, filename=None, test_split='cont'):
		self.df = None
		self.list_of_profiles = {}
		self.test_split = test_split # obsolete 

		self.filename = None
		if filename is not None:
			self.filename = filename
			self.read_csv()

	def read_csv(self):
		# print("Reading file")
		self.df = pd.read_csv(self.filename)
		# self.df = self.df.sort_values(['timestamp'], ascending=True)
		# print(self.df['timestamp'])

	def reset_df(self, filename):
		self.filename = filename
		self.read_csv()

	'''------------------------------  SETTERS ------------------------------ '''

	''' Stores entire dataframe as a location window for a given profile
	
	This function is used if user wants to store all profile data in one window without dividing
	it into smaller windows. Ideal function for when the stics daily data file is divided 
	already by profile and  shift to read the shift file directly without subdividing it further
	'''
	def set_single_window_for_profile(self, profile_id, directory=None, export_type='shift', stics=True):
		list_of_rows = self.df.index[self.df['profile_id'] == profile_id].tolist()
		start = list_of_rows[0]
		end = list_of_rows[-1]
		profile_data = self.df.loc[start:end]
		profile = pf.LocationProfile(profile_id, list_of_rows)
		window = self.set_profile_window(profile_id, profile_data, start, end)
		window.directory = directory
		if export_type == 'shift':
			window.directory = self.set_window_directory(directory)
		profile.windows[tf.reformat_dates(profile_data['timestamp'][start])] = window
		self.list_of_profiles[profile_id] = profile

	''' Creates a single location profile for a given participant
	
	This function is used to divide the data for a profile into windows. The data can be 
	split by two methods:
	1) By Shift: this is specific to STICS data to divide data in correspondence to clinical data
	collection from shifts (3 PM - 11 PM EST, 11 PM - 7 AM EST, 7 AM - 3 PM EST).
	2) By timeframe and threshold: this was used to create windows from the USF dataset based on
	length of windows and number of points per window. This can still be used on the 'shift' files
	for the STICS dataset to divide the data into even smaller windows.
	'''
	def set_single_profile(self, profile_id, byshift=False, save=False, filter=True, stics=True, **kwargs):
		list_of_rows = self.df.index[self.df['profile_id'] == profile_id].tolist()
		# print(list_of_rows)
		profile_data = self.df.loc[list_of_rows[0]:list_of_rows[-1]]
		print(profile_id, profile_data)
		if byshift: profile_data = self.df
		# if save:
		# 	savepath = "data/new/cont/%s/%s/" % (tf, str(threshold))
		# 	Path(savepath).mkdir(parents=True, exist_ok=True)
		# 	filename = savepath + profile_id + '.csv'
		windows = self.get_windows(profile_data, list_of_rows[0], save=save, test_split=self.test_split, 
								   byshift=byshift, **kwargs)
		if profile_id in self.list_of_profiles: profile = self.get_single_profile(profile_id)
		else: profile = pf.LocationProfile(profile_id, list_of_rows)
		for i in windows:
			# if not bydate: 
			window_id = tf.reformat_dates(i[3])
			# window_id = i[3][0]
			if window_id in profile.windows:
				self.append_profile_window(profile.windows, window_id, profile_data, i=i, stics=stics)
			else: 
				start = list_of_rows[0]+i[1]
				end = list_of_rows[0]+i[2]

				window = self.set_profile_window(profile_id, profile_data, start, end, i=i, filter=filter, stics=stics)
				profile.windows[window_id] = window
		self.list_of_profiles[profile_id] = profile

	def set_profile_window(self, profile_id, profile_data, start, end, stics=True, filter=True, i=None):
		if i == None: i = [0, 0, -1]
		window = pf.LocationWindow(profile_id=profile_id, rows=(start, end))
		window.x = profile_data['x'][i[1]:i[2]]
		window.y = profile_data['y'][i[1]:i[2]]
		window.ts_utc = profile_data['timestamp'][i[1]:i[2]]
		if stics:
			window.room = profile_data['room'][i[1]:i[2]]
			window.geofence_type = profile_data['geofence_type'][i[1]:i[2]]
			window.confidence = profile_data['room_probability'][i[1]:i[2]]
			window.directory = self.set_window_directory(str(self.filename))
		# window.train = [i[-1]]

		# If we want to filter, and if we have enough points (threshold can be adjusted)
		if filter and window.x.size > 20:
			window = windowFilter.filter_points_by_mean_speed(window, stics=stics)
			# window = filterPoints.filter_points_by_distance(window)
			# window = filterPoints.filter_points_by_confidence(window)
		return window

	''' Appends new data to existing window data structure
	
	This function is used to concatenate windows when the data for a particular window is
	divided between two files. For example, profile 123 has daily data on 2021-10-15 and
	2021-10-16. The last shift for the first file ends at 2021-10-15 03:00:00, and the second
	file begins at 2021-10-16 04:00:00. Therefore, the last hour of data between the first and last file
	must be combined. This function will create a window for the last hour of data in the first file,
	and then when reading the second file, will append the data to the created window.
	'''
	def append_profile_window(self, windows, window_id, profile_data, i, stics=True):
		windows[window_id].x = pd.concat([windows[window_id].x, profile_data['x'][i[1]:i[2]]])
		windows[window_id].y = pd.concat([windows[window_id].y, profile_data['y'][i[1]:i[2]]])
		windows[window_id].ts_utc = pd.concat([windows[window_id].ts_utc, profile_data['timestamp'][i[1]:i[2]]])
		if stics:
			windows[window_id].room = pd.concat([windows[window_id].room, profile_data['room'][i[1]:i[2]]])
			windows[window_id].geofence_type = pd.concat([windows[window_id].geofence_type, profile_data['geofence_type'][i[1]:i[2]]])
			windows[window_id].confidence = pd.concat([windows[window_id].confidence, profile_data['room_probability'][i[1]:i[2]]])
			windows[window_id].directory = self.set_window_directory(str(self.filename))

	def set_all_profiles(self, window=False, timeframe=900, threshold=None, csvwriter=None, filter=True, stics=True, save=False):
		for profile_id in self.df.profile_id.unique():
			self.set_single_profile(profile_id, timeframe=timeframe, threshold=threshold, 
									csvwriter=csvwriter, save=save, stics=stics, filter=filter)


	def extract_path_for_single_window(self, profile, window_id):
		window = profile.windows[window_id]
		paths = tf.extract_paths_from_window(window)
		return paths
	
	def create_windows_from_paths(self, profile, window_id):
		original_window = profile.windows[window_id]
		# print("UNSPLICED: ", original_window.x)
		path_windows = []
		paths = self.extract_path_for_single_window(profile, window_id)
		for path in paths:
			spliced_window = self.splice_window(original_window, path[0], path[1])
			path_windows.append(spliced_window)
		return path_windows

	def splice_window(self, window, start, end):
		spliced_window = copy.deepcopy(window)
		spliced_window.x = window.x.iloc[start:end]
		spliced_window.y = window.y.iloc[start:end]
		spliced_window.ts_utc = window.ts_utc.iloc[start:end]
		spliced_window.room                                                 = window.room.iloc[start:end]
		spliced_window.geofence_type = window.geofence_type.iloc[start:end]
		spliced_window.confidence = window.confidence.iloc[start:end]
		return spliced_window

	''' This function will set the directory property for a window so that the export_windows_to_csv()
	function can export the window to the appropriate file directory.
	'''
	def set_window_directory(self, directory):
		export_path = directory.split('/')[:]
		export_path.insert(-1, 'shift/')
		export_path = '/'.join(export_path[:-1])
		Path(export_path).mkdir(parents=True, exist_ok=True)

		return export_path

	'''------------------------------  GETTERS ------------------------------ '''

	def get_single_profile(self, profile_id):
		return self.list_of_profiles[profile_id]

	def get_profile_list(self):
		for profile_id in self.df.profile_id.unique():
			print(profile_id)

	'''Returns a generator which will extract windows based on preference'''
	def get_windows(self, df, start_row, timeframe=900, threshold=None,save=False, csvwriter=None, 
					stics=True, byshift=False, test_split='cont', filename='default.csv'):
		if byshift: return tf.generate_STICS_windows_by_shift(df, start_row, save=save, 
						  csvwriter=csvwriter, test_split=test_split)		
		# This returns the original function used by the USF dataset when this file was originally coded
		return tf.generate_windows_by_timeframe_and_threshold(df, start_row, tf=timeframe, 
					threshold=threshold, save=save, csvwriter=csvwriter, stics=stics, 
					test_split=test_split, filename=filename)

	def get_window_keys(self, profile_id=None, profile=None):
		# print(profile)
		if profile is None: profile = self.get_single_profile(profile_id)
		for window_id, window in profile.windows.items():
			print('Profile Keys: \n', window_id)
		return profile.windows.items()

	def export_windows_to_csv(self, profile_id=None, profile=None, change_dir=True):
		# print(profile)
		if profile is None: profile = self.get_single_profile(profile_id)
		for window_id, window in profile.windows.items():
			profile_id_series = [window.profile_id]*len(window.x)
			profile_id_series = pd.Series(profile_id_series, name='profile_id', index=window.x.index)
			list_to_export = [profile_id_series, window.x, window.y, window.ts_utc, 
							  window.room, window.geofence_type, window.confidence]
			exported_data = pd.concat(list_to_export, axis=1)
			if change_dir:
				window.directory = window.directory + window_id + '.csv'
			exported_data.to_csv(window.directory)
			print("Exported to: ", window.directory)
			exported_data = None

	def export_windows_to_image_from_raw_data(self, profile_id, profile=None):
		if profile is None: profile = self.get_single_profile(profile_id)
		for window_id, window in profile.windows.items():
			print(window.directory)
			if '.csv' in window.directory: window.directory = window.directory.replace('.csv', '.png')
			split_directory = window.directory.split('/')
			split_directory.insert(12, 'images')
			window.directory = '/'.join(split_directory)
			Path('/'.join(split_directory[0:-1])).mkdir(parents=True, exist_ok=True)
			# print(window.directory)
			self.plot_window_on_map(window, save_directory=window.directory)

	''' Animates a list of profiles 
	input:  [[profile_id, window_key]]
	output: [[profile_id, x, y, ts_utc]]
	'''
	def animateProfileWindow(self, profile_info, floorplan=None, save=False, profiles=None):
		profiles = []
		for details in profile_info:
			filename = '_'.join(details)
			profile = self.get_single_profile(details[0])
			window = profile.windows[details[1]]
			x = round(window.x, 2)
			y = round(window.y, 2)
			ts_ms = tf.get_ts_seconds(window.ts_utc)
			profiles.append([x, y, ts_ms])
		animation.launch_animation_new(profiles, floorplan=floorplan, filename=filename, save=save)

	# TODO: Separate into function to impute 1 point and multiple points
	# TODO: Impute for windows of profile
	def impute_series(self, profile_id=None, start=0, end=-1, window=2, threshold=10, profile=None):
		profile = profile
		if profile == None:
			profile = self.get_single_profile(profile_id)
		ts_ms = tf.get_ts_seconds(profile.ts_utc[start:end+1])
		if end == -1:
			ts_ms = tf.get_ts_seconds(profile.ts_utc[start:])
		points_to_impute = tf.find_imputation_indices(ts_ms, threshold=threshold)
		# self.plot_localization_data(type_of_plot=X_VS_Y, profile_id=profile_id)
		num_points_added = 0
		while len(points_to_impute) > 0:
			idx, diff = points_to_impute.pop(0)
			idx = idx + num_points_added
			new_ts, new_x, new_y = tf.impute_point(idx, round(diff, 2), window, profile)
			num_points_added += len(new_x)
			profile.list_of_rows = list(range(profile.list_of_rows[0], profile.list_of_rows[-1] + 1 + len(new_x)))
			profile.x = tf.insert_point(profile.x, new_x, idx, profile.list_of_rows)
			profile.y = tf.insert_point(profile.y, new_y, idx, profile.list_of_rows)
			profile.ts_utc = tf.insert_point(profile.ts_utc, new_ts, idx, profile.list_of_rows) 
		# self.plot_localization_data(type_of_plot=X_VS_Y, profile=profile)

	# metric = 'temporal' or 'spatiial'
	def calculate_gaps_in_window(self, profile_info, metric, threshold):
		with open(profile_info[0][0] + '_' + profile_info[0][1] + '.csv', 'w') as writefile:
			csvwriter = csv.writer(writefile, delimiter=',')
			for details in profile_info:
				filename = '_'.join(details)
				profile = self.get_single_profile(details[0])
				window = profile.windows[details[1]]
				if metric == 'temporal':
					csvwriter.writerow(['idx', 'ts1', 'ts2', 'delta'])
					timestamps = tf.get_ts_seconds(window.ts_utc)
					points_to_impute = tf.find_imputation_indices(timestamps, threshold=threshold)
					for point_to_impute in points_to_impute:
						# print(len(points_to_impute))
						point_to_impute[0] = tf.reformat_dates(tf.strftime_usf(point_to_impute[0]))
						point_to_impute[1] = tf.reformat_dates(tf.strftime_usf(point_to_impute[1]))
						print(point_to_impute)
						csvwriter.writerow(point_to_impute)
				else: # metric == 'spatial'
					x_points_to_impute = tf.find_imputation_indices(window.x.tolist(), threshold=threshold)
					y_points_to_impute = tf.find_imputation_indices(window.y.tolist(), threshold=threshold)
					for x_point_to_impute, y_point_to_impute in zip(x_points_to_impute, y_points_to_impute):
						point_to_impute = x_point_to_impute + y_point_to_impute
						csvwriter.writerow(point_to_impute)

	def calculate_all_time_gaps_in_windows(self, profile_id):
		profile = self.get_single_profile(profile_id)
		for window_id, window in profile.windows.items():
			self.calculate_gaps_in_window([[profile_id, window_id]], 'temporal', 5)

	
	def plot_window_on_map(self, window, save_directory=None, filter=False, floorplan=MAP):
		fig, ax = plt.subplots()
		plt.xlabel('X Position')
		plt.ylabel('Y Position')
		starting_number_of_points = window.confidence.size
		if not window.x.empty:
			if filter:
				window = windowFilter.filter_points_by_distance(window)
				window = windowFilter.filter_points_by_mean_speed(window)
				window = windowFilter.filter_points_by_confidence(window)
				# number_of_points_removed = starting_number_of_points - window.confidence.size
				# percent_points_removed = np.round(100*number_of_points_removed/starting_number_of_points, 1)
				# print('NUM POINTS BEFORE FILTER: ', starting_number_of_points)
				# print('NUM POINTS AFTER FILTER: ', window.confidence.size)
				# print('%i points were removed, i.e. %0.2f percent of starting points' % (number_of_points_removed, percent_points_removed))
			im = plt.imread(floorplan)
			implot = plt.imshow(im)
			colour_sequence = list(range(len(window.x)))
			test = ax.scatter(window.x * 28 + 57, -window.y * 27.7 + 58, c=colour_sequence, s=1, cmap='rainbow')
			divider = make_axes_locatable(ax)
			cax = divider.append_axes('right', size='5%', pad=0.05)
			fig.colorbar(test, ax=ax, orientation='vertical')
			# ax.scatter(window.x * 28 + 57, -window.y * 27.7 + 58, s=1)
			plt.xlim(0, X_LIM)
			plt.ylim(0, Y_LIM)
			plt.axis('off')
			plt.savefig(save_directory, format='png', dpi=1200, bbox_inches='tight')


	'''--------------------- FEATURE EXTRACTION METHODS ---------------------'''

	def extract_all_features(self, profile_id, windows=None):
		ft = featureTable.FeatureTable(lpc=self)
		feature_table = ft.extract_all_features(profile_id, windows=windows)
		return feature_table

	def extract_all_features_from_all_profiles(self):
		feature_tables = []
		for profile_id in self.list_of_profiles:
			print("Extracting features for profile: ", profile_id)
			feature_table = ft.extract_all_features(profile_id)
			# print(feature_table)
			feature_tables.append(feature_table)
		result = pd.concat(feature_tables)
		return result

	'''--------------------- USF DATA SPECIFIC FUNCTIONS --------------------'''

	def plot_profile_windows(self, profile_id=None, profile=None,x_lim=None, 
							y_lim=None, timeframe=900, threshold=None):
		if profile is None: profile = self.get_single_profile(profile_id)
		print(profile.windows.items())
		for window_id, window in profile.windows.items():
			window_id = window_id.replace('/', '_')
			window_id = window_id.replace(' ', '_')
			window_id = window_id.replace(':', '_')
			plot_usf_data.plot_continuous_window(window, timeframe=timeframe, threshold=threshold, 
											test_split=self.test_split, x_lim=x_lim, y_lim=y_lim)
	
	def plot_all_profile_windows(self, timeframe=900, threshold=None, standardize=False):
		KEARNS_X_LIM = (-2.0, 30)
		KEARNS_Y_LIM = (-5.0, 10)
		# if we want all images to be scaled correctly based on the floorplan
		if standardize: x_lim, y_lim = KEARNS_X_LIM, KEARNS_Y_LIM
		for profile_id in self.list_of_profiles:
			self.plot_profile_windows(profile_id=profile_id, timeframe=timeframe, x_lim=x_lim, 
										y_lim=y_lim, threshold=threshold) 

	mmse_scores =  {'SP025': "0", 'SP016': "0", 
					'SP023': "0", 'SP013': "1",
					'SP014': "1", 'SP009': "1", 
					'SP002': "1", 'SP026': "1",
					'SP019': "1", 'SP017': "2"}

	def assign_mmse_score(self, profile_id):
		profile = self.get_single_profile(profile_id)
		if profile_id in self.mmse_scores:
			return self.mmse_scores[profile_id]