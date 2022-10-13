Space-time Indices for Clinical Support Data Pipeline
================================================

by Tamim Faruk (<tamim.faruk@mail.utoronto.ca>)

Overview
--------

The STICS repo contains various tools and scripts written in python3 used for analyzing 
the STICS real-time location database. These tools include a parser with the functionality
to divide the location data for each participant in the database, filter their data,
plot the data, and extract features from their data. Additionally, there are various 
scripts that use machine learning models and other tools that can be found in this
repo.

Getting Started
------------

Install the packages in this repo using:

    pip3 install -r requirements.txt

Some packages are used that were not captured by requirements.txt, but they are very few
and can be installed manually if errors arise.

Files in Root Directory
--------

### LocationProfileCollector.py
A class that collects/stores profile data for a csv file, or list of csv files; 
This class contains helper methods to split the data into windows of fixed duration,
filter the data, and extract features from it as well.

### LocationProfile.py
This file currently contains two essential classes.
    
* LocationWindow 
* LocationProfile

A LocationWindow implementation of Python's data class module to create a "window" struct that 
stores the location data. The LocationProfile is a collection of LocationWindows for a 
a particular participant.

### organize_stics_data.py
This script contains various functions to divide and organize the downloaded raw location data

* divide_by_date(): Takes daily, raw RTLS data, which contains data for all participants for each
date and separates the data for each participant into a distinct file with their daily location data
* divide_into_shifts(): Takes the daily RTLS data for each participant and divides it further into 
three shifts in correspondence with the schedule PAS is recorded
* read_shift_data_and_export_images(): Takes the daily RTLS data that is divided into shifts to 
generate corresponding images of their path
* read_pas_results(): Creates csv that relates PAS scores to date and shift
extract_feature_table_with_PAS_scores(): Combines features extracted from shift-level windows 
with PAS Scores 


Helper Files
--------

### timefunc.py
Timestamp manipulation functions and window segmentations based on timestamps
    
* Manipulate timestamps by converting from UTC to local time
* Divide data into windows of fixed duration
* Divide data into shift-windows for STICS specific data 

When the data is divided into windows of fixed duration, the timestamp for each subsequent window 
is simply the window starting timestamp + window length. There are options for using thresholds 
for the minimum number of points required to be present a window for acceptance.

When the data is divided into 'shift-windows', the data is divided into predefined windows based on
the shift in which the PAS score is collected. The shifts are from 7 AM - 3 PM, 3 PM - 11 PM, and 
11 PM - 7 AM (all in EST).

### featureExtractor.py
Contains mathematical functions to calculate each feature for a given location window. 
These features are described elsewhere.

### featureTable.py

A class which collects and organizes all features into a dictionary

### windowFilter.py

Contains three methods to filter the data

* The first method is to filter based on confidence. Confidence is the probability that
a given location data point is accurate.
* The second method is to filter based on a hard distance threshold
* The third and most useful method is to filter based on speed. A moving-window traverses
the data and the average speed of the points in the moving-window are compared to the 
instantaneous speed of the oncoming point with the last point in the moving-window. If
the instantaneous speed exceeds the average speed of the moving-window by a threshold
then the oncoming point is discarded.

Only the third method is used for filtering right now as the other two lead to some distortions
of the trajectory.

### animation.py
Given a window of location data, contains methods to generate animation of movement trajectory

Deprecated
----------

### plot_usf_data.py
Given a window of location data, contains methods to plot the data into image.

This was originally for the USF dataset. Not used anymore, deprecated. 

How to Use
----------

See the file tenera_demo.py for an example of how to use the various utility in this repository.
