import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class LocationWindow:
	profile_id: str
	rows: tuple # remove later
	x: list = None
	y: list = None
	z: list = None # remove later
	room: list = None
	geofence_type: list = None
	confidence: list = None
	ts_utc: list = None
	train: str = None # obsolete
	directory: str = None
	
class LocationProfile:
	def __init__(self, profileId, list_of_rows):
		self.windows = {}