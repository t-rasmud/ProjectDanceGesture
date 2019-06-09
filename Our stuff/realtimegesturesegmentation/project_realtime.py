# This cell includes the major classes used in our classification analyses
import matplotlib.pyplot as plt # needed for plotting
import numpy as np # numpy is primary library for numeric array (and matrix) handling
import scipy as sp
from scipy import stats, signal
import random
from sklearn import svm # needed for svm
from sklearn.metrics import confusion_matrix
import itertools
from os import listdir
import ntpath
import os
from matplotlib.gridspec import GridSpec

import sys, serial, argparse
from time import sleep
from collections import deque
import itertools
import math

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.spatial import distance
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from matplotlib.gridspec import GridSpec

# Each accelerometer log file gets parsed and made into a SensorData object
class SensorData:
    
    # Constructors in Python look like this (strangely enough)
    # All arguments are numpy arrays except sensorType, which is a str
    def __init__(self, sensorType, currentTimeMs, sensorTimestampMs, x, y, z):
        self.sensorType = sensorType
        
        # On my mac, I could cast as straight-up int but on Windows, this failed
        # This is because on Windows, a long is 32 bit but on Unix, a long is 64bit
        # So, forcing to int64 to be safe. See: https://stackoverflow.com/q/38314118
        self.currentTimeMs = currentTimeMs.astype(np.int64)
        
        # sensorTimestampMs comes from the Arduino function 
        # https://www.arduino.cc/reference/en/language/functions/time/millis/
        # which returns the number of milliseconds passed since the Arduino board began running the current program.
        self.sensorTimestampMs = sensorTimestampMs.astype(np.int64)
        
        self.x = x.astype(float)
        self.y = y.astype(float)
        self.z = z.astype(float)
   
        # Calculate the magnitude of the signal
        self.mag = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        
        self.sampleLengthInSecs = (self.currentTimeMs[-1] - self.currentTimeMs[0]) / 1000.0
        self.samplesPerSecond = len(self.currentTimeMs) / self.sampleLengthInSecs 
        
    # Returns a dict of numpy arrays for each axis of the accel + magnitude
    def get_data(self):
        return {"x":self.x, "y":self.y, "z":self.z, "mag":self.mag}
    
    # Returns a dict of numpy arrays for each axis of the accel + magnitude
    def get_processed_data(self):
        return {"x_p":self.x_p, "y_p":self.y_p, "z_p":self.z_p, "mag_p":self.mag_p}
    
    # Creates a new padded version of each data array with zeroes. Throws exception
    # if newArrayLength smaller than the current data array (and thus nothing to pad)
    # See: https://docs.scipy.org/doc/numpy-1.14.1/reference/generated/numpy.pad.html
    def pad_with_zeros(self, newArrayLength):
        self.signalLengthBeforePadding = len(self.x)
        arrayLengthDiff = newArrayLength - len(self.x)
        if arrayLengthDiff < 0:
            raise ValueError("New array length '{}' must be larger than current array length '{}".
                             format(newArrayLength, len(self.x)))
        
        # np.pad allows us to pad either the left side, right side, or both sides of an array
        # in this case, we are padding only the right side. 
        # See: https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html
        self.x_padded = np.pad(self.x, (0, arrayLengthDiff), 'constant', constant_values=0)
        self.y_padded = np.pad(self.y, (0, arrayLengthDiff), 'constant', constant_values=0)
        self.z_padded = np.pad(self.z, (0, arrayLengthDiff), 'constant', constant_values=0)
        self.mag_padded = np.pad(self.mag, (0, arrayLengthDiff), 'constant', constant_values=0)

# A trial is one gesture recording and includes an accel SensorData object
# In the future, this could be expanded to include other recorded sensors (e.g., a gyro)
# that may be recorded simultaneously
class Trial:
    
    # We actually parse the sensor log files in the constructor--this is probably bad practice
    # But offers a relatively clean solution
    def __init__(self, gestureName, endTimeMs, trialNum, accelLogFilenameWithPath):
        self.gestureName = gestureName
        self.trialNum = trialNum
        self.endTimeMs = endTimeMs
        self.accelLogFilenameWithPath = accelLogFilenameWithPath
        self.accelLogFilename = os.path.basename(accelLogFilenameWithPath)
        
        # unpack=True puts each column in its own array, see https://stackoverflow.com/a/20245874
        # I had to force all types to strings because auto-type inferencing failed
        parsedAccelLogData = np.genfromtxt(accelLogFilenameWithPath, delimiter=',', 
                              dtype=str, encoding=None, skip_header=1, unpack=True)
        
        # The asterisk is really cool in Python. It allows us to "unpack" this variable
        # into arguments needed for the SensorData constructor. Google for "tuple unpacking"
        self.accel = SensorData("Accelerometer", *parsedAccelLogData)
    
    # Utility function that returns the end time as a nice string
    def getEndTimeMsAsString(self):
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.endTimeMs / 1000))
    
    def __str__(self):
         return "'{}' : Trial {} from {}".format(self.gestureName, self.trialNum, self.accelLogFilename)

# Container for a single set of gestures and trials
class GestureSet:
    
    def __init__(self, gesture_log_path, map_gestures_to_trials):
        self.path = gesture_log_path
        self.map_gestures_to_trials = map_gestures_to_trials 

    # returns the longest trial (based on num rows recorded and not clock time)
    def get_longest_trial(self):
        longest_trial_length = -1
        longest_trial = None
        for gesture_name, trial_list in self.map_gestures_to_trials.items():
            for trial in trial_list:
                if longest_trial_length < len(trial.accel.x):
                    longest_trial_length = len(trial.accel.x)
                    longest_trial = trial
        return longest_trial
    
    # returns the base path
    def get_base_path(self):
        return os.path.basename(os.path.normpath(self.path))
    
    # returns the number of gestures
    def get_num_gestures(self):
        return len(self.map_gestures_to_trials)
    
    # returns trials for a gesture name
    def get_trials_for_gesture(self, gesture_name):
        return self.map_gestures_to_trials[gesture_name]
    
    # creates an aggregate signal based on *all* trials for this gesture
    # TODO: in future could add in an argument, which takes a list of trial nums
    # to use to produce aggregate signal
    def create_aggregate_signal(self, gesture_name, signal_var_name):
        trials = self.get_trials_for_gesture(gesture_name)
        aggregate_signal = None
        trial_signals = []
        trial_signals_original = []
        first_trial = None
        first_trial_signal = None
        
        max_length = -1
        for trial in trials:
            trial_signal = getattr(trial.accel, signal_var_name)
            if max_length < len(trial_signal):
                max_length = len(trial_signal)
            
        for i in range(len(trials)):
            if i == 0:
                first_trial = trials[i]
                trial_signal = getattr(first_trial.accel, signal_var_name)
                trial_signal_mod = np.copy(trial_signal)

                trial_signals.append(trial_signal_mod)
                trial_signals_original.append(trial_signal)
                
                array_length_diff = max_length - len(trial_signal_mod)
                trial_signal_mod = np.pad(trial_signal_mod, (0, array_length_diff), 'mean')  

                aggregate_signal = trial_signal_mod
                first_trial_signal = trial_signal_mod
            else:

                cur_trial = trials[i]
                cur_trial_signal = getattr(trial.accel, signal_var_name) 
                trial_signals_original.append(cur_trial_signal)
                
                array_length_diff = max_length - len(cur_trial_signal)
                cur_trial_signal_mod = np.pad(cur_trial_signal, (0, array_length_diff), 'mean') 

                cur_trial_signal_mod = get_aligned_signal_cutoff_and_pad(cur_trial_signal_mod, first_trial_signal)
                trial_signals.append(cur_trial_signal_mod)
                aggregate_signal += cur_trial_signal_mod
        
        mean_signal = aggregate_signal / len(trial_signals) 
        return mean_signal

    # Returns the minimum number of trials across all gestures (just in case we accidentally recorded a 
    # different number. We should have the same number of trials across all gestures)
    def get_min_num_of_trials(self):
        minNumTrials = -1 
        for gestureName, trialSet in self.map_gestures_to_trials.items():
            if minNumTrials == -1 or minNumTrials > len(trialSet):
                minNumTrials = len(trialSet)
        return minNumTrials

    # returns the total number of trials
    def get_total_num_of_trials(self):
        numTrials = 0 
        for gestureName, trialSet in self.map_gestures_to_trials.items():
            numTrials = numTrials + len(trialSet)
        return numTrials
    
    # get random gesture name
    def get_random_gesture_name(self):
        gesture_names = list(self.map_gestures_to_trials.keys())
        rand_gesture_name = gesture_names[random.randint(0, len(gesture_names) - 1)]
        return rand_gesture_name
    
    # get random trial
    def get_random_trial(self):
        rand_gesture_name = self.get_random_gesture_name()
        print("rand_gesture_name", rand_gesture_name)
        trials_for_gesture = self.map_gestures_to_trials[rand_gesture_name]
        return trials_for_gesture[random.randint(0, len(trials_for_gesture) - 1)]
    
    # returns a sorted list of gesture names
    def get_gesture_names_sorted(self):
        return sorted(self.map_gestures_to_trials.keys())
    
    # prettify the str()
    def __str__(self):
         return "'{}' : {} gestures and {} total trials".format(self.path, self.get_num_gestures(), self.get_total_num_of_trials())

# Returns all csv filenames in the given directory
# Currently excludes any filenames with 'fulldatastream' in the title
def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) and "fulldatastream" not in filename ]


# Parses and creates Trial objects for all csv files in the given dir
# Returns a dict() mapping (str: gestureName) to (list: Trial objects)
def parse_and_create_gesture_trials_individual( path_to_dir ):
    csvFilenames = find_csv_filenames(path_to_dir)
    
    print("Found {} csv files in {}".format(len(csvFilenames), path_to_dir))
    
    mapGestureNameToTrialList = dict()
    mapGestureNameToMapEndTimeMsToMapSensorToFile = dict()
    for csvFilename in csvFilenames:
        # parse filename into meaningful parts
        filenameNoExt = os.path.splitext(csvFilename)[0];
        filenameParts = filenameNoExt.split("_")
        gestureName = filenameParts[0]
        timeMs = filenameParts[1]
        numRows = int(filenameParts[2])
        sensorName = "Accelerometer" # currently only one sensor but could expand to more
        
        print("gestureName={} timeMs={} numRows={}".format(gestureName, timeMs, numRows))
        
        if gestureName not in mapGestureNameToMapEndTimeMsToMapSensorToFile:
            mapGestureNameToMapEndTimeMsToMapSensorToFile[gestureName] = dict()
        
        if timeMs not in mapGestureNameToMapEndTimeMsToMapSensorToFile[gestureName]:
            mapGestureNameToMapEndTimeMsToMapSensorToFile[gestureName][timeMs] = dict()
        
        mapGestureNameToMapEndTimeMsToMapSensorToFile[gestureName][timeMs][sensorName] = csvFilename
        # print (mapGestureNameToMapEndTimeMsToMapSensorToFile)
    
    print("Found {} gestures".format(len(mapGestureNameToMapEndTimeMsToMapSensorToFile)))
    
    # track the longest array so we can resize accordingly (by padding with zeros currently)
    maxArrayLength = -1
    trialWithMostSensorEvents = None
    
    # Now we need to loop through the data and sort each gesture set by timems values 
    # (so that we have trial 1, 2, 3, etc.)
    for gestureName, mapEndTimeMsToMapSensorToFile in mapGestureNameToMapEndTimeMsToMapSensorToFile.items():
        gestureTrialNum = 0
        mapGestureNameToTrialList[gestureName] = list()
        for endTimeMs in sorted(mapEndTimeMsToMapSensorToFile.keys()):
            mapSensorToFile = mapEndTimeMsToMapSensorToFile[endTimeMs]
            
            accelFilenameWithPath = os.path.join(path_to_dir, mapSensorToFile["Accelerometer"])
            gestureTrial = Trial(gestureName, endTimeMs, gestureTrialNum, accelFilenameWithPath)
            mapGestureNameToTrialList[gestureName].append(gestureTrial)
            
            if maxArrayLength < len(gestureTrial.accel.x):
                maxArrayLength = len(gestureTrial.accel.x)
                trialWithMostSensorEvents = gestureTrial
            
            gestureTrialNum = gestureTrialNum + 1
        
        print("Found {} trials for '{}'".format(len(mapGestureNameToTrialList[gestureName]), gestureName))
    
    # CSE599TODO: You'll want to loop through the sensor signals and preprocess them
    # Some things to explore: padding signal to equalize length between trials, smoothing, detrending, and scaling
    for gestureName, trialList in mapGestureNameToTrialList.items():
        for trial in trialList: 
            # preprocess each signal
            x = 0; # no-op just delete this
          
    return mapGestureNameToTrialList


# Parses and creates Trial objects for all csv files in the given dir
# Returns a dict() mapping (str: gestureName) to (list: Trial objects)
def parse_and_create_gesture_trials( path_to_dir ):
    csvFilenames = find_csv_filenames(path_to_dir)
    
    print("Found {} csv files in {}".format(len(csvFilenames), path_to_dir))
    
    mapGestureNameToTrialList = dict()
    mapGestureNameToMapEndTimeMsToMapSensorToFile = dict()
    for csvFilename in csvFilenames:
        
        # parse filename into meaningful parts
        # print(csvFilename)
        filenameNoExt = os.path.splitext(csvFilename)[0];
        
        filenameParts = filenameNoExt.split("_")
        gestureName = None
        timeMs = None
        numRows = None
        sensorName = "Accelerometer" # currently only one sensor but could expand to more
            
        # Added this conditional on May 15, 2019 because Windows machines created differently formatted
        # filenames from Macs. Windows machines automatically replaced the character "'"
        # with "_", which affects filenames like "Midair Zorro 'Z'_1556730840228_206.csv"
        # which come out like "Midair Zorro _Z__1557937136974_211.csv" instead
        if '__' in filenameNoExt:
            filename_parts1 = filenameNoExt.split("__")
            gestureName = filename_parts1[0]
            gestureName = gestureName.replace('_',"'")
            gestureName += "'"
            
            filename_parts2 = filename_parts1[1].split("_")
            timeMs = filename_parts2[0]
            numRows = filename_parts2[1]
        else:
            filenameParts = filenameNoExt.split("_")
            gestureName = filenameParts[0]
            timeMs = filenameParts[1]
            numRows = int(filenameParts[2])
        
        # print("gestureName={} timeMs={} numRows={}".format(gestureName, timeMs, numRows))
        
        if gestureName not in mapGestureNameToMapEndTimeMsToMapSensorToFile:
            mapGestureNameToMapEndTimeMsToMapSensorToFile[gestureName] = dict()
        
        if timeMs not in mapGestureNameToMapEndTimeMsToMapSensorToFile[gestureName]:
            mapGestureNameToMapEndTimeMsToMapSensorToFile[gestureName][timeMs] = dict()
        
        mapGestureNameToMapEndTimeMsToMapSensorToFile[gestureName][timeMs][sensorName] = csvFilename
        # print (mapGestureNameToMapEndTimeMsToMapSensorToFile)
    
    print("Found {} gestures".format(len(mapGestureNameToMapEndTimeMsToMapSensorToFile)))
    
    # track the longest array so we can resize accordingly (by padding with zeros currently)
    maxArrayLength = -1
    trialWithMostSensorEvents = None
    
    # Now we need to loop through the data and sort each gesture set by timems values 
    # (so that we have trial 1, 2, 3, etc. in order)
    for gestureName, mapEndTimeMsToMapSensorToFile in mapGestureNameToMapEndTimeMsToMapSensorToFile.items():
        gestureTrialNum = 0
        mapGestureNameToTrialList[gestureName] = list()
        for endTimeMs in sorted(mapEndTimeMsToMapSensorToFile.keys()):
            mapSensorToFile = mapEndTimeMsToMapSensorToFile[endTimeMs]
            
            accelFilenameWithPath = os.path.join(path_to_dir, mapSensorToFile["Accelerometer"])
            gestureTrial = Trial(gestureName, endTimeMs, gestureTrialNum, accelFilenameWithPath)
            mapGestureNameToTrialList[gestureName].append(gestureTrial)
            
            if maxArrayLength < len(gestureTrial.accel.x):
                maxArrayLength = len(gestureTrial.accel.x)
                trialWithMostSensorEvents = gestureTrial
            
            gestureTrialNum = gestureTrialNum + 1
        
        print("Found {} trials for '{}'".format(len(mapGestureNameToTrialList[gestureName]), gestureName))
    
    # Perform some preprocessing
    listSamplesPerSec = list()
    listTotalSampleTime = list()
    print("Max trial length across all gesture is '{}' Trial {} with {} sensor events. Padding all arrays to match".
          format(trialWithMostSensorEvents.gestureName, trialWithMostSensorEvents.trialNum, maxArrayLength))
    
    for gestureName, trialList in mapGestureNameToTrialList.items():
        for trial in trialList: 

            listSamplesPerSec.append(trial.accel.samplesPerSecond)
            listTotalSampleTime.append(trial.accel.sampleLengthInSecs)
            
            # preprocess signal before classification and store in new arrays
            trial.accel.x_p = preprocess(trial.accel.x, maxArrayLength)
            trial.accel.y_p = preprocess(trial.accel.y, maxArrayLength)
            trial.accel.z_p = preprocess(trial.accel.z, maxArrayLength)
            trial.accel.mag_p = preprocess(trial.accel.mag, maxArrayLength)
            
            
    print("Avg samples/sec across {} sensor files: {:0.1f}".format(len(listSamplesPerSec), sum(listSamplesPerSec)/len(listSamplesPerSec)))
    print("Avg sample length across {} sensor files: {:0.1f}s".format(len(listTotalSampleTime), sum(listTotalSampleTime)/len(listTotalSampleTime)))
    print()
    return mapGestureNameToTrialList

# Performs some basic preprocesing on rawSignal and returns the preprocessed signal in a new array
def preprocess(rawSignal, maxArrayLength):
    meanFilterWindowSize = 10
    arrayLengthDiff = maxArrayLength - len(rawSignal)

    # CSE599 TODO: add in your own preprocessing here
    # Just smoothing the signal for now with a mean filter
    smoothed = np.convolve(rawSignal, np.ones((meanFilterWindowSize,))/meanFilterWindowSize, mode='valid')
    return smoothed

# Returns the leafs in a path
# From: https://stackoverflow.com/a/8384788
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

# From: https://stackoverflow.com/questions/800197/how-to-get-all-of-the-immediate-subdirectories-in-python
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

# Utility function to extract gesture name from filename
def extract_gesture_name( filename ):
    # leaf = path_leaf(filename)
    tokenSplitPos = filename.index('_')
    gestureName = filename[:tokenSplitPos]
    return gestureName

# Returns the minimum number of trials across all gestures (just in case we accidentally recorded a 
# different number. We should have 5 or 10 each for the A2 assignment)
def get_min_num_of_trials( mapGestureToTrials ):
    minNumTrials = -1 
    for gestureName, trialSet in mapGestureToTrials.items():
        if minNumTrials == -1 or minNumTrials > len(trialSet):
            minNumTrials = len(trialSet)
    return minNumTrials

# returns the total number of trials
def get_total_num_of_trials (mapGestureToTrials):
    numTrials = 0 
    for gestureName, trialSet in mapGestureToTrials.items():
        numTrials = numTrials + len(trialSet)
    return numTrials

# Helper function to align signals. 
# Returns a shifted signal of a based on cross correlation and a roll function
def get_aligned_signal(a, b):
    corr = signal.correlate(a, b, mode='full')
    index_shift = len(a) - np.argmax(corr)
    a_shifted = np.roll(a, index_shift - 1) 
    return a_shifted

# Returns a shifted signal of a based on cross correlation and padding
def get_aligned_signal_cutoff_and_pad(a, b):
    corr = signal.correlate(a, b, mode='full')
    index_shift = len(a) - np.argmax(corr)
    index_shift_abs = abs(index_shift - 1)
    a_shifted_cutoff = None
    if (index_shift - 1) < 0:
        a_shifted_cutoff = a[index_shift_abs:]
        a_shifted_cutoff = np.pad(a_shifted_cutoff, (0, index_shift_abs), 'mean')
    else:
        a_shifted_cutoff = np.pad(a, (index_shift_abs,), 'mean')
        a_shifted_cutoff = a_shifted_cutoff[:len(a)]
    return a_shifted_cutoff

# calculate zero crossings
# See: https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python
# TODO: in future, could have a min_width detection threshold that ignores 
# any changes < min_width samples after an initial zero crossing was detected
# TODO: could also have a mininum height after the zero crossing (withing some window)
# to eliminate noise
def calc_zero_crossings(s):
    # I could not get the speedier solutions to work reliably so here's a 
    # custom non-Pythony solution
    cur_pt = s[0]
    zero_crossings = []
    for ind in range(1, len(s)):
        next_pt = s[ind]
        
        if ((next_pt < 0 and cur_pt > 0) or (next_pt > 0 and cur_pt < 0)):
            zero_crossings.append(ind)
        elif cur_pt == 0 and next_pt > 0:
            # check for previous points less than 0
            # as soon as tmp_pt is not zero, we are done
            tmp_pt = cur_pt
            walk_back_idx = ind
            while(tmp_pt == 0 and walk_back_idx > 0):
                walk_back_idx -= 1
                tmp_pt = s[walk_back_idx]
            
            if tmp_pt < 0:
                zero_crossings.append(ind)
        elif cur_pt == 0 and next_pt < 0:
            # check for previous points greater than 0
            # as soon as tmp_pt is not zero, we are done
            tmp_pt = cur_pt
            walk_back_idx = ind
            while(tmp_pt == 0 and walk_back_idx > 0):
                walk_back_idx -= 1
                tmp_pt = s[walk_back_idx]
            
            if tmp_pt > 0:
                zero_crossings.append(ind)
            
        cur_pt = s[ind]
    return zero_crossings

# Load the data
root_gesture_log_path = '/Users/rashmi/project_local/ProjectDanceGesture/Our stuff/realtimegesturesegmentation/GestureLogs/' # this dir should have a set of gesture sub-directories
print(get_immediate_subdirectories(root_gesture_log_path))
gesture_log_paths = get_immediate_subdirectories(root_gesture_log_path)
map_gesture_sets = dict()
selected_gesture_set = None
selected_gesture_set_Jon_easy = None
selected_gesture_set_Jon_hard = None
for gesture_log_path in gesture_log_paths:
    
    path_to_gesture_log = os.path.join(root_gesture_log_path, gesture_log_path)
    print("Reading in:", path_to_gesture_log)
    map_gestures_to_trials = parse_and_create_gesture_trials(path_to_gesture_log)
    gesture_set = GestureSet(gesture_log_path, map_gestures_to_trials)
    map_gesture_sets[gesture_set.get_base_path()] = gesture_set
    if "Rashmi" in gesture_log_path:
        selected_gesture_set = gesture_set
        
if selected_gesture_set is not None:
    print("The selected gesture set:", selected_gesture_set)

def get_gesture_set_with_str(str):
    for base_path, gesture_set in map_gesture_sets.items():
        if str in base_path:
            return gesture_set
    return None    

# plot class
class AccelPlot:

    ARDUINO_CSV_INDEX_TIMESTAMP = 0
    ARDUINO_CSV_INDEX_X = 1
    ARDUINO_CSV_INDEX_Y = 2
    ARDUINO_CSV_INDEX_Z = 3

    # constr
    def __init__(self, fig, ax, txt, score_txt, str_port, ax2, ax3, ax4, ax5, baud_rate=9600, max_length=100):
        # open serial port
        self.ser = serial.Serial(str_port, 9600)

        self.fig = fig
        self.ax = ax
        self.txt = txt
        self.score_txt = score_txt

        self.ax2 = ax2
        self.ax3 = ax3
        self.ax4 = ax4
        self.ax5 = ax5

        self.bestAggregateGesture = None

        self.data = list()
        num_values_to_plot = 4
        for i in range(0, num_values_to_plot):
            buf = deque()
            self.data.append(buf)

        self.x = self.data[0]
        self.y = self.data[1]
        self.z = self.data[2]
        self.mag = self.data[3]
        self.time = deque()

        self.max_length = max_length # max length to show

        # segmentation stuff
        self.window_length = 30
        self.window_step = 10
        self.window_buffer = deque()
        self.current_event = None #tuple (time list, val list)

        self.event_counter = 0

    def __add_to_buffer(self, buf, val):
        if len(buf) < self.max_length:
            buf.append(val)
        else:
            buf.popleft()
            buf.append(val)


    def add_data(self, csv_data):
        self.__add_to_buffer(self.time, csv_data[AccelPlot.ARDUINO_CSV_INDEX_TIMESTAMP])
        self.__add_to_buffer(self.x, csv_data[AccelPlot.ARDUINO_CSV_INDEX_X])
        self.__add_to_buffer(self.y, csv_data[AccelPlot.ARDUINO_CSV_INDEX_Y])
        self.__add_to_buffer(self.z, csv_data[AccelPlot.ARDUINO_CSV_INDEX_Z])
        mag = math.sqrt(csv_data[AccelPlot.ARDUINO_CSV_INDEX_X] ** 2 + 
            csv_data[AccelPlot.ARDUINO_CSV_INDEX_Y] ** 2 + 
            csv_data[AccelPlot.ARDUINO_CSV_INDEX_Z]** 2) 
        self.__add_to_buffer(self.mag, mag)

        # add mag to window buffer used for segmentation
        self.window_buffer.append(mag)


    def segment_event(self):
        segment_result = None
        if len(self.window_buffer) >= self.window_length:
            # you may need/want to change these tolerances
            min_max_begin_segment_threshold = 90 
            min_max_continue_segment_threshold = 25 #lower threshold for continuing event
            min_event_length_ms = 600

            # analyze the buffer
            s = np.array(self.window_buffer)
            min_max_diff = abs(np.max(s) - np.min(s))

            if min_max_diff > min_max_begin_segment_threshold and self.current_event is None:
                print("begin segment!", min_max_diff)
                
                start_idx = len(self.time) - self.window_length
                end_idx = len(self.time)
                
                t = list(itertools.islice(self.time, start_idx, end_idx))
                s = list(itertools.islice(self.mag, start_idx, end_idx))
                x_seg = list(itertools.islice(self.x, start_idx, end_idx))
                y_seg = list(itertools.islice(self.y, start_idx, end_idx))
                z_seg = list(itertools.islice(self.z, start_idx, end_idx))

                # preprocess signal before classification and store in new arrays
                x_p = list(preprocess(x_seg, len(x_seg)))
                y_p = list(preprocess(y_seg, len(y_seg)))
                z_p = list(preprocess(z_seg, len(z_seg)))
                mag_p = list(preprocess(s, len(s)))

                self.ax.axvline(self.time[-self.window_length], ls='--', color='black', linewidth=1, alpha=0.8)
                self.current_event = (t, s, x_seg, y_seg, z_seg, x_p, y_p, z_p, mag_p)
            elif self.current_event is not None:
                # we are in the middle or end of a potential event
                if min_max_diff >= min_max_continue_segment_threshold: 
                    print("continue segment", min_max_diff)
       
                    start_idx = len(self.time) - self.window_step
                    end_idx = len(self.time)
                    
                    t = list(itertools.islice(self.time, start_idx, end_idx))
                    s = list(itertools.islice(self.mag, start_idx, end_idx))
                    x_seg = list(itertools.islice(self.x, start_idx, end_idx))
                    y_seg = list(itertools.islice(self.y, start_idx, end_idx))
                    z_seg = list(itertools.islice(self.z, start_idx, end_idx))

                    # preprocess signal before classification and store in new arrays
                    x_p = list(preprocess(x_seg, len(x_seg)))
                    y_p = list(preprocess(y_seg, len(y_seg)))
                    z_p = list(preprocess(z_seg, len(z_seg)))
                    mag_p = list(preprocess(s, len(s)))

                    self.current_event[0].extend(t)
                    self.current_event[1].extend(s)
                    self.current_event[2].extend(x_seg)
                    self.current_event[3].extend(y_seg)
                    self.current_event[4].extend(z_seg)
                    self.current_event[5].extend(x_p)
                    self.current_event[6].extend(y_p)
                    self.current_event[7].extend(z_p)
                    self.current_event[8].extend(mag_p)
                elif min_max_diff < min_max_continue_segment_threshold:
                    print("finish segment", min_max_diff)
                    event_time = self.current_event[0]
                    event_length_ms = event_time[-1] - event_time[0]
                    if event_length_ms > min_event_length_ms:
                        self.ax.axvspan(event_time[0], event_time[-1], color='red', alpha=0.4)
                        self.ax.axvline(event_time[-1], ls='--', color='black', linewidth=1, alpha=0.8)
                    else:
                        print("discarded event for being too short")

                    segment_result = {'time' : self.current_event[0],
                                      'mag' : self.current_event[1],
                                      'x' : self.current_event[2],
                                      'y' : self.current_event[3],
                                      'z' : self.current_event[4],
                                      'x_p' : self.current_event[5],
                                      'y_p' : self.current_event[6],
                                      'z_p' : self.current_event[7],
                                      'mag_p' : self.current_event[8]}
                
                    self.current_event = None # clear events

            new_length = self.window_length - self.window_step
            while len(self.window_buffer) > new_length:
                self.window_buffer.popleft()

        return segment_result
    
    def graphComparison(self, segment_result, gesture_name, 
                        gestureDictionary, plt_lines_xp, plt_lines_yp, 
                        plt_lines_zp, plt_lines_magp):
        aggGesture = gestureDictionary[gesture_name]
        aggGesturex = aggGesture['x_p']
        aggGesturey = aggGesture['y_p']
        aggGesturez = aggGesture['z_p']
        aggGesturemag = aggGesture['mag_p']
        t = segment_result['time']
        x_p = segment_result['x_p']
        y_p = segment_result['y_p']
        z_p = segment_result['z_p']
        mag_p = segment_result['mag_p']
            

    
    def classify_event(self, segment_result):
        # print("classify event", segment_result)
        t = segment_result['time']
        x = segment_result['x']
        y = segment_result['y']
        z = segment_result['z']
        mag = segment_result['mag']
        x_p = segment_result['x_p']
        y_p = segment_result['y_p']
        z_p = segment_result['z_p']
        mag_p = segment_result['mag_p']

        if self.event_counter == 0:
            print("Dance move: Pulling")
            currentSignal = np.array(x_p)
            signalToCompare = np.array(pulling_nicole_agg_x_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_x_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # nicole_dist_x_p, path = fastdtw(alignedSignal_x_p, pulling_nicole_agg_x_p, dist=euclidean)
            nicole_euclid_distance_x = distance.euclidean(alignedSignal_x_p, signalToComparePad)

            currentSignal = np.array(y_p)
            singnalToCompare = np.array(pulling_nicole_agg_y_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_y_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # nicole_dist_y_p, path = fastdtw(alignedSignal_y_p, pulling_nicole_agg_y_p, dist=euclidean)
            nicole_euclid_distance_y = distance.euclidean(alignedSignal_y_p, signalToComparePad)

            currentSignal = np.array(z_p)
            singnalToCompare = np.array(pulling_nicole_agg_z_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_z_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # nicole_dist_z_p, path = fastdtw(alignedSignal_z_p, pulling_nicole_agg_z_p, dist=euclidean)
            nicole_euclid_distance_z = distance.euclidean(alignedSignal_z_p, signalToComparePad)

            currentSignal = np.array(mag_p)
            singnalToCompare = np.array(pulling_nicole_agg_mag_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_mag_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # nicole_dist_mag_p, path = fastdtw(alignedSignal_mag_p, pulling_nicole_agg_mag_p, dist=euclidean)
            nicole_euclid_distance_mag = distance.euclidean(alignedSignal_mag_p, signalToComparePad)

            # score_comp_to_nicole = nicole_dist_x_p + nicole_dist_y_p + nicole_dist_z_p + nicole_dist_mag_p
            score_comp_to_nicole = nicole_euclid_distance_x + nicole_euclid_distance_y + nicole_euclid_distance_z + nicole_euclid_distance_mag

            currentSignal = np.array(x_p)
            singnalToCompare = np.array(pulling_rashmi_agg_x_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_x_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # rashmi_dist_x_p, path = fastdtw(alignedSignal_x_p, pulling_rashmi_agg_x_p, dist=euclidean)
            rashmi_euclid_distance_x = distance.euclidean(alignedSignal_x_p, signalToComparePad)

            currentSignal = np.array(y_p)
            singnalToCompare = np.array(pulling_rashmi_agg_y_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_y_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # rashmi_dist_y_p, path = fastdtw(alignedSignal_y_p, pulling_rashmi_agg_y_p, dist=euclidean)
            rashmi_euclid_distance_y = distance.euclidean(alignedSignal_y_p, signalToComparePad)

            currentSignal = np.array(z_p)
            singnalToCompare = np.array(pulling_rashmi_agg_z_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_z_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # rashmi_dist_z_p, path = fastdtw(alignedSignal_z_p, pulling_rashmi_agg_z_p, dist=euclidean)
            rashmi_euclid_distance_z = distance.euclidean(alignedSignal_z_p, signalToComparePad)
            
            currentSignal = np.array(mag_p)
            singnalToCompare = np.array(pulling_rashmi_agg_mag_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_mag_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # rashmi_dist_mag_p, path = fastdtw(alignedSignal_mag_p, pulling_rashmi_agg_mag_p, dist=euclidean)
            rashmi_euclid_distance_mag = distance.euclidean(alignedSignal_mag_p, signalToComparePad)

            # score_comp_to_rashmi = rashmi_dist_x_p + rashmi_dist_y_p + rashmi_dist_z_p + rashmi_dist_mag_p
            score_comp_to_rashmi = rashmi_euclid_distance_x + rashmi_euclid_distance_y + rashmi_euclid_distance_z + rashmi_euclid_distance_mag

            if score_comp_to_nicole < score_comp_to_rashmi:
                print("Matched with Nicole")
                print(score_comp_to_nicole)
                self.score_txt.set_text("Score: " + str(score_comp_to_nicole))
            else:
                print("Matched with Rashmi")
                print(score_comp_to_rashmi)
                self.score_txt.set_text("Score: " + str(score_comp_to_rashmi))
            self.event_counter = self.event_counter + 1
            self.txt.set_text("Do the dance move: Dont pay Attention")

        elif self.event_counter == 1:
            print("Dance move: Dont pay Attention")
            currentSignal = np.array(x_p)
            signalToCompare = np.array(dpa_nicole_agg_x_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_x_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # nicole_dist_x_p, path = fastdtw(alignedSignal_x_p, dpa_nicole_agg_x_p, dist=euclidean)
            nicole_euclid_distance_x = distance.euclidean(alignedSignal_x_p, signalToComparePad)

            currentSignal = np.array(y_p)
            singnalToCompare = np.array(dpa_nicole_agg_y_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_y_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # nicole_dist_y_p, path = fastdtw(alignedSignal_y_p, dpa_nicole_agg_y_p, dist=euclidean)
            nicole_euclid_distance_y = distance.euclidean(alignedSignal_y_p, signalToComparePad)

            currentSignal = np.array(z_p)
            singnalToCompare = np.array(dpa_nicole_agg_z_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_z_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # nicole_dist_z_p, path = fastdtw(alignedSignal_z_p, dpa_nicole_agg_z_p, dist=euclidean)
            nicole_euclid_distance_z = distance.euclidean(alignedSignal_z_p, signalToComparePad)

            currentSignal = np.array(mag_p)
            singnalToCompare = np.array(dpa_nicole_agg_mag_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_mag_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # nicole_dist_mag_p, path = fastdtw(alignedSignal_mag_p, dpa_nicole_agg_mag_p, dist=euclidean)
            nicole_euclid_distance_mag = distance.euclidean(alignedSignal_mag_p, signalToComparePad)

            # score_comp_to_nicole = nicole_dist_x_p + nicole_dist_y_p + nicole_dist_z_p + nicole_dist_mag_p
            score_comp_to_nicole = nicole_euclid_distance_x + nicole_euclid_distance_y + nicole_euclid_distance_z + nicole_euclid_distance_mag

            currentSignal = np.array(x_p)
            singnalToCompare = np.array(dpa_rashmi_agg_x_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_x_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # rashmi_dist_x_p, path = fastdtw(alignedSignal_x_p, dpa_rashmi_agg_x_p, dist=euclidean)
            rashmi_euclid_distance_x = distance.euclidean(alignedSignal_x_p, signalToComparePad)

            currentSignal = np.array(y_p)
            singnalToCompare = np.array(dpa_rashmi_agg_y_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_y_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # rashmi_dist_y_p, path = fastdtw(alignedSignal_y_p, dpa_rashmi_agg_y_p, dist=euclidean)
            rashmi_euclid_distance_y = distance.euclidean(alignedSignal_y_p, signalToComparePad)

            currentSignal = np.array(z_p)
            singnalToCompare = np.array(dpa_rashmi_agg_z_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_z_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # rashmi_dist_z_p, path = fastdtw(alignedSignal_z_p, dpa_rashmi_agg_z_p, dist=euclidean)
            rashmi_euclid_distance_z = distance.euclidean(alignedSignal_z_p, signalToComparePad)
            
            currentSignal = np.array(mag_p)
            singnalToCompare = np.array(dpa_rashmi_agg_mag_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_mag_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # rashmi_dist_mag_p, path = fastdtw(alignedSignal_mag_p, dpa_rashmi_agg_mag_p, dist=euclidean)
            rashmi_euclid_distance_mag = distance.euclidean(alignedSignal_mag_p, signalToComparePad)

            # score_comp_to_rashmi = rashmi_dist_x_p + rashmi_dist_y_p + rashmi_dist_z_p + rashmi_dist_mag_p
            score_comp_to_rashmi = rashmi_euclid_distance_x + rashmi_euclid_distance_y + rashmi_euclid_distance_z + rashmi_euclid_distance_mag

            if score_comp_to_nicole < score_comp_to_rashmi:
                print("Matched with Nicole")
                print(score_comp_to_nicole)
                self.score_txt.set_text("Score: " + str(score_comp_to_nicole))
            else:
                print("Matched with Rashmi")
                print(score_comp_to_rashmi)
                self.score_txt.set_text("Score: " + str(score_comp_to_rashmi))
            self.event_counter = self.event_counter + 1
            self.txt.set_text("Do the dance move: Flip")

        elif self.event_counter == 2:
            print("Dance move: Flip")
            currentSignal = np.array(x_p)
            signalToCompare = np.array(flip_nicole_agg_x_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_x_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # nicole_dist_x_p, path = fastdtw(alignedSignal_x_p, flip_nicole_agg_x_p, dist=euclidean)
            nicole_euclid_distance_x = distance.euclidean(alignedSignal_x_p, signalToComparePad)

            currentSignal = np.array(y_p)
            singnalToCompare = np.array(flip_nicole_agg_y_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_y_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # nicole_dist_y_p, path = fastdtw(alignedSignal_y_p, flip_nicole_agg_y_p, dist=euclidean)
            nicole_euclid_distance_y = distance.euclidean(alignedSignal_y_p, signalToComparePad)

            currentSignal = np.array(z_p)
            singnalToCompare = np.array(flip_nicole_agg_z_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_z_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # nicole_dist_z_p, path = fastdtw(alignedSignal_z_p, flip_nicole_agg_z_p, dist=euclidean)
            nicole_euclid_distance_z = distance.euclidean(alignedSignal_z_p, signalToComparePad)

            currentSignal = np.array(mag_p)
            singnalToCompare = np.array(flip_nicole_agg_mag_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_mag_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # nicole_dist_mag_p, path = fastdtw(alignedSignal_mag_p, flip_nicole_agg_mag_p, dist=euclidean)
            nicole_euclid_distance_mag = distance.euclidean(alignedSignal_mag_p, signalToComparePad)

            # score_comp_to_nicole = nicole_dist_x_p + nicole_dist_y_p + nicole_dist_z_p + nicole_dist_mag_p
            score_comp_to_nicole = nicole_euclid_distance_x + nicole_euclid_distance_y + nicole_euclid_distance_z + nicole_euclid_distance_mag

            currentSignal = np.array(x_p)
            singnalToCompare = np.array(flip_rashmi_agg_x_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_x_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # rashmi_dist_x_p, path = fastdtw(alignedSignal_x_p, flip_rashmi_agg_x_p, dist=euclidean)
            rashmi_euclid_distance_x = distance.euclidean(alignedSignal_x_p, signalToComparePad)

            currentSignal = np.array(y_p)
            singnalToCompare = np.array(flip_rashmi_agg_y_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_y_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # rashmi_dist_y_p, path = fastdtw(alignedSignal_y_p, flip_rashmi_agg_y_p, dist=euclidean)
            rashmi_euclid_distance_y = distance.euclidean(alignedSignal_y_p, signalToComparePad)

            currentSignal = np.array(z_p)
            singnalToCompare = np.array(flip_rashmi_agg_z_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_z_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # rashmi_dist_z_p, path = fastdtw(alignedSignal_z_p, flip_rashmi_agg_z_p, dist=euclidean)
            rashmi_euclid_distance_z = distance.euclidean(alignedSignal_z_p, signalToComparePad)
            
            currentSignal = np.array(mag_p)
            singnalToCompare = np.array(flip_rashmi_agg_mag_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_mag_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # rashmi_dist_mag_p, path = fastdtw(alignedSignal_mag_p, flip_rashmi_agg_mag_p, dist=euclidean)
            rashmi_euclid_distance_mag = distance.euclidean(alignedSignal_mag_p, signalToComparePad)

            # score_comp_to_rashmi = rashmi_dist_x_p + rashmi_dist_y_p + rashmi_dist_z_p + rashmi_dist_mag_p
            score_comp_to_rashmi = rashmi_euclid_distance_x + rashmi_euclid_distance_y + rashmi_euclid_distance_z + rashmi_euclid_distance_mag

            if score_comp_to_nicole < score_comp_to_rashmi:
                print("Matched with Nicole")
                print(score_comp_to_nicole)
                self.score_txt.set_text("Score: " + str(score_comp_to_nicole))
            else:
                print("Matched with Rashmi")
                print(score_comp_to_rashmi)
                self.score_txt.set_text("Score: " + str(score_comp_to_rashmi))

            self.event_counter = self.event_counter + 1
            self.txt.set_text("Do the dance move: Clap")

        elif self.event_counter == 3:
            print("Dance move: Clap")
            currentSignal = np.array(x_p)
            signalToCompare = np.array(clap_nicole_agg_x_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_x_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # nicole_dist_x_p, path = fastdtw(alignedSignal_x_p, clap_nicole_agg_x_p, dist=euclidean)
            nicole_euclid_distance_x = distance.euclidean(alignedSignal_x_p, signalToComparePad)

            currentSignal = np.array(y_p)
            singnalToCompare = np.array(clap_nicole_agg_y_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_y_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # nicole_dist_y_p, path = fastdtw(alignedSignal_y_p, clap_nicole_agg_y_p, dist=euclidean)
            nicole_euclid_distance_y = distance.euclidean(alignedSignal_y_p, signalToComparePad)

            currentSignal = np.array(z_p)
            singnalToCompare = np.array(clap_nicole_agg_z_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_z_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # nicole_dist_z_p, path = fastdtw(alignedSignal_z_p, clap_nicole_agg_z_p, dist=euclidean)
            nicole_euclid_distance_z = distance.euclidean(alignedSignal_z_p, signalToComparePad)

            currentSignal = np.array(mag_p)
            singnalToCompare = np.array(clap_nicole_agg_mag_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_mag_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # nicole_dist_mag_p, path = fastdtw(alignedSignal_mag_p, clap_nicole_agg_mag_p, dist=euclidean)
            nicole_euclid_distance_mag = distance.euclidean(alignedSignal_mag_p, signalToComparePad)

            # score_comp_to_nicole = nicole_dist_x_p + nicole_dist_y_p + nicole_dist_z_p + nicole_dist_mag_p
            score_comp_to_nicole = nicole_euclid_distance_x + nicole_euclid_distance_y + nicole_euclid_distance_z + nicole_euclid_distance_mag

            currentSignal = np.array(x_p)
            singnalToCompare = np.array(clap_rashmi_agg_x_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_x_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # rashmi_dist_x_p, path = fastdtw(alignedSignal_x_p, clap_rashmi_agg_x_p, dist=euclidean)
            rashmi_euclid_distance_x = distance.euclidean(alignedSignal_x_p, signalToComparePad)

            currentSignal = np.array(y_p)
            singnalToCompare = np.array(clap_rashmi_agg_y_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_y_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # rashmi_dist_y_p, path = fastdtw(alignedSignal_y_p, clap_rashmi_agg_y_p, dist=euclidean)
            rashmi_euclid_distance_y = distance.euclidean(alignedSignal_y_p, signalToComparePad)

            currentSignal = np.array(z_p)
            singnalToCompare = np.array(clap_rashmi_agg_z_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_z_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # rashmi_dist_z_p, path = fastdtw(alignedSignal_z_p, clap_rashmi_agg_z_p, dist=euclidean)
            rashmi_euclid_distance_z = distance.euclidean(alignedSignal_z_p, signalToComparePad)
            
            currentSignal = np.array(mag_p)
            singnalToCompare = np.array(clap_rashmi_agg_mag_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_mag_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # rashmi_dist_mag_p, path = fastdtw(alignedSignal_mag_p, clap_rashmi_agg_mag_p, dist=euclidean)
            rashmi_euclid_distance_mag = distance.euclidean(alignedSignal_mag_p, signalToComparePad)

            # score_comp_to_rashmi = rashmi_dist_x_p + rashmi_dist_y_p + rashmi_dist_z_p + rashmi_dist_mag_p
            score_comp_to_rashmi = rashmi_euclid_distance_x + rashmi_euclid_distance_y + rashmi_euclid_distance_z + rashmi_euclid_distance_mag

            if score_comp_to_nicole < score_comp_to_rashmi:
                print("Matched with Nicole")
                print(score_comp_to_nicole)
                self.score_txt.set_text("Score: " + str(score_comp_to_nicole))
            else:
                print("Matched with Rashmi")
                print(score_comp_to_rashmi)
                self.score_txt.set_text("Score: " + str(score_comp_to_rashmi))

            self.event_counter = self.event_counter + 1
            self.txt.set_text("Do the dance move: Elephant Arm")

        elif self.event_counter == 4:
            print("Dance move: Elephant Arm")
            currentSignal = np.array(x_p)
            signalToCompare = np.array(ea_nicole_agg_x_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_x_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # nicole_dist_x_p, path = fastdtw(alignedSignal_x_p, ea_nicole_agg_x_p, dist=euclidean)
            nicole_euclid_distance_x = distance.euclidean(alignedSignal_x_p, signalToComparePad)

            currentSignal = np.array(y_p)
            singnalToCompare = np.array(ea_nicole_agg_y_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_y_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # nicole_dist_y_p, path = fastdtw(alignedSignal_y_p, ea_nicole_agg_y_p, dist=euclidean)
            nicole_euclid_distance_y = distance.euclidean(alignedSignal_y_p, signalToComparePad)

            currentSignal = np.array(z_p)
            singnalToCompare = np.array(ea_nicole_agg_z_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_z_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # nicole_dist_z_p, path = fastdtw(alignedSignal_z_p, ea_nicole_agg_z_p, dist=euclidean)
            nicole_euclid_distance_z = distance.euclidean(alignedSignal_z_p, signalToComparePad)

            currentSignal = np.array(mag_p)
            singnalToCompare = np.array(ea_nicole_agg_mag_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_mag_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # nicole_dist_mag_p, path = fastdtw(alignedSignal_mag_p, ea_nicole_agg_mag_p, dist=euclidean)
            nicole_euclid_distance_mag = distance.euclidean(alignedSignal_mag_p, signalToComparePad)

            # score_comp_to_nicole = nicole_dist_x_p + nicole_dist_y_p + nicole_dist_z_p + nicole_dist_mag_p
            score_comp_to_nicole = nicole_euclid_distance_x + nicole_euclid_distance_y + nicole_euclid_distance_z + nicole_euclid_distance_mag

            currentSignal = np.array(x_p)
            singnalToCompare = np.array(ea_rashmi_agg_x_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_x_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # rashmi_dist_x_p, path = fastdtw(alignedSignal_x_p, ea_rashmi_agg_x_p, dist=euclidean)
            rashmi_euclid_distance_x = distance.euclidean(alignedSignal_x_p, signalToComparePad)

            currentSignal = np.array(y_p)
            singnalToCompare = np.array(ea_rashmi_agg_y_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_y_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # rashmi_dist_y_p, path = fastdtw(alignedSignal_y_p, ea_rashmi_agg_y_p, dist=euclidean)
            rashmi_euclid_distance_y = distance.euclidean(alignedSignal_y_p, signalToComparePad)

            currentSignal = np.array(z_p)
            singnalToCompare = np.array(ea_rashmi_agg_z_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_z_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # rashmi_dist_z_p, path = fastdtw(alignedSignal_z_p, ea_rashmi_agg_z_p, dist=euclidean)
            rashmi_euclid_distance_z = distance.euclidean(alignedSignal_z_p, signalToComparePad)
            
            currentSignal = np.array(mag_p)
            singnalToCompare = np.array(ea_rashmi_agg_mag_p)
            signalToComparePad = signalToCompare
            currentSignalPad = currentSignal
            if signalToCompare.shape[0] < currentSignal.shape[0]:
                signalToComparePad = np.pad(signalToCompare, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            else:
                currentSignalPad = np.pad(currentSignal, (0, abs(signalToCompare.shape[0] - currentSignal.shape[0])), 'mean')
            alignedSignal_mag_p = get_aligned_signal_cutoff_and_pad(currentSignalPad, signalToComparePad)
            # rashmi_dist_mag_p, path = fastdtw(alignedSignal_mag_p, ea_rashmi_agg_mag_p, dist=euclidean)
            rashmi_euclid_distance_mag = distance.euclidean(alignedSignal_mag_p, signalToComparePad)

            # score_comp_to_rashmi = rashmi_dist_x_p + rashmi_dist_y_p + rashmi_dist_z_p + rashmi_dist_mag_p
            score_comp_to_rashmi = rashmi_euclid_distance_x + rashmi_euclid_distance_y + rashmi_euclid_distance_z + rashmi_euclid_distance_mag

            if score_comp_to_nicole < score_comp_to_rashmi:
                print("Matched with Nicole")
                print(score_comp_to_nicole)
                self.score_txt.set_text("Score: " + str(score_comp_to_nicole))
            else:
                print("Matched with Rashmi")
                print(score_comp_to_rashmi)
                self.score_txt.set_text("Score: " + str(score_comp_to_rashmi))
            self.event_counter = 0
            self.txt.set_text("Do the dance move: Pulling")

    # update plot
    def update(self, frameNum, args, plt_lines, plt_linesx, plt_linesy, plt_linesz, plt_linesmag):
        try:
            while self.ser.in_waiting:
                line = self.ser.readline()
                line = line.decode('utf-8')
                data = line.split(",")
                data = [int(val.strip()) for val in line.split(",")]
                #print(data)
                self.add_data(data)

                segment_result = self.segment_event()
                if segment_result != None:
                    cls_result = self.classify_event(segment_result)

                    plt_linesx[0].set_data(segment_result['time'], segment_result['x_p'])
                    plt_linesy[0].set_data(segment_result['time'], segment_result['y_p'])
                    plt_linesz[0].set_data(segment_result['time'], segment_result['z_p'])
                    plt_linesmag[0].set_data(segment_result['time'], segment_result['mag_p'])

                    bestGesture = self.bestAggregateGesture;
                    longest = max(segment_result['time'][-1] - segment_result['time'][0], bestGesture['time'][-1])
                    plt_linesx[1].set_data(bestGesture['time'], bestGesture['x_p'])
                    plt_linesy[1].set_data(bestGesture['time'], bestGesture['y_p'])
                    plt_linesz[1].set_data(bestGesture['time'], bestGesture['z_p'])
                    plt_linesmag[1].set_data(bestGesture['time'], bestGesture['mag_p'])

                # plot the data
                for i in range(0, len(plt_lines)):
                    plt_lines[i].set_data(self.time, self.data[i])
                self.ax.set_xlim(self.time[0], self.time[-1])

        except KeyboardInterrupt:
            print('exiting')

        # except Exception as e:
        #     print('Error '+ str(e))

        #return a0,
        return plt_lines

    # clean up
    def close(self):
        # close serial
        self.ser.flush()
        self.ser.close()

#Create Aggregate (Nicole)
nicole_gestures = get_gesture_set_with_str("Nicole")
nicole_dictionary = {}
nicole_dictionary["pulling"] = {}
nicole_dictionary["Dont pay Attention"] = {}
nicole_dictionary["Flip"] = {}
nicole_dictionary['Clap'] = {}
nicole_dictionary['Elephant Arm'] = {}
print("Nicole:", nicole_gestures)

currDictionary = nicole_dictionary['pulling']
pulling_nicole_agg_x_p = nicole_gestures.create_aggregate_signal("Pulling", "x_p")
currDictionary['x_p'] = pulling_nicole_agg_x_p
pulling_nicole_agg_y_p = nicole_gestures.create_aggregate_signal("Pulling", "y_p")
currDictionary['y_p'] = pulling_nicole_agg_y_p
pulling_nicole_agg_z_p = nicole_gestures.create_aggregate_signal("Pulling", "z_p")
currDictionary['z_p'] = pulling_nicole_agg_z_p
pulling_nicole_agg_mag_p = nicole_gestures.create_aggregate_signal("Pulling", "mag_p")
currDictionary['mag_p'] = pulling_nicole_agg_mag_p

currDictionary = nicole_dictionary['Dont pay Attention']
dpa_nicole_agg_x_p = nicole_gestures.create_aggregate_signal("Dont pay Attention", "x_p")
currDictionary['x_p'] = dpa_nicole_agg_x_p
dpa_nicole_agg_y_p = nicole_gestures.create_aggregate_signal("Dont pay Attention", "y_p")
currDictionary['y_p'] = dpa_nicole_agg_y_p
dpa_nicole_agg_z_p = nicole_gestures.create_aggregate_signal("Dont pay Attention", "z_p")
currDictionary['z_p'] = dpa_nicole_agg_z_p
dpa_nicole_agg_mag_p = nicole_gestures.create_aggregate_signal("Dont pay Attention", "mag_p")
currDictionary['mag_p'] = dpa_nicole_agg_mag_p

currDictionary = nicole_dictionary['Flip']
flip_nicole_agg_x_p = nicole_gestures.create_aggregate_signal("Flip", "x_p")
currDictionary['x_p'] = flip_nicole_agg_x_p
flip_nicole_agg_y_p = nicole_gestures.create_aggregate_signal("Flip", "y_p")
currDictionary['y_p'] = flip_nicole_agg_y_p
flip_nicole_agg_z_p = nicole_gestures.create_aggregate_signal("Flip", "z_p")
currDictionary['z_p'] = flip_nicole_agg_z_p
flip_nicole_agg_mag_p = nicole_gestures.create_aggregate_signal("Flip", "mag_p")
currDictionary['mag_p'] = flip_nicole_agg_mag_p

currDictionary = nicole_dictionary['Clap']
clap_nicole_agg_x_p = nicole_gestures.create_aggregate_signal("Clap", "x_p")
currDictionary['x_p'] = clap_nicole_agg_x_p 
clap_nicole_agg_y_p = nicole_gestures.create_aggregate_signal("Clap", "y_p")
currDictionary['y_p'] = clap_nicole_agg_y_p 
clap_nicole_agg_z_p = nicole_gestures.create_aggregate_signal("Clap", "z_p")
currDictionary['z_p'] = clap_nicole_agg_z_p 
clap_nicole_agg_mag_p = nicole_gestures.create_aggregate_signal("Clap", "mag_p")
currDictionary['mag_p'] = clap_nicole_agg_mag_p 

currDictionary = nicole_dictionary['Elephant Arm']
ea_nicole_agg_x_p = nicole_gestures.create_aggregate_signal("Elephant Arm", "x_p")
currDictionary['x_p'] = ea_nicole_agg_x_p
ea_nicole_agg_y_p = nicole_gestures.create_aggregate_signal("Elephant Arm", "y_p")
currDictionary['y_p'] = ea_nicole_agg_y_p
ea_nicole_agg_z_p = nicole_gestures.create_aggregate_signal("Elephant Arm", "z_p")
currDictionary['z_p'] = ea_nicole_agg_z_p
ea_nicole_agg_mag_p = nicole_gestures.create_aggregate_signal("Elephant Arm", "mag_p")
currDictionary['mag_p'] = ea_nicole_agg_mag_p

#Create Aggregate (Rashmi)
rashmi_gestures = get_gesture_set_with_str("Rashmi")
print("Rashmi:", rashmi_gestures)
rashmi_dictionary = {}
rashmi_dictionary["pulling"] = {}
rashmi_dictionary["Dont pay Attention"] = {}
rashmi_dictionary["Flip"] = {}
rashmi_dictionary['Clap'] = {}
rashmi_dictionary['Elephant Arm'] = {}
currDictionary = rashmi_dictionary["pulling"]
pulling_rashmi_agg_x_p = rashmi_gestures.create_aggregate_signal("Pulling", "x_p")
currDictionary['x_p'] = pulling_rashmi_agg_x_p
pulling_rashmi_agg_y_p = rashmi_gestures.create_aggregate_signal("Pulling", "y_p")
currDictionary['y_p'] = pulling_rashmi_agg_y_p
pulling_rashmi_agg_z_p = rashmi_gestures.create_aggregate_signal("Pulling", "z_p")
currDictionary['z_p'] = pulling_rashmi_agg_z_p
pulling_rashmi_agg_mag_p = rashmi_gestures.create_aggregate_signal("Pulling", "mag_p")
currDictionary['mag_p'] = pulling_rashmi_agg_mag_p

currDictionary = rashmi_dictionary['Dont pay Attention']
dpa_rashmi_agg_x_p = rashmi_gestures.create_aggregate_signal("Dont pay Attention", "x_p")
currDictionary['x_p'] = dpa_rashmi_agg_x_p
dpa_rashmi_agg_y_p = rashmi_gestures.create_aggregate_signal("Dont pay Attention", "y_p")
currDictionary['y_p'] = dpa_rashmi_agg_y_p
dpa_rashmi_agg_z_p = rashmi_gestures.create_aggregate_signal("Dont pay Attention", "z_p")
currDictionary['z_p'] = dpa_rashmi_agg_z_p
dpa_rashmi_agg_mag_p = rashmi_gestures.create_aggregate_signal("Dont pay Attention", "mag_p")
currDictionary['mag_p'] = dpa_rashmi_agg_mag_p

currDictionary = rashmi_dictionary["Flip"]
flip_rashmi_agg_x_p = rashmi_gestures.create_aggregate_signal("Flip", "x_p")
currDictionary['x_p'] = flip_rashmi_agg_x_p
flip_rashmi_agg_y_p = rashmi_gestures.create_aggregate_signal("Flip", "y_p")
currDictionary['y_p'] = flip_rashmi_agg_y_p
flip_rashmi_agg_z_p = rashmi_gestures.create_aggregate_signal("Flip", "z_p")
currDictionary['z_p'] = flip_rashmi_agg_z_p
flip_rashmi_agg_mag_p = rashmi_gestures.create_aggregate_signal("Flip", "mag_p")
currDictionary['mag_p'] = flip_rashmi_agg_mag_p

currDictionary = rashmi_dictionary['Clap']
clap_rashmi_agg_x_p = rashmi_gestures.create_aggregate_signal("Clap", "x_p")
currDictionary['x_p'] = clap_rashmi_agg_x_p
clap_rashmi_agg_y_p = rashmi_gestures.create_aggregate_signal("Clap", "y_p")
currDictionary['y_p'] = clap_rashmi_agg_y_p
clap_rashmi_agg_z_p = rashmi_gestures.create_aggregate_signal("Clap", "z_p")
currDictionary['z_p'] = clap_rashmi_agg_z_p
clap_rashmi_agg_mag_p = rashmi_gestures.create_aggregate_signal("Clap", "mag_p")
currDictionary['mag_p'] = clap_rashmi_agg_mag_p

currDictionary = rashmi_dictionary["Elephant Arm"]
ea_rashmi_agg_x_p = rashmi_gestures.create_aggregate_signal("Elephant Arm", "x_p")
currDictionary['x_p'] = ea_rashmi_agg_x_p 
ea_rashmi_agg_y_p = rashmi_gestures.create_aggregate_signal("Elephant Arm", "y_p")
currDictionary['y_p'] = ea_rashmi_agg_y_p 
ea_rashmi_agg_z_p = rashmi_gestures.create_aggregate_signal("Elephant Arm", "z_p")
currDictionary['z_p'] = ea_rashmi_agg_z_p 
ea_rashmi_agg_mag_p = rashmi_gestures.create_aggregate_signal("Elephant Arm", "mag_p")
currDictionary['mag_p'] = ea_rashmi_agg_mag_p 

def main():
	# python serial_plotter.py --port /dev/cu.usbmodem14601
	# windows: python lserial_plotter.py --port COM5	
    # create parser

    parser = argparse.ArgumentParser(description="Accel Serial Plotter")

    # add expected arguments
    parser.add_argument('--port', dest='port', required=True, help='the serial port for incoming data')
    parser.add_argument('--max_len', dest='max_len', required=False, default=770, type=int, 
        help='the number of samples to plot at a time')

    # parse args
    args = parser.parse_args()

    # strPort = '/dev/tty.usbserial-A7006Yqh'
    str_port = str(args.port)

    print('Reading from serial port: {}'.format(str_port))

    # plot parameters
    
    fig = plt.figure(figsize=(10, 5), tight_layout=True)
    gs = GridSpec(2, 4, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_ylabel("x_p")
    ax2.set_xlabel("time")
    #ax1.setTitle("real time gestures")
    #ax2.setTitle("x_p comparison")
    ax3=fig.add_subplot(gs[1, 1])
    ax3.set_ylabel("y_p")
    ax3.set_xlabel("time")
    ax4=fig.add_subplot(gs[1,2])
    ax4.set_ylabel("z_p")
    ax4.set_xlabel("time")
    ax5= fig.add_subplot(gs[1,3])
    ax5.set_ylabel("mag_p")
    ax5.set_xlabel("time")
    #ax3.setTitle("y_p comparison")
    #ax4.setTitle("z_p comparison")
    #ax5.setTitle("mag_p comparison")
    #ax = plt.axes(xlim=(0, args.max_len), ylim=(0, 1023))

    # place a text box in upper left in axes coords
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    txt = ax1.text(0.05, 0.95, "Do the dance move: Pulling", transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    score_txt = ax1.text(0.05, 0.25, "Score: ", transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    fig.align_labels()
    ax1.set_ylim((0, 1500))
    ax2.set_ylim((0, 1500))
    ax3.set_ylim((0, 1500))
    ax4.set_ylim((0, 1500))
    ax5.set_ylim((0, 1500))
    #ax = plt.axes(ax1, ylim=(0, 1500))
    #plt.show()


    accel_plot = AccelPlot(fig, ax1, txt, score_txt, str_port, ax2, ax3, ax4, ax5, max_length=args.max_len)

    # set up animation
  
    lines = list()
    num_vals = 4 # x,y,z,mag
    labels = ['x', 'y', 'z', 'mag']
    alphas = [0.8, 0.8, 0.8, 0.9]
    for i in range(0, num_vals):
        line2d, = ax1.plot([1], [1], label=labels[i], alpha=alphas[i])
        lines.append(line2d)

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)

    xpLines = list()
    num_vals_xp = 2
    labels_subplotsx = ['current gesture', 'aggregate gesture']
    labels_subplotsy = ['current gesture', 'aggregate gesture']
    labels_subplotsz = ['current gesture', 'aggregate gesture']
    labels_subplotsmag = ['current gesture', 'aggregate gesture']
    alphassub = [0.8, 0.8]
    ypLines = list()
    zpLines = list()
    magpLines = list()
    for i in range(0, num_vals_xp):
        line2dx = ax2.plot([1], [1], label=labels_subplotsx[i], alpha=alphassub[i])
        xpLines.append(line2dx)
        line2dy = ax3.plot([1], [1], label=labels_subplotsy[i], alpha=alphassub[i])
        ypLines.append(line2dy)
        line2dz = ax4.plot([1], [1], label=labels_subplotsz[i], alpha=alphassub[i])
        zpLines.append(line2dz)
        line2dmag = ax5.plot([1], [1], label=labels_subplotsmag[i], alpha=alphassub[i])
        magpLines.append(line2dmag)

    handles1, labels1 = ax3.get_legend_handles_labels()
    ax2.legend(handles1, labels1)
    handles2, labels2 = ax3.get_legend_handles_labels()
    ax3.legend(handles2, labels2)
    handles, labels = ax4.get_legend_handles_labels()
    ax4.legend(handles, labels)
    handles, labels = ax5.get_legend_handles_labels()
    ax5.legend(handles, labels)

    # for more on animation function, see https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
    anim = animation.FuncAnimation(fig, accel_plot.update,
                                   fargs=(args, lines, xpLines, ypLines, zpLines, magpLines), # could consider adding blit=True
                                   interval=50) #interval=50 is 20fps

    # show plot
    plt.show()

    # clean up
    accel_plot.close()

    print('Exiting...')


# call main
if __name__ == '__main__':
    main()