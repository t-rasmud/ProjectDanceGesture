# Interactive Dance Tutorial

The goal of this project is to provide real time feedback to users on how well they perform a dance gesture.
We use a predefined set of 5 gestures and provide feedback corresponding to these by taking input from
an accelerometer tied around a user's fingers.

# Usage Instructions

Arduino code is located in the folder : `Our stuff/GestureRecorder/Arduino` <br>
Real time gesture segmentation code: `Our stuff/realtimegesturesegmentation/project_realtime.py`

From the terminal, run the commands

```
cd Our stuff/realtimegesturesegmentation
py project_realtime.py --port <port number>
```
