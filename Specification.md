# Specification

This is a specification for claude code to generate code for me


## Task

this environment is jetson orin nano, i need you to implement a system will be used for navigation without using GPS

In the first stage,
please create programs that will first download maps from internet those maps might have GPS coordination, i need maps for Taipei, Taiwan, then I would send you a photo, and your program will identify the photo location.

once your program has identified the given photo GPS location, i need you to print out the location using gps location readings.

in the second stage, I will let your program to be ran in the jetson nano orion, the photo will be captured by a camera, the jetson and camera will be installed in a F450 drone
when the drone lift up to vertain height, for example 30 meters, it will capture a photo, and send into your program. your program has to identify the photo location and send to pixhawk then