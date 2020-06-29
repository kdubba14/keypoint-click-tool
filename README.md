# Keypoint Click Tool
### Made for: The data collection process of labeling self-obtained data on keypoints
### Keypoints used in:
- Facial Recognition
- Body Pose Recognition
- Hand Pose Recognition

#### Generates a file called keypoint_labels.json once finished

***A sample folder will be in the 'hands' folder***

**NOTE:** The files are sorted in alphabetical order before running
**NOTE:** At the moment a maximum of 21 keypoints are allowed in the program
##### **NOTE:** Please refer to Keypoint-Example.png for the order of keypoints

================================================

## How To Run: 
- Move folder full of non-processed images into ```keypoint-click-tool``` folder
    - The tool will look for an ```images``` folder by default
    - To specify another folder use the ```--read``` tag with the name of the folder
    - Example: ```--read other-folder-name```
- ```pip install opencv-python``` (or pip3 for some Mac users)
- Run the tool using ```python keypoint-clicker.py```


### Flags:
- Read Flag
    - ```--read other-folder-name``` - Used to specify if the tool should read images from another folder
- Begin Flag
    - ```--begin IMG_NAME.jpg``` - Used to specify to start reading at a certain file name


### Commands:
- 'n' Key
    - Used to go to the next image (usually used once all keypoints are selected)
- 'b'
    - Used to go back a keypoint
- 'q'
    - Used to quit the program