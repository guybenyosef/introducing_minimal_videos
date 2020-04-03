# Introducing Minimal Videos


### Minimal videos

TDB

### Data
This folder contains minimal video files used for our MTurk experiments, 
arranged by sub-folders for each minimal video. 
 
Each sub-folder (e.g., ‘Mopping1’) contains:
* A single minimal video file 
(e.g., `v_MoppingFloor_g11_c03_inds_46_51_size_19x19_rate_2_O_TL.gif`)
* The original video clip from which the minimal video was taken:
(e.g., `v_MoppingFloor_g11_c03.avi` – taken from the UCF101 dataset)
* A subfolder contains the spatial and temporal sub-minimal versions. 
* Spatial sub-minimal files are identified by the file name suffix. The suffix indicates the type of video frame reduction:
    * BL = Bottom Left crop (e.g., `v_MoppingFloor_g11_c03_inds_46_51_size_16x16_rate_2_O_TL_BL.gif`)
    * BR = Bottom Right crop (e.g., `v_MoppingFloor_g11_c03_inds_46_51_size_16x16_rate_2_O_TL_BR.gif`)
    * TL = Top Left crop (e.g., `v_MoppingFloor_g11_c03_inds_46_51_size_16x16_rate_2_O_TL_TL.gif`)
    * TR = Top Right crop (e.g., `v_MoppingFloor_g11_c03_inds_46_51_size_16x16_rate_2_O_TL_TR.gif`)
    * scl = reduce resolution by 20% (e.g., `v_MoppingFloor_g11_c03_inds_46_51_size_16x16_rate_2_O_TL_scl.gif`)
* Temporal sub-minimal files are the two frames of the minimal video. They are identified by their frame index in the original video. 
(e.g., `v_MoppingFloor_g11_c03_TL_frame46_size_19x19.png` and `v_MoppingFloor_g11_c03_TL_frame51_size_19x19.png` to indicate the first and second frames respectively).

The Mechanical Turk data file contains psychophysics data (recognition rate in percentage) for the minimal videos and their sub-minimal versions. 
It also contains additional details about each minimal video including frame size, frame rate, and frame index. 


### Code

TBD

### Papers
This repo contains data and code for the following papers:
* Guy Ben-Yosef, Gabriel Kreiman, and Shimon Ullman. **Minimal videos: Trade-off between spatial and temporal information in human and machine vision**. *Cognition*, in press.
* Guy Ben-Yosef, Gabriel Kreiman, and Shimon Ullman. **What can human minimal videos tell us about dynamic recognition models?** *Workshop on Bridging AI and Cognitive Science, International Conference on Learning Representations (ICLR)*, 2020 
