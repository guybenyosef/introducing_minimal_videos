# Introducing Minimal Videos

### Data
This folder contains minimal video files used for our MTurk experiments, 
arranged by sub-folders for each minimal video. 
 
Each sub-folder (e.g., ‘Mopping1’) contains:
•	A single minimal video file 
(e.g., `v_MoppingFloor_g11_c03_inds_46_51_size_19x19_rate_2_O_TL.gif`)
•	The original video clip from which the minimal video was taken:
(e.g., `v_MoppingFloor_g11_c03.avi` – taken from the UCF101 dataset)
•	A subfolder contains the spatial and temporal sub-minimal versions. 
•	Spatial sub-minimal files are identified by the file name suffix. The suffix indicates the type of video frame reduction:
o	BL = Bottom Left crop
(e.g., `v_MoppingFloor_g11_c03_inds_46_51_size_16x16_rate_2_O_TL_BL.gif`)
o	BR = Bottom Right crop
(e.g., `v_MoppingFloor_g11_c03_inds_46_51_size_16x16_rate_2_O_TL_BR.gif`)
o	TL = Top Left crop
(e.g., `v_MoppingFloor_g11_c03_inds_46_51_size_16x16_rate_2_O_TL_TL.gif`)
o	TR = Top Right crop
(e.g., `v_MoppingFloor_g11_c03_inds_46_51_size_16x16_rate_2_O_TL_TR.gif`)
o	scl = reduce resolution by 20%
(e.g., `v_MoppingFloor_g11_c03_inds_46_51_size_16x16_rate_2_O_TL_scl.gif`)
•	Temporal sub-minimal files are the two frames of the minimal video. They are identified by their frame index in the original video. 
•	(e.g., `v_MoppingFloor_g11_c03_TL_frame46_size_19x19.png` and `v_MoppingFloor_g11_c03_TL_frame51_size_19x19.png` to indicate the first and second frames respectively).

The Mechanical Turk data file describes additional details about each minimal video including frame size, frame rate, and frame index. 
It also contains the psycho-physics results (recognition rate in percentage) for the minimal videos and their sub-minimal versions. 


### Code

TBD