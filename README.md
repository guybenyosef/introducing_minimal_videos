# Introducing Minimal Videos


### Minimal videos

### The minimal videos data:
| Class index   | Minimal Image Type | frame Size | # images  | Examples  | Hard negative examples |
|:-------------:|:------------------:|:----------:|:---------:| --------- | -----------------------
| 1             | Rowing             |   *30x30*  |   *TBD*   |![10](raw_data/Rowing1/minimal/v_Rowing_g10_c05_inds_56_5_rate_2_O_BR.gif) ![10](raw_data/Rowing6/minimal/v_Rowing_g21_c03_size_30_bbox_99_176_169_246_inds_39_69_rate_2_O.gif) ![10](raw_data/Rowing3/minimal//v_Rowing_g09_c01_size_30_inds_91_123_rate_2_O.gif)       |
| 2             | Biking             |   *30x30*  |   *TBD*   |![10](raw_data/Biking1/minimal/v_Biking_g15_c04_inds_20_26_size_16x16_rate_2_O_TL_TR.gif), ![10](raw_data/Biking2/minimal/v_Biking_g15_c04_inds_20_26_size_14x14_rate_2_O_BL.gif), ![10](raw_data/Biking3/minimal/v_Biking_g03_c01_size_20_inds_113_120_rate_2_O.gif)       |
| 3             | Playing Violin     |   *30x30*  |   *TBD*   |![10](raw_data/PlayingViolin1/minimal/v_PlayingViolin_g11_c02_inds_16_26_size_14x14_rate_4_O_scl_BL.gif), ![10](raw_data/PlayingViolin2/minimal/v_PlayingViolin_g22_c04_inds_16_21_size_12x12_rate_5_O_BR.gif), ![10](raw_data/PlayingViolin3/minimal/v_PlayingViolin_g15_c04_inds_30_36_size_15x15_rate_2_O_BR_BL.gif)       |
| 4             | Mopping            |   *30x30*  |   *TBD*   |![10](raw_data/Mopping1/minimal/v_MoppingFloor_g11_c03_inds_46_51_size_19x19_rate_2_O_TL.gif), ![10](raw_data/Mopping2/minimal/v_MoppingFloor_g11_c01_size_30_bbox_28_218_45_235_inds_18_44_rate_2_actualSize_26_size__O_TR.gif), ![10](samples/42.png)       |


### Data
The `raw_data` folder contains minimal video files arranged by sub-folders. 
 
Each sub-folder (e.g., ‘Mopping1’) contains:
* The folder `minimal` with a single minimal video file 
(e.g., `v_MoppingFloor_g11_c03_inds_46_51_size_19x19_rate_2_O_TL.gif`)
* The original video clip from which the minimal video was taken
(e.g., `v_MoppingFloor_g11_c03.avi` downloaded from the UCF101 dataset)
* The folder `sub_minimal` with the spatial and temporal sub-minimal versions. 
* Spatial sub-minimal files are identified by the file name suffix. The suffix indicates the type of video frame reduction:
    * BL = Bottom Left crop (e.g., `v_MoppingFloor_g11_c03_inds_46_51_size_16x16_rate_2_O_TL_BL.gif`)
    * BR = Bottom Right crop (e.g., `v_MoppingFloor_g11_c03_inds_46_51_size_16x16_rate_2_O_TL_BR.gif`)
    * TL = Top Left crop (e.g., `v_MoppingFloor_g11_c03_inds_46_51_size_16x16_rate_2_O_TL_TL.gif`)
    * TR = Top Right crop (e.g., `v_MoppingFloor_g11_c03_inds_46_51_size_16x16_rate_2_O_TL_TR.gif`)
    * scl = reduce resolution by 20% (e.g., `v_MoppingFloor_g11_c03_inds_46_51_size_16x16_rate_2_O_TL_scl.gif`)
* Temporal sub-minimal files are the two frames of the minimal video. They are identified by their frame index in the original video. 
(e.g., `v_MoppingFloor_g11_c03_TL_frame46_size_19x19.png` and `v_MoppingFloor_g11_c03_TL_frame51_size_19x19.png` to indicate the first and second frames respectively).

The file `mturk.xlsx` contains Mechanical Turk psychophysics data (recognition rate in percentage) for the minimal videos and their sub-minimal versions. 
It also contains additional details about each minimal video including frame size, frame rate, and frame index. 


### Code

##### Requirements
* pytorch 1.1 or more
* sklearn
* scikit-image
* tqdm

##### Random crop for non-class examples at minimal video style 
In your `data/` folder, create a new folder `negatives`, with sub-folders `negatives/train` and `negatives/test`. Then run
```bash
python randomCrop.py -i path/to/your/video/dataset/train -o data/nonfour/train
python randomCrop.py -i path/to/your/video/dataset/test -o data/nonfour/test
```
where `path/to/your/video/dataset/` is the path to some real-world videos dataset (e.g., UCF101).

You can also run this script with additional parameters, e.g.,  
```bash
python randomCrop.py -i /Users/gby/data/minimal_images/negatives/nonhorse_large/0/ -o /Users/gby/data/minimal_images/negtives/nonhorse/ -ns 10 -lm 400
```
where `ns` is the number of sets created to contain the negative examples (mutually excluded), `lm` is the total number of generated 
negative examples.   

To extract negative video examples from for specific action category at UCF101 dataset, run e.g.,
```buildoutcfg
python randomCrop.py -i ucf_rowing -o data/minimal/negatives_video/nonrowing -ns 1 -lm 100000000
```
where `data/minimal/negatives_video/nonrowing` is the folder in which the negative video examples will be stored.

Finally, two modes to search and crop frame windows are supported: a `sliding window` mode and a `selective search` mode.
To use the selective search mode (default and recommended option) clone this repo:
```bash
git clone https://github.com/ChenjieXu/selective_search.git
```

### Papers
This repo contains data and code for the following papers:
* Guy Ben-Yosef, Gabriel Kreiman, and Shimon Ullman. **Minimal videos: Trade-off between spatial and temporal information in human and machine vision**. *Cognition*, in press.
* Guy Ben-Yosef, Gabriel Kreiman, and Shimon Ullman. **What can human minimal videos tell us about dynamic recognition models?** *Workshop on Bridging AI and Cognitive Science, International Conference on Learning Representations (ICLR)*, 2020 
