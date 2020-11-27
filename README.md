# Introducing Minimal Videos


### Minimal videos
Objects and their parts can be visually recognized from purely spatial or purely temporal information but the 
mechanisms integrating space and time are poorly understood. Here we show that visual recognition of objects and actions 
can be achieved by efficiently combining spatial and motion cues in configurations where each source on its own is 
insufficient for recognition. This analysis is obtained by 
identifying minimal videos: these are short and tiny video clips in which objects, parts, and actions can be reliably recognized, 
but any reduction in either space or time makes them unrecognizable. Human recognition in minimal videos is invariably accompanied by 
full interpretation of the internal components of the video. State-of-the-art deep convolutional networks for dynamic recognition 
cannot replicate human behavior in these configurations. The gap between human and machine vision demonstrated here is due to critical 
mechanisms for full spatiotemporal interpretation that are lacking in current computational models. 
Dynamic figure is provided [here](figures/fig1.mp4).
### The minimal videos data summary:
|Minimal Video Type  | Minimal Video examples (Recognizable) | Spatial sub-minimal (Not recognizable)| Temporal sub-minimal (Not recognizable) | Hard negative examples (Confusing DNNs)|
|:------------------:| ------------------------------------- | ------------------------------------- | --------------------------------------- | -------------------------------------------|
| Rowing             |![10](raw_data/Rowing1/minimal/v_Rowing_g10_c05_inds_56_5_rate_2_O_BR.gif), ![10](raw_data/Rowing6/minimal/v_Rowing_g21_c03_size_30_bbox_99_176_169_246_inds_39_69_rate_2_O.gif), ![10](raw_data/Rowing3/minimal//v_Rowing_g09_c01_size_30_inds_91_123_rate_2_O.gif)       | ![10](raw_data/Rowing1/sub_minimal/v_Rowing_g10_c05_inds_56_5_rate_2_O_BR_TL.gif), ![10](raw_data/Rowing6/sub_minimal/v_Rowing_g21_c03_size_30_bbox_99_176_169_246_inds_39_69_rate_2_actualSize_26_O_TL.gif), ![10](raw_data/Rowing3/sub_minimal/v_Rowing_g09_c01_size_30_bbox_149_190_163_219_inds_91_123_rate_2_O_TR.gif) | ![10](raw_data/Rowing1/sub_minimal/v_Rowing_g10_c05_BR_frame5_size_19x19.png), ![10](raw_data/Rowing6/sub_minimal/v_Rowing_g21_c03_frame39_bbox_99_176_169_246_size_30x30.png), ![10](raw_data/Rowing3/sub_minimal/v_Rowing_g09_c01_frame123_bbox_149_190_163_219_size_30x30.png) | ![10](hardneg/demo_hardneg_rowing/v_CliffDiving_g25_c04_size_30_bbox_23_214_58_249_inds_11_26_rate_2_O.gif), ![10](hardneg/demo_hardneg_rowing/v_Haircut_g25_c04_size_30_bbox_41_175_12_146_inds_25_64_rate_2_O.gif), ![10](hardneg/demo_hardneg_rowing/v_HammerThrow_g25_c05_size_30_bbox_94_222_75_203_inds_29_34_rate_2_O.gif)
| Biking             |![10](raw_data/Biking1/minimal/v_Biking_g15_c04_inds_20_26_size_16x16_rate_2_O_TL_TR.gif), ![10](raw_data/Biking2/minimal/v_Biking_g15_c04_inds_20_26_size_14x14_rate_2_O_BL.gif), ![10](raw_data/Biking3/minimal/v_Biking_g03_c01_size_20_inds_113_120_rate_2_O.gif)       | ![10](raw_data/Biking1/sub_minimal/v_Biking_g15_c04_inds_20_26_size_14x14_rate_2_O_TL_TR_BR.gif), ![10](raw_data/Biking2/sub_minimal/v_Biking_g15_c04_inds_20_26_size_12x12_rate_2_O_BL_scl.gif), ![10](raw_data/Biking3/sub_minimal/v_Biking_g03_c01_inds_113_120_size_17x17_rate_2_O_BR.gif)| ![10](raw_data/Biking1/sub_minimal/v_Biking_g15_c04_TL_TR_frame20_size_16x16.png), ![10](raw_data/Biking2/sub_minimal/v_Biking_g15_c04_BL_frame26_size_14x14.png), ![10](raw_data/Biking3/sub_minimal/v_Biking_g03_c01_frame120_size_20x20.png) |
| Playing Violin     |![10](raw_data/PlayingViolin1/minimal/v_PlayingViolin_g11_c02_inds_16_26_size_14x14_rate_4_O_scl_BL.gif), ![10](raw_data/PlayingViolin2/minimal/v_PlayingViolin_g22_c04_inds_16_21_size_12x12_rate_5_O_BR.gif), ![10](raw_data/PlayingViolin3/minimal/v_PlayingViolin_g15_c04_inds_30_36_size_15x15_rate_2_O_BR_BL.gif)       | ![10](raw_data/PlayingViolin1/sub_minimal/v_PlayingViolin_g11_c02_inds_16_26_size_12x12_rate_4_O_scl_BL_TL.gif), ![10](raw_data/PlayingViolin2/sub_minimal/v_PlayingViolin_g22_c04_inds_16_21_size_10x10_rate_5_O_BR_TR.gif), ![10](raw_data/PlayingViolin3/sub_minimal/v_PlayingViolin_g15_c04_inds_30_36_size_13x13_rate_2_O_BR_BL_TL.gif) | ![10](raw_data/PlayingViolin1/sub_minimal/v_PlayingViolin_g11_c02_scl_BL_frame26_size_14x14.png), ![10](raw_data/PlayingViolin2/sub_minimal/v_PlayingViolin_g22_c04_BR_frame16_size_12x12.png), ![10](raw_data/PlayingViolin3/sub_minimal/v_PlayingViolin_g15_c04_BR_BL_frame30_size_15x15.png)|
| Mopping            |![10](raw_data/Mopping1/minimal/v_MoppingFloor_g11_c03_inds_46_51_size_19x19_rate_2_O_TL.gif), ![10](raw_data/Mopping2/minimal/v_MoppingFloor_g11_c01_size_30_bbox_28_218_45_235_inds_18_44_rate_2_actualSize_26_size__O_TR.gif)      | ![10](raw_data/Mopping1/sub_minimal/v_MoppingFloor_g11_c03_inds_46_51_size_16x16_rate_2_O_TL_scl.gif), ![10](raw_data/Mopping2/sub_minimal/v_MoppingFloor_g11_c01_size_30_bbox_28_218_45_235_inds_18_44_rate_2_actualSize_23_size__O_TR_BL.gif)      | ![10](raw_data/Mopping1/sub_minimal/v_MoppingFloor_g11_c03_TL_frame46_size_19x19.png), ![10](raw_data/Mopping1/sub_minimal/v_MoppingFloor_g11_c03_TL_frame51_size_19x19.png)|


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
* tensorboard
* sacred

###### Random crop for non-class examples at minimal video style 
To generate non-class video examples used for training minimal videos classifiers run, e.g.,
```bash
python randomCrop.py -i path/to/your/video/dataset -o path/to/your/output/folder
```
where `path/to/your/video/dataset/` is a folder containing real-world videos files (e.g., videos from UCF101), and 
`path/to/your/output/folder` is a folder in which the generated video examples are saved.  

You can also run this script with additional parameters, e.g.,  
```bash
python randomCrop.py -i path/to/your/video/dataset -o path/to/your/output/folder -ns 10 -lm 400
```
where `ns` is the number of sets created to contain the negative examples (mutually excluded), and `lm` is the maximal number of generated 
negative examples.   

To extract negative video examples for specific action category at UCF101 dataset, run e.g.,
```bash
python randomCrop.py -i ucf_but_rowing -o data/minimal/negatives_video/nonrowing -ns 1 -lm 100000000 -fi 2
```
where `data/minimal/negatives_video/nonrowing` is the folder in which the negative video examples will be stored. You would also need to 
download the [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php) and update the ucf101 path at `CONSTS.py`.

Finally, the script for generating of negative examples includes two modes for cropping frame windows: 
a `sliding window` mode and a `selective search` mode.
To use the selective search mode (default and recommended option) clone [this repo](https://github.com/ChenjieXu/selective_search.git):
```bash
git clone https://github.com/ChenjieXu/selective_search.git
```
and update the path to your local copy at `CONSTS.py`

###### Training a video classifier
Run the `mini` option:
```bash
python train.py with mini
```

###### Evaluating classification results 
Eval on naive set:
```bash
python eval.py with weights=storage/logs/RowingOrNot/1/ResNet3D18/1/weights_RowingOrNot_ResNet3D18_best.pth
``` 
or on a small naive set:
```bash
python eval.py with weights=storage/logs/RowingOrNot/1/ResNet3D18/1/weights_RowingOrNot_ResNet3D18_best.pth subset=277
```
(notice additional flags such as `plot`)

Eval on a small hard negatives set:
```bash
python eval.py with weights=storage/logs/RowingOrNot/1/ResNet3D18/1/weights_RowingOrNot_ResNet3D18_best.pth hard
```

Eval on a set of spatial sub-minimal videos:
```bash
python eval.py with weights=storage/logs/RowingOrNot/1/ResNet3D18/9/weights_RowingOrNot_ResNet3D18_best.pth submirc
```

### Papers
If you are using this repo please cite the following paper:
* Guy Ben-Yosef, Gabriel Kreiman, and Shimon Ullman. [**Minimal videos: Trade-off between spatial and temporal information in human and machine vision**](https://doi.org/10.1016/j.cognition.2020.104263), *Cognition*, 201, 104263, August 2020
[**PDF**](https://www.researchgate.net/publication/340796972_Minimal_videos_Trade-off_between_spatial_and_temporal_information_in_human_and_machine_vision)

Other relevant papers:          
* Guy Ben-Yosef, Gabriel Kreiman, and Shimon Ullman. [**What can human minimal videos tell us about dynamic recognition models?**](https://baicsworkshop.github.io/pdf/BAICS_1.pdf) *Workshop on Bridging AI and Cognitive Science, International Conference on Learning Representations (ICLR)*, 2020 
