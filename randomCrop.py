import os
import cv2
import sys
import argparse
import CONSTS
import random
import glob
import numpy as np
import imageio

from utils import make_dataset, make_dataset_txtfile

if os.path.exists(CONSTS.SELECTIVE_SEARCH_DIR):
    sys.path.insert(1, CONSTS.SELECTIVE_SEARCH_DIR)
    import selective_search

def argument_parser():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('-m', '--mode', default='selective', help='', type=str)
    parser.add_argument('-i', '--input_path', default='ucf_but_rowing', help='', type=str)
    parser.add_argument('-o', '--output_path', default='tmp', help='', type=str)
    parser.add_argument('-lm', '--limit', default=int(1e4), help='', type=int)
    parser.add_argument('-s', '--minimalImage_size', default=30, help='', type=int)
    parser.add_argument('-ns', '--num_sets', default=1, help='', type=int)
    parser.add_argument('-fi', '--frame_intervals', default=20, help='', type=int)

    return parser.parse_args()

def write_window_to_file(window, minimalImage_size, detection_indx, output_path, raw_name):
    # crop and save:
    window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
    window = cv2.resize(window, (minimalImage_size, minimalImage_size))
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    outputfilename = os.path.join(output_path, "%s_id%07d.png" % (raw_name, detection_indx))
    cv2.imwrite(outputfilename, window)
    if detection_indx % 1000 == 0:
        print("Detected windows.. %d" % detection_indx)

def write_windows_to_file(window1, window2, minimalImage_size, gen_frame_rate, detection_indx, output_path, raw_name):
    # crop and save:
    window1 = cv2.cvtColor(window1, cv2.COLOR_BGR2GRAY)
    window2 = cv2.cvtColor(window2, cv2.COLOR_BGR2GRAY)
    window1 = cv2.resize(window1, (minimalImage_size, minimalImage_size))
    window2 = cv2.resize(window2, (minimalImage_size, minimalImage_size))

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # cv2.imwrite(os.path.join(output_path, "%s_frame1.png" % raw_name), window1)
    # cv2.imwrite(os.path.join(output_path, "%s_frame2.png" % raw_name), window2)

    outputfilename = os.path.join(output_path, "%s_O.gif" % raw_name)
    imageio.mimsave(outputfilename, [window1, window2], duration=1/gen_frame_rate)

    if detection_indx % 1000 == 0:
        print("Detected windows.. %d" % detection_indx)


def selective_search_detection(input_examples, output_path, minimalImage_size, gen_frame_rate=2, frame_intervals=20, limit=np.Inf):

    detection_indx = 0
    chunck_size = min(1000, int(limit / 1000))

    for img_path in input_examples:
        if os.path.exists(img_path):
            # read the image and define the stepSize and window size (width,height)
            print("processing %s" % img_path)
            #image = cv2.imread(img_path)  # your image path
            cap = cv2.VideoCapture(img_path)
            raw_name = os.path.splitext(os.path.basename(img_path))[0]

            fps = cap.get(cv2.CAP_PROP_FPS)
            # Default resolutions of the frame are obtained.The default resolutions are system dependent.
            # We convert the resolutions from float to integer.
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_number_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#            for fram_indx in range(0, total_number_frames, frame_intervals):
            for ii in range(0, int(total_number_frames/frame_intervals)):
                fram_indx = np.random.randint(0, total_number_frames-1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, fram_indx)
                ret, first_frame = cap.read()

                #with np.errstate(divide='ignore', invalid='ignore'):
                boxes = selective_search.selective_search(first_frame, mode='fast')
                # boxes_filter (optional)
                boxes = selective_search.box_filter(boxes, min_size=20, topN=80)
                random.shuffle(boxes)
                if limit < 10001:
                    limit_per_image = 100
                    boxes = boxes[:limit_per_image]

                for x1, y1, x2, y2 in boxes:

                    window_frame1 = first_frame[x1:x2, y1:y2, :] # [x:x + w_width, y:y + w_height, :]

                    if window_frame1.size > 0:

                        # randomize frame 2:
                        second_fram_indx = np.random.randint(fram_indx, total_number_frames)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, second_fram_indx)
                        ret, second_frame = cap.read()
                        window_frame2 = second_frame[x1:x2, y1:y2, :]

                        chunck_indx = int(detection_indx / chunck_size)
                        #'v_MoppingFloor_g05_c01_size_30_bbox_29_207_37_215_inds_17_55_rate_2_O'
                        video_file_name = raw_name + '_size_' + str(minimalImage_size) \
                                          + '_bbox_' + str(x1)+'_' + str(x2)+'_' + str(y1)+'_' + str(y2)+'_' \
                                          + '_inds_' + '_' + str(fram_indx) + '_' + str(second_fram_indx) + '_rate_' + str(gen_frame_rate)
                        write_windows_to_file(window_frame1, window_frame2, minimalImage_size, gen_frame_rate, detection_indx, os.path.join(output_path, str(chunck_indx)), video_file_name)
                        detection_indx += 1
                        if detection_indx > limit:
                            return

    # # show all windows
    # plt.show()
    print("Total %d detected windows were saved to %s!" % (detection_indx, output_path))


def sliding_window_detection(input_examples, minimalImage_size, output_path, limit):

    detection_indx = 0

    for img_path in input_examples:
        # read the image and define the stepSize and window size (width,height)
        image = cv2.imread(img_path)  # your image path

        for window_size in range(minimalImage_size, 200, 10):

            for stepSize in range(minimalImage_size, minimalImage_size*2, 5):

                (w_width, w_height) = (window_size, window_size)  # window size

                for x in range(0, image.shape[1] - w_width, stepSize):

                    for y in range(0, image.shape[0] - w_height, stepSize):

                        window = image[x:x + w_width, y:y + w_height, :]

                        # classify content of the window with your classifier and

                        if window.size > 0:
                            write_window_to_file(window, minimalImage_size, detection_indx)
                            detection_indx += 1
                            if detection_indx > limit:
                                return

                        # determine if the window includes an object (cell) or not      # draw window on image
                        # cv2.rectangle(tmp, (x, y), (x + w_width, y + w_height), (255, 0, 0), 2)  # draw rectangle on image
                        # plt.imshow(np.array(tmp).astype('uint8'))

    # # show all windows
    # plt.show()
    print("Total %d detected windows were saved to %s!" % (detection_indx, output_path))


def gen_data(mode, input_filenames, output_path, num_sets, minimalImage_size, gen_frame_rate, frame_intervals, limit):

    random.shuffle(input_filenames)

    set_size = int(len(input_filenames) / num_sets)

    for set_indx in range(num_sets):
        set_output_path = os.path.join(output_path, str(set_indx))
        os.makedirs(set_output_path)

        files_subset = input_filenames[set_indx * set_size : (set_indx + 1) * set_size]

        if mode == 'selective':
            selective_search_detection(files_subset, set_output_path, minimalImage_size, gen_frame_rate, frame_intervals, limit)
        elif mode == 'sliding':
            sliding_window_detection(files_subset, minimalImage_size, set_output_path, limit)
        else:
            os.error("incorrect mode name")

def get_ucf_class_filenames(ucf_folder_path, category='Rowing'):
    list_of_ucf101_categories = os.listdir(ucf_folder_path)
    if category in list_of_ucf101_categories:
        ucf_category_filenames = make_dataset(os.path.join(ucf_folder_path, category), ext='avi')
    return ucf_category_filenames


def get_ucf_all_but_class_filenames(ucf_folder_path, category='Rowing'):
    list_of_ucf101_categories = os.listdir(ucf_folder_path)
    all_but_category_ucf_classification_names = []
    for ucf_category in list_of_ucf101_categories:
        if ucf_category != category:
            all_but_category_ucf_classification_names.extend(make_dataset(os.path.join(ucf_folder_path,ucf_category), ext='avi'))
    return all_but_category_ucf_classification_names

# ===========================
# Main
# ===========================
if __name__ == '__main__':

    args = argument_parser()

    input_path = args.input_path
    output_path = args.output_path

    if input_path == 'ucf_rowing':
        # get all rowing ucf files:
        input_filenames = get_ucf_class_filenames(ucf_folder_path=CONSTS.UCF_DIR, category='Rowing')
    elif input_path == 'ucf_but_rowing':
        # get all nonrowing ucf files:
        input_filenames = get_ucf_all_but_class_filenames(ucf_folder_path=CONSTS.UCF_DIR, category='Rowing')
    elif input_path == 'ucf_tennis':
        input_filenames = get_ucf_class_filenames(ucf_folder_path=CONSTS.UCF_DIR, category='TennisSwing')
    elif input_path == 'ucf_hammer':
        input_filenames = get_ucf_class_filenames(ucf_folder_path=CONSTS.UCF_DIR, category='HammerThrow')

    else:
        input_filenames = make_dataset(dir=input_path)

    gen_data(mode=args.mode, input_filenames=input_filenames, output_path=output_path,
            num_sets=args.num_sets, minimalImage_size=args.minimalImage_size, gen_frame_rate=2, frame_intervals=args.frame_intervals, limit=args.limit)






    # # Filter box proposals
    # # Feel free to change parameters
    # boxes_filter = selective_search.box_filter(boxes, min_size=20, topN=80)
    #
    # # draw rectangles on the original image
    # fig, ax = plt.subplots(figsize=(6, 6))
    # ax.imshow(image)
    # for x1, y1, x2, y2 in boxes_filter:
    #     bbox = mpatches.Rectangle(
    #         (x1, y1), (x2-x1), (y2-y1), fill=False, edgecolor='red', linewidth=1)
    #     ax.add_patch(bbox)
    #
    # plt.axis('off')
    # plt.show()

