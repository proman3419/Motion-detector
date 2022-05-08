import cv2
import sys


MIN_PARAMS_CNT = 4
GAUSS_KERNEL_VAL = 5
DILATED_ITERATIONS_CNT = 15
OUTPUT_WIDTH = 1366
OUTPUT_HEGIHT = 768


def resize_and_show_frame(frame, frame_i, title):
    frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEGIHT))
    cv2.imshow(f'Frame {frame_i} - {title}', frame)


def main_loop(debug, sensitivity, cap, frame_dims, previous_frame, curr_frame):
    i = 0
    while cap.isOpened():
        diff_frame = cv2.absdiff(previous_frame, curr_frame)
        gray_frame = cv2.cvtColor(diff_frame, cv2.COLOR_BGR2GRAY)
        blured_frame = cv2.GaussianBlur(gray_frame, (GAUSS_KERNEL_VAL, 
                                                     GAUSS_KERNEL_VAL), 0)
        _, thresh_frame = cv2.threshold(blured_frame, thresh_value, 255, 
                                        cv2.THRESH_BINARY)
        dilated_frame = cv2.dilate(thresh_frame, None, 
                                   iterations=DILATED_ITERATIONS_CNT)
        contoured_frame, _ = cv2.findContours(dilated_frame, cv2.RETR_TREE, 
                                              cv2.CHAIN_APPROX_SIMPLE)
        resize_and_show_frame(frame, i, 'Result')
        i += 1


if __name__ == '__main__':
    if len(sys.argv) < MIN_PARAMS_CNT or sys.argv[1] == 'help':  
        print('python3 motion_detectory.py <stream_src[url/path]> <debug[0/1]> <sensitivity[0..1]')

    stream_src = sys.argv[1]
    debug = bool(int(sys.argv[2]))
    sensitivity = float(sys.argv[3])

    cap = cv2.VideoCapture(stream_src)
    frame_dims = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    previous_frame = cap.read()
    curr_frame = cap.read()

    main_loop(debug, sensitivity, cap, frame_dims, previous_frame, curr_frame)
