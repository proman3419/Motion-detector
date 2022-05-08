import cv2
import sys

MIN_PARAMS_CNT = 4
GAUSS_KERNEL_VAL = 5
DILATED_ITERATIONS_CNT = 15
OUTPUT_WIDTH = 600
OUTPUT_HEGIHT = 400
URL = 'https://imageserver.webcamera.pl/rec/krakow1/latest.mp4'
TRUE = 1
THRESH_VALUE = 40

# TODO len(frame) ustawiac
INPUT_WIDTH = 2000
INPUT_HEIGHT = 1300

START_WIDTH = 0
START_HEIGHT = 0
END_WIDTH = INPUT_WIDTH
END_HEIGHT = INPUT_HEIGHT
SMALLEST_AREA = 2000

SENSITIVE_OF_CONTOURS = [(START_WIDTH, START_HEIGHT, END_WIDTH, END_HEIGHT // 2, SMALLEST_AREA),
                         (START_WIDTH, END_HEIGHT // 2, END_WIDTH, END_HEIGHT, 250)]


def resize_and_show_frame(frame, title):
    frame = cv2.resize(frame, (600, 400))
    cv2.imshow(title, frame)


def create_contours(curr_frame, contours):
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        for start_width, start_height, end_width, end_height, smallest_area in SENSITIVE_OF_CONTOURS:
            if start_width <= x and x + w <= end_width and start_height <= y and y + h <= end_height:
                if cv2.contourArea(contour) < smallest_area:
                    continue
                else:
                    cv2.rectangle(curr_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(curr_frame, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 3)


def debug(curr_frame, next_frame, diff, gray, blur, thresh, dilated):
    resize_and_show_frame("Curr frame", curr_frame)
    resize_and_show_frame("Next frame", next_frame)
    resize_and_show_frame("Diff", diff)
    resize_and_show_frame("Gray", gray)
    resize_and_show_frame("Blur", blur)
    resize_and_show_frame("Thresh", thresh)
    resize_and_show_frame("Dilated", dilated)
    resize_and_show_frame("Result", curr_frame)


def main_loop(is_debug, sensitivity, cap, frame_dims, curr_frame, next_frame):
    while cap.isOpened():
        diff_frame = cv2.absdiff(curr_frame, next_frame)
        gray_frame = cv2.cvtColor(diff_frame, cv2.COLOR_BGR2GRAY)
        blured_frame = cv2.GaussianBlur(gray_frame, (GAUSS_KERNEL_VAL,
                                                     GAUSS_KERNEL_VAL), 0)
        _, thresh_frame = cv2.threshold(blured_frame, THRESH_VALUE, 255,
                                        cv2.THRESH_BINARY)
        dilated_frame = cv2.dilate(thresh_frame, None,
                                   iterations=DILATED_ITERATIONS_CNT)
        contours, _ = cv2.findContours(dilated_frame, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if is_debug:
            debug(curr_frame, next_frame, diff_frame, gray_frame, blured_frame, thresh_frame, dilated_frame)
        else:
            create_contours(curr_frame, contours)
            resize_and_show_frame(curr_frame, 'Result')

        curr_frame = next_frame
        _, next_frame = cap.read()

        if cv2.waitKey(40) == 27:
            break


if __name__ == '__main__':
    if len(sys.argv) < MIN_PARAMS_CNT or sys.argv[1] == 'help':
        print('python3 motion_detectory.py <stream_src[url/path]> <debug[0/1]> <sensitivity[0..1]')

    # stream_src = sys.argv[1]
    stream_src = URL
    # debug = bool(int(sys.argv[2]))
    is_debug = TRUE
    # sensitivity = float(sys.argv[3])
    sensitivity = TRUE

    cap = cv2.VideoCapture(stream_src)
    frame_dims = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    _, curr_frame = cap.read()
    _, next_frame = cap.read()

    main_loop(is_debug, sensitivity, cap, frame_dims, curr_frame, next_frame)
