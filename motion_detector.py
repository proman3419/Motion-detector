from SensitivityArea import SensitivityArea
import cv2
import sys


MIN_PARAMS_CNT = 4
GAUSS_KERNEL_VAL = 5
DILATED_ITERATIONS_CNT = 15
OUTPUT_WIDTH = 600
OUTPUT_HEGIHT = 400
URL = 'https://imageserver.webcamera.pl/rec/krakow1/latest.mp4'
THRESH_VALUE = 40

# TODO len(frame) ustawiac
INPUT_WIDTH = 2000
INPUT_HEIGHT = 1300

START_WIDTH = 0
START_HEIGHT = 0
END_WIDTH = INPUT_WIDTH
END_HEIGHT = INPUT_HEIGHT
SMALLEST_AREA = 2000

INTERFACE_COLOR = (0, 255, 0)
SENSITIVITY_AREA_COLOR = (255, 0, 0)

SENSITIVE_OF_CONTOURS = [(START_WIDTH, START_HEIGHT, END_WIDTH, END_HEIGHT // 2, SMALLEST_AREA),
                         (START_WIDTH, END_HEIGHT // 2, END_WIDTH, END_HEIGHT, 250)]


def preprocess_sensitivity_areas(sensitivity_areas_raw):
    sensitivity_areas = []
    for e in sensitivity_areas_raw.split('|'):
        x_min, x_max, y_min, y_max, min_size = map(float, e.split(','))
        sensitivity_areas.append(SensitivityArea(x_min, x_max, y_min, y_max, min_size))
    return sensitivity_areas


def resize_and_show_frame(frame, title):
    frame = cv2.resize(frame, (600, 400))
    cv2.imshow(title, frame)


def create_contours(sensitivity_areas, curr_frame, contours):
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        for sa in sensitivity_areas:
            if sa.x_min <= x and x + w <= sa.x_max and \
               sa.y_min <= y and y + h <= sa.y_max and \
               cv2.contourArea(contour) >= sa.min_size:
                cv2.rectangle(curr_frame, (x, y), (x+w, y+h), INTERFACE_COLOR, 2)
                cv2.putText(curr_frame, 'MOTION DETECTED', (15, 35), 
                            cv2.FONT_HERSHEY_DUPLEX, 1, INTERFACE_COLOR, 3)


def mark_sensitivity_areas(sensitivity_areas, curr_frame):
    for sa in sensitivity_areas:
        cv2.rectangle(curr_frame, (int(sa.x_min), int(sa.y_min)), 
                      (int(sa.x_max), int(sa.y_max)), SENSITIVITY_AREA_COLOR, 2)
    pass


def debug(curr_frame, diff_frame, gray_frame, blur_frame, thresh_frame, dilated_frame):
    resize_and_show_frame(diff_frame, "Diff")
    resize_and_show_frame(gray_frame, "Gray")
    resize_and_show_frame(blur_frame, "Blur")
    resize_and_show_frame(thresh_frame, "Thresh")
    resize_and_show_frame(dilated_frame, "Dilated")
    resize_and_show_frame(curr_frame, "Result")


def main_loop(is_debug, sensitivity_areas, cap, frame_dims, curr_frame, next_frame):
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
        create_contours(sensitivity_areas, curr_frame, contours)
        mark_sensitivity_areas(sensitivity_areas, curr_frame)

        if is_debug:
            debug(curr_frame, diff_frame, gray_frame, blured_frame, thresh_frame, dilated_frame)
        else:
            resize_and_show_frame(curr_frame, 'Result')

        curr_frame = next_frame
        _, next_frame = cap.read()

        if cv2.waitKey(40) == 27:
            break


if __name__ == '__main__':
    if len(sys.argv) < MIN_PARAMS_CNT or sys.argv[1] == 'help':
        print('python3 motion_detectory.py <stream_src[url/path]> <debug[0/1]> <sensitivity_areas[x_min,x_max,y_min,y_max,min_size|...]>')

    # stream_src = sys.argv[1]
    stream_src = URL
    # debug = bool(int(sys.argv[2]))
    is_debug = False
    # sensitivity_areas = preprocess_sensitivity_areas(sys.argv[3])
    sensitivity_areas = preprocess_sensitivity_areas("0,100,0,100,300|0,1200,400,900,150")

    cap = cv2.VideoCapture(stream_src)
    frame_dims = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    _, curr_frame = cap.read()
    _, next_frame = cap.read()

    main_loop(is_debug, sensitivity_areas, cap, frame_dims, curr_frame, next_frame)
