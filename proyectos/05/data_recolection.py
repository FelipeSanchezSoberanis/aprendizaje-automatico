import os
from datetime import datetime
import numpy.typing as npt
import cv2 as cv
from enum import Enum


class Figures(Enum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2

    def to_string(self) -> str:
        if self == Figures.ROCK:
            return "rock"
        elif self == Figures.PAPER:
            return "paper"
        elif self == Figures.SCISSORS:
            return "scissors"
        else:
            raise Exception("Invalid enum")


HOME_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(HOME_DIR, "data")
DATA_DIR_PER_FIGURE: dict[Figures, str] = {
    Figures.ROCK: os.path.join(DATA_DIR, Figures.ROCK.to_string()),
    Figures.PAPER: os.path.join(DATA_DIR, Figures.PAPER.to_string()),
    Figures.SCISSORS: os.path.join(DATA_DIR, Figures.SCISSORS.to_string()),
}
IMAGES_PER_FIGURE = 100
IMAGE_DIMENSION = 400


def increase_brightness(frame: npt.NDArray, value: int) -> npt.NDArray:
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv.merge((h, s, v))
    frame = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)

    return frame


def calculate_centered_square(
    image: npt.NDArray, size: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    top_margin = (image.shape[0] - size) // 2
    left_margin = (image.shape[1] - size) // 2
    rect_start_point = (0 + left_margin, 0 + top_margin)
    rect_end_point = (size + left_margin, size + top_margin)

    return rect_start_point, rect_end_point


def save_figure_frame(index: int, figure: Figures, frame: npt.NDArray):
    username = os.environ.get("USER", os.environ.get("USERNAME"))
    date_time_str = datetime.now().strftime("%m%d%Y%H%M%S%f")
    file_name = f"{username}-{date_time_str}-{figure.to_string()}-{str(index + 1).zfill(3)}.png"

    rect_start_point, rect_end_point = calculate_centered_square(frame, IMAGE_DIMENSION)

    cv.imwrite(
        os.path.join(DATA_DIR_PER_FIGURE[figure], file_name),
        frame[rect_start_point[1] : rect_end_point[1], rect_start_point[0] : rect_end_point[0]],
    )


def annotate_frame(frame: npt.NDArray, figure: Figures, index: int) -> npt.NDArray:
    color_red = (0, 0, 255)
    font = cv.FONT_HERSHEY_SIMPLEX

    rect_start_point, rect_end_point = calculate_centered_square(frame, IMAGE_DIMENSION)

    annotated_frame = cv.flip(frame, 1)

    annotated_frame = cv.rectangle(annotated_frame, rect_start_point, rect_end_point, color_red, 3)

    annotated_frame = cv.putText(
        annotated_frame, f"{figure.to_string()} {index}", (5, 25), font, 1, color_red
    )

    return annotated_frame


def ensure_dirs():
    for dir in DATA_DIR_PER_FIGURE.values():
        if not os.path.isdir(dir):
            os.makedirs(dir)


def main():
    ensure_dirs()

    camera = cv.VideoCapture(0)
    framerate: int = camera.get(5)
    frames_kept_bright = framerate / 3
    wait_time_between_figures = 5  # in seconds

    for figure in DATA_DIR_PER_FIGURE:
        saved_images_counter = 0
        frame_counter = -wait_time_between_figures * framerate

        while saved_images_counter < IMAGES_PER_FIGURE + 1:
            ret, frame = camera.read()
            original_frame = frame.copy()

            if not ret:
                raise Exception("Camera error. Closing.")

            annotated_frame = annotate_frame(frame, figure, saved_images_counter)

            if frame_counter >= 0:
                if frame_counter % framerate == 0:
                    if saved_images_counter < IMAGES_PER_FIGURE:
                        save_figure_frame(saved_images_counter, figure, original_frame)
                    saved_images_counter += 1
                elif frame_counter % framerate <= frames_kept_bright:
                    annotated_frame = increase_brightness(annotated_frame, 255)

            cv.imshow("video", annotated_frame)

            frame_counter += 1

            cv.waitKey(1)

    camera.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
