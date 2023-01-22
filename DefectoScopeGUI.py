import tkinter as tk
from tkinter import simpledialog as simple_dlg
from tkinter.dialog import Dialog as tk_dialog
from PIL import Image as PIL_Image
from PIL import ImageTk as PIL_Image_Tk
import cv2
import numpy as np
import os

WINDOW_SIZE = (1024, 600)
VIDEO_FRAME_SIZE = (768, 480)  # WVGA
BUTTON_SIZE = (200, 100)
BACKGROUND_COLOR = '#8C9787'

plane_id = ''
record_on = False


def bgr_to_tk_image(image):
    b, g, r = cv2.split(image)
    img = cv2.merge((r, g, b))
    img = PIL_Image.fromarray(img)
    img_tk = PIL_Image_Tk.PhotoImage(image=img)
    return img_tk


def start_action(root):
    global plane_id
    plane_id = simple_dlg.askstring('Input airplane name', 'ID', parent=root)
    print(plane_id)
    if plane_id is not None:
        capture_button['state'] = 'normal'
        record_button['state'] = 'normal'
        new_button['state'] = 'disabled'
        stop_button['state'] = 'normal'
        test_button['state'] = 'disabled'


def capture_action():
    print('Capture')


def record_action():
    global record_on
    if not record_on:
        record_button['image'] = record_button_on_icon
        record_on = True
    else:
        record_button['image'] = record_button_off_icon
        record_on = False


def stop_action():
    capture_button['state'] = 'disabled'
    if record_on:
        record_action()
    record_button['state'] = 'disabled'
    new_button['state'] = 'normal'
    stop_button['state'] = 'disabled'
    test_button['state'] = 'normal'
    print('Stop')


def select_test_dialog(root):
    d = tk_dialog(root, {'title': 'Choose Test Type',
                         'text': '',
                         'bitmap': '',
                         'default': 0,
                         'strings': ('Classifier',
                                     'Detector')})
    return ('Classifier', 'Detector')[d.num]


def test_action(root):
    test_type = select_test_dialog(root)
    if test_type is not None:
        print('test', test_type)


def load_icon(icon_name):
    return tk.PhotoImage(file=os.path.join('img', icon_name + '.png'))


def create_left_menu_button(icon, index, action):
    button = tk.Button(main_window,
                       image=icon, border='0',
                       bg=BACKGROUND_COLOR,
                       activebackground=BACKGROUND_COLOR)
    button.place(x=5 + VIDEO_FRAME_SIZE[0] + 20, y=15 + BUTTON_SIZE[1] * index + 15 * index,
                 width=BUTTON_SIZE[0], height=BUTTON_SIZE[1])
    button.config(command=action)
    return button


def on_closing(root):
    stop_action()
    print('Exit')
    root.destroy()


if __name__ == '__main__':
    main_window = tk.Tk()
    screen_width = main_window.winfo_screenwidth()
    screen_height = main_window.winfo_screenheight()

    xc = int((screen_width / 2) - (WINDOW_SIZE[0] / 2))
    yc = int((screen_height / 2) - (WINDOW_SIZE[1] / 2))
    main_window.geometry('{}x{}+{}+{}'.format(WINDOW_SIZE[0], WINDOW_SIZE[1], xc, yc))
    main_window.resizable(False, False)
    main_window.configure(bg=BACKGROUND_COLOR)
    # main_window.wm_attributes('-fullscreen', 'True')

    frame = np.random.randint(0, 255, size=(VIDEO_FRAME_SIZE[1], VIDEO_FRAME_SIZE[0], 3), dtype=np.uint8)
    frame_tk = bgr_to_tk_image(frame)

    video_out = tk.Label(main_window,
                         image=frame_tk,
                         width=VIDEO_FRAME_SIZE[0],
                         height=VIDEO_FRAME_SIZE[1],
                         borderwidth=2,
                         relief='ridge')
    video_out.place(x=5, y=5)

    new_button_icon = load_icon('button_new')
    new_button = create_left_menu_button(new_button_icon, 0, lambda: start_action(main_window))

    capture_button_icon = load_icon('button_capture')
    capture_button = create_left_menu_button(capture_button_icon, 1, capture_action)
    capture_button['state'] = 'disable'

    record_button_off_icon = load_icon('button_record_off')
    record_button_on_icon = load_icon('button_record_on')
    record_button = create_left_menu_button(record_button_off_icon, 2, record_action)
    record_button['state'] = 'disable'

    stop_button_icon = load_icon('button_stop')
    stop_button = create_left_menu_button(stop_button_icon, 3, stop_action)
    stop_button['state'] = 'disabled'

    test_button_icon = load_icon('button_test')
    test_button = create_left_menu_button(test_button_icon, 4, lambda: test_action(main_window))

    main_window.protocol("WM_DELETE_WINDOW", lambda: on_closing(main_window))
    main_window.update()
    main_window.mainloop()
