import tkinter as tk
from tkinter import simpledialog as simple_dlg
from tkinter.dialog import Dialog as tk_dialog
from PIL import Image as PIL_Image
from PIL import ImageTk as PIL_Image_Tk
import cv2
import numpy as np
import os
import net_test
from detector import classifier
from detector import detector_t
import datetime
from detector import analyzer
import video_cam
import multiprocessing as mp
from database import defects_list
from database import defects_base

WINDOW_SIZE = (1024, 600)
VIDEO_FRAME_SIZE = (768, 480)  # WVGA
BUTTON_SIZE = (200, 100)
BACKGROUND_COLOR = '#8C9787'

MODEL_DETECTOR_PATH = 'detector\\models\\rcnn\\300.torch'
MODEL_CLASSIFIER_PATH = 'detector\\models\\classification_model.pth'

defect_database = defects_base.DefectsBase()

net_analyzer = None
camera = None
stop_event = mp.Event()
capture_event = mp.Event()
record_event = mp.Event()
vid_writer = None
video_panel_image = None
analyze_started = False
bounding_boxes = []
defect_types = []
defect_scores = []

plane_id = {
    'name': "",
    'id': '',
    'date': '',
    'time': '',
    'has_data': False,
}

record_on = False


def bgr_to_tk_image(image):
    b, g, r = cv2.split(image)
    img = cv2.merge((r, g, b))
    img = PIL_Image.fromarray(img)
    img_tk = PIL_Image_Tk.PhotoImage(image=img)
    return img_tk


def start_action(root):
    global plane_id
    global net_analyzer
    plane_id['name'] = simple_dlg.askstring('Input airplane name', 'Name', parent=root)
    plane_id['id'] = simple_dlg.askstring('Input airplane serial number', 'ID', parent=root)
    plane_id['date'] = datetime.datetime.now().strftime('%d.%m.%Y')
    plane_id['time'] = datetime.datetime.now().strftime('%H:%M:%S')

    plane_dir = os.path.join('data_analyzed', plane_id['name'] + '_' + plane_id['id'])
    if not os.path.exists(plane_dir):
        os.mkdir(plane_dir)
    plane_dir = os.path.join(plane_dir, plane_id['date'] + '_' + plane_id['time'].replace(':', '_'))
    os.mkdir(plane_dir)
    plane_id['results'] = plane_dir
    plane_id['capture_counter'] = 0
    plane_id['record_counter'] = 0
    plane_id['defects'] = defects_list.AirCraftDefectsList(plane_id['id'], plane_id['name'])
    print(plane_id)
    if plane_id['name'] is not None:
        capture_button['state'] = 'normal'
        record_button['state'] = 'normal'
        new_button['state'] = 'disabled'
        stop_button['state'] = 'normal'
        test_button['state'] = 'disabled'
        net_analyzer = analyzer.DefectAnalyzer(MODEL_CLASSIFIER_PATH, MODEL_DETECTOR_PATH)


def capture_action():
    global capture_event
    capture_event.set()
    print('Capture')


def record_action():
    global record_on
    global record_event
    global vid_writer
    global plane_id

    if not record_on:
        record_button['image'] = record_button_on_icon
        record_on = True
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        vid_writer = cv2.VideoWriter(os.path.join(plane_id['results'], str(plane_id['record_counter']) + '.mp4'),
                                     fourcc, 30, (1280, 720))
        record_event.set()
    else:
        record_button['image'] = record_button_off_icon
        record_on = False
        record_event.clear()
        vid_writer.release()
        plane_id['record_counter'] += 1


def stop_action():
    global net_analyzer
    global plane_id

    capture_button['state'] = 'disabled'
    if record_on:
        record_action()
    record_button['state'] = 'disabled'
    new_button['state'] = 'normal'
    stop_button['state'] = 'disabled'
    test_button['state'] = 'normal'
    if net_analyzer is not None:
        net_analyzer.stop()
        net_analyzer = None

    if plane_id['has_data']:
        defect_database.add(plane_id['defects'])
        defect_database.report(plane_id['results'], plane_id['name'], plane_id['id'])
        plane_id['has_data'] = False
    print('Stop')


def select_test_dialog(root):
    d = tk_dialog(root, {'title': 'Choose Test Type',
                         'text': '',
                         'bitmap': '',
                         'default': 0,
                         'strings': ('Classifier',
                                     'Detector')})
    return ('Classifier', 'Detector')[d.num]


def center_window(w, size):
    s_w = w.winfo_screenwidth()
    s_h = w.winfo_screenheight()
    s_xc = int((s_w / 2) - (size[0] / 2))
    s_yc = int((s_h / 2) - (size[1] / 2))
    w.geometry('{}x{}+{}+{}'.format(size[0], size[1], s_xc, s_yc))
    return w


def wait_window(root):
    win = tk.Toplevel(root)
    win = center_window(win, (400, 160))
    win.resizable(False, False)
    win.configure(bg=BACKGROUND_COLOR)
    win.transient()
    win.overrideredirect(True)
    win.grab_set()
    lbl = tk.Label(win, text='Wait! Test in progress...', bg=BACKGROUND_COLOR, font=("Arial", 23))
    lbl.place(relx=0.5, rely=0.5, anchor="center")
    return win


def report_window(root, text):
    win = tk.Toplevel(root)
    win = center_window(win, (900, 500))
    win.resizable(False, False)
    win.configure(bg=BACKGROUND_COLOR)
    win.transient()
    win.title('Results!')
    win.grab_set()
    tk.Label(win, text=text, anchor='w', justify=tk.LEFT, bg=BACKGROUND_COLOR, font=("Courier", 14)).pack()
    btn = tk.Button(win, text='Ok', command=win.destroy)
    btn.place(x=400, y=400, width=100, height=70, anchor='center')
    return win


def wait_classifier_test_end(root, wait_win):
    if net_test.is_ready():
        res = net_test.report()
        wait_win.destroy()
        report_text = '*' * 80 + '\n' + \
                      'Testing results:\n' + \
                      '{:<50}: {:20d}\n'.format('Total images tested', res['total_images']) + \
                      '{:<50}: {:20d}\n'.format('True Positive checks', res['tp_n']) + \
                      '{:<50}: {:20d}\n'.format('False Negative checks', res['fn_n']) + \
                      '{:<50}: {:20d}\n'.format('False Positive checks', res['fp_n']) + \
                      '{:<50}: {:19.3f}%\n'.format('Precision', res['precision']) + \
                      '{:<50}: {:19.3f}%\n'.format('Recall', res['recall']) + \
                      '{:<50}: {:19.3f}%\n\n'.format('Probability of wrong classification', res['prob_error']) + \
                      '{:<50}: {:18.3f}ms\n'.format('Average timing for classification', res['average_t']) + \
                      '{:<50}: {:18.3f}ms\n'.format('Maximum timing for classification', res['max_t']) + \
                      '{:<50}: {:18.3f}ms'.format('Standard deviation of timing for classification', res['std_t'])
        report_window(root, report_text)
        test_button['state'] = 'normal'
    else:
        root.after(100, wait_classifier_test_end, root, wait_win)


def wait_detector_test_end(root, wait_win):
    if net_test.is_ready():
        res = net_test.report()
        wait_win.destroy()
        report_text = '*' * 80 + '\n' + \
                      'Testing results:\n' + \
                      '{:<50}: {:20d}\n'.format('Total images tested', res['total_images']) + \
                      '{:<50}: {:19.2f}%\n\n'.format('Mean Average Precision (mAP50)', res['mAP50']) + \
                      '{:<50}: {:18.3f}ms\n'.format('Average timing for classification', res['average_t']) + \
                      '{:<50}: {:18.3f}ms\n'.format('Maximum timing for classification', res['max_t']) + \
                      '{:<50}: {:18.3f}ms'.format('Standard deviation of timing for classification', res['std_t'])
        report_window(root, report_text)
        test_button['state'] = 'normal'
    else:
        root.after(100, wait_detector_test_end, root, wait_win)


def test_action(root):
    test_type = select_test_dialog(root)
    if test_type is not None:
        if test_type == 'Classifier':
            test_button['state'] = 'disabled'
            net_test.run_test(classifier.Classifier, MODEL_CLASSIFIER_PATH)
            test_wait_win = wait_window(root)
            root.after(100, wait_classifier_test_end, root, test_wait_win)
        elif test_type == 'Detector':
            test_button['state'] = 'disabled'
            net_test.run_test(detector_t.Detector, MODEL_DETECTOR_PATH)
            test_wait_win = wait_window(root)
            root.after(100, wait_detector_test_end, root, test_wait_win)


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
    stop_event.set()
    if camera is not None:
        camera.stop()
    stop_action()
    print('Exit')
    root.destroy()


def analyze_loop(root, video_panel):
    global camera
    global analyze_started
    global bounding_boxes
    global defect_types
    global defect_scores
    global plane_id
    global vid_writer

    img = camera.get()
    if img is not None:
        if net_analyzer is not None:
            if not analyze_started:
                analyze_started = True
                net_analyzer.test_start(img)
            else:
                res = net_analyzer.get()
                if res is not None:
                    bounding_boxes = res['bboxes']
                    defect_scores = res['probs']
                    defect_types = res['types']
                    analyze_started = False
            for def_type, def_score, bbox in zip(defect_types, defect_scores, bounding_boxes):
                x1, y1, x2, y2 = bbox
                cv2.rectangle(img,
                              (x1, y1),
                              (x2, y2),
                              color=(0, 0, 255),
                              thickness=2)
                cv2.putText(img,
                            '{:.2f}% {:}'.format(def_score * 100.0, def_type),
                            (x1, y1 - 10),
                            fontFace=cv2.FONT_ITALIC,
                            fontScale=0.8,
                            thickness=1,
                            color=(255, 0, 0))
        if capture_event.is_set():
            cv2.imwrite(os.path.join(plane_id['results'], str(plane_id['capture_counter']) + '.jpg'), img)
            plane_id['capture_counter'] += 1
            if len(bounding_boxes) > 0:
                plane_id['defects'].add(defects_list.DefectFrame(img, bounding_boxes, defect_scores, defect_types))
                plane_id['has_data'] = True
            capture_event.clear()

        if record_event.is_set():
            vid_writer.write(img)
            if len(bounding_boxes):
                plane_id['defects'].add(defects_list.DefectFrame(img, bounding_boxes, defect_scores, defect_types))
                plane_id['has_data'] = True
        img = cv2.resize(img, VIDEO_FRAME_SIZE)
        update_action(video_panel, img)
    if not stop_event.is_set():
        root.after(ms=10, func=lambda: analyze_loop(root, video_panel))


def update_action(video_panel, img):
    global video_panel_image
    video_panel_image = bgr_to_tk_image(img)
    video_panel.configure(image=video_panel_image)
    video_panel.image = video_panel_image


if __name__ == '__main__':
    main_window = tk.Tk()
    main_window = center_window(main_window, WINDOW_SIZE)
    main_window.resizable(False, False)
    main_window.configure(bg=BACKGROUND_COLOR)
    # main_window.wm_attributes('-fullscreen', 'True')

    frame = np.random.randint(0, 255, size=(VIDEO_FRAME_SIZE[1], VIDEO_FRAME_SIZE[0], 3), dtype=np.uint8)
    video_panel_image = bgr_to_tk_image(frame)

    video_out = tk.Label(main_window,
                         image=video_panel_image,
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

    camera = video_cam.VideoCamera(0)
    main_window.after(ms=10, func=lambda: analyze_loop(main_window, video_out))

    main_window.update()
    main_window.mainloop()
