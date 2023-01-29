import multiprocessing as mp
import cv2


def run_loop(class_model, detect_model, queue_in, queue_out, stop):
    print('analyzer started!')

    from detector import classifier
    from detector import detector_t

    net_classifier = classifier.Classifier(class_model)
    net_detector = detector_t.Detector(detect_model)

    while not stop.is_set():
        img = queue_in.get(block=True)
        if img is not None:
            result = {
                'image': None,
                'bboxes': [],
                'scores': [],
                'types': [],
                'probs': [],
            }
            bboxes, scores = net_detector.test(img)
            result['bboxes'] = bboxes
            result['scores'] = scores
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                img_small = img[y1:y2, x1:x2]
                defect_type, prob = net_classifier.test(img_small)
                result['types'].append(defect_type)
                result['probs'].append(prob)
                cv2.rectangle(img,
                              (x1, y1),
                              (x2, y2),
                              color=(0, 0, 255),
                              thickness=2)
                cv2.putText(img,
                            '{:.2f}% {:}'.format(prob * 100.0, defect_type),
                            (x1, y1 - 10),
                            fontFace=cv2.FONT_ITALIC,
                            fontScale=0.4,
                            thickness=1,
                            color=(255, 0, 0))
            result['image'] = img
            if not queue_out.full():
                queue_out.put(result)
    print('Detector loop stopped!')


class DefectAnalyzer:
    def __init__(self, classifier_model_path, detector_model_path):
        self.classifier_model_path = classifier_model_path
        self.detector_model_path = detector_model_path
        self.image_queue = mp.Queue(10)
        self.stop_event = mp.Event()
        self.result_queue = mp.Queue(10)
        self.detector_proc = mp.Process(target=run_loop,
                                        args=(
                                            self.classifier_model_path,
                                            self.detector_model_path,
                                            self.image_queue,
                                            self.result_queue,
                                            self.stop_event)
                                        )
        self.detector_proc.start()

    def test_start(self, image):
        if not self.image_queue.full():
            self.image_queue.put(image)

    def get(self):
        if not self.result_queue.empty():
            return self.result_queue.get()
        else:
            return None

    def stop(self):
        self.stop_event.set()
        self.image_queue.put(None)
        self.detector_proc.join()
