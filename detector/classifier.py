import torch
import cv2
import torchvision.transforms as transforms
import detector.classifier_net.utils as utils
import time
import os
import torch.nn.functional as nn_functional
import numpy as np


class Classifier:
    def __init__(self, path_to_saved_model):
        self.target_device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = utils.load_model(self.target_device, path_to_saved_model)
        self.labels = [
            'Class_1',
            'Class_2',
            'Class_3',
            'Class_4',
            'Class_5',
        ]
        # define preprocess transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    def test(self, image_to_test):
        image_bw = cv2.cvtColor(image_to_test, cv2.COLOR_BGR2RGB)
        image_bw = self.transform(image_bw)
        # add batch dimension
        image_bw = torch.unsqueeze(image_bw, 0)
        with torch.no_grad():
            outputs = self.model(image_bw.to(self.target_device))
            probs = nn_functional.softmax(outputs, dim=-1)
        predicted_class_ = self.labels[torch.argmax(probs)]
        probability_ = torch.max(probs)
        return predicted_class_, probability_

    def verification(self):
        test_images_paths = [
            os.path.abspath(os.path.join(os.path.curdir, 'dataset/classification/train')),
            os.path.abspath(os.path.join(os.path.curdir, 'dataset/classification/validation'))
        ]
        images_paths = []
        for sub_folder_path in test_images_paths:
            for test_images_path in os.listdir(sub_folder_path):
                images_folder = os.path.join(sub_folder_path, test_images_path)
                for f in os.listdir(images_folder):
                    if os.path.isfile(os.path.join(images_folder, f)) and '.jpg' in f.lower():
                        images_paths.append(os.path.join(images_folder, f))
        infer_times = []
        infer_valid = []
        infer_non_valid = []
        # pre-heat classifier
        image = cv2.imread(images_paths[0])
        _, _ = self.test(image)
        for im_path in images_paths:
            image = cv2.imread(im_path)
            # get the ground truth class
            gt_class = im_path.split('\\')[-2]
            t0 = time.time()
            prediction, probability = self.test(image)
            t1 = time.time()
            infer_times.append((t1 - t0) * 1000.0)
            infer_valid.append(1 if gt_class == prediction else 0)
            infer_non_valid.append(1 if probability < 0.5 else 0)
        n_total = len(infer_valid)
        n_true_positives = sum(infer_valid)
        n_false_negatives = sum(infer_non_valid)
        n_false_positives = n_total - sum(infer_valid)
        accuracy = n_true_positives / n_total * 100.0
        precision = n_true_positives / (n_true_positives + n_false_positives) * 100.0
        recall = n_true_positives / (n_true_positives + n_false_negatives) * 100.0
        probability_error = (n_false_positives + n_false_negatives) / n_total * 100.0
        average_timing = sum(infer_times) / n_total
        max_timing = max(infer_times)
        std_timing = np.std(np.array(infer_times))
        verification_result = {
            'total_images': n_total,
            'tp_n': n_true_positives,
            'fn_n': n_false_negatives,
            'fp_n': n_false_positives,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'prob_error': probability_error,
            'average_t': average_timing,
            'max_t': max_timing,
            'std_t': std_timing,
        }
        return verification_result


if __name__ == '__main__':
    net = Classifier('models\\classification_model.pth')
    res = net.verification()
    print('*' * 80)
    print('Testing results:')
    print('Total images tested:', res['total_images'])
    print('True Positive checks: {}'.format(res['tp_n']))
    print('False Negative checks: {}'.format(res['fn_n']))
    print('False Positive checks: {}'.format(res['fp_n']))
    print('Precision: {:.3f}%'.format(res['precision']))
    print('Recall: {:.3f}%'.format(res['recall']))
    print('Probability of wrong classification: {:.3f}%'.format(res['prob_error']))
    print()
    print('Average timing for classification: {:.3f}ms'.format(res['average_t']))
    print('Maximum timing for classification: {:.3f}ms'.format(res['max_t']))
    print('Standard deviation of timing for classification: {:.3f}ms'.format(res['std_t']))
