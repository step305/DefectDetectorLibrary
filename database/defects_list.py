import datetime


class DefectFrame:
    def __init__(self, image, boxes, scores, types):
        self.image = image
        self.boxes = boxes
        self.scores = scores
        self.types = types


class AirCraftDefectsList:
    def __init__(self, serial_num='0000', name='plane'):
        self.defects = []
        self.serial_num = serial_num
        self.name = name
        self.date = datetime.datetime.now()

    def add(self, defect):
        self.defects.append(defect)
