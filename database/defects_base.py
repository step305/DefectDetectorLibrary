import datetime
import os.path
import pickle
import sqlite3

import cv2
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from database import defects_list


def add_text_to_pdf_center(canvas, text, y):
    width, height = A4
    text_width = stringWidth(text, fontName='GOST', fontSize=14)
    pdf_text_object = canvas.beginText((width - text_width) / 2.0, y)
    pdf_text_object.textOut(text)


def add_text_to_pdf_left(canvas, text, x, y):
    pdf_text_object = canvas.beginText(x, y)
    pdf_text_object.textOut(text)


def footer(canvas, date, width, page):
    canvas.drawCentredString(width / 2, 40, '{}.{}.{}'.format(date.day,
                                                              date.month,
                                                              date.year))
    canvas.drawCentredString(width / 2, 60, 'ООО СК-"Роботикс"')
    canvas.drawCentredString(width / 2, 20, 'стр. {}'.format(page))
    canvas.line(width / 2 - 150, 53, width / 2 + 150, 53)


class DefectsBase:
    def __init__(self, path_to_base='defects_base.db'):
        try:
            self.db_connection = sqlite3.connect(path_to_base)
            self.cursor = self.db_connection.cursor()
            sql_req = 'CREATE TABLE IF NOT EXISTS airplanes (id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT DEFAULT 0,' \
                      'name TEXT NOT NULL,' \
                      'serial TEXT NOT NULL,' \
                      'comment TEXT DEFAULT "",' \
                      'UNIQUE (name, serial) ON CONFLICT  IGNORE);'
            self.cursor.execute(sql_req)
            sql_req = 'CREATE TABLE IF NOT EXISTS defects (id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT DEFAULT 0,' \
                      'airplane_name TEXT NOT NULL,' \
                      'air_plane_serial TEXT NOT NULL,' \
                      'date TIMESTAMP NOT NULL,' \
                      'defect_data BLOB,' \
                      'comment TEXT DEFAULT "");'
            self.cursor.execute(sql_req)
            self.db_connection.commit()
            self.cursor.close()
        except sqlite3.Error as e:
            print('Error opening/creating database', e)

    def all(self):
        self.cursor = self.db_connection.cursor()
        sql_req = 'SELECT name, serial FROM airplanes;'
        self.cursor.execute(sql_req)
        data = self.cursor.fetchall()
        result = {
            'name': [],
            'serial': [],
        }
        for dat in data:
            name_, serial_ = dat
            result['name'].append(name_)
            result['serial'].append(serial_)
        self.cursor.close()
        return result

    def add(self, aircraft_defects_list: defects_list.AirCraftDefectsList):
        self.cursor = self.db_connection.cursor()
        sql_req = 'INSERT OR IGNORE INTO airplanes (name, serial) VALUES (?,?);'
        self.cursor.execute(sql_req, (aircraft_defects_list.name, aircraft_defects_list.serial_num))

        date_now = datetime.datetime.now()
        for defect in aircraft_defects_list.defects:
            sql_req = 'INSERT INTO defects (airplane_name, air_plane_serial, date, defect_data, comment) ' \
                      'VALUES (?, ?, ?, ?, ?);'
            jpeg_defect = defect
            _, jpeg_defect.img = cv2.imencode('.jpg', jpeg_defect.image)
            self.cursor.execute(sql_req, (aircraft_defects_list.name,
                                          aircraft_defects_list.serial_num,
                                          aircraft_defects_list.date,
                                          pickle.dumps(jpeg_defect),
                                          ''))
        self.db_connection.commit()
        self.cursor.close()

    def get(self, aircraft_name, aircraft_serial):
        self.cursor = self.db_connection.cursor()
        sql_req = 'SELECT MAX(date) FROM defects WHERE airplane_name=? AND air_plane_serial=?;'
        self.cursor.execute(sql_req, (aircraft_name, aircraft_serial))
        date, = self.cursor.fetchone()

        sql_req = 'SELECT defect_data FROM defects WHERE airplane_name=? AND air_plane_serial=? AND date=?;'
        self.cursor.execute(sql_req, (aircraft_name, aircraft_serial, date))
        defects_resp = self.cursor.fetchall()
        date = datetime.datetime.strptime(date.split('.')[0], '%Y-%m-%d %H:%M:%S')

        air_craft = defects_list.AirCraftDefectsList(aircraft_serial, aircraft_name)
        air_craft.date = date
        for defect_resp in defects_resp:
            defect_pickled, = defect_resp
            defect = pickle.loads(defect_pickled)
            defect.img = cv2.imdecode(defect.img, cv2.IMREAD_COLOR)
            air_craft.defects.append(defect)
        self.cursor.close()
        return air_craft

    def report(self, report_path, aircraft_name, aircraft_serial):
        defects_table = {'Class_1': 'не определен', 'Class_2': '"трещины"', 'Class_3': '"риски"', 'Class_4': '"вмятины"',
                         'Class_5': '"коррозия"'}
        air_craft = self.get(aircraft_name, aircraft_serial)
        pdfmetrics.registerFont(TTFont('GOST', 'database\\GOSTtypeB.ttf'))

        pdf_canvas = canvas.Canvas(os.path.join(report_path,
                                                'air_craft_{}_serial_{}.pdf'.format(aircraft_name, aircraft_serial),
                                                ), pagesize=A4)
        width, height = A4
        pdf_canvas.setLineWidth(3)
        pdf_canvas.setFont('GOST', size=14)

        report_day = air_craft.date
        page_num = 1
        footer(pdf_canvas, report_day, width, page_num)

        pdf_canvas.drawCentredString(width / 2, 755, 'Отчет по ВС {}'.format(air_craft.name))
        pdf_canvas.drawCentredString(width / 2, 740, 'серийный номер {}'.format(air_craft.serial_num))
        pdf_canvas.line(width / 2 - 240, 732, width / 2 + 240, 732)
        pdf_canvas.line(width / 2 - 230, 730, width / 2 + 230, 730)

        total_defects = [len(defect.types) for defect in air_craft.defects]
        count_defects = sum(total_defects)

        pdf_canvas.drawString(75, 650, 'Обнаружено дефектов: {} на {} кадрах'.format(count_defects, len(total_defects)))
        pdf_canvas.showPage()

        for defect in air_craft.defects:
            pdf_canvas.setFont('GOST', size=14)
            cv2.imwrite('temp.jpg', cv2.resize(defect.img, (int(width) - 100, int(height / 2))))
            pic = ImageReader('temp.jpg')
            pdf_canvas.drawImage(pic, 50, 100)
            uniq_defects = {i: defect.types.count(i) for i in defect.types}
            str_defects = []
            for defect_type in uniq_defects.keys():
                str_defects.append('дефектов типа {} - {}'.format(defects_table[defect_type],
                                                                  uniq_defects[defect_type]))
            pdf_canvas.drawString(75, 710, 'Обнаружено дефектов:')
            y = 710 - 20
            for line in str_defects:
                pdf_canvas.drawString(100, y, '{}'.format(line))
                y -= 20
            page_num += 1
            footer(pdf_canvas, report_day, width, page_num)
            pdf_canvas.showPage()
        pdf_canvas.save()

    def close(self):
        self.db_connection.close()
