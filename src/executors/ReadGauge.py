import os
import sys
import math
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.base.model import BoundingBox
from sdks.novavision.src.helper.executor import Executor
from capsules.ReadGauge.src.utils.utils import load_model
from capsules.ReadGauge.src.utils.response import build_response
from capsules.ReadGauge.src.models.PackageModel import PackageModel, Detection


class ReadGauge(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.image = self.request.get_param("inputImage")
        self.device = self.request.get_param("ConfigDevice")
        self.conf_thres = self.request.get_param("ConfidenceThreshold")
        self.iou_thres = self.request.get_param("IOUThreshold")
        self.model = self.bootstrap.get("model")
        
        self.gauge_min_value = self.request.get_param("GaugeMinimumValue")
        self.gauge_max_value = self.request.get_param("GaugeMaximumValue")
        self.use_longer_arc = self.request.get_param("UseLongerArc")

        self.namedict = {
            0: "center",
            1: "gauge",
            2: "max",
            3: "min",
            4: "tip"
        }
    @staticmethod
    def bootstrap(config: dict) -> dict:
        model = load_model(config=config)
        return {"model": model}

    def get_center_point(self, bbox):
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def calculate_angle(self, center, point):
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        angle = math.atan2(-dy, dx)  # negative dy because y axis different in image
        return math.degrees(angle)

    def normalize_angle(self, angle):
        while angle < 0:
            angle += 360
        while angle >= 360:
            angle -= 360
        return angle

    def calculate_distance(self, p1, p2): # euclidean
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def calculate_gauge_reading(self, center_pos, tip_pos, min_pos, max_pos):
        tip_angle = self.normalize_angle(self.calculate_angle(center_pos, tip_pos))
        min_angle = self.normalize_angle(self.calculate_angle(center_pos, min_pos))
        max_angle = self.normalize_angle(self.calculate_angle(center_pos, max_pos))

        tip_radius = self.calculate_distance(center_pos, tip_pos)
        min_radius = self.calculate_distance(center_pos, min_pos)
        max_radius = self.calculate_distance(center_pos, max_pos)
        gauge_scale_radius = (min_radius + max_radius) / 2

        # determine direction
        ccw_range = (max_angle - min_angle) % 360
        cw_range = (min_angle - max_angle) % 360

        # counterclockwise / clockwise
        if self.use_longer_arc:
            chosen_direction = "CCW" if ccw_range >= cw_range else "CW"
        else:
            chosen_direction = "CCW" if ccw_range <= cw_range else "CW"

        if chosen_direction == "CCW":
            angle_range = ccw_range
            tip_offset_raw = (tip_angle - min_angle) % 360
            direction = "CCW"

            if tip_offset_raw <= angle_range:
                tip_offset = tip_offset_raw
            else:
                dist_to_min_backwards = 360 - tip_offset_raw
                dist_beyond_max = tip_offset_raw - angle_range
                tip_offset = 0 if dist_to_min_backwards < dist_beyond_max else angle_range
        else:
            angle_range = cw_range
            tip_offset_raw = (min_angle - tip_angle) % 360
            direction = "CW"

            if tip_offset_raw <= angle_range:
                tip_offset = tip_offset_raw
            else:
                dist_to_min_backwards = 360 - tip_offset_raw
                dist_beyond_max = tip_offset_raw - angle_range
                tip_offset = 0 if dist_to_min_backwards < dist_beyond_max else angle_range

        # linear interpolation
        if angle_range != 0:
            reading = self.gauge_min_value + (tip_offset / angle_range) * (self.gauge_max_value - self.gauge_min_value)
        else:
            reading = self.gauge_min_value

        return {
            'reading': reading,
            'tip_angle': tip_angle,
            'min_angle': min_angle,
            'max_angle': max_angle,
            'tip_radius': tip_radius,
            'min_radius': min_radius,
            'max_radius': max_radius,
            'gauge_scale_radius': gauge_scale_radius,
            'angle_range': angle_range,
            'tip_offset': tip_offset,
            'direction': direction,
            'percentage': (tip_offset / angle_range * 100) if angle_range != 0 else 0
        }

    def output_result(self, output, img_uid):
        output = output[0].cpu().numpy()
        bboxes = output.boxes.data
        detection_list = []
        
        for i in range(len(bboxes)):
            bbox = BoundingBox(
                left=bboxes[i][0],
                top=bboxes[i][1],
                width=bboxes[i][2] - bboxes[i][0],
                height=bboxes[i][3] - bboxes[i][1],
            )
            new_detect = Detection(
                boundingBox=bbox,
                confidence=bboxes[i][4],
                classLabel=self.namedict[int(bboxes[i][5])],
                classId=int(bboxes[i][5]),
                imgUID=img_uid,
            )
            detection_list.append(new_detect)
        
        return detection_list

    def infer(self, image):
        return self.model.predict(
            image,
            conf=float(self.conf_thres),
            iou=float(self.iou_thres),
        )

    def detection_inference(self, img):
        output = self.infer(np.array(img.value))
        return self.output_result(output, img.uID)

    def extract_positions_from_detections(self, detections):
        positions = {
            'center': None,
            'tip': None,
            'min': None,
            'max': None
        }
        
        for detection in detections:
            class_label = detection.classLabel.lower()
            
            bbox = detection.boundingBox
            x1 = bbox.left
            y1 = bbox.top
            x2 = bbox.left + bbox.width
            y2 = bbox.top + bbox.height
            center_point = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            if class_label == 'center':
                positions['center'] = center_point
            elif class_label == 'tip':
                positions['tip'] = center_point
            elif class_label == 'min':
                positions['min'] = center_point
            elif class_label == 'max':
                positions['max'] = center_point
        
        return positions

    def run(self):
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        
        self.detections = self.detection_inference(img)
        positions = self.extract_positions_from_detections(self.detections)
        
        if all([positions['center'], positions['tip'], positions['min'], positions['max']]):
            calculation_result = self.calculate_gauge_reading(
                positions['center'],
                positions['tip'],
                positions['min'],
                positions['max']
            )
            
            self.reading = calculation_result
        else:
            missing = [k for k, v in positions.items() if v is None]
            print(f"Cannot calculate reading - missing: {', '.join(missing)}")
            self.reading = {}
        
        packageModel = build_response(context=self)
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()