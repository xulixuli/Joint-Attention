import pycrfsuite
import numpy as np
import cv2

from files.logger import logger

# from files.run_length_filter import RunLengthFilter

LABELS = {
    "cards": 1,
    "dice": 2,
    "key": 3,
    "map": 4,
    "ball": 5,  # phone
    "face": 6,  # ball

    "none": 0,
    "uncertain": 0
}
REV_LABELS = {
    1: "cards",
    2: "dice",
    3: "key",
    4: "map",
    5: "ball",
    6: "face",

    0: "none"
}


class DoStuff:
    def __init__(self, glass_id, confidence_threshold, num_objects, object_detect, debug):
        self.last_frame_processed = 0
        self.glass_id = glass_id
        self.confidence_threshold = confidence_threshold
        self.object_detect = object_detect
        self.num_objects = num_objects
        # self.run_length_filter = []
        # for i in range(self.num_objects):
        # self.run_length_filter.append(RunLengthFilter())
        self.debug = debug

        # CRF
        # self.CRF_filter = CRFFilter()
        self.x_temp = 0
        self.y_temp = 0
        self.pprev = {
            "pprev_cards_distance": 10000,
            "pprev_dice_distance": 10000,
            "pprev_key_distance": 10000,
            "pprev_map_distance": 10000,
            "pprev_ball_distance": 10000,
            "pprev_face_distance": 10000,

            "pprev_is_cards": False,
            "pprev_is_dice": False,
            "pprev_is_key": False,
            "pprev_is_map": False,
            "pprev_is_ball": False,
            "pprev_is_face": False,

            "pprev_gaze_displacement": 0
        }
        self.previous = {
            "previous_cards_distance": 10000,
            "previous_dice_distance": 10000,
            "previous_key_distance": 10000,
            "previous_map_distance": 10000,
            "previous_ball_distance": 10000,
            "previous_face_distance": 10000,

            "previous_is_cards": False,
            "previous_is_dice": False,
            "previous_is_key": False,
            "previous_is_map": False,
            "previous_is_ball": False,
            "previous_is_face": False,

            "previous_gaze_displacement": 0
        }
        self.current = {
            "cards_distance": 10000,
            "dice_distance": 10000,
            "key_distance": 10000,
            "map_distance": 10000,
            "ball_distance": 10000,
            "face_distance": 10000,

            "is_cards": False,
            "is_dice": False,
            "is_key": False,
            "is_map": False,
            "is_ball": False,
            "is_face": False,

            "gaze_displacement": 0
        }

    def do_some_stuff(self, world_proxy, eye_0_proxy, eye_1_proxy, common_data_proxy):
        logger.info('Starting Do_Stuff...')
        # print('do')

        while True:

            world = world_proxy.get_values()  # world: [0]glass_id, [1]timestamp, [2]index, [3]frame
            pupil_0 = eye_0_proxy.get_values()  # pupil: [0]glass_id, [1]timestamp, [2]eye_id, [3]norm_pos, [4]confidence
            pupil_1 = eye_1_proxy.get_values()
            # print(world)

            # if world[0] is None or pupil_0[0] is None or pupil_1[0] is None or self.last_frame_processed == world[2]:
            #     continue

            if (world[0] is None) or (pupil_0[0] is None and pupil_1[0] is None):
                # print(world[2],pupil_0[3],pupil_1[3],'b')
                continue
            # print('do')
            logger.info("Frame - {}, Timestamp - {}".format(world[2], world[1]))
            # logger.info("Eye_Id - {}, Norm_Pos - {}, Confidence - {}, Timestamp - {}".format(pupil_0[2], pupil_0[3], pupil_0[4], pupil_0[1]))
            # logger.info("Eye_Id - {}, Norm_Pos - {}, Confidence - {}, Timestamp - {}".format(pupil_1[2], pupil_1[3], pupil_1[4], pupil_1[1]))

            if pupil_0[0] is None:
                pupil_0 = pupil_1
            elif pupil_1[0] is None:
                pupil_1 = pupil_0

            # print(world[0],pupil_0[3],pupil_1[3],'a')
            if pupil_0[4] > self.confidence_threshold and pupil_1[4] > self.confidence_threshold:
                pupil_loc = np.mean([pupil_0[3], pupil_1[3]], axis=0)
                confidence = np.mean([pupil_0[4], pupil_1[4]])
            elif pupil_0[4] > pupil_1[4]:
                pupil_loc = pupil_0[3]
                confidence = pupil_0[4]
            else:
                pupil_loc = pupil_1[3]
                confidence = pupil_1[4]

            try:
                detections = self.object_detect.perform_detect(world[3])
                detections = denormalize_detections(detections, confidence)
            except Exception as e:
                raise e

            if self.debug:
                self.display_image(detections, pupil_loc, world[3])

            self.last_frame_processed = world[2]

            # run_length_output = self.perform_run_length(detections, pupil_loc)
            # common_data_proxy.set_values(self.glass_id, world[1], world[2], run_length_output[0], run_length_output[1])

            crf_output = self.perform_crf(detections, pupil_loc)
            common_data_proxy.set_values(self.glass_id, world[1], world[2], crf_output[0], crf_output[1])
            # common_data_proxy.set_values(self.glass_id, world[1], world[2], crf_output[0])

    def perform_crf(self, detections, pupil_loc):
        # output = [0] * self.num_objects
        hit = [0] * self.num_objects

        print("num_objects:", self.num_objects)

        x = pupil_loc[0]
        y = pupil_loc[1]
        d = ((x - self.x_temp) ** 2 + (y - self.y_temp) ** 2) ** 0.5
        print("x:", x, "y:", y, "d:", d)

        self.current.update({"gaze_displacement": d})

        self.x_temp = x
        self.y_temp = y

        for detection in detections:
            index = self.object_detect.get_alt_names().index(detection[0]) + 1
            # print(index, "--", detection[0])

            bounds = detection[2]  # bounds: [0]bbox_x, [1]bbox_y, [2]bbox_w, [3]bbox_h

            dist = ((x - bounds[0]) ** 2 + (y - bounds[1]) ** 2) ** 0.5
            dial = ((bounds[2] / 2) ** 2 + (bounds[3] / 2) ** 2) ** 0.5
            norm_dist = dist / dial

            # self.current.update({detection[0] + "_distance": norm_dist})
            self.current.update({REV_LABELS[index] + "_distance": norm_dist})

            x1 = bounds[0] - bounds[2] / 2
            x2 = bounds[0] + bounds[2] / 2
            y1 = bounds[1] - bounds[3] / 2
            y2 = bounds[1] + bounds[3] / 2
            xin = (x1 <= x <= x2)
            yin = (y1 <= y <= y2)
            # print("bbox_x:", bounds[0], "bbox_y:", bounds[1], "bbox_w:", bounds[2], "bbox_h:", bounds[3])
            if xin and yin:
                hit[index - 1] = 1
                # self.current.update({"is_" + detection[0]: True})
                self.current.update({"is_" + REV_LABELS[index]: True})
                break

        feature = self.current
        feature.update(self.previous)
        feature.update(self.pprev)
        features = [feature]

        # training
        tagger = pycrfsuite.Tagger()
        tagger.open('exp_4')

        pred = tagger.tag(features)[0]
        logger.info("features - {}".format(features))
        logger.info("pred - {}".format(pred))
        print("features:", features)
        print("pred:", pred)

        # logger.info("CRF features - {}".format(self.nex))
        # output[LABELS[pred]] = 1

        # logger.info("hit scan output - {}".format(hit))
        # logger.info("CRF output - {}".format(output))
        # print("hit scan output. - {}".format(hit))
        # print("CRF output - {}".format(output))

        self.pprev.update({
            "pprev_cards_distance": self.previous["previous_cards_distance"],
            "pprev_dice_distance": self.previous["previous_dice_distance"],
            "pprev_key_distance": self.previous["previous_key_distance"],
            "pprev_map_distance": self.previous["previous_map_distance"],
            "pprev_ball_distance": self.previous["previous_ball_distance"],
            "pprev_face_distance": self.previous["previous_face_distance"],

            "pprev_is_cards": self.previous["previous_is_cards"],
            "pprev_is_dice": self.previous["previous_is_dice"],
            "pprev_is_key": self.previous["previous_is_key"],
            "pprev_is_map": self.previous["previous_is_map"],
            "pprev_is_ball": self.previous["previous_is_ball"],
            "pprev_is_face": self.previous["previous_is_face"],

            "pprev_gaze_displacement": self.previous["previous_gaze_displacement"]
        })
        self.previous.update({
            "previous_cards_distance": self.current["cards_distance"],
            "previous_dice_distance": self.current["dice_distance"],
            "previous_key_distance": self.current["key_distance"],
            "previous_map_distance": self.current["map_distance"],
            "previous_ball_distance": self.current["ball_distance"],
            "previous_face_distance": self.current["face_distance"],

            "previous_is_cards": self.current["is_cards"],
            "previous_is_dice": self.current["is_dice"],
            "previous_is_key": self.current["is_key"],
            "previous_is_map": self.current["is_map"],
            "previous_is_ball": self.current["is_ball"],
            "previous_is_face": self.current["is_face"],

            "previous_gaze_displacement": self.current["gaze_displacement"]
        })
        self.current.update({
            "cards_distance": 10000,
            "dice_distance": 10000,
            "key_distance": 10000,
            "map_distance": 10000,
            "ball_distance": 10000,
            "face_distance": 10000,

            "is_cards": False,
            "is_dice": False,
            "is_key": False,
            "is_map": False,
            "is_ball": False,
            "is_face": False,

            "gaze_displacement": 0
        })

        # return output, hit
        return pred, hit

    def display_image(self, detections, pupil_loc, frame):
        tmp = frame
        for detection in detections:
            bounds = detection[2]
            x1 = bounds[0] - bounds[2] / 2
            x2 = bounds[0] + bounds[2] / 2
            y1 = bounds[1] - bounds[3] / 2
            y2 = bounds[1] + bounds[3] / 2
            cv2.rectangle(tmp, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            label = detection[0] + ' - ' + str(round(detection[1], 4))
            cv2.putText(tmp, text=label, org=(int(x1), int(y1)), fontFace=3, fontScale=.5, color=(255, 0, 0),
                        thickness=1)

        cv2.imshow('frame.world_{}'.format(self.glass_id),
                   cv2.circle(tmp, (int(pupil_loc[0]), int(pupil_loc[1])), 15, (0, 0, 255), -1))
        cv2.waitKey(1)


def denormalize_detections(detections, confidence):
    if detections is None or len(detections) == 0:
        return detections

    detections_new = []
    x_norm_base = 1280 / 416
    y_norm_base = 720 / 416
    for detection in detections:
        bounds = detection[2]
        x1 = bounds[0] * x_norm_base
        y1 = bounds[1] * y_norm_base
        width = bounds[2] * x_norm_base
        height = bounds[3] * y_norm_base + (1 - confidence / 100) * 25
        if detection[0] == 0 or detection[0] == 4 or detection[0] == 5:
            width = width + (1 - confidence / 100) * 15
            height = height + (1 - confidence / 100) * 15
        else:
            width = width + (1 - confidence / 100) * 30
            height = height + (1 - confidence / 100) * 30

        bounds_new = (x1, y1, width, height)

        detections_new.append((detection[0], detection[1], bounds_new))

    return detections_new
