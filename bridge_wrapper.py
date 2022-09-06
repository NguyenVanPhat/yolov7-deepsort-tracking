'''
A Moduele which binds Yolov7 repo with Deepsort with modifications
'''

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # comment out below line to enable tensorflow logging outputs
import time
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.compat.v1 import \
    ConfigProto  # DeepSORT official implementation uses tf1.x so we have to do some modifications to avoid errors

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

# import from helpers
from tracking_helpers import read_class_names, create_box_encoder
from detection_helpers import *

# load configuration for object detector
config = ConfigProto()
config.gpu_options.allow_growth = True


def pprint(name_variable, variable):
    print(
        "\n------------------------------------------ " + "BIẾN " + name_variable + " ------------------------------------------")
    try:
        print("TYPE: " + "---" + str(type(variable)) + "---")
    except:
        print("ko hien thi duoc TYPE()")
    try:
        print("LEN: " + "---" + str(len(variable)) + "---")
    except:
        print("ko hien thi duoc LEN()")
    try:
        print("SHAPE: " + "---" + str(variable.shape) + "---")
    except:
        print("ko hien thi duoc SHAPE()")
    try:
        print("VALUE: ", variable)
    except:
        print("ko hien thi duoc VALUE")
    finally:
        print(
            "------------------------------------------ KẾT THÚC BIẾN {0} ------------------------------------------".format(
                name_variable))


class YOLOv7_DeepSORT:
    """
    Class to Wrap ANY detector  of YOLO type with DeepSORT
    """

    def __init__(self, reID_model_path: str, detector, max_cosine_distance: float = 0.4, nn_budget: float = None,
                 nms_max_overlap: float = 1.0,
                 coco_names_path: str = "./io_data/input/classes/coco.names", ):
        """
        args:
            reID_model_path: Path of the model which uses generates the embeddings for the cropped area for Re identification
            detector: object of YOLO models or any model which gives you detections as [x1,y1,x2,y2,scores, class]
            max_cosine_distance: Cosine Distance threshold for "SAME" person matching - xem "https://www.youtube.com/watch?v=sPu-V5Qy3CY" để hiểu thêm.
            nn_budget:  If not None, fix samples per class to at most this number. Removes the oldest samples when the budget is reached.
            nms_max_overlap: Maximum NMs allowed for the tracker
            coco_file_path: File wich contains the path to coco naames
        """
        self.detector = detector
        self.coco_names_path = coco_names_path
        self.nms_max_overlap = nms_max_overlap
        # "read_class_names()" sẽ trả ra 1 dict có len = 80; value = {0: 'person', 1: 'bicycle', 2: 'car'...}
        self.class_names = read_class_names()

        # initialize deep sort
        self.encoder = create_box_encoder(reID_model_path, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance,
                                                           nn_budget)  # calculate cosine distance metric
        # Khởi tạo Tracker của DeepSORT
        self.tracker = Tracker(metric)  # initialize tracker

    def track_video(self, video: str, output: str, skip_frames: int = 0, show_live: bool = False,
                    count_objects: bool = False, verbose: int = 0):
        '''
        Track any given webcam or video
        args: 
            video: path to input video or set to 0 for webcam
            output: path to output video
            skip_frames: Skip every nth frame. After saving the video, it'll have very visuals experience due to skipped frames
            show_live: Whether to show live video tracking. Press the key 'q' to quit
            skip_frames: sẽ cắt ngắn video từ frame thứ bao nhiêu, nếu để "0" là sẽ ko cắt video
            count_objects: count objects being
            tracked on screen (hiển thị dòng chữ ở góc trái trên cùng video số lượng object đang theo dõi)
            verbose: print details on the screen allowed values 0,1,2
        '''
        # Dùng OpenCV để đọc video từ đường dẫn "video"
        try:  # begin video capture
            vid = cv2.VideoCapture(int(video))
        except:
            vid = cv2.VideoCapture(video)

        out = None
        # thiết lập các thông số để lưu video vào đường dẫn đầu ra "output"
        if output:  # get video ready to save locally if flag is set
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))  # by default VideoCapture returns float instead of int
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(output, codec, fps, (width, height))

        # "frame_num" dùng để đếm số frame hiện tại trong video
        frame_num = 0
        # Khối xử lý chính của Chương trình - Vòng lặp chạy qua từng Frame video và xử lý từng frame đó
        while True:  # while video is running
            # "frame" có type = numpy.ndarray; shape = (1080, 1920, 3);
            return_value, frame = vid.read()
            # "return_value = False" nghĩa là đã chạy hết Video sẽ dừng loop While
            if not return_value:
                print('Video has ended or failed!')
                break
            frame_num += 1
            print("\n FRAME = ", frame_num)
            # nếu "skip_frames" có giá trị, thì khi Frame chạy đến vị trí "skip_frames" quy định sẽ chạy..
            # lệnh "continue" khi đó sẽ bỏ qua khối xử lý bên dưới và quay lại loop while bên trên cho..
            # đến hết video, đồng nghĩa video đầu ra sẽ ko có các frame từ "skip_frames" trở đi.
            if skip_frames and not frame_num % skip_frames: continue  # skip every nth frame. When every frame is not important, you can use this to fasten the process
            if verbose >= 1: start_time = time.time()

            # -----------------------------------------PUT ANY DETECTION MODEL HERE -----------------------------------------------------------------
            # "frame.copy()" cho ra kết quả y hệt "frame"
            # "yolo_dets" có type = numpy.ndarray; shape = (n, 6); là n là số lượng object phát hiện được trong..
            # 1 frame truyền vào và "6" là 6 giá trị thuộc tính của mỗi đối tượng
            # mỗi object là 1 row trong yolo_dets có dạng ("bb" viết tắt của bounding boxes):..
            # [toạ_độ_X_top_left_bb, toạ_độ_Y_top_left_bb, toạ_độ_X_bottom_right_bb, toạ_độ_Y_bottom_right_bb, confident, classes]
            # --> lưu ý: những thông số kỹ thuật trên (toạ độ tâm, chiều cao...) đều chưa được mã hoá thành định dạng..
            # theo chuẩn của Yolo, vẫn đang trình bày dưới dạng giá trị pixel và giá trị float
            yolo_dets = self.detector.detect(frame.copy(), plot_bb=False)  # Get the detections
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Nếu ko detect được object trong frame thì "yolo_dets" sẽ có "value = None"
            # "scores" là confidence trong Yolo (% dự đoán chính xác đối tượng)
            if yolo_dets is None:
                bboxes = []
                scores = []
                classes = []
                num_objects = 0

            else:
                # "bboxes" đóng vai trò là tạo độ của bounding boxes
                bboxes = yolo_dets[:, :4]
                bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]  # convert from xyxy to xywh
                bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]

                scores = yolo_dets[:, 4]
                # "classes" có type = numpy.ndarray; shape = (n,); nếu object là người thì sẽ có value = 0
                classes = yolo_dets[:, -1]
                # "num_objects" có type = int; value = n
                num_objects = bboxes.shape[0]
            # ---------------------------------------- DETECTION PART COMPLETED ---------------------------------------------------------------------
            names = []
            for i in range(num_objects):  # loop through objects and use class index to get class name
                class_indx = int(classes[i])
                # "class_name" có type = str; value = "person"; đóng vai trò dịch các số index thành tên classes ví dụ: "person", "car"..
                # để bỏ các classes đó vào list "names"
                class_name = self.class_names[class_indx]
                # giả sử "classes" có value = [0, 0, 38]
                # thì "names" sau khi kết thức loop sẽ có value = ["person", "person", "tennis racket"]
                names.append(class_name)

            names = np.array(names)
            count = len(names)

            if count_objects:
                cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1.5, (0, 0, 0), 2)

            # ---------------------------------- DeepSORT tracker work starts here ------------------------------------------------------------
            # "frame" có type = numpy.ndarray; shape = (1080, 1920, 3);
            # "bboxes" có type = numpy.ndarray; shape = (6, 4);
            # "features" có type = numpy.ndarray; shape = (6, 128); là một vectơ đặc trưng mô tả đối tượng có trong frame này.
            features = self.encoder(frame,
                                    bboxes)  # encode detections and feed to tracker. [No of BB / detections per frame, embed_size]
            # "detections" có type = list; len = n; mỗi phần tử trong list "detections" là 1 bounding boxes đại diện cho 1 object..
            # được detect trong frame này.
            # Mỗi phần tử trong list "detections" có type = <class 'deep_sort.detection.Detection'>
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                          zip(bboxes, scores, names,
                              features)]  # [No of BB per frame] deep_sort.detection.Detection object

            # "cmap" có type = <class 'matplotlib.colors.ListedColormap'>;
            cmap = plt.get_cmap('tab20b')  # initialize color map
            # "np.linspace(0, 1, 20)" tạo ra 1 numpy.ndarray có len = 20
            # "colors" có type = list; len = 20; mỗi phần tử trong list này là 1 tuple..
            # có value ví dụ như = (0.22, 0.23, 0.47) đại diện cho mã màu cho object
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
            # "boxs" có type = numpy.ndarray; shape = (6, 4); có định dạng Bounding box (top left x, top left y, width, height)
            boxs = np.array([d.tlwh for d in detections])  # run non-maxima supression below
            # "scores" có type = numpy.ndarray; len = n; value = [0.92801, 0.92623, 0.91129, 0.89674, 0.89323, 0.85231]
            scores = np.array([d.confidence for d in detections])
            # "classes" có type = numpy.ndarray; len = n; value = ['person' 'person'...]
            classes = np.array([d.class_name for d in detections])
            # "nms_max_overlap" có type = float; value = 1.0. là --Maximum NMs allowed for the tracker--
            # "indices" có type = list; len = 6; value = [0, 1, 2, 3, 4, 5]
            indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
            # "detections" có type = list; len = 6; chứa 6 phần tử là class 'deep_sort.detection.Detection'
            detections = [detections[i] for i in indices]

            # Gọi Hàm predict() của Class "Tracker" trong file ./deepsort/tracker.py; vô tracker.py để biết chi tiết
            # Bước gọi hàm "predict()" sử dụng KalmanFilter để dự đoán trước các attribute của mỗi object được detect trong frame
            self.tracker.predict()  # Call the tracker
            # Update bouding boxes của các object cho Kalman
            self.tracker.update(detections)  # updtate using Kalman Gain

            for track in self.tracker.tracks:  # update new findings AKA tracks
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                class_name = track.get_class()

                color = colors[int(track.track_id) % len(colors)]  # draw bbox on screen
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
                              (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color,
                              -1)
                cv2.putText(frame, class_name + " : " + str(track.track_id), (int(bbox[0]), int(bbox[1] - 11)), 0, 0.6,
                            (255, 255, 255), 1, lineType=cv2.LINE_AA)

                if verbose == 2:
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(
                        str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

            # -------------------------------- Tracker work ENDS here -----------------------------------------------------------------------
            if verbose >= 1:
                fps = 1.0 / (time.time() - start_time)  # calculate frames per second of running detections
                if not count_objects:
                    print(f"Processed frame no: {frame_num} || Current FPS: {round(fps, 2)}")
                else:
                    print(
                        f"Processed frame no: {frame_num} || Current FPS: {round(fps, 2)} || Objects tracked: {count}")

            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if output: out.write(result)  # save output video

            if show_live:
                cv2.imshow("Output Video", result)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

        cv2.destroyAllWindows()
