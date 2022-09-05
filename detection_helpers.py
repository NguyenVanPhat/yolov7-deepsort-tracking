
import cv2
import torch
from numpy import random

from models.experimental import attempt_load
from utils.datasets import letterbox, np
from utils.general import check_img_size, non_max_suppression, apply_classifier,scale_coords, xyxy2xywh
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier,TracedModel

def pprint(name_variable, variable):
    print("\n------------------------------------------ "+"BIẾN "+name_variable+" ------------------------------------------")
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
        print("------------------------------------------ KẾT THÚC BIẾN {0} ------------------------------------------".format(name_variable))

class Detector:
    def __init__(self, conf_thres:float = 0.25, iou_thresh:float = 0.45, agnostic_nms:bool = False, save_conf:bool = False, classes:list = None):
        '''
        args:
        conf_thres: Thresholf for Classification
        iou_thres: Thresholf for IOU box to consider
        agnostic_nms: whether to use Class-Agnostic NMS
        save_conf: whether to save confidences in 'save_txt' labels afters inference
        classes: Filter by class from COCO. can be in the format [0] or [0,1,2] etc
        '''
        self.device = select_device("cuda" if torch.cuda.is_available() else 'cpu')
        self.conf_thres = conf_thres
        self.iou_thres = iou_thresh
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.save_conf = save_conf


    def load_model(self, weights:str, img_size:int = 640, trace:bool = True, classify:bool = False):
        '''
        weights: Path to the model
        img_size: Input image size of the model
        trace: Whether to trace the model or not
        classify: whether to load the second stage classifier model or not
        '''
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(img_size, s=self.stride)  # check img_size

        if trace:
            self.model = TracedModel(self.model, self.device, img_size)

        if self.half:
            self.model.half()  # to FP1
        
        # Run inference for CUDA just once
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

        # Second-stage classifier
        self.classify = classify
        if classify:
            self.modelc = load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()

         # Get names and colors of Colors for BB creation
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]



    @torch.no_grad()
    def detect(self, source, plot_bb:bool =True):
        '''
        source: Path to image file, video file, link or text etc
        plot_bb: whether to plot the bounding box around image or return the prediction
        '''
        # "img" có type = <class 'numpy.ndarray'>; shape = (3, 384, 640);
        # "im0" có type = <class 'numpy.ndarray'>; shape = (1080, 1920, 3);
        img, im0 = self.load_image(source)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3: # Single batch -> single image
            img = img.unsqueeze(0)

        # Inference
        # "pred" có TYPE=<class 'torch.Tensor'>, SHAPE=[1, 15120, 85]
        # Với mỗi img(3, 384, 640) truyền vào model(img, augment=opt.augment) sẽ trả ra 1 tuple có LEN=2:
        # - Phần tử đầu tiên là được gán vào biến "pred" và có là 1 Tensor có SHAPE=[1, 15120, 85]
        # - Phần tử thứ 2 là 1 list có LEN=3 chứa 3 Tensor khác nhau:
        #   + Tensor 1 có SHAPE=[1, 3, 48, 80, 85]
        #   + Tensor 2 có SHAPE=[1, 3, 24, 40, 85]
        #   + Tensor 3 có SHAPE=[1, 3, 12, 20, 85]
        # --> chú ý: các thông số này áp dụng với thiết đặt chỉ detect 1 classes là ["person"].
        # trong ví dụ video đầu vào có 6 người được phát hiện
        pred = self.model(img, augment=False)[0] # We don not need any augment during inference time


        # Apply NMS
        # Apply NMS viết tắt của Non-Maximum Suppression: là 1 method trong thị giác máy tính giúp chọn...
        # một thực thể duy nhất trong nhiều thực thể chồng chéo lên nhau (thực thể trong bài này là...
        # bounding boxes). Phương pháp là loại bỏ các thực thể nằm dưới một giới hạn xác xuất(%) đã cho sẵn..
        # nếu giới hạn xác xuất càng cao thì càng khắt khe trong việc chọn thực thể hay nói cách khác..
        # số lượng thực thể được chọn sẽ ít đi, chỉ những thực thể có xác xuất cao đáng tin cậy mới được chọn.
        # ----------------------------------------------------------------------------------------
        # "opt.conf_thres" mặc định là 0.25(float)
        # "opt.iou_thres" mặc định là 0.45(float)
        # "opt.classes" mặc định là None(nghĩa là detect tất các classes có thể)
        # ----------------------------------------------------------------------------------------
        # với mỗi "pred"([1, 15120, 85]) bên trên non_max_suppression() sẽ trả ra một list of detections..
        # list này chứa duy nhất 1 Tensor có SHAPE=[n, 6] với n ở đây là số đối tượng phát hiện được..
        # trong image/frame. số 6 là số lượng thông số kỹ thuật cho mỗi object được phát hiện cụ thể..
        # mỗi object là 1 row trong Tensor có dạng ("bb" viết tắt của bounding boxes):..
        # [toạ_độ_X_top_left_bb, toạ_độ_Y_top_left_bb, toạ_độ_X_bottom_right_bb, toạ_độ_Y_bottom_right_bb, confident, classes]
        # --> lưu ý: những thông số kỹ thuật trên (toạ độ tâm, chiều cao...) đều chưa được mã hoá thành định dạng..
        # theo chuẩn của Yolo, vẫn đang trình bày dưới dạng giá trị pixel và giá trị float
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)

        # Apply Classifier
        # "classify" mặc định là False
        if self.classify:
            pred = apply_classifier(pred, self.modelc, img, im0) # I thnk we need to add a new axis to im0


        # Post - Process detections
        det = pred[0]# detections per image but as we have  just 1 image, it is the 0th index
        # nếu Detect được bất kỳ object nào thì sẽ chạy "if len(det):" nếu ko sẽ return "None" bằng lệnh dưới
        if len(det):
            # Rescale boxes from img_size to im0 size
            # "img.shape" có type: <class 'torch.Size'>; value: [1, 3, 384, 640]
            # "im0.shape" có type: <class 'tuple'>; value: (1080, 1920, 3)
            # "scale_coords()" sẽ chuyển giá trị phần tử trong "det" đang phù hợp với "img.shape"..
            # thành giá trị mới phù hợp cho "im0.shape"..
            # -> Ví dụ: img.shape[2:]=[384, 640] có det=[523.44592, 69.83182, 561.39929, 143.79759, 0.90688, 0.00000]..
            # với im0.shape=(1080, 1920, 3) có det=[1570.0, 173.0, 1684.0, 395.0, 0.906877, 0.0]
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                # "plot_bb" mặc định là False, nếu để thành True sẽ khiến chương trình chạy lỗi.
                if plot_bb:  # Add bbox to image   # save_img
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)
                    
            # "det.detach().cpu().numpy()" có type = numpy.ndarray; shape = (6, 6)
            return im0 if plot_bb else det.detach().cpu().numpy()

        return im0 if plot_bb else None # just in case there's no detection, return the original image. For tracking purpose plot_bb has to be False always
        


    def load_image(self, img0):
        '''
        Load and pre process the image
        args: img0: Path of image or numpy image in 'BGR" format
        '''
        # "isinstance" kiểm tra xem "img0" có phải là đường dẫn image/video hay ko để sử dụng "cv2.imread()"..
        # để đọc, nhưng trong bài này ảnh được được đọc và truyền hẳn vào luôn.
        if isinstance(img0, str): img0 = cv2.imread(img0)  # BGR
        assert img0 is not None, 'Image Not Found '

        # Padded resize
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img, img0
    

    def save_txt(self, det, im0_shape, txt_path):
        '''
        Save the results of an image in a .txt file
        args:
            det: detecttions from the model
            im0_shape: Shape of Original image
            txt_path: File of the text path
        '''
        gn = torch.tensor(im0_shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in reversed(det):
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
            with open(txt_path + '.txt', 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')
