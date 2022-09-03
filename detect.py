import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box, plot_one_box_center_point
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


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


def detect(save_img=False):
    # "save_txt" có Type: bool, mặc định là False
    # "trace" có Type: bool, mặc định là True
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging() # print thông tin hệ thống như: "YOLOR 🚀 c5a68aa torch 1.12.1+cu113 CPU..."
    device = select_device(opt.device)
    # "half" = False nếu  đang sử dụng cpu và "half" = "tên gpu" nếu có sử dụng GPU
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    # khởi tạo mô hình và nạp trọng số đã truyền vào
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # "trace" mặc định = True, Nếu là False thì model sẽ vẫn là "model = attempt_load()" như ở trên
    if trace:
        # khởi tạo Trace model và kế thừa tất cả attribute của model cũ là "model = attempt_load()"
        # vẫn chưa biết vai trò của model này trong Yolov7
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        # "dataset": có type: <class 'utils.datasets.LoadImages'>
        # dùng để lấy ra dữ liệu theo từng frame thành 4 biến "path, img, im0s, vid_cap"
        # lấy frame tới đâu xử lý ngày tới đó
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    # "hasattr" kiểm tra xem đối tượng có attribute 'module' hay ko
    # Nếu "model = attempt_load()" hoặc "model = TracedModel()" sẽ chỉ sử dụng được "model.names"
    # biến "names" lúc này sẽ get attribute "names" của model
    # biến "names" có TYPE=list, LEN=80, VALUE=['person', 'bicycle', 'car'...]
    names = model.module.names if hasattr(model, 'module') else model.names
    # "colors" có TYPE=list, LEN=80, VALUE=[[239, 9, 93], [153, 225, 34], [152, 194, 182]...]
    # "colors" chứa màu cho từng classes, list colors sẽ tự động thay đổi mỗi lần chạy file detect.py
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    # định nghĩa "inference": là quá trình phân loại và localization(bản địa hoá) từng đối
    # tượng xuất hiện trong image.
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    # đặt t0 là thời gian bắt đầu chạy
    t0 = time.time()
    # path là đường dẫn tới image/video đầu vào, không bị thay đổi trong suốt vòng lặp
    # -----------------------------------------------
    # im0s là image đầu vào, đóng vai trò là image gốc để đối chiếu
    # nếu là video thì im0s sẽ lần lượt là từng frame trong video...
    # hình ảnh im0s có type: (numpy.ndarray)
    # có shape = (1080, 1920, 3)
    # -----------------------------------------------
    # img là image đầu vào đã được resize và xử lý để có shape phù hợp với thuật toán...
    # nếu là video thì img sẽ lần lượt là từng frame trong video...
    # hình ảnh img có type: (numpy.ndarray)
    # giả sử đầu vào có shape = (1080, 1920, 3) thì img sẽ chỉ còn shape = (3, 384, 640)
    # -----------------------------------------------
    # vid_cap có type: class 'cv2.VideoCapture'
    for path, img, im0s, vid_cap in dataset:
        # đang chỉnh sửa từng image từ trong "dataset" trước khi đưa vào model để predict
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        # "pred" có TYPE=<class 'torch.Tensor'>, SHAPE=[1, 15120, 85]
        # Với mỗi img(3, 384, 640) truyền vào model(img, augment=opt.augment) sẽ trả ra 1 tuple có LEN=2:
        # - Phần tử đầu tiên là được gán vào biến "pred" và có là 1 Tensor có SHAPE=[1, 15120, 85]
        # - Phần tử thứ 2 là 1 list có LEN=3 chứa 3 Tensor khác nhau:
        #   + Tensor 1 có SHAPE=[1, 3, 48, 80, 85]
        #   + Tensor 2 có SHAPE=[1, 3, 24, 40, 85]
        #   + Tensor 3 có SHAPE=[1, 3, 12, 20, 85]
        # --> chú ý: các thông số này áp dụng với thiết đặt chỉ detect 1 classes là ["person"].
        # trong ví dụ video đầu vào có 6 người được phát hiện
        pred = model(img, augment=opt.augment)[0]
        # "time_synchronized()" để lấy thời giạn chạy hiện tại
        t2 = time_synchronized()

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
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        # enumerate(pred) sẽ lấy ra được Tensor[n, 6] nằm trong list "pred"
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                # im0 chính là im0s là image đầu vào gốc để đối chiếu
                # frame là số frame hiện tại trong tất cả frame có trong video..
                # ví dụ: frame đầu tiên thì "frame = 1"
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            # print("\nframe = ", frame)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                # "img.shape" có type: <class 'torch.Size'>; value: [1, 3, 384, 640]
                # "im0.shape" có type: <class 'tuple'>; value: (1080, 1920, 3)
                # "scale_coords()" sẽ chuyển giá trị phần tử trong "det" đang phù hợp với "img.shape"..
                # thành giá trị mới phù hợp cho "im0.shape"..
                # -> Ví dụ: img.shape[2:]=[384, 640] có det=[523.44592, 69.83182, 561.39929, 143.79759, 0.90688, 0.00000]..
                # với im0.shape=(1080, 1920, 3) có det=[1570.0, 173.0, 1684.0, 395.0, 0.906877, 0.0]
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # Trong Ví dụ này chỉ phát hiện được 6 person nên "det[:, -1]=[0, 0, 0, 0, 0, 0]"
                # "det[:, -1].unique()" tìm ra các giá trị khác nhau trong Tensor, nếu nhiều phần tử..
                #  có value giống nhau chỉ lấy 1 ra làm đại diện..
                # -> Ví dụ: tensor = [0, 0, 5] thì tensor.unique() trả ra 1 tensor khác có value=[0, 5]
                for c in det[:, -1].unique():
                    # "n" có value = 6 đại diện cho số lượng object được dectect
                    # "s" có type=string; value="6 persons"
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    # "save_img" mặc định là True
                    # "view_img" mặc định là False và nếu đặt True thì vẫn ko show dc trên Colab
                    if save_img or view_img:  # Add bbox to image
                        # "label" có type = string; value = "person 0.76"
                        # value có dạng <Tên classes> + <confidence>;
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                '''
                if source.endswith(".mp4"):
                    print("This is Video File")
                    # Write results
                    # "reversed()" đảo ngược thứ tự row của Tensor truyền vào
                    # -> Ví dụ: - det = [[0, 1, 2],
                    #                  [3, 4, 5]]
                    #           - reversed(det) = [[3, 4, 5],
                    #                            [0, 1, 2]]
                    # "conf" nhận confidence của từng row trong "reversed(det)"
                    # "cls" nhận classes của từng row trong "reversed(det)"
                    # "*xyxy" trả ra n list (n là số lượng object được phát hiện), mỗi list chứa 4 tensor..
                    # mỗi tensor này chứa 4 giá trị đầu tiên trong mỗi row "det"..
                    # nhưng khi gọi "xyxy" sẽ chỉ trả về giá trị tương ứng với "conf" và "cls"
                    def xyxy2xywh_for_det_previous(det):
                        det_previous_temp = []
                        # for *xyxy, conf, cls in reversed(det):
                        for i in range(len(reversed(det))):
                            xywh = (xyxy2xywh(torch.tensor(reversed(det)[i][:4]).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            xywh.append(reversed(det)[i][-2].item())
                            xywh.append(reversed(det)[i][-1].item())
                            # line = (xywh, reversed(det)[i][-2], reversed(det)[i][-1])  # label format
                            det_previous_temp.append(xywh)
                        return det_previous_temp


                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        # "save_img" mặc định là True
                        # "view_img" mặc định là False và nếu đặt True thì vẫn ko show dc trên Colab
                        if save_img or view_img:  # Add bbox to image
                            # "label" có type = string; value = "person 0.76"
                            # value có dạng <Tên classes> + <confidence>;
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                    if frame > 1:
                        for i in range(len(reversed(det))):
                                xywh_current = (xyxy2xywh(torch.tensor(reversed(det)[i][:4]).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                xywh_current.append(reversed(det)[i][-2].item())
                                xywh_current.append(reversed(det)[i][-1].item())
                                pprint("xywh_current", xywh_current)
                                # xywh_previous = det_previous[i][:4]
                                # pprint("xywh_previous", xywh_previous)
                                print("i = ", i)
                                pprint("det_previous", det_previous[i])
                                plot_one_box_center_point(xywh_current, det_previous[i], im0, color=colors[int(cls)], line_thickness=3)
                                det_previous = xyxy2xywh_for_det_previous(det)
                    else:
                        det_previous = xyxy2xywh_for_det_previous(det)
                        # pprint("det_previous first time", det_previous)
                else:
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        # "save_img" mặc định là True
                        # "view_img" mặc định là False và nếu đặt True thì vẫn ko show dc trên Colab
                        if save_img or view_img:  # Add bbox to image
                            # "label" có type = string; value = "person 0.76"
                            # value có dạng <Tên classes> + <confidence>;
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                '''


            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences (% accuracy predict of class) in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    # print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
