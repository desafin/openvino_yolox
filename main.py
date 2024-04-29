import cv2
from openvino.runtime import Core
import numpy as np
# COCO 데이터셋 기준 클래스 이름
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]
def nms(boxes, scores, overlapThresh):
    if len(boxes) == 0:
        return []

    # float형 데이터가 아닌 경우
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return pick

def preprocess_input(frame, height, width):
    # 이미지 전처리: 크기 조정, 차원 변경
    image = cv2.resize(frame, (width, height))
    image = image.transpose((2, 0, 1))  # HWC -> CHW
    image = image.reshape(1, 3, height, width)
    return image


# 모델 출력 후처리 중 NMS 적용 부분
def postprocess_output(frame, results, threshold=0.9, nms_threshold=0.4, max_area=100000000):
    boxes = []
    confidences = []
    class_ids = []

    for detection in results[0]:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > threshold:
            center_x = int(detection[0] * frame.shape[1])
            center_y = int(detection[1] * frame.shape[0])
            width = int(detection[2] * frame.shape[1])
            height = int(detection[3] * frame.shape[0])
            left = int(center_x - width / 2)
            top = int(center_y - height / 2)

            area = width * height  # 객체의 면적 계산

            if area < max_area:  # 면적이 최대 허용 면적보다 작은 경우에만 추가
                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = nms(np.array(boxes), np.array(confidences), nms_threshold)

    for i in idxs:
        left, top, width, height = boxes[i]
        cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
        label = f'{COCO_CLASSES[class_ids[i]]}: {confidences[i]:.2f}'
        cv2.putText(frame, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)



# 모델 파일 경로 설정
model_xml = "FP32/yolox-tiny.xml"
model_bin = "FP32/yolox-tiny.bin"

# OpenVINO 모델 로드
ie = Core()
model = ie.read_model(model=model_xml, weights=model_bin)
compiled_model = ie.compile_model(model=model, device_name="CPU")

# 입력 및 출력 크기 얻기
input_blob = next(iter(compiled_model.inputs))
output_blob = next(iter(compiled_model.outputs))
batch_size, channels, height, width = input_blob.shape

# 웹캠 준비
cap = cv2.VideoCapture(0)
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 입력 데이터 전처리
        input_image = preprocess_input(frame, height, width)

        # 모델 추론
        results = compiled_model.infer_new_request({input_blob.any_name: input_image})

        # 결과 데이터 확인 (디버깅을 위한 로그 출력)
        #print("Results shape:", results[output_blob].shape)
        print("Results data:", results[output_blob])

        # 출력 데이터 후처리
        postprocess_output(frame, results[output_blob])

        # 결과 화면 표시
        cv2.imshow("YOLOX Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()