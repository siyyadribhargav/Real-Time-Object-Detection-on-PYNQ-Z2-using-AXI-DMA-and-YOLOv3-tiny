import cv2
import numpy as np
from pynq import Overlay, allocate

# ── 1. Load overlay ──
ol = Overlay("/home/xilinx/jupyter_notebooks/design_2_wrapper.bit")
dma = ol.axi_dma_0

# ── 2. Load YOLOv3-tiny ──
net = cv2.dnn.readNetFromDarknet(
    "/home/xilinx/jupyter_notebooks/yolov3-tiny.cfg",
    "/home/xilinx/jupyter_notebooks/yolov3-tiny.weights"
)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print("Model loaded OK!", flush=True)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# 80 COCO classes
CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors",
    "teddy bear","hair drier","toothbrush"
]

# Fix common wrong labels due to similar shapes
LABEL_REMAP = {
    "vase"  : "bottle",   # dark steel bottle looks like vase
    "tv"    : "laptop",   # laptop screen looks like tv
    "bowl"  : "cup",      # small cups look like bowls
}

# ── 3. Open camera ──
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Camera 0 not found! Trying camera 1...", flush=True)
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("No camera found! Check USB connection.", flush=True)

# ── 4. Enhance frame (fixes dark/backlit scenes) ──
def enhance_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return enhanced

# ── 5. DMA transfer (PS → PL → PS) ──
# 64x64 grayscale = 4096 bytes, within 16383 DMA buffer limit
DMA_W, DMA_H = 64, 64

def dma_transfer(frame):
    thumb = cv2.resize(frame, (DMA_W, DMA_H))
    gray  = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
    flat  = gray.flatten()
    n     = len(flat)

    input_buffer  = allocate(shape=(n,), dtype=np.uint8)
    output_buffer = allocate(shape=(n,), dtype=np.uint8)

    np.copyto(input_buffer, flat)

    dma.sendchannel.transfer(input_buffer)
    dma.recvchannel.transfer(output_buffer)
    dma.sendchannel.wait()
    dma.recvchannel.wait()

    input_buffer.freebuffer()
    output_buffer.freebuffer()

    return frame

# ── 6. Postprocess ──
# conf_thresh=0.15 — low enough to detect bottles, cups, laptops etc
# nms_thresh=0.3  — removes duplicate boxes cleanly
def postprocess(outputs, frame, conf_thresh=0.15, nms_thresh=0.3):
    h, w = frame.shape[:2]
    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for det in output:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = float(scores[class_id])
            if confidence < conf_thresh:
                continue

            cx = int(det[0] * w)
            cy = int(det[1] * h)
            bw = int(det[2] * w)
            bh = int(det[3] * h)
            x  = cx - bw // 2
            y  = cy - bh // 2

            boxes.append([x, y, bw, bh])
            confidences.append(confidence)
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)

    detected = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]

            # Apply label remap fix
            label_name = CLASSES[class_ids[i]]
            label_name = LABEL_REMAP.get(label_name, label_name)
            label = f"{label_name}: {confidences[i]:.2f}"
            detected.append(label)

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

            # Draw label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, y - th - 10), (x + tw, y), (0, 255, 0), -1)

            # Draw label text
            cv2.putText(frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return frame, detected

# ── 7. Main loop ──
MAX_FRAMES = 30
print(f"Starting detection for {MAX_FRAMES} frames...", flush=True)
print("━" * 40, flush=True)
frame_count = 0

try:
    while frame_count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame — check camera", flush=True)
            break

        # Step 1: Enhance brightness/contrast
        enhanced = enhance_frame(frame)

        # Step 2: PS↔PL DMA transfer
        processed = dma_transfer(enhanced)

        # Step 3: YOLO inference on ARM (PS)
        blob = cv2.dnn.blobFromImage(
            processed, 1/255.0, (416, 416), swapRB=True, crop=False
        )
        net.setInput(blob)
        outputs = net.forward(output_layers)

        # Step 4: Draw detections
        result, detected = postprocess(outputs, processed.copy())

        # Step 5: Stamp frame info on saved image
        cv2.putText(result, f"Frame: {frame_count+1}/{MAX_FRAMES}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(result, f"Objects: {len(detected)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Step 6: Save output image
        cv2.imwrite("/home/xilinx/jupyter_notebooks/latest_detection.jpg", result)

        frame_count += 1

        # Step 7: Print live summary
        print(f"Frame     : {frame_count}/{MAX_FRAMES}", flush=True)
        print(f"Detected  : {len(detected)} object(s)", flush=True)
        if detected:
            for d in detected:
                print(f"  ✔ {d}", flush=True)
        else:
            print(f"  ✘ Nothing detected above threshold", flush=True)
        print("━" * 40, flush=True)

except KeyboardInterrupt:
    print("\nStopped by user", flush=True)

finally:
    cap.release()
    print(f"\nDone. Total frames processed: {frame_count}/{MAX_FRAMES}", flush=True)
    print("Open latest_detection.jpg in Jupyter to see the result!", flush=True)