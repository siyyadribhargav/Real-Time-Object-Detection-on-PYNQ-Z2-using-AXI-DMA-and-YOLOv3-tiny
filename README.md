# FPGA-Based Real-Time Object Detection on PYNQ-Z2

## Project Overview
Real-time object detection system using PYNQ-Z2 board with PS+PL co-design.
- **PS (ARM CPU):** Camera capture, CLAHE enhancement, YOLOv3-tiny inference
- **PL (FPGA):** AXI DMA data transfer via Vivado block design

## Hardware
- Board: PYNQ-Z2 (Xilinx Zynq-7000 xc7z020clg400-1)
- Camera: USB Camera

## Tools Used
- Vivado 2023.1
- PYNQ v2.7
- Python 3.8
- OpenCV 4.2
- YOLOv3-tiny (Darknet format)

## IP Blocks Used in Vivado
- ZYNQ7 Processing System (HP0 port enabled)
- AXI DMA
- AXI4-Stream Data FIFO (loopback)
- AXI Interconnect x2
- Processor System Reset
- XLConcat

## How to Run
1. Copy `design_2_wrapper.bit` and `design_2.hwh` to PYNQ board
2. Copy `yolov3-tiny.cfg` and `yolov3-tiny.weights` to PYNQ board
3. Open Jupyter Notebook on PYNQ (http://<board-ip>:9090)
4. Run `object_detection.py`
5. Open `latest_detection.jpg` to see results

## Results
- Detected up to 8 objects in a single frame
- Classes: person, laptop, bottle, chair, cup, mouse, keyboard
- Speed: 1-3 FPS on ARM Cortex-A9
- Confidence threshold: 0.15
- NMS threshold: 0.30