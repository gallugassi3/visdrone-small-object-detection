# Experiment Log

Every training run gets a row. Conclusions drive the next experiment.

| Run | Changes vs previous | mAP50 | mAP50-95 | Epoch time | Conclusion |
|-----|--------------------|-------|----------|------------|------------|
| baseline_640_smoke | 5 epochs, imgsz=640, batch=8 | TBD | TBD | TBD | Pipeline verified; GPU power-throttles 120W→40W at ~90°C; ~4.4/8GB VRAM used → batch=16 viable |