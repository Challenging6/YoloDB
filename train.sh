python train.py  \
    --img 640 \
    --batch 2 \
    --epochs 5 \
    --cfg ./models/db.yaml \
    --data ./data/icdar2015.yaml \
    --workers 0 \
    # --weights yolov5s.pt