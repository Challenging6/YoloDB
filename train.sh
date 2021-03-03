python -u train.py  \
    --img 640 \
    --batch 16 \
    --epochs 180 \
    --cfg ./models/db.yaml \
    --data ./data/icdar2015.yaml \
    --adam \
    --workers 8 # >train.log 2>&1 &
    # --weights yolov5s.pt