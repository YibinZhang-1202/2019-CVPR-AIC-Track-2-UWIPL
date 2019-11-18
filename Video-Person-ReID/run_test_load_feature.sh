CUDA_VISIBLE_DEVICES=1 \
python main_video_person_reid.py \
    --train-batch 16 \
    --workers 0 \
    --seq-len 4 \
    --arch resnet50ta_surface_nu \
    --width 224 \
    --height 224 \
    --dataset aictrack2 \
    --use-surface \
    --evaluate \
    --pretrained-model log/ta_surface_nu_checkpoint_ep300.pth.tar \
    --save-dir log-test \
    --gpu-devices 1 \
    --re-ranking \
    --metadata-model v2m100 \
    --load-feature \
    --feature-dir log-test/feature_ep0300