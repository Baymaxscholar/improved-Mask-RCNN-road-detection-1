windows10+CUDA11.1+CUDNN=8.0.4 PYTORCH=1.9.1

training model:
python tools/train.py configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py

verification image：
python tools/test.py ./configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py ./work_dirs/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco/latest.pth --show-dir ./result
python tools/test.py ./configs/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x.py ./work_dirs/dynamic_rcnn_r50_fpn_1x/latest.pth --show-dir ./result

Accuracy：
python tools/analysis_tools/analyze_logs.py plot_curve ./work_dirs/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco/20220901_101324.log.json --keys acc --legend acc --out acc.jpg

flops：
python tools/analysis_tools/get_flops.py ./configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py

fps
python ./tools/analysis_tools/benchmark.py ./configs/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x.py ./work_dirs/dynamic_rcnn_r50_fpn_1x1/latest.pth

Generate .pkl file：
python tools/test.py work_dirs/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py  work_dirs/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco/latest.pth  --out results.pkl

bbox generates results：
python tools/test.py  work_dirs/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py  work_dirs/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco/latest.pth  --format-only  --options "jsonfile_prefix=./results"

bbox result graph：
python tools/analysis_tools/coco_error_analysis.py results.bbox.json results  --ann=D:data/coco/annotations/instances_val2017.json

segm result graph;
python tools/analysis_tools/coco_error_analysis.py results.segm.json results1  --ann=D:data/coco/annotations/instances_val2017.json

classification loss：
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco/20220901_101324.log.json --keys loss_cls --legend loss_cls

Classification loss saved as pdf:
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco/20220901_101324.log.json --keys loss_cls loss_bbox --out losses.pdf
performance statistics.：
python tools/test.py configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py work_dirs/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco/latest.pth --eval bbox segm

log analysis：
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco/20211226_093405.log.json

iou value：
python PR.py

View data augmentation:
python .\tools\misc\browse_dataset.py configs/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x.py
map

python ./tools/analysis_tools/analyze_logs.py plot_curve ./work_dirs/yolox_s_8x8_300e_coco/20220705_155440.log.json --keys mAP  --out out2.jpg --eval-interval 10







