_base_ = [
    '_base_/models/retinanet_r50_fpn.py',
    '_base_/datasets/coco_detection.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['mmdet.models.backbones.dmf'],
    allow_failed_imports=False)
model = dict(
    backbone=dict(
        type='dmf_198',
        style='pytorch',
        init_cfg = dict(type = "Pretrained", checkpoint='configs/pretrained/checkpoint-467.pth')),
    neck=dict(
        type='FPN',
        in_channels=[20, 36, 88, 120],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)