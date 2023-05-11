_base_ = [
    '_base_/models/mask_rcnn_r50_fpn.py',
    '_base_/datasets/coco_instance.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]
custom_imports = dict(
    imports=['mmdet.models.backbones.dmf'],
    allow_failed_imports=False)
model = dict(
    backbone=dict(
        type='dmf_499',
        style='pytorch',
        init_cfg=dict(type = "Pretrained", checkpoint='configs/pretrained/checkpoint-441.pth')),
    neck=dict(
        type='FPN',
        in_channels=[40, 72, 176, 240],
        out_channels=256,
        num_outs=5))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)