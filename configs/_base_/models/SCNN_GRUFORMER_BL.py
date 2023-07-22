# model settings


model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='SCNN_GRUFORMER_BL',
        pretrained=None,
        batchNorm=True, 
        output_layers=None,
        init_std=0.005,
        num_heads=8,
        clip_n =3,

        dim = 4096, 
        clip_len=16,
        tau=100., 
        threshold=0.75,
        init_values=1e-5,
        ),
    
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'),

    cls_head=dict(
        type='I3DHead',
        # num_classes=1000,
        # num_classes=300,
        num_classes=114,
        # num_classes=101,
        # in_channels=4096,
        in_channels=4096,
        spatial_type=None,
        dropout_ratio=0.5,
        init_std=0.01),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='score'))
