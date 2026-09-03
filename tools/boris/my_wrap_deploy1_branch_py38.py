#!/usr/bin/env python3

import logging
import os
import os.path as osp
import time
from copy import deepcopy
from functools import partial

import mmengine
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process, set_start_method

from mmdeploy.apis import (create_calib_input_data, extract_model,
                           get_predefined_partition_cfg, torch2onnx,
                           visualize_model)
from mmdeploy.apis.core import PIPELINE_MANAGER
from mmdeploy.apis.utils import to_backend
from mmdeploy.backend.sdk.export_info import export2SDK
from mmdeploy.utils import (IR, Backend, get_backend, get_calib_filename,
                            get_ir_config, get_partition_config,
                            get_root_logger, load_config, target_wrapper)

# ── paths ──────────────────────────────────────────────────────────────

NEW_EXPEREMENTAL_PARAMS = True
USE_FP16 = False #30.6 even worse
ORT_OPTIMIZE_ALL = False # 31.4 FPS --> 31.6 FPS same
FPS_calc = True

work_dir = '/home/borisef/temp/out_mmdeploy_try'
img = '/home/borisef/projects/mm/mmdeploy/demo/resources/human-pose.jpg'
deploy_cfg_path = (
    '/home/borisef/projects/mm/mmdeploy/configs/mmpose/'
    'pose-detection_onnxruntime_static.py'
)

if(NEW_EXPEREMENTAL_PARAMS):
    checkpoint_path = (
        #'/home/borisef/projects/mm/mmpose/tools/atraf/borisef/work_dirs/hrnet_UDP_w32_try4/epoch_18.pth'
        '/home/borisef/projects/mm/mmpose/tools/atraf/borisef/work_dirs/hrnet_UDP_w32_try_changes/epoch_12.pth'
    )

    model_cfg_path = (
       # '/home/borisef/projects/mm/mmpose/tools/atraf/borisef/work_dirs/hrnet_UDP_w32_try4/td-hm_hrnet-w32_udp-8xb64-210e_coco-384x288_try1_for_onnx.py'
       # '/home/borisef/projects/mm/mmpose/tools/atraf/borisef/work_dirs/hrnet_UDP_w32_try4/td-hm_hrnet-w32_udp-8xb64-210e_coco-384x288_try1.py'
        '/home/borisef/projects/mm/mmpose/tools/atraf/borisef/work_dirs/hrnet_UDP_w32_try_changes/td-hm_hrnet-w32_udp-8xb64-210e_coco-384x288_try_changes.py'
    )
    # model_cfg_path = (
    #     '/home/borisef/projects/mm/mmdeploy/temp/'
    #     'td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
    # )

else:
    checkpoint_path = (
        '/home/borisef/projects/mm/mmdeploy/temp/'
        'td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth'
    )

    model_cfg_path = (
        '/home/borisef/projects/mm/mmdeploy/temp/'
        'td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
    )



device = 'cuda'
dump_info = True
show = True
log_level = logging.INFO


def create_process(name, target, args, kwargs, ret_value=None):
    logger = get_root_logger()
    logger.info(f'{name} start.')
    log_lvl = logger.level

    wrap_func = partial(target_wrapper, target, log_lvl, ret_value)

    process = Process(target=wrap_func, args=args, kwargs=kwargs)
    process.start()
    process.join()

    if ret_value is not None:
        if ret_value.value != 0:
            logger.error(f'{name} failed.')
            exit(1)
        else:
            logger.info(f'{name} success.')


def _ensure_metainfo(model_cfg):
    """Make sure mmdeploy can extract dataset metainfo from the config.

    mmdeploy's ``create_input`` calls ``_get_dataset_metainfo`` which reads
    ``<dataloader>.dataset.type`` and looks up its ``METAINFO``. When the
    dataset is wrapped in ``ConcatDataset`` / ``RepeatDataset`` /
    ``ClassBalancedDataset`` (as in the experimental training config) no
    METAINFO is found, the function returns ``None`` and ``data.update(None)``
    raises ``TypeError: 'NoneType' object is not iterable``.

    Unwrap such wrappers down to their first concrete child so a real dataset
    (e.g. ``CocoDataset``) is exposed.

    MMPose's dataset defaults can contain a relative ``from_file`` entry (for
    example ``configs/_base_/datasets/coco.py``). MMPose resolves that path
    relative to the current working directory, which fails when this script is
    started from the mmdeploy repository. Copy the dataset's default metainfo
    into the config and make its file path absolute.
    """
    import mmpose.datasets  # noqa: F401
    from mmpose.registry import DATASETS

    def resolve_metainfo_file(path):
        if osp.isabs(path) or osp.exists(path):
            return path

        # Editable/source MMPose installations keep `configs` beside the
        # Python package; this is independent of the caller's current cwd.
        import mmpose
        mmpose_root = osp.dirname(osp.dirname(osp.abspath(mmpose.__file__)))
        candidate = osp.join(mmpose_root, path)
        if osp.exists(candidate):
            return candidate

        # Also support a config located anywhere below the MMPose checkout.
        config_file = getattr(model_cfg, 'filename', None)
        if config_file:
            parent = osp.dirname(osp.abspath(config_file))
            while parent != osp.dirname(parent):
                candidate = osp.join(parent, path)
                if osp.exists(candidate):
                    return candidate
                parent = osp.dirname(parent)
        return path

    wrappers = ('ConcatDataset', 'RepeatDataset', 'ClassBalancedDataset')
    for name in ('test_dataloader', 'val_dataloader', 'train_dataloader'):
        if name not in model_cfg:
            continue
        ds = model_cfg[name].get('dataset', None)
        while isinstance(ds, dict) and ds.get('type') in wrappers:
            if ds.get('datasets'):
                ds = ds['datasets'][0]
            elif 'dataset' in ds:
                ds = ds['dataset']
            else:
                break
        if ds is not None:
            metainfo = ds.get('metainfo')
            if metainfo is None:
                dataset_class = DATASETS.get(ds.get('type'))
                metainfo = deepcopy(
                    getattr(dataset_class, 'METAINFO', {})
                    if dataset_class is not None else {})
            else:
                metainfo = deepcopy(metainfo)

            if isinstance(metainfo, dict) and isinstance(
                    metainfo.get('from_file'), str):
                metainfo['from_file'] = resolve_metainfo_file(
                    metainfo['from_file'])
            if metainfo:
                ds['metainfo'] = metainfo
            model_cfg[name]['dataset'] = ds
    return model_cfg


def _overlay_classifier_predictions(onnx_path, img_path, model_cfg,
                                     deploy_cfg, output_images):
    """Run ONNX inference and draw classifier predictions on saved images."""
    classifiers = model_cfg.get('model', {}).get('head', {}).get(
        'classifiers', [])
    if not classifiers:
        return

    import cv2
    import numpy as np
    import onnxruntime

    from mmdeploy.apis.utils import build_task_processor
    from mmdeploy.utils import get_input_shape

    session = onnxruntime.InferenceSession(onnx_path)
    task_processor = build_task_processor(model_cfg, deploy_cfg, 'cpu')
    input_shape = get_input_shape(deploy_cfg)
    _, input_tensor = task_processor.create_input(img_path, input_shape)
    if isinstance(input_tensor, (list, tuple)):
        input_tensor = torch.stack(input_tensor)
    input_np = input_tensor.cpu().float().numpy()

    input_name = session.get_inputs()[0].name
    if 'float16' in session.get_inputs()[0].type:
        input_np = input_np.astype(np.float16)
    outputs = session.run(None, {input_name: input_np})

    lines = []
    for i, clf in enumerate(classifiers):
        logits = outputs[i + 1][0]
        exp_l = np.exp(logits - logits.max())
        probs = exp_l / exp_l.sum()
        pred_idx = int(np.argmax(probs))
        pred_score = float(probs[pred_idx])
        labels = clf.get('labels', None)
        name = clf['field_name']
        label_str = labels[pred_idx] if labels else str(pred_idx)
        lines.extend([
            f'{name}: {label_str}',
            f'{name} score:',
            f'{pred_score:.8f}',
        ])

    for img_file in output_images:
        if not osp.exists(img_file):
            continue
        vis_img = cv2.imread(img_file)
        y = 30
        for line in lines:
            cv2.putText(vis_img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            y += 30
        cv2.imwrite(img_file, vis_img)


def _overlay_pytorch_classifier_predictions(img_path, model_cfg, deploy_cfg,
                                            checkpoint, output_image):
    """Run native PyTorch inference and annotate one saved image.

    ``HeatmapHeadWithClassifiers.predict`` stores its softmax probabilities in
    ``PoseDataSample.pred_classifiers``. This deliberately uses ``test_step``
    rather than the export-only forward rewriter, so no ONNX code or outputs
    participate in the PyTorch visualization.
    """
    classifiers = model_cfg.get('model', {}).get('head', {}).get(
        'classifiers', [])
    if not classifiers or not osp.exists(output_image):
        return

    import cv2

    from mmdeploy.apis.utils import build_task_processor
    from mmdeploy.utils import get_input_shape

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    task_processor = build_task_processor(model_cfg, deploy_cfg, dev)
    model = task_processor.build_pytorch_model(checkpoint)
    model.eval()

    input_shape = get_input_shape(deploy_cfg)
    data, _ = task_processor.create_input(img_path, input_shape)
    with torch.no_grad():
        result = model.test_step(data)[0]

    predictions = getattr(result, 'pred_classifiers', None)
    if not predictions:
        get_root_logger().warning(
            'PyTorch result has no pred_classifiers; skipping classifier '
            'overlay for %s.', output_image)
        return

    lines = []
    for clf in classifiers:
        name = clf['field_name']
        prediction = predictions.get(name)
        if prediction is None:
            get_root_logger().warning(
                'PyTorch result has no prediction for classifier %s.', name)
            continue
        label = prediction.get('pred_label')
        if label is None:
            label = str(prediction['pred_class'])
        score = prediction['pred_score']
        lines.extend([
            f'{name}: {label}',
            f'{name} score:',
            f'{score:.8f}',
        ])

    vis_img = cv2.imread(output_image)
    y = 30
    for line in lines:
        cv2.putText(vis_img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        y += 30
    cv2.imwrite(output_image, vis_img)


def _run_pytorch_fps_benchmark(img_path, model_cfg, deploy_cfg, checkpoint,
                               num_runs=100):
    """Run PyTorch inference num_runs times and report FPS. Returns (fps, ms_per_frame, device_str)."""
    from mmdeploy.apis.utils import build_task_processor
    from mmdeploy.utils import get_input_shape

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    task_processor = build_task_processor(model_cfg, deploy_cfg, dev)
    model = task_processor.build_pytorch_model(checkpoint)
    model.eval()

    input_shape = get_input_shape(deploy_cfg)
    data, _ = task_processor.create_input(img_path, input_shape)

    with torch.no_grad():
        model.test_step(data)  # warm-up
        if dev == 'cuda':
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(num_runs):
            model.test_step(data)
        if dev == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    fps = num_runs / elapsed
    ms_per_frame = elapsed / num_runs * 1000
    device_str = f'GPU (CUDA)' if dev == 'cuda' else 'CPU'
    logger = get_root_logger()
    logger.info(
        f'[PyTorch FPS] {num_runs} runs on {device_str} in {elapsed:.3f}s → '
        f'{fps:.1f} FPS  ({ms_per_frame:.2f} ms/frame)'
    )
    return fps, ms_per_frame, device_str


def _run_fps_benchmark(onnx_path, img_path, model_cfg, deploy_cfg,
                       num_runs=100):
    """Run ONNX inference num_runs times and report FPS. Returns (fps, ms_per_frame, device_str)."""
    import ctypes
    import numpy as np
    import onnxruntime

    from mmdeploy.apis.utils import build_task_processor
    from mmdeploy.utils import get_input_shape

    # cuDNN lives inside the torch package dir; pre-load it with RTLD_GLOBAL so
    # that ORT's CUDA provider can find it (it's not on the system LD_LIBRARY_PATH).
    _cudnn = osp.join(osp.dirname(torch.__file__), 'lib', 'libcudnn.so.8')
    if osp.exists(_cudnn):
        ctypes.CDLL(_cudnn, mode=ctypes.RTLD_GLOBAL)

    session = onnxruntime.InferenceSession(
        onnx_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    providers = session.get_providers()
    device_str = 'GPU (CUDA)' if 'CUDAExecutionProvider' in providers else 'CPU'

    task_processor = build_task_processor(model_cfg, deploy_cfg, 'cpu')
    input_shape = get_input_shape(deploy_cfg)
    _, input_tensor = task_processor.create_input(img_path, input_shape)
    if isinstance(input_tensor, (list, tuple)):
        input_tensor = torch.stack(input_tensor)
    input_np = input_tensor.cpu().float().numpy()

    input_name = session.get_inputs()[0].name
    if 'float16' in session.get_inputs()[0].type:
        input_np = input_np.astype(np.float16)

    session.run(None, {input_name: input_np})  # warm-up

    t0 = time.perf_counter()
    for _ in range(num_runs):
        session.run(None, {input_name: input_np})
    elapsed = time.perf_counter() - t0

    fps = num_runs / elapsed
    ms_per_frame = elapsed / num_runs * 1000
    logger = get_root_logger()
    logger.info(
        f'[FPS_calc] {num_runs} runs on {device_str} in {elapsed:.3f}s → '
        f'{fps:.1f} FPS  ({ms_per_frame:.2f} ms/frame)'
    )
    return fps, ms_per_frame, device_str


def main():
    set_start_method('spawn', force=True)
    logger = get_root_logger()
    logger.setLevel(log_level)

    pipeline_funcs = [
        torch2onnx, extract_model, create_calib_input_data
    ]
    PIPELINE_MANAGER.enable_multiprocess(True, pipeline_funcs)
    PIPELINE_MANAGER.set_log_level(log_level, pipeline_funcs)

    deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)

    if USE_FP16:
        deploy_cfg['backend_config'] = dict(
            type='onnxruntime',
            precision='fp16',
            common_config=dict(
                min_positive_val=1e-7,
                max_finite_val=1e4,
                keep_io_types=False,
                disable_shape_infer=False,
                op_block_list=None,
                node_block_list=None))

    if ORT_OPTIMIZE_ALL:
        os.environ['MMDEPLOY_ORT_OPTIMIZE_ALL'] = '1'
        from mmdeploy.backend.onnxruntime.atraf.wrapper_extended import ORTWrapperExtended  # noqa: F401, E501

    classifiers = model_cfg.get('model', {}).get('head', {}).get(
        'classifiers', [])
    if classifiers:
        output_names = ['output']
        for clf in classifiers:
            output_names.append(clf['field_name'])
        deploy_cfg['onnx_config']['output_names'] = output_names

    # unwrap ConcatDataset/etc. so dataset metainfo can be extracted; pass this
    # Config object (not the path) to the APIs below
    model_cfg = _ensure_metainfo(model_cfg)

    mmengine.mkdir_or_exist(osp.abspath(work_dir))

    if dump_info:
        export2SDK(
            deploy_cfg,
            model_cfg,
            work_dir,
            pth=checkpoint_path,
            device=device)

    ret_value = mp.Value('d', 0, lock=False)

    # convert to IR
    ir_config = get_ir_config(deploy_cfg)
    ir_save_file = ir_config['save_file']
    ir_type = IR.get(ir_config['type'])
    if ir_type != IR.ONNX:
        raise ValueError(f'Unexpected IR type {ir_type}, expected ONNX')

    torch2onnx(
        img,
        work_dir,
        ir_save_file,
        deploy_cfg,
        model_cfg,
        checkpoint_path,
        device=device)

    ir_files = [osp.join(work_dir, ir_save_file)]

    fps_result = None
    pytorch_fps_result = None
    if FPS_calc:
        pytorch_fps_result = _run_pytorch_fps_benchmark(
            img, model_cfg, deploy_cfg, checkpoint_path)
        fps_result = _run_fps_benchmark(
            osp.join(work_dir, ir_save_file), img, model_cfg, deploy_cfg)

    # partition model (if configured)
    partition_cfgs = get_partition_config(deploy_cfg)
    if partition_cfgs is not None:
        if 'partition_cfg' in partition_cfgs:
            partition_cfgs = partition_cfgs.get('partition_cfg', None)
        else:
            assert 'type' in partition_cfgs
            partition_cfgs = get_predefined_partition_cfg(
                deploy_cfg, partition_cfgs['type'])

        origin_ir_file = ir_files[0]
        ir_files = []
        for partition_cfg in partition_cfgs:
            save_file = partition_cfg['save_file']
            save_path = osp.join(work_dir, save_file)
            start = partition_cfg['start']
            end = partition_cfg['end']
            dynamic_axes = partition_cfg.get('dynamic_axes', None)

            extract_model(
                origin_ir_file,
                start,
                end,
                dynamic_axes=dynamic_axes,
                save_file=save_path)

            ir_files.append(save_path)

    # calibration data (if configured)
    calib_filename = get_calib_filename(deploy_cfg)
    if calib_filename is not None:
        calib_path = osp.join(work_dir, calib_filename)
        create_calib_input_data(
            calib_path,
            deploy_cfg_path,
            model_cfg,
            checkpoint_path,
            dataset_cfg=None,
            dataset_type='val',
            device=device)

    # convert to backend
    backend = get_backend(deploy_cfg)

    PIPELINE_MANAGER.set_log_level(log_level, [to_backend])
    if backend == Backend.TENSORRT:
        PIPELINE_MANAGER.enable_multiprocess(True, [to_backend])

    backend_files = to_backend(
        backend,
        ir_files,
        work_dir=work_dir,
        deploy_cfg=deploy_cfg,
        log_level=log_level,
        device=device)

    # visualize backend model
    test_img = img
    extra = dict(
        backend=backend,
        output_file=osp.join(work_dir, f'output_{backend.value}.jpg'),
        show_result=show)

    create_process(
        f'visualize {backend.value} model',
        target=visualize_model,
        args=(model_cfg, deploy_cfg_path, backend_files, test_img,
              device),
        kwargs=extra,
        ret_value=ret_value)

    # visualize pytorch model
    create_process(
        'visualize pytorch model',
        target=visualize_model,
        args=(model_cfg, deploy_cfg_path, [checkpoint_path],
              test_img, device),
        kwargs=dict(
            backend=Backend.PYTORCH,
            output_file=osp.join(work_dir, 'output_pytorch.jpg'),
            show_result=show),
        ret_value=ret_value)

    # Each visualization is annotated from its own inference backend.
    onnx_path = osp.join(work_dir, ir_save_file)
    onnx_output_image = osp.join(work_dir, f'output_{backend.value}.jpg')
    _overlay_classifier_predictions(
        onnx_path, test_img, model_cfg, deploy_cfg, [onnx_output_image])

    pytorch_output_image = osp.join(work_dir, 'output_pytorch.jpg')
    _overlay_pytorch_classifier_predictions(
        test_img, model_cfg, deploy_cfg, checkpoint_path,
        pytorch_output_image)

    if pytorch_fps_result is not None or fps_result is not None:
        logger.info('─' * 55)
        logger.info('  FPS SUMMARY')
        if pytorch_fps_result is not None:
            fps, ms, dev = pytorch_fps_result
            logger.info(f'  PyTorch : {dev:12s}  {fps:6.1f} FPS  ({ms:.2f} ms/frame)')
        if fps_result is not None:
            fps, ms, dev = fps_result
            logger.info(f'  ONNX    : {dev:12s}  {fps:6.1f} FPS  ({ms:.2f} ms/frame)')
        if pytorch_fps_result is not None and fps_result is not None:
            speedup = fps_result[0] / pytorch_fps_result[0]
            logger.info(f'  Speedup : {speedup:.2f}x  (ONNX vs PyTorch)')
        logger.info('─' * 55)
    logger.info('All process success.')


if __name__ == '__main__':
    main()
