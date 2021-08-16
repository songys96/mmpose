import torch
import onnx
import onnxruntime
import warnings
import cv2
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from test_utils import _get_max_preds

img_metas = [
    {
        "image_file": '',
        "center": np.array([128.5, 128.5]),
        "scale": np.array([1.6, 1.6]),
        "rotation": 0,
        "bbox_score": 1,
        'flip_pairs': [[5, 6], [7, 8], [9, 10], [11, 12]],
        "bbox_id": 0
     }
]




def transform_preds(coords, center, scale, output_size, use_udp=False):
    """Get final keypoint predictions from heatmaps and apply scaling and
    translation to map them back to the image.

    Note:
        num_keypoints: K

    Args:
        coords (np.ndarray[K, ndims]):

            * If ndims=2, corrds are predicted keypoint location.
            * If ndims=4, corrds are composed of (x, y, scores, tags)
            * If ndims=5, corrds are composed of (x, y, scores, tags,
              flipped_tags)

        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        use_udp (bool): Use unbiased data processing

    Returns:
        np.ndarray: Predicted coordinates in the images.
    """
    assert coords.shape[1] in (2, 4, 5)
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    # Recover the scale which is normalized by a factor of 200.
    scale = scale * 200.0

    if use_udp:
        scale_x = scale[0] / (output_size[0] - 1.0)
        scale_y = scale[1] / (output_size[1] - 1.0)
    else:
        scale_x = scale[0] / output_size[0]
        scale_y = scale[1] / output_size[1]

    target_coords = np.ones_like(coords)
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5

    return target_coords



def keypoints_from_heatmaps(heatmaps,
                            center,
                            scale,
                            unbiased=False,
                            post_process='default',
                            kernel=11,
                            valid_radius_factor=0.0546875,
                            use_udp=False,
                            target_type='GaussianHeatmap'):

    # Avoid being affected
    heatmaps = heatmaps.copy()

    # detect conflicts
    if unbiased:
        assert post_process not in [False, None, 'megvii']
    if post_process in ['megvii', 'unbiased']:
        assert kernel > 0
    if use_udp:
        assert not post_process == 'megvii'

    # normalize configs
    if post_process is False:
        warnings.warn(
            'post_process=False is deprecated, '
            'please use post_process=None instead', DeprecationWarning)
        post_process = None
    elif post_process is True:
        if unbiased is True:
            warnings.warn(
                'post_process=True, unbiased=True is deprecated,'
                " please use post_process='unbiased' instead",
                DeprecationWarning)
            post_process = 'unbiased'
        else:
            warnings.warn(
                'post_process=True, unbiased=False is deprecated, '
                "please use post_process='default' instead",
                DeprecationWarning)
            post_process = 'default'
    elif post_process == 'default':
        if unbiased is True:
            warnings.warn(
                'unbiased=True is deprecated, please use '
                "post_process='unbiased' instead", DeprecationWarning)
            post_process = 'unbiased'

    # start processing
    if post_process == 'megvii':
        heatmaps = _gaussian_blur(heatmaps, kernel=kernel)

    N, K, H, W = heatmaps.shape
    if use_udp:
        if target_type.lower() == 'GaussianHeatMap'.lower():
            preds, maxvals = _get_max_preds(heatmaps)
            preds = post_dark_udp(preds, heatmaps, kernel=kernel)
        elif target_type.lower() == 'CombinedTarget'.lower():
            for person_heatmaps in heatmaps:
                for i, heatmap in enumerate(person_heatmaps):
                    kt = 2 * kernel + 1 if i % 3 == 0 else kernel
                    cv2.GaussianBlur(heatmap, (kt, kt), 0, heatmap)
            # valid radius is in direct proportion to the height of heatmap.
            valid_radius = valid_radius_factor * H
            offset_x = heatmaps[:, 1::3, :].flatten() * valid_radius
            offset_y = heatmaps[:, 2::3, :].flatten() * valid_radius
            heatmaps = heatmaps[:, ::3, :]
            preds, maxvals = _get_max_preds(heatmaps)
            index = preds[..., 0] + preds[..., 1] * W
            index += W * H * np.arange(0, N * K / 3)
            index = index.astype(np.int).reshape(N, K // 3, 1)
            preds += np.concatenate((offset_x[index], offset_y[index]), axis=2)
        else:
            raise ValueError('target_type should be either '
                             "'GaussianHeatmap' or 'CombinedTarget'")
    else:
        preds, maxvals = _get_max_preds(heatmaps)
        if post_process == 'unbiased':  # alleviate biased coordinate
            # apply Gaussian distribution modulation.
            heatmaps = np.log(
                np.maximum(_gaussian_blur(heatmaps, kernel), 1e-10))
            for n in range(N):
                for k in range(K):
                    preds[n][k] = _taylor(heatmaps[n][k], preds[n][k])
        elif post_process is not None:
            # add +/-0.25 shift to the predicted locations for higher acc.
            for n in range(N):
                for k in range(K):
                    heatmap = heatmaps[n][k]
                    px = int(preds[n][k][0])
                    py = int(preds[n][k][1])
                    if 1 < px < W - 1 and 1 < py < H - 1:
                        diff = np.array([
                            heatmap[py][px + 1] - heatmap[py][px - 1],
                            heatmap[py + 1][px] - heatmap[py - 1][px]
                        ])
                        preds[n][k] += np.sign(diff) * .25
                        if post_process == 'megvii':
                            preds[n][k] += 0.5

    # Transform back to the image
    for i in range(N):
        preds[i] = transform_preds(
            preds[i], center[i], scale[i], [W, H], use_udp=use_udp)

    if post_process == 'megvii':
        maxvals = maxvals / 255.0 + 0.5

    return preds, maxvals

def decode(img_metas, output, **kwargs):
    """Decode keypoints from heatmaps.

    Args:
        img_metas (list(dict)): Information about data augmentation
            By default this includes:
            - "image_file: path to the image file
            - "center": center of the bbox
            - "scale": scale of the bbox
            - "rotation": rotation of the bbox
            - "bbox_score": score of bbox
        output (np.ndarray[N, K, H, W]): model predicted heatmaps.
    """
    batch_size = len(img_metas)

    if 'bbox_id' in img_metas[0]:
        bbox_ids = []
    else:
        bbox_ids = None

    c = np.zeros((batch_size, 2), dtype=np.float32)
    s = np.zeros((batch_size, 2), dtype=np.float32)
    image_paths = []
    score = np.ones(batch_size)
    for i in range(batch_size):
        c[i, :] = img_metas[i]['center']
        s[i, :] = img_metas[i]['scale']
        image_paths.append(img_metas[i]['image_file'])

        if 'bbox_score' in img_metas[i]:
            score[i] = np.array(img_metas[i]['bbox_score']).reshape(-1)
        if bbox_ids is not None:
            bbox_ids.append(img_metas[i]['bbox_id'])

    preds, maxvals = keypoints_from_heatmaps(
        output,
        c,
        s)

    all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
    all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
    all_preds[:, :, 0:2] = preds[:, :, 0:2]
    all_preds[:, :, 2:3] = maxvals
    all_boxes[:, 0:2] = c[:, 0:2]
    all_boxes[:, 2:4] = s[:, 0:2]
    all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
    all_boxes[:, 5] = score

    result = {}

    result['preds'] = all_preds
    result['boxes'] = all_boxes
    result['image_paths'] = image_paths
    result['bbox_ids'] = bbox_ids

    return result


def alignCoord(coord):
    print(coord)
    x = coord[0,:]
    y = coord[1,:]
    return x,y

def show(img, pred):
    plt.imshow(img)
    plt.scatter([20,30],[30,40])
    plt.show()




def main(video, save="", show=False):
    onnx_file = '/home/butlely/PycharmProjects/mmlab/mmpose/poodle_w32/save_model.onnx'
    model = onnx.load(onnx_file)
    check = onnx.checker.check_model(model)
    graph = onnx.helper.printable_graph(model.graph)
    #
    ort_session = onnxruntime.InferenceSession(onnx_file)

    input_name = ort_session.get_inputs()[0].name

    result_path = save

    if video == "":
        print("[webcam 시작]")
        vs = cv2.VideoCapture(0)
    else:
        print("[video 시작]")
        vs = cv2.VideoCapture(video)

    writer = None

    while True:
        ret, frame = vs.read()
        if frame is None:
            break

        # my code
        img = frame.transpose(2,0,1).astype(np.float32)
        img = img.reshape(1,3,256,256)
        result = ort_session.run(None, {input_name: img})
        heatmap = result[0]
        res = decode(img_metas, heatmap)
        coord = res['preds'][0]
        x, y = alignCoord(coord)
        print(x)
        break
        # 프레임 출력
        if show:
            cv2.imshow("frame", frame)

            # 'q' 키를 입력하면 종료
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        if save:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(result_path, fourcc, 25, (frame.shape[1], frame.shape[0]), True)

            # 비디오 저장
            if writer is not None:
                writer.write(frame)

    # 종료
    vs.release()
    cv2.destroyAllWindows()

vid_path = "/home/butlely/PycharmProjects/mmlab/mmpose/poodle_w32/poodle_test_256.mp4"
main(vid_path)
raise ValueError







