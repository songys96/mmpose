import mmcv

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile:
    """Loading image from file.

    Args:
        color_type (str): Flags specifying the color type of a loaded image,
          candidates are 'color', 'grayscale' and 'unchanged'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='rgb'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order

    def __call__(self, results):
        """Loading image from file."""
        image_file = results['image_file']
        img = mmcv.imread(image_file, self.color_type, self.channel_order)

        if img is None:
            raise ValueError(f'Fail to read {image_file}')
        results['img'] = img

        # customizing
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt

        def show(img, keypoints):
            plt.imshow(img)
            xs, ys = alignKeypoint(keypoints)
            plt.scatter(xs, ys)
            plt.show()

        def alignKeypoint(keypoints):
            xs = []
            ys = []
            for keypoint in keypoints:
                xs.append(keypoint[0])
                ys.append(keypoint[1])
            return xs, ys

        def resizeData(results):
            # img (h, w, c)
            """
            ['image_file', 'center', 'scale', 'bbox', 'rotation', 'joints_3d', 'joints_3d_visible', 'dataset', 'bbox_score', 'bbox_id', 'ann_info', 'img'])
            """

            x, y, w, h = results['bbox']
            img = results['img']
            joints_3d = results['joints_3d']

            # transfer

            joints_3d_clipped = np.subtract(joints_3d, [x, y, 0])
            joints_3d_clipped[np.where(joints_3d_clipped<0)] = 0
            joints_3d_ratio = np.divide(joints_3d_clipped, [w, h, 1])

            img_clipped = img[y:y+h, x:x+w]

            if h > w:
                fx = w / h
                w_new = int(256 * fx)
                pad = int((256-w_new)/2)
                img_resize = cv2.resize(img_clipped, dsize=(w_new, 256))
                img_pad = np.pad(img_resize, ((0,0), (pad, 256-w_new-pad), (0,0)), 'constant', constant_values=0)
                joints_3d_resize = np.multiply(joints_3d_ratio, [w_new,256,1])
                joints_3d_resize[:,0] += pad
                joints_3d_resize[:,0][np.where(joints_3d_resize[:,0]==pad)] = 0


            else:
                fy = h / w
                h_new = int(256 * fy)
                pad = int((256-h_new)/2)
                img_resize = cv2.resize(img_clipped, dsize=(256, h_new))
                img_pad = np.pad(img_resize, ((pad, 256-h_new-pad),(0,0), (0,0)), 'constant', constant_values=0)

                joints_3d_resize = np.multiply(joints_3d_ratio, [256,h_new,1])
                joints_3d_resize[:,1] += pad
                joints_3d_resize[:,1][np.where(joints_3d_resize[:,1]==pad)] = 0

            results['img'] = img_pad
            results['joints_3d'] = joints_3d_resize
            results['center'] = np.array([128,128])
            results['bbox'] = [0,0,256,256]

            return results

        results = resizeData(results)

        return results

