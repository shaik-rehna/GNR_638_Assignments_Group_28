import os
import time
import cv2
import my_framework as mf


class ImageFolderDataset:

    def __init__(self, root_dir):

        self.root_dir = root_dir
        self.samples = []
        self.class_to_idx = {}
        self.input_channels = None

        start_time = time.time()
        self._load_dataset()
        end_time = time.time()

        print("Dataset loading time:",
              round(end_time - start_time, 4),
              "seconds")

    def _load_dataset(self):

        classes = sorted(os.listdir(self.root_dir))

        for idx, class_name in enumerate(classes):
            self.class_to_idx[class_name] = idx

        for class_name in classes:

            class_path = os.path.join(self.root_dir, class_name)

            if not os.path.isdir(class_path):
                continue

            label = self.class_to_idx[class_name]

            for filename in os.listdir(class_path):

                if not filename.lower().endswith(".png"):
                    continue

                img_path = os.path.join(class_path, filename)

                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

                if img is None:
                    continue

                # --------------------------------------------------
                # 1) True grayscale image (H, W)
                # --------------------------------------------------
                if len(img.shape) == 2:
                    img = img[:, :, None]

                # --------------------------------------------------
                # 2) Remove alpha channel if present
                # --------------------------------------------------
                if img.shape[2] == 4:
                    img = img[:, :, :3]

                # --------------------------------------------------
                # 3) Detect fake RGB grayscale (all channels equal)
                # --------------------------------------------------
                if img.shape[2] == 3:
                    ch0 = img[:, :, 0]
                    ch1 = img[:, :, 1]
                    ch2 = img[:, :, 2]

                    if (ch0 == ch1).all() and (ch1 == ch2).all():
                        img = ch0[:, :, None]

                # --------------------------------------------------
                # 4) Resize to 32x32
                # --------------------------------------------------
                # Resize to 32x32
                img = cv2.resize(img, (32, 32))

                # If grayscale after resize, restore channel dim
                if len(img.shape) == 2:
                    img = img[:, :, None]
                # --------------------------------------------------
                # 5) Normalize
                # --------------------------------------------------
                img = img.astype("float32") / 255.0

                # --------------------------------------------------
                # 6) HWC → CHW
                # --------------------------------------------------
                img = img.transpose(2, 0, 1)

                # --------------------------------------------------
                # 7) Set input_channels once
                # --------------------------------------------------
                if self.input_channels is None:
                    self.input_channels = img.shape[0]

                data = img.flatten().tolist()

                tensor = mf.Tensor(
                    data,
                    [self.input_channels, 32, 32],
                    False
                )

                self.samples.append((tensor, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]