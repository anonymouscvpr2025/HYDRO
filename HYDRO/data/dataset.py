from torchvision.transforms.functional import center_crop, gaussian_blur, resize
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import torch.nn.functional as F
import json
import os
import numpy as np
import random
import tqdm
from skimage import io
from PIL import Image


class SwapDataset(Dataset):
    def __init__(self,
                 root_dir='aligned_shrink_75_v2',
                 seg_root='aligned_shrink_75_seg',
                 same_ratio=0.2, transform=None, batch_size=10, iterations=50000, seed=9001, train_size=1.0):
        self.root_dir = root_dir
        self.seg_root = seg_root
        self.same_ratio = same_ratio
        self.transform = transform
        self.identities = os.listdir(root_dir)
        random.Random(seed).shuffle(self.identities)

        self.train_identities = self.identities[:int(train_size * len(self.identities))]
        self.val_identities = self.identities[int(train_size * len(self.identities)):]

        self.same_ticker = 0
        self.batch_size = batch_size
        self.iterations = iterations

    def __len__(self):
        return self.iterations

    def __getitem__(self, _):
        target_identity = np.random.randint(len(self.train_identities))
        if np.random.uniform() > self.same_ratio:
            source_identity = np.random.randint(len(self.train_identities))
        else:
            source_identity = target_identity
        if self.same_ticker == (self.batch_size - 1):
            source_identity = target_identity

        if source_identity == target_identity:
            same_flag = 1
            self.same_ticker = 0
        else:
            same_flag = 0

        try:
            targets = os.listdir(self.root_dir + self.train_identities[target_identity])
            sources = os.listdir(self.root_dir + self.train_identities[source_identity])

            target_idx = np.random.randint(len(targets))
            source_idx = np.random.randint(len(sources))

            img_target_path = self.root_dir + self.train_identities[target_identity] + '/' + targets[target_idx]
            img_source_path = self.root_dir + self.train_identities[source_identity] + '/' + sources[source_idx]
            if same_flag == 1:
                img_source_path = img_target_path

            # Image data
            target = io.imread(img_target_path)[:, :, :3]
            source = io.imread(img_source_path)[:, :, :3]
            mask = -1
            if self.seg_root is not None:
                img_mask_path = self.seg_root + 'face/' + self.train_identities[target_identity] + '/' + targets[
                    target_idx]
                img_hair_path = self.seg_root + 'hair/' + self.train_identities[target_identity] + '/' + targets[
                    target_idx]
                mask = io.imread(img_mask_path).astype('float32')
                hair = io.imread(img_hair_path).astype('float32')

            if self.transform is not None:
                if self.seg_root is not None:
                    t = self.transform(image=target, mask=mask, hair_mask=hair)
                    mask = t['mask'].unsqueeze(0) / 255.0
                    target = t['image']
                    hair = t['hair_mask'].unsqueeze(0) / 255.0

                    mask = center_crop(mask, 240)
                    mask = resize(mask, 256)
                    mask = gaussian_blur(mask, 15, sigma=2.3)
                    mask = mask * (1 - hair)

                    t = self.transform(image=source)
                    source = t['image']
                else:
                    t = self.transform(image=target)
                    target = t['image']

                    t = self.transform(image=source)
                    source = t['image']

            self.prev_target = target
            self.prev_source = source
            self.prev_mask = mask
            self.prev_flag = same_flag

            self.same_ticker += 1

            return target, source, mask, same_flag

        except Exception as e:
            self.same_ticker += 1
            return self.prev_target, self.prev_source, self.prev_mask, self.prev_flag

    def get_val_item(self, _):
        target_identity = np.random.randint(len(self.val_identities))
        if np.random.uniform() > self.same_ratio:
            source_identity = np.random.randint(len(self.val_identities))
        else:
            source_identity = target_identity
        if self.same_ticker == (self.batch_size - 1):
            source_identity = target_identity

        if source_identity == target_identity:
            same_flag = 1
            self.same_ticker = 0
        else:
            same_flag = 0

        try:
            targets = os.listdir(self.root_dir + self.val_identities[target_identity])
            sources = os.listdir(self.root_dir + self.val_identities[source_identity])

            target_idx = np.random.randint(len(targets))
            source_idx = np.random.randint(len(sources))

            img_target_path = self.root_dir + self.val_identities[target_identity] + '/' + targets[target_idx]
            img_source_path = self.root_dir + self.val_identities[source_identity] + '/' + sources[source_idx]

            if same_flag == 1:
                img_source_path = img_target_path

            # Image data
            target = io.imread(img_target_path)[:, :, :3]
            source = io.imread(img_source_path)[:, :, :3]
            mask = -1

            if self.seg_root is not None:
                img_mask_path = self.seg_root + 'face/' + self.val_identities[target_identity] + '/' + targets[
                    target_idx]
                img_hair_path = self.seg_root + 'hair/' + self.val_identities[target_identity] + '/' + targets[
                    target_idx]
                mask = io.imread(img_mask_path).astype('float32')
                hair = io.imread(img_hair_path).astype('float32')

            if self.transform is not None:
                if self.seg_root is not None:
                    t = self.transform(image=target, mask=mask, hair_mask=hair)
                    mask = t['mask'].unsqueeze(0) / 255.0
                    target = t['image']
                    hair = t['hair_mask'].unsqueeze(0) / 255.0

                    mask = center_crop(mask, 240)
                    mask = resize(mask, 256)
                    mask = gaussian_blur(mask, 15, sigma=2.3)
                    mask = mask * (1 - hair)

                    t = self.transform(image=source)
                    source = t['image']
                else:
                    t = self.transform(image=target)
                    target = t['image']

                    t = self.transform(image=source)
                    source = t['image']

            self.prev_target_val = target
            self.prev_source_val = source
            self.prev_mask_val = mask
            self.prev_flag_val = same_flag

            self.same_ticker += 1

            return target, source, mask, same_flag

        except Exception as e:
            self.same_ticker += 1
            return self.prev_target_val, self.prev_source_val, self.prev_mask_val, self.prev_flag_val


class FaceDataset(Dataset):
    def __init__(self,
                 root_dir='aligned_shrink_75_v2',
                 transform=None, iterations=500000, seed=9002, train_size=1.0):
        self.root_dir = root_dir
        self.transform = transform
        self.identities = os.listdir(root_dir)
        random.Random(seed).shuffle(self.identities)

        self.train_identities = self.identities[:int(train_size * len(self.identities))]
        self.val_identities = self.identities[int(train_size * len(self.identities)):]
        self.iterations = iterations

    def __len__(self):
        return self.iterations

    def __getitem__(self, _):
        target_identity = np.random.randint(len(self.train_identities))
        source_identity = np.random.randint(len(self.train_identities))

        try:
            targets = os.listdir(self.root_dir + self.train_identities[target_identity])
            sources = os.listdir(self.root_dir + self.train_identities[source_identity])

            target_idx = np.random.randint(len(targets))
            source_idx = np.random.randint(len(sources))

            img_target_path = self.root_dir + self.train_identities[target_identity] + '/' + targets[target_idx]
            img_source_path = self.root_dir + self.train_identities[source_identity] + '/' + sources[source_idx]

            # Image data
            target = Image.open(img_target_path)
            source = Image.open(img_source_path)

            if self.transform is not None:
                target = self.transform(target)
                source = self.transform(source)

            self.prev_target = target
            self.prev_source = source

            return target, source

        except Exception as e:
            return self.__getitem__(-1)


class SwapDatasetNoSame(Dataset):
    def __init__(self,
                 root_dir='aligned_shrink_75_v2',
                 seg_root='aligned_shrink_75_seg',
                 same_ratio=0.2, transform=None, batch_size=10, iterations=50000):
        self.root_dir = root_dir
        self.seg_root = seg_root
        self.same_ratio = same_ratio
        self.transform = transform
        self.identities = os.listdir(root_dir)

        self.prev_source = None
        self.prev_target = None
        self.prev_mask = None

        self.same_ticker = 0
        self.batch_size = batch_size
        self.iterations = iterations

    def __len__(self):
        return self.iterations

    def __getitem__(self, _):
        target_identity = np.random.randint(len(self.identities))
        source_identity = np.random.randint(len(self.identities))

        try:
            targets = os.listdir(self.root_dir + self.identities[target_identity])
            sources = os.listdir(self.root_dir + self.identities[source_identity])

            target_idx = np.random.randint(len(targets))
            source_idx = np.random.randint(len(sources))

            img_target_path = self.root_dir + self.identities[target_identity] + '/' + targets[target_idx]
            img_source_path = self.root_dir + self.identities[source_identity] + '/' + sources[source_idx]
            img_mask_path = self.seg_root + 'face/' + self.identities[target_identity] + '/' + targets[target_idx]
            img_hair_path = self.seg_root + 'hair/' + self.identities[target_identity] + '/' + targets[target_idx]

            # Image data
            target = io.imread(img_target_path)[:, :, :3]
            source = io.imread(img_source_path)[:, :, :3]
            mask = io.imread(img_mask_path).astype('float32')
            hair = io.imread(img_hair_path).astype('float32')

            if self.transform is not None:
                t = self.transform(image=target, mask=mask, hair_mask=hair)
                mask = t['mask'].unsqueeze(0) / 255.0
                target = t['image']
                hair = t['hair_mask'].unsqueeze(0) / 255.0

                mask = center_crop(mask, 240)
                mask = resize(mask, 256)
                mask = gaussian_blur(mask, 15, sigma=2.3)
                mask = mask * (1 - hair)

                t = self.transform(image=source)
                source = t['image']

            self.prev_target = target
            self.prev_source = source
            self.prev_mask = mask

            self.same_ticker += 1

            return target, source, target, mask

        except Exception as e:
            self.same_ticker += 1
            return self.prev_target, self.prev_source, self.prev_target, self.prev_mask


class SwapDatasetDifferentTargetSource(Dataset):
    def __init__(self,
                 root_dir='aligned_shrink_75_v2',
                 seg_root='aligned_shrink_75_seg',
                 same_ratio=0.2, transform=None, batch_size=10, iterations=50000):
        self.root_dir = root_dir
        self.seg_root = seg_root
        self.same_ratio = same_ratio
        self.transform = transform
        self.identities = os.listdir(root_dir)

        self.prev_source = None
        self.prev_target = None
        self.prev_target_source = None
        self.prev_mask = None

        self.same_ticker = 0
        self.batch_size = batch_size
        self.iterations = iterations

    def __len__(self):
        return self.iterations

    def __getitem__(self, _):
        target_identity = np.random.randint(len(self.identities))
        source_identity = np.random.randint(len(self.identities))

        try:
            targets = os.listdir(self.root_dir + self.identities[target_identity])
            sources = os.listdir(self.root_dir + self.identities[source_identity])

            target_idx = np.random.randint(len(targets))
            target_source_idx = np.random.randint(len(targets))
            source_idx = np.random.randint(len(sources))

            img_target_path = self.root_dir + self.identities[target_identity] + '/' + targets[target_idx]
            img_target_source_path = self.root_dir + self.identities[target_identity] + '/' + targets[target_source_idx]
            img_source_path = self.root_dir + self.identities[source_identity] + '/' + sources[source_idx]
            img_mask_path = self.seg_root + 'face/' + self.identities[target_identity] + '/' + targets[target_idx]
            img_hair_path = self.seg_root + 'hair/' + self.identities[target_identity] + '/' + targets[target_idx]

            # Image data
            target = io.imread(img_target_path)[:, :, :3]
            target_source = io.imread(img_target_source_path)[:, :, :3]
            source = io.imread(img_source_path)[:, :, :3]
            mask = io.imread(img_mask_path).astype('float32')
            hair = io.imread(img_hair_path).astype('float32')

            if self.transform is not None:
                t = self.transform(image=target, mask=mask, hair_mask=hair)
                mask = t['mask'].unsqueeze(0) / 255.0
                target = t['image']
                hair = t['hair_mask'].unsqueeze(0) / 255.0

                mask = center_crop(mask, 240)
                mask = resize(mask, 256)
                mask = gaussian_blur(mask, 15, sigma=2.3)
                mask = mask * (1 - hair)

                t = self.transform(image=source)
                source = t['image']

                t = self.transform(image=target_source)
                target_source = t['image']

            self.prev_target = target
            self.prev_target_source = target_source
            self.prev_source = source
            self.prev_mask = mask

            self.same_ticker += 1

            return target, source, target_source, mask

        except Exception as e:
            self.same_ticker += 1
            return self.prev_target, self.prev_source, self.prev_target_source, self.prev_mask


class VRADataset(Dataset):
    def __init__(self,
                 root_dir='D:/DFGC-VRA/',
                 transform=None,
                 batch_size=10,
                 iterations=50000,
                 n_frames=5,
                 n_class_bins=10,
                 mix_up_chance=0.5,
                 split_size=0.2,
                 seed=42,
                 train_mode=True):
        self.root_dir = root_dir
        self.transform = transform
        self.n_class_bins = n_class_bins

        import pandas as pd

        self.file_paths = pd.read_csv(root_dir + 'label/train_set.csv', delimiter=',')

        self.video_files = self.file_paths['file'].values
        self.labels = self.file_paths['mos'].values
        self.labels = (self.labels - np.min(self.labels)) / (np.max(self.labels) - np.min(self.labels)) * 2 - 1
        self.labels_class = ((self.labels + 1) * (n_class_bins - 1) / 2).astype('int32')
        self.weights = len(self.labels) / (n_class_bins * np.bincount(self.labels_class))

        # pre-shuffle for validation split
        assert len(self.video_files) == len(self.labels)

        # Reproducability / comparison
        np.random.seed(seed)

        p = np.random.permutation(len(self.file_paths))

        self.video_files = self.video_files[p]
        self.labels = self.labels[p]
        self.labels_class = self.labels_class[p]

        data_length = len(self.video_files)
        train_length = int(data_length * (1 - split_size))

        self.train_files = self.video_files[:train_length]
        self.train_labels = self.labels[:train_length]
        self.train_labels_class = self.labels_class[:train_length]

        self.val_files = self.video_files[train_length:]
        self.val_labels = self.labels[train_length:]
        self.val_labels_class = self.labels_class[train_length:]

        self.n_frames = n_frames
        self.batch_size = batch_size
        self.iterations = iterations
        self.mix_up_chance = mix_up_chance

        self.train_mode = train_mode

    def __len__(self):
        if self.train_mode:
            return len(self.train_files)
        else:
            return len(self.val_files)

    def _val_load_data(self, idx):
        file_name = self.val_files[idx]
        label = np.asarray(self.val_labels[idx], dtype='float32')
        cls = self.val_labels_class[idx]
        loss_weight = np.asarray(self.weights[cls], dtype='float32')

        cls_onehot = np.zeros(shape=(self.n_class_bins,), dtype='float32')
        cls_onehot[cls] = 1

        frames = os.listdir(self.root_dir + f"preprocessed/{file_name}")

        start_idx = np.random.randint(len(frames) - self.n_frames + 1)

        images = []

        for i in range(self.n_frames):
            image = io.imread(self.root_dir + f"preprocessed/{file_name}/{frames[start_idx + i]}")[:, :, :3]

            if self.transform is not None:
                image = self.transform(image)

            images.append(image.unsqueeze(0))

        images_torch = torch.concat(images, dim=0)
        label_torch = torch.from_numpy(label)
        cls_torch = torch.from_numpy(cls_onehot)
        weight_torch = torch.from_numpy(loss_weight)

        return images_torch, label_torch, cls_torch, weight_torch

    def _load_data(self, idx):
        file_name = self.train_files[idx]
        label = np.asarray(self.train_labels[idx], dtype='float32')
        cls = self.train_labels_class[idx]
        loss_weight = np.asarray(self.weights[cls], dtype='float32')

        cls_onehot = np.zeros(shape=(self.n_class_bins,), dtype='float32')
        cls_onehot[cls] = 1

        frames = os.listdir(self.root_dir + f"preprocessed/{file_name}")

        start_idx = np.random.randint(len(frames) - self.n_frames + 1)

        images = []

        for i in range(self.n_frames):
            image = io.imread(self.root_dir + f"preprocessed/{file_name}/{frames[start_idx + i]}")[:, :, :3]

            if self.transform is not None:
                image = self.transform(image)

            images.append(image.unsqueeze(0))

        images_torch = torch.concat(images, dim=0)
        label_torch = torch.from_numpy(label)
        cls_torch = torch.from_numpy(cls_onehot)
        weight_torch = torch.from_numpy(loss_weight)

        weight_torch = weight_torch / 10 + 1.0

        return images_torch, label_torch, cls_torch, weight_torch

    def __getitem__(self, idx):
        if self.train_mode:
            images_torch, label_torch, cls_torch, weight_torch = self._load_data(idx)

            if np.random.uniform() < self.mix_up_chance:
                idx_mu = np.random.randint(self.__len__())

                images_torch_mu, label_torch_mu, cls_torch_mu, weight_torch_mu = self._load_data(idx_mu)

                alpha = np.random.beta(0.2, 0.2)

                images_torch = images_torch * alpha + (1 - alpha) * images_torch_mu
                label_torch = label_torch * alpha + (1 - alpha) * label_torch_mu
                cls_torch = cls_torch * alpha + (1 - alpha) * cls_torch_mu
                weight_torch = weight_torch * alpha + (1 - alpha) * weight_torch_mu

            return images_torch, label_torch, cls_torch, weight_torch
        else:
            return self._val_load_data(idx)


class FIVADataset(Dataset):
    def __init__(self,
                 transform=None,
                 iterations=50000,
                 seed=9001,
                 refine_paths="heuristic_error_path.npy",
                 refine_error="heuristic_error_list.npy"):
        # load heuristic errors of the images along with their paths
        error_list = np.load(refine_error)
        paths_list = np.load(refine_paths)

        # stats
        err_sigma = np.std(error_list)
        err_mu = np.mean(error_list)

        # find end index of images with a specific error threshold
        partitions = [np.sum(np.where(error_list > (err_mu - err_sigma) + i * err_sigma * 0.5, 1, 0)) for i in range(7)]

        # add start index
        partitions += [0]
        partitions.reverse()

        # divide the data paths into 7 partitions depending on its error
        paths_dict = {}
        top_k = np.argsort(-1 * error_list)
        for p in range(len(partitions) - 1):
            paths_dict[p] = paths_list[top_k[partitions[p]:partitions[p + 1]]]

        self.top_k_paths = paths_dict

        # sampliing probabilities, images with high error is more likely to get sampled
        self.prob = np.asarray([9, 9, 7, 6, 5, 5, 5])
        self.prob = self.prob / np.sum(self.prob)

        print(self.prob)

        self.transform = transform

        self.iterations = iterations

    def __len__(self):
        return self.iterations

    def __getitem__(self, _):
        partition_idx = np.random.choice(7, 1, p=self.prob)[0]
        paths = self.top_k_paths[partition_idx]

        try:
            img_target_path = self.top_k_paths[partition_idx][np.random.randint(len(self.top_k_paths[partition_idx]))]

            # Image data
            target = io.imread(img_target_path)[:, :, :3]

            if self.transform is not None:
                t = self.transform(image=target)
                target = t['image']

            self.prev_target = target

            return target

        except Exception as e:
            return self.prev_target


class GeometryFaceDataset(Dataset):
    def __init__(self,
                 root_dir='D:/vggface2/v2/aligned_train/',
                 seg_root='D:/vggface2/v2/aligned_train_seg/',
                 cam_root='D:/vggface2/v2/3d_data_train/',
                 transform=None,
                 seed=9001,
                 iterations=500000):
        self.root_dir = root_dir
        self.seg_root = seg_root
        self.cam_root = cam_root
        self.transform = transform
        self.train_images = os.listdir(root_dir)
        self.train_masks = os.listdir(seg_root)
        self.train_cams = os.listdir(cam_root)
        self.iterations = iterations

    def __len__(self):
        return self.iterations

    def __getitem__(self, _):
        target_identity = np.random.randint(len(self.train_images))

        try:
            # Image data
            face = io.imread(self.root_dir + self.train_images[target_identity])[:, :, :3]
            if self.transform is not None:
                if self.seg_root is not None:
                    mask = io.imread(self.seg_root + self.train_masks[target_identity]).astype('float32')

                    t = self.transform(image=face, mask=mask)
                    mask = t['mask'].unsqueeze(0) / 255.0
                    face = t['image']

                    if self.cam_root is not None:
                        with open(self.cam_root + self.train_cams[target_identity], 'r') as j_f:
                            im_3d_info = json.load(j_f)

                            intrinsics = torch.tensor(im_3d_info['intrinsics'])
                            cam2world = torch.tensor(im_3d_info['pose'])

                            cond = torch.concat((cam2world, intrinsics), dim=0)

                else:
                    t = self.transform(image=face)
                    face = t['image']

            self.prev_face = face
            self.prev_mask = mask
            self.prev_cond = cond

            return face, mask, cond

        except Exception as e:
            raise e
            return self.prev_face, self.prev_mask, self.prev_cond


class GeometryFaceSwapDataset(Dataset):
    def __init__(self,
                 root_dir='D:/vggface2/v2/aligned_train/',
                 seg_root='D:/vggface2/v2/aligned_train_seg/',
                 cam_root='D:/vggface2/v2/3d_data_train/',
                 transform=None,
                 seed=9001,
                 iterations=500000):
        self.root_dir = root_dir
        self.seg_root = seg_root
        self.cam_root = cam_root
        self.transform = transform
        self.train_images = os.listdir(root_dir)
        self.train_masks = os.listdir(seg_root)
        self.train_cams = os.listdir(cam_root)
        self.iterations = iterations

    def __len__(self):
        return self.iterations

    def __getitem__(self, _):
        target_identity = np.random.randint(len(self.train_images))
        source_identity = np.random.randint(len(self.train_images))

        try:
            # Image data
            target_face = io.imread(self.root_dir + self.train_images[target_identity])[:, :, :3]
            source_face = io.imread(self.root_dir + self.train_images[source_identity])[:, :, :3]
            if self.transform is not None:
                if self.seg_root is not None:
                    target_mask = io.imread(self.seg_root + self.train_masks[target_identity]).astype('float32')
                    source_mask = io.imread(self.seg_root + self.train_masks[source_identity]).astype('float32')

                    t = self.transform(image=target_face, mask=target_mask)
                    target_mask = t['mask'].unsqueeze(0) / 255.0
                    target_face = t['image']

                    t = self.transform(image=source_face, mask=source_mask)
                    source_mask = t['mask'].unsqueeze(0) / 255.0
                    source_face = t['image']

                    if self.cam_root is not None:
                        with open(self.cam_root + self.train_cams[target_identity], 'r') as j_f:
                            im_3d_info = json.load(j_f)

                            intrinsics = torch.tensor(im_3d_info['intrinsics'])
                            cam2world = torch.tensor(im_3d_info['pose'])

                            target_camera_parameters = torch.concat((cam2world, intrinsics), dim=0)

                        with open(self.cam_root + self.train_cams[source_identity], 'r') as j_f:
                            im_3d_info = json.load(j_f)

                            intrinsics = torch.tensor(im_3d_info['intrinsics'])
                            cam2world = torch.tensor(im_3d_info['pose'])

                            source_camera_parameters = torch.concat((cam2world, intrinsics), dim=0)

                else:
                    t = self.transform(image=target_face)
                    target_face = t['image']
                    target_mask = None
                    target_camera_parameters = None

                    t = self.transform(image=source_face)
                    source_face = t['image']
                    source_mask = None
                    source_camera_parameters = None

            self.prev_target_face = target_face
            self.prev_target_mask = target_mask
            self.prev_target_camera_parameters = target_camera_parameters

            self.prev_source_face = source_face
            self.prev_source_mask = source_mask
            self.prev_source_camera_parameters = source_camera_parameters

            return target_face, source_face, target_mask, source_mask, \
                   target_camera_parameters, source_camera_parameters

        except Exception as e:
            return self.prev_target_face, self.prev_source_face, self.prev_target_mask, self.prev_source_mask, \
                   self.prev_target_camera_parameters, self.prev_source_camera_parameters


class FaceswapDataset(Dataset):
    def __init__(self,
                 root_dir='aligned_shrink_75_v2',
                 seg_root='aligned_shrink_75_seg',
                 transform=None,
                 iterations=500000,
                 seed=9001):
        self.root_dir = root_dir
        self.seg_root = seg_root
        self.transform = transform
        self.train_images = os.listdir(root_dir)
        self.train_masks = os.listdir(seg_root)
        self.iterations = iterations

    def __len__(self):
        return self.iterations

    def __getitem__(self, _):
        target_identity = np.random.randint(len(self.train_images))
        source_identity = np.random.randint(len(self.train_images))

        try:
            # Image data
            target = io.imread(self.root_dir + self.train_images[target_identity])[:, :, :3]
            source = io.imread(self.root_dir + self.train_images[source_identity])[:, :, :3]
            if self.transform is not None:
                if self.seg_root is not None:
                    target_mask = io.imread(self.seg_root + self.train_masks[target_identity]).astype('float32')
                    source_mask = io.imread(self.seg_root + self.train_masks[target_identity]).astype('float32')

                    t = self.transform(image=target, mask=target_mask)
                    target_mask = t['mask'].unsqueeze(0) / 255.0
                    target = t['image']

                    target_mask = center_crop(target_mask, 240)
                    target_mask = resize(target_mask, 256)
                    target_mask = gaussian_blur(target_mask, 15, sigma=2.3)

                    t = self.transform(image=source, mask=source_mask)
                    source_mask = t['mask'].unsqueeze(0) / 255.0
                    source = t['image']

                    source_mask = center_crop(source_mask, 240)
                    source_mask = resize(source_mask, 256)
                    source_mask = gaussian_blur(source_mask, 15, sigma=2.3)

                else:
                    t = self.transform(image=target)
                    target = t['image']
                    target_mask = None

                    t = self.transform(image=source)
                    source = t['image']
                    source_mask = None

            self.prev_target = target
            self.prev_target_mask = target_mask
            self.prev_source = source
            self.prev_source_mask = source_mask

            return target, target_mask, source, source_mask

        except Exception as e:
            return self.prev_target, self.prev_target_mask, self.prev_source, self.prev_source_mask


class AnonymizationDataset(Dataset):
    def __init__(self,
                 root_dir='aligned_shrink_75_v2',
                 transform=None,
                 iterations=500000):
        self.root_dir = root_dir
        self.transform = transform
        self.identities = os.listdir(root_dir)
        random.shuffle(self.identities)
        self.iterations = iterations

    def __len__(self):
        return self.iterations

    def __getitem__(self, _):
        target_identity = np.random.randint(len(self.identities))

        try:
            targets = os.listdir(self.root_dir + self.identities[target_identity])

            target_idx = np.random.randint(len(targets))

            img_target_path = self.root_dir + self.identities[target_identity] + '/' + targets[target_idx]

            # Image data
            target = Image.open(img_target_path)
            #target = io.imread(img_target_path)[:, :, :3]

            if self.transform is not None:
                target = self.transform(target)

            self.prev_target = target

            return target

        except Exception as e:
            raise e
            return self.prev_target

    def shuffle(self):
        random.shuffle(self.identities)


class DeIDDataset(Dataset):
    def __init__(self,
                 root_dir='aligned_shrink_75_v2',
                 transform=None,
                 iterations=5000000,
                 ):
        self.root_dir = root_dir
        self.transform = transform
        self.identities = os.listdir(root_dir)
        self.iterations = iterations

    def __len__(self):
        return self.iterations

    def __getitem__(self, _):
        target_identity = np.random.randint(len(self.identities))

        target_faces = os.listdir(os.path.join(self.root_dir, self.identities[target_identity]))

        target_face = np.random.randint(len(target_faces))

        try:
            # Image data
            target = np.array(Image.open(os.path.join(self.root_dir,
                                                      self.identities[target_identity],
                                                      target_faces[target_face])).convert('RGB'))
            if self.transform is not None:
                t = self.transform(image=target)
                target = t['image']

            return target

        except Exception as e:
            return self.__getitem__(np.random.randint(self.__len__()))


class HVAEDataset(Dataset):
    def __init__(self,
                 root_dir='E:/DATASETS/vggface2/train_aligned/aligned/',
                 transform=None,
                 iterations=5000000,
                 ):
        self.root_dir = root_dir
        self.transform = transform
        self.identities = os.listdir(root_dir)
        self.iterations = iterations

    def __len__(self):
        return self.iterations

    def __getitem__(self, _):
        target_identity = np.random.randint(len(self.identities))

        target_faces = os.listdir(os.path.join(self.root_dir, self.identities[target_identity]))

        target_face = np.random.randint(len(target_faces))

        try:
            # Image data
            target = Image.open(os.path.join(self.root_dir, self.identities[target_identity],
                                             target_faces[target_face])).convert('RGB')
            if self.transform is not None:
                target = self.transform(target)

            return target

        except Exception as e:
            raise e
            return self.__getitem__(np.random.randint(self.__len__()))


class MultiDeIDDataset(Dataset):
    def __init__(self,
                 root_dir='aligned_shrink_75_v2',
                 transform=None,
                 iterations=5000000,
                 ):
        if not isinstance(root_dir, list):
            self.root_dir = [root_dir]
        else:
            self.root_dir = root_dir
        self.identities = {}
        for dataset in self.root_dir:
            self.identities[dataset] = os.listdir(dataset)
        self.transform = {}
        if transform is None:
            for dataset in self.root_dir:
                self.transform[dataset] = None
        self.transform = transform
        self.iterations = iterations

    def __len__(self):
        return self.iterations

    def __getitem__(self, _):

        dataset_idx = np.random.randint(len(self.root_dir))
        dataset = self.root_dir[dataset_idx]
        dataset_identities = self.identities[dataset]

        target_identity = np.random.randint(len(dataset_identities))

        target_faces = os.listdir(os.path.join(dataset, dataset_identities[target_identity]))

        target_face = np.random.randint(len(target_faces))

        try:
            # Image data
            target = np.array(Image.open(os.path.join(dataset,
                                                      dataset_identities[target_identity],
                                                      target_faces[target_face])).convert('RGB'))
            if self.transform[dataset] is not None:
                t = self.transform[dataset](image=target)
                target = t['image']

            return target

        except Exception as e:
            return self.__getitem__(np.random.randint(self.__len__()))


class FSDataset(Dataset):
    def __init__(self,
                 root_dir='aligned_shrink_75_v2',
                 transform=None,
                 iterations=500000):
        self.root_dir = root_dir
        self.transform = transform
        self.identities = os.listdir(root_dir)
        random.shuffle(self.identities)
        self.iterations = iterations

    def __len__(self):
        return self.iterations

    def __getitem__(self, _):
        target_identity = np.random.randint(len(self.identities))
        source_identity = np.random.randint(len(self.identities))

        try:
            targets = os.listdir(self.root_dir + self.identities[target_identity])
            sources = os.listdir(self.root_dir + self.identities[source_identity])

            target_idx = np.random.randint(len(targets))
            source_idx = np.random.randint(len(sources))

            img_target_path = self.root_dir + self.identities[target_identity] + '/' + targets[target_idx]
            img_source_path = self.root_dir + self.identities[source_identity] + '/' + sources[source_idx]

            # Image data
            target = Image.open(img_target_path)
            source = Image.open(img_source_path)

            if self.transform is not None:
                target = self.transform(target)
                source = self.transform(source)

            return target, source

        except Exception as e:
            return self.__getitem__(-1)

    def shuffle(self):
        random.shuffle(self.identities)


class IdentityRetrievalDataset(Dataset):
    def __init__(self,
                 root_dir='',
                 meta_dir='',
                 transform=None):
        self.root_dir = root_dir.replace('/', '\\')
        self.meta_dir = meta_dir.replace('/', '\\')
        self.transform = transform
        self.identities = os.listdir(root_dir)

        self.sample_paths = []

        for root, dirs, files in os.walk(self.root_dir):
            for name in files:
                if name[-4:] == '.png' or name[-4:] == '.jpg': self.sample_paths.append(os.path.join(root, name))

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        current_file_path = self.sample_paths[idx]
        file_path_splits = current_file_path.split('\\')
        current_id = file_path_splits[-2]
        current_file = file_path_splits[-1]

        # Get the face image
        target = Image.open(current_file_path)

        # Get the meta data for the true identity
        true_identity = np.load(os.path.join(self.meta_dir, current_id, current_file[:-4] + '.npy'))

        if self.transform is not None:
            target = self.transform(target)

        return target, true_identity

    def shuffle(self):
        random.shuffle(self.identities)


class GenerateEvalDataset(Dataset):
    def __init__(self,
                 root_dir='',
                 transform=None,
                 target_root_dir=None,
                 face_swap_mode=False):
        self.root_dir = root_dir.replace('/', '\\')
        self.transform = transform
        self.identities = os.listdir(root_dir)

        self.face_swap_mode = face_swap_mode

        self.sample_paths = []

        for root, dirs, files in tqdm.tqdm(os.walk(self.root_dir)):
            for name in files:
                if name[-4:] == '.png' or name[-4:] == '.jpg': self.sample_paths.append(os.path.join(root, name))

        self.target_root_dir = None
        if target_root_dir is not None:
            self.target_root_dir = target_root_dir.replace('/', '\\')

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        current_file_path = self.sample_paths[idx]
        file_path_splits = current_file_path.split('\\')
        current_id = file_path_splits[-2]
        current_file = file_path_splits[-1]

        # Get the face image
        target = Image.open(current_file_path)

        if self.transform is not None:
            target = self.transform(target)

        if self.target_root_dir is None:
            if not self.face_swap_mode:
                return target, current_file, current_id
            else:
                try:
                    source_id = self.identities[np.random.randint(len(self.identities))]

                    while source_id == current_id:
                        source_id = self.identities[np.random.randint(len(self.identities))]

                    source_files = os.listdir(os.path.join(self.root_dir, source_id))
                    source_file = source_files[np.random.randint(len(source_files))]

                    source = Image.open(os.path.join(self.root_dir, source_id, source_file))

                    if self.transform is not None:
                        source = self.transform(source)
                except Exception as e:
                    print(e)
                    print(source_id)
                    raise e

                return target, current_file, current_id, source, source_file, source_id
        else:
            unmanipulated_target = Image.open(current_file_path.replace(self.root_dir, self.target_root_dir))

            if self.transform is not None:
                unmanipulated_target = self.transform(unmanipulated_target)

            return target, unmanipulated_target, current_file, current_id

    def shuffle(self):
        random.shuffle(self.identities)


class CompareDataset(Dataset):
    def __init__(self,
                 root_dir='',
                 transform=None,):
        self.root_dir = root_dir
        self.transform = transform
        self.methods = os.listdir(root_dir)

        self.sample_paths = []
        self.sample_method = []

        for method in self.methods:
            images = os.listdir(os.path.join(root_dir, method, "target/"))
            for im in images:
                self.sample_paths.append(os.path.join(root_dir, method, "target/", im))
                self.sample_method.append(method)

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        current_file_path = self.sample_paths[idx]
        current_method = self.sample_method[idx]

        # Get the face image
        target = Image.open(current_file_path)

        if self.transform is not None:
            target = self.transform(target)

        return target, current_method

    def shuffle(self):
        random.shuffle(self.identities)


class EvalDataset(Dataset):
    def __init__(self,
                 target_root_dir='',
                 source_root_dir='',
                 transform=None,):
        self.target_root_dir = target_root_dir.replace('/', '\\')
        self.source_root_dir = source_root_dir.replace('/', '\\')
        self.transform = transform
        self.identities = os.listdir(target_root_dir)

        self.sample_paths = []

        for root, dirs, files in tqdm.tqdm(os.walk(self.target_root_dir)):
            for name in files:
                if name[-4:] == '.png' or name[-4:] == '.jpg': self.sample_paths.append(os.path.join(root, name))

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        current_file_path = self.sample_paths[idx]
        file_path_splits = current_file_path.split('\\')
        current_id = file_path_splits[-2]
        current_file = file_path_splits[-1]

        # Get the face image
        target = Image.open(current_file_path)
        source = Image.open(current_file_path.replace(self.target_root_dir, self.source_root_dir))

        if self.transform is not None:
            target = self.transform(target)
            source = self.transform(source)

        return target, source, current_file, current_id

    def shuffle(self):
        random.shuffle(self.identities)


class LFWProtocolDatasetGenuine(Dataset):
    def __init__(self,
                 root_dir_target='',
                 root_dir_genuine='',
                 transform=None,
                 protocol_file="pairsDevTest_genuine.txt",
                 imposter_mode=False):
        self.root_dir_target = root_dir_target.replace('/', '\\')
        self.root_dir_genuine = root_dir_genuine.replace('/', '\\')
        self.transform = transform
        self.identities = os.listdir(root_dir_target)
        self.imposter_mode = imposter_mode

        lfw_imposter_file = open(protocol_file, 'r')
        pairs = lfw_imposter_file.readlines()

        self.target_paths = []
        self.genuine_paths = []

        print("Mapping all file paths..")

        for pair in pairs:
            pair_split = pair.split('\t')
            if len(pair_split) == 3:
                try:
                    genuine_path = os.path.join(self.root_dir_genuine, pair_split[0])
                    genuine_index = int(pair_split[2]) - 1
                    genuine_image_list = os.listdir(genuine_path)

                    self.genuine_paths.append(os.path.join(genuine_path, genuine_image_list[genuine_index]))

                    target_path = os.path.join(self.root_dir_target, pair_split[0])
                    target_index = int(pair_split[1]) - 1
                    target_image_list = os.listdir(target_path)

                    self.target_paths.append(os.path.join(target_path, target_image_list[target_index]))

                except Exception as e:
                    print(e)
                    print(target_path)

    def __len__(self):
        return len(self.target_paths)

    def __getitem__(self, idx):
        target_file_path = self.target_paths[idx]
        genuine_file_path = self.genuine_paths[idx]

        target = Image.open(target_file_path)
        genuine = Image.open(genuine_file_path)

        if self.transform is not None:
            target = self.transform(target)
            genuine = self.transform(genuine)

        return genuine, target

    def shuffle(self):
        random.shuffle(self.identities)



