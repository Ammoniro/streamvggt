import PIL
import numpy as np
import torch
import random
import itertools
import cv2
import torchvision.transforms as T
from dust3r.datasets.base.easy_dataset import EasyDataset
from dust3r.datasets.utils.transforms import ImgNorm, SeqColorJitter
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates
import dust3r.datasets.utils.cropping as cropping
from dust3r.datasets.utils.corr import extract_correspondences_from_pts3d
from .NNfill import fill_in_fast

PATTERN_IDS = {
    'random': 0,
    'velodyne': 1,
    'sfm': 2,
    'downscale': 3,
    'cubic': 4,
    "distance": 5
}

def get_ray_map(c2w1, c2w2, intrinsics, h, w):
    c2w = np.linalg.inv(c2w1) @ c2w2
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    grid = np.stack([i, j, np.ones_like(i)], axis=-1)
    ro = c2w[:3, 3]
    rd = np.linalg.inv(intrinsics) @ grid.reshape(-1, 3).T
    rd = (c2w @ np.vstack([rd, np.ones_like(rd[0])])).T[:, :3].reshape(h, w, 3)
    rd = rd / np.linalg.norm(rd, axis=-1, keepdims=True)
    ro = np.broadcast_to(ro, (h, w, 3))
    ray_map = np.concatenate([ro, rd], axis=-1)
    return ray_map

def add_noise(dep, input_noise):
    # add noise
    # the noise can be "0.1" (fixed probablity) or "0.0~0.1" (uniform in the range)
    if input_noise != "0.0":
        if '~' in input_noise:
            noise_prob_low, noise_prob_high = input_noise.split('~')
            noise_prob_low, noise_prob_high = float(noise_prob_low), float(noise_prob_high)
        else:
            noise_prob_low, noise_prob_high = float(input_noise), float(input_noise)

        noise_prob = np.random.uniform(noise_prob_low, noise_prob_high)
        noise_mask = torch.tensor(np.random.binomial(n=1, p=noise_prob, size=dep.shape))
        depth_min, depth_max = np.percentile(dep, 10), np.percentile(dep, 90)
        noise_values = torch.tensor(np.random.uniform(depth_min, depth_max, size=dep.shape)).float()

        dep[noise_mask == 1] = noise_values[noise_mask == 1]

    return dep


class BaseMultiViewDataset(EasyDataset):
    """Define all basic options.

    Usage:
        class MyDataset (BaseMultiViewDataset):
            def _get_views(self, idx, rng):
                # overload here
                views = []
                views.append(dict(img=, ...))
                return views
    """

    def __init__(
        self,
        *,  # only keyword arguments
        num_views=None,
        split=None,
        resolution=None,  # square_size or (width, height) or list of [(width,height), ...]
        transform=ImgNorm,
        aug_crop=False,
        n_corres=0,
        nneg=0,
        seed=None,
        allow_repeat=False,
        seq_aug_crop=False,
    ):
        assert num_views is not None, "undefined num_views"
        self.num_views = num_views
        self.split = split
        self._set_resolutions(resolution)

        self.n_corres = n_corres
        self.nneg = nneg
        assert (
            self.n_corres == "all"
            or isinstance(self.n_corres, int)
            or (
                isinstance(self.n_corres, list) and len(self.n_corres) == self.num_views
            )
        ), f"Error, n_corres should either be 'all', a single integer or a list of length {self.num_views}"
        assert (
            self.nneg == 0 or self.n_corres != "all"
        ), "nneg should be 0 if n_corres is all"

        self.is_seq_color_jitter = False
        if isinstance(transform, str):
            transform = eval(transform)
        if transform == SeqColorJitter:
            transform = SeqColorJitter()
            self.is_seq_color_jitter = True
        self.transform = transform

        self.aug_crop = aug_crop
        self.seed = seed
        self.allow_repeat = allow_repeat
        self.seq_aug_crop = seq_aug_crop

    def __len__(self):
        return len(self.scenes)

    @staticmethod
    def efficient_random_intervals(
        start,
        num_elements,
        interval_range,
        fixed_interval_prob=0.8,
        weights=None,
        seed=42,
    ):
        if random.random() < fixed_interval_prob:
            intervals = random.choices(interval_range, weights=weights) * (
                num_elements - 1
            )
        else:
            intervals = [
                random.choices(interval_range, weights=weights)[0]
                for _ in range(num_elements - 1)
            ]
        return list(itertools.accumulate([start] + intervals))

    def sample_based_on_timestamps(self, i, timestamps, num_views, interval=1):
        time_diffs = np.abs(timestamps - timestamps[i])
        ids_candidate = np.where(time_diffs < interval)[0]
        ids_candidate = np.sort(ids_candidate)
        if (self.allow_repeat and len(ids_candidate) < num_views // 3) or (
            len(ids_candidate) < num_views
        ):
            return []
        ids_sel_list = []
        ids_candidate_left = ids_candidate.copy()
        while len(ids_candidate_left) >= num_views:
            ids_sel = np.random.choice(ids_candidate_left, num_views, replace=False)
            ids_sel_list.append(sorted(ids_sel))
            ids_candidate_left = np.setdiff1d(ids_candidate_left, ids_sel)

        if len(ids_candidate_left) > 0 and len(ids_candidate) >= num_views:
            ids_sel = np.concatenate(
                [
                    ids_candidate_left,
                    np.random.choice(
                        np.setdiff1d(ids_candidate, ids_candidate_left),
                        num_views - len(ids_candidate_left),
                        replace=False,
                    ),
                ]
            )
            ids_sel_list.append(sorted(ids_sel))

        if self.allow_repeat:
            ids_sel_list.append(
                sorted(np.random.choice(ids_candidate, num_views, replace=True))
            )

        # add sequences with fixed intervals (all possible intervals)
        pos_i = np.where(ids_candidate == i)[0][0]
        curr_interval = 1
        stop = len(ids_candidate) < num_views
        while not stop:
            pos_sel = [pos_i]
            count = 0
            while len(pos_sel) < num_views:
                if count % 2 == 0:
                    curr_pos_i = pos_sel[-1] + curr_interval
                    if curr_pos_i >= len(ids_candidate):
                        stop = True
                        break
                    pos_sel.append(curr_pos_i)
                else:
                    curr_pos_i = pos_sel[0] - curr_interval
                    if curr_pos_i < 0:
                        stop = True
                        break
                    pos_sel.insert(0, curr_pos_i)
                count += 1
            if not stop and len(pos_sel) == num_views:
                ids_sel = sorted([ids_candidate[pos] for pos in pos_sel])
                if ids_sel not in ids_sel_list:
                    ids_sel_list.append(ids_sel)
            curr_interval += 1
        return ids_sel_list

    @staticmethod
    def blockwise_shuffle(x, rng, block_shuffle):
        if block_shuffle is None:
            return rng.permutation(x).tolist()
        else:
            assert block_shuffle > 0
            blocks = [x[i : i + block_shuffle] for i in range(0, len(x), block_shuffle)]
            shuffled_blocks = [rng.permutation(block).tolist() for block in blocks]
            shuffled_list = [item for block in shuffled_blocks for item in block]
            return shuffled_list

    def get_seq_from_start_id(
        self,
        num_views,
        id_ref,
        ids_all,
        rng,
        min_interval=1,
        max_interval=25,
        video_prob=0.5,
        fix_interval_prob=0.5,
        block_shuffle=None,
    ):
        """
        args:
            num_views: number of views to return
            id_ref: the reference id (first id)
            ids_all: all the ids
            rng: random number generator
            max_interval: maximum interval between two views
        returns:
            pos: list of positions of the views in ids_all, i.e., index for ids_all
            is_video: True if the views are consecutive
        """
        assert min_interval > 0, f"min_interval should be > 0, got {min_interval}"
        assert (
            min_interval <= max_interval
        ), f"min_interval should be <= max_interval, got {min_interval} and {max_interval}"
        assert id_ref in ids_all
        pos_ref = ids_all.index(id_ref)
        all_possible_pos = np.arange(pos_ref, len(ids_all))

        remaining_sum = len(ids_all) - 1 - pos_ref

        if remaining_sum >= num_views - 1:
            if remaining_sum == num_views - 1:
                assert ids_all[-num_views] == id_ref
                return [pos_ref + i for i in range(num_views)], True
            max_interval = min(max_interval, 2 * remaining_sum // (num_views - 1))
            intervals = [
                rng.choice(range(min_interval, max_interval + 1))
                for _ in range(num_views - 1)
            ]

            # if video or collection
            if rng.random() < video_prob:
                # if fixed interval or random
                if rng.random() < fix_interval_prob:
                    # regular interval
                    fixed_interval = rng.choice(
                        range(
                            1,
                            min(remaining_sum // (num_views - 1) + 1, max_interval + 1),
                        )
                    )
                    intervals = [fixed_interval for _ in range(num_views - 1)]
                is_video = True
            else:
                is_video = False

            pos = list(itertools.accumulate([pos_ref] + intervals))
            pos = [p for p in pos if p < len(ids_all)]
            pos_candidates = [p for p in all_possible_pos if p not in pos]
            pos = (
                pos
                + rng.choice(
                    pos_candidates, num_views - len(pos), replace=False
                ).tolist()
            )

            pos = (
                sorted(pos)
                if is_video
                else self.blockwise_shuffle(pos, rng, block_shuffle)
            )
        else:
            # assert self.allow_repeat
            uniq_num = remaining_sum
            new_pos_ref = rng.choice(np.arange(pos_ref + 1))
            new_remaining_sum = len(ids_all) - 1 - new_pos_ref
            new_max_interval = min(max_interval, new_remaining_sum // (uniq_num - 1))
            new_intervals = [
                rng.choice(range(1, new_max_interval + 1)) for _ in range(uniq_num - 1)
            ]

            revisit_random = rng.random()
            video_random = rng.random()

            if rng.random() < fix_interval_prob and video_random < video_prob:
                # regular interval
                fixed_interval = rng.choice(range(1, new_max_interval + 1))
                new_intervals = [fixed_interval for _ in range(uniq_num - 1)]
            pos = list(itertools.accumulate([new_pos_ref] + new_intervals))

            is_video = False
            if revisit_random < 0.5 or video_prob == 1.0:  # revisit, video / collection
                is_video = video_random < video_prob
                pos = (
                    self.blockwise_shuffle(pos, rng, block_shuffle)
                    if not is_video
                    else pos
                )
                num_full_repeat = num_views // uniq_num
                pos = (
                    pos * num_full_repeat
                    + pos[: num_views - len(pos) * num_full_repeat]
                )
            elif revisit_random < 0.9:  # random
                pos = rng.choice(pos, num_views, replace=True)
            else:  # ordered
                pos = sorted(rng.choice(pos, num_views, replace=True))
        assert len(pos) == num_views
        return pos, is_video

    def get_img_and_ray_masks(self, is_metric, v, rng, p=[0.8, 0.15, 0.05]):
        # generate img mask and raymap mask
        if v == 0 or (not is_metric):
            img_mask = True
            raymap_mask = False
        else:
            rand_val = rng.random()
            if rand_val < p[0]:
                img_mask = True
                raymap_mask = False
            elif rand_val < p[0] + p[1]:
                img_mask = False
                raymap_mask = True
            else:
                img_mask = True
                raymap_mask = True
        return img_mask, raymap_mask

    def get_stats(self):
        return f"{len(self)} groups of views"

    def __repr__(self):
        resolutions_str = "[" + ";".join(f"{w}x{h}" for w, h in self._resolutions) + "]"
        return (
            f"""{type(self).__name__}({self.get_stats()},
            {self.num_views=},
            {self.split=},
            {self.seed=},
            resolutions={resolutions_str},
            {self.transform=})""".replace(
                "self.", ""
            )
            .replace("\n", "")
            .replace("   ", "")
        )

    def _get_views(self, idx, resolution, rng, num_views):
        raise NotImplementedError()

    def __getitem__(self, idx):
        # print("Receiving:" , idx)
        if isinstance(idx, (tuple, list, np.ndarray)):
            # the idx is specifying the aspect-ratio
            idx, ar_idx, nview = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0
            nview = self.num_views

        assert nview >= 1 and nview <= self.num_views
        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, "_rng"):
            seed = torch.randint(0, 2**32, (1,)).item()
            self._rng = np.random.default_rng(seed=seed)

        if self.aug_crop > 1 and self.seq_aug_crop:
            self.delta_target_resolution = self._rng.integers(0, self.aug_crop)

        # over-loaded code
        resolution = self._resolutions[
            ar_idx
        ]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        views = self._get_views(idx, resolution, self._rng, nview)
        assert len(views) == nview

        if "camera_pose" not in views[0]:
            views[0]["camera_pose"] = np.ones((4, 4), dtype=np.float32)
        first_view_camera_pose = views[0]["camera_pose"]
        transform = SeqColorJitter() if self.is_seq_color_jitter else self.transform

        for v, view in enumerate(views):
            assert (
                "pts3d" not in view
            ), f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
            view["idx"] = (idx, ar_idx, v)

            # encode the image
            width, height = view["img"].size

            view["true_shape"] = np.int32((height, width))
            view["img"] = transform(view["img"])
            view["sky_mask"] = view["depthmap"] < 0

            assert "camera_intrinsics" in view
            if "camera_pose" not in view:
                view["camera_pose"] = np.full((4, 4), np.nan, dtype=np.float32)
            else:
                assert np.isfinite(
                    view["camera_pose"]
                ).all(), f"NaN in camera pose for view {view_name(view)}"

            ray_map = get_ray_map(
                first_view_camera_pose,
                view["camera_pose"],
                view["camera_intrinsics"],
                height,
                width,
            )
            view["ray_map"] = ray_map.astype(np.float32)

            assert "pts3d" not in view
            assert "valid_mask" not in view
            assert np.isfinite(
                view["depthmap"]
            ).all(), f"NaN in depthmap for view {view_name(view)}"
            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)

            view["pts3d"] = pts3d
            view["valid_mask"] = valid_mask & np.isfinite(pts3d).all(axis=-1)

            view['mask_sp'] = view['mask_sp'].numpy().astype(bool) & view['valid_mask']
            view['depth_sp'] = view['depth_sp'].numpy().astype(np.float32) * view['mask_sp']

            # check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"
            K = view["camera_intrinsics"]

        if self.n_corres > 0:
            ref_view = views[0]
            for view in views:
                corres1, corres2, valid = extract_correspondences_from_pts3d(
                    ref_view, view, self.n_corres, self._rng, nneg=self.nneg
                )
                view["corres"] = (corres1, corres2)
                view["valid_corres"] = valid

        # last thing done!
        for view in views:
            view["rng"] = int.from_bytes(self._rng.bytes(4), "big")
        return views

    def _set_resolutions(self, resolutions):
        assert resolutions is not None, "undefined resolution"

        if not isinstance(resolutions, list):
            resolutions = [resolutions]

        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                width, height = resolution
            assert isinstance(
                width, int
            ), f"Bad type for {width=} {type(width)=}, should be int"
            assert isinstance(
                height, int
            ), f"Bad type for {height=} {type(height)=}, should be int"
            self._resolutions.append((width, height))

    def _crop_resize_if_necessary(
        self, image, depthmap, intrinsics, resolution, rng=None, info=None
    ):
        """This function:
        - first downsizes the image with LANCZOS inteprolation,
          which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W - cx)
        min_margin_y = min(cy, H - cy)
        assert min_margin_x > W / 5, f"Bad principal point in view={info}"
        assert min_margin_y > H / 5, f"Bad principal point in view={info}"
        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        image, depthmap, intrinsics = cropping.crop_image_depthmap(
            image, depthmap, intrinsics, crop_bbox
        )

        # transpose the resolution if necessary
        W, H = image.size  # new size

        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        if self.aug_crop > 1:
            target_resolution += (
                rng.integers(0, self.aug_crop)
                if not self.seq_aug_crop
                else self.delta_target_resolution
            )
        image, depthmap, intrinsics = cropping.rescale_image_depthmap(
            image, depthmap, intrinsics, target_resolution
        )

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = cropping.camera_matrix_of_crop(
            intrinsics, image.size, resolution, offset_factor=0.5
        )
        crop_bbox = cropping.bbox_from_intrinsics_in_out(
            intrinsics, intrinsics2, resolution
        )
        image, depthmap, intrinsics2 = cropping.crop_image_depthmap(
            image, depthmap, intrinsics, crop_bbox
        )

        return image, depthmap, intrinsics2

    class ToNumpy:
        def __call__(self, sample):
            return np.array(sample)

    def get_sparse_depth(self, dep, pattern_raw, match_density=True, rgb_np=None, input_noise="0.0"):
        if isinstance(dep, np.ndarray):
            t_dep = T.Compose([
                self.ToNumpy(),
                T.ToTensor()
            ])
            dep = t_dep(dep).to(torch.float32)
        dep = torch.clone(dep)

        channel, height, width = dep.shape

        assert channel == 1

        idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)
        num_idx = len(idx_nnz)

        # the patterns can have format "0.8*100~2000+0.2*sift"
        all_weights = []
        all_patterns = []
        for pattern_item in pattern_raw.split('+'):
            if '*' in pattern_item:
                weight, pattern = pattern_item.split('*')
            else:
                weight, pattern = 1.0, pattern_item

            all_weights.append(weight)
            all_patterns.append(pattern)
        pattern = np.random.choice(all_patterns, p=all_weights)
        # further parse if needed
        if '~' in pattern and "downscale" not in pattern and "cubic" not in pattern and "distance" not in pattern:
            num_start, num_end = pattern.split('~')
            num_start = int(num_start)
            num_end = int(num_end)
            pattern = str(np.random.randint(num_start, num_end))

        if pattern.isdigit():
            num_sample = int(pattern)

            if match_density:
                # we want a uniform density
                num_sample_normalized = max(int(round(num_idx * num_sample / (height * width))), 5)
                idx_sample = torch.randperm(num_idx)[:num_sample_normalized]
            else:
                idx_sample = torch.randperm(num_idx)[:num_sample]

            idx_nnz = idx_nnz[idx_sample[:]]

            mask = torch.zeros((channel * height * width))
            mask[idx_nnz] = 1.0
            mask = mask.view((channel, height, width))

            dep = add_noise(dep, input_noise)
            dep_sp = dep * mask.type_as(dep)
            pattern_id = PATTERN_IDS['random']
        
        elif "distance" in pattern:
            factor = pattern.split("distance")[-1]
            
            if '~' in factor:
                dbegin, dend = factor.split('~')
                r_left, r_right = int(dbegin), int(dend)
            else:
                factor = int(factor)
                r_left, r_right = 1e-3, factor
                
            range_mask = torch.logical_and(
                dep >= r_left, dep <= r_right
            )
            range_mask = torch.logical_and(
                range_mask, dep > 1e-3
            )
            in_range = torch.nonzero(range_mask.view(-1) > 0.5, as_tuple=False)
            num_in_range = len(in_range)
            idx_sample = torch.randperm(num_in_range)[:2000]
            in_range = in_range[idx_sample[:]]
            mask = torch.zeros((channel * height * width))
            mask[in_range] = 1.0
            mask = mask.view((channel, height, width))
            
            dep_sp = dep * mask.type_as(dep)
            pattern_id = PATTERN_IDS['distance']
            
        elif "downscale" in pattern:
            factor = pattern.split("downscale")[-1]
            
            if '~' in factor:
                sbegin, send = factor.split('~')
                s_start, s_end = int(sbegin), int(send)
                factor = np.random.randint(s_start, s_end)
            else:
                factor = int(factor)
                
            dep = add_noise(dep, input_noise)
            # We first obtain the low-resolution picture.
            h, w = dep.shape[-2:]
            lh, lw = h // factor, w // factor
            ldep = F.interpolate(dep.unsqueeze(0), size=(lh, lw), mode='bilinear', align_corners=True)
            
            # Obtain the coodinates of the sparse points.
            sh, sw = h / lh, w / lw
            ih, iw = (sh * torch.arange(lh)).long(), (sw * torch.arange(lw)).long()
            
            # We then obtain the sparse mask of the high-resolution image.
            low_mask = torch.zeros_like(dep, dtype=torch.bool)
            low_mask[..., ih[:, None], iw] = True
            
            # Fill in all low-resolution pixels to the higher-res one.
            dep_sp = torch.zeros_like(dep, dtype=torch.float32)
            dep_sp[low_mask] = ldep.flatten()
            # Filter the sparse points with valid mask. 
            mask = torch.torch.logical_and(low_mask, dep > 0.0001) # low-res valid mask.
            
            dep_sp = dep_sp * mask.type_as(dep_sp)
            pattern_id = PATTERN_IDS['downscale']

        elif pattern == "velodyne":
            # sample a virtual baseline
            #%#
            train_depth_velodyne_random_baseline = True
            if train_depth_velodyne_random_baseline:
                baseline_horizontal = np.random.choice([1.0, -1.0]) * np.random.uniform(0.03, 0.06)
                baseline_vertical = np.random.uniform(-0.02, 0.02)
            else:
                baseline_horizontal = 0.0
                baseline_vertical = 0.0

            # the target view canvas need to be slightly bigger
            target_view_expand_factor = 1.5
            height_expanded = int(target_view_expand_factor * height)
            width_expanded = int(target_view_expand_factor * width)

            # sample a virtual intrinsics
            w_c = np.random.uniform(-0.5*width, 1.5*width)
            h_c = np.random.uniform(0.5*height, 0.7*height)
            # w_c = 0.5 * width
            # h_c = 0.5 * height
            focal = np.random.uniform(1.5*height, 2.0*height)
            Km = np.eye(3)
            Km[0, 0] = focal
            Km[1, 1] = focal
            Km[0, 2] = w_c
            Km[1, 2] = h_c

            Km_target = np.copy(Km)
            Km_target[0, 2] += (target_view_expand_factor - 1.0) / 2.0 * width
            Km_target[1, 2] += (target_view_expand_factor - 1.0) / 2.0 * height

            dep_np = dep.numpy()

            # unproject every depth to a virtual neighboring view
            _, v, u = np.nonzero(dep_np)
            z = dep_np[0, v, u]
            points3D_source = np.linalg.inv(Km) @ (np.vstack([u, v, np.ones_like(u)]) * z) # 3 x N
            points3D_target = np.copy(points3D_source)
            points3D_target[0] -= baseline_horizontal # move in the x direction
            points3D_target[1] -= baseline_vertical # move in the y direction

            points2D_target = Km_target @ points3D_target
            depth_target = points2D_target[2]
            points2D_target = points2D_target[0:2] / (points2D_target[2:3] + 1e-8)  # 2 x N

            # 2 x N_valid
            points2D_target = np.round(points2D_target).astype(int)
            points2D_target_valid = points2D_target[:, ((points2D_target[0] >= 0) & (points2D_target[0] < width_expanded) &
                                                        (points2D_target[1] >= 0) & (points2D_target[1] < height_expanded))]

            # N_valid
            depth_target_valid = depth_target[((points2D_target[0] >= 0) & (points2D_target[0] < width_expanded) &
                                                        (points2D_target[1] >= 0) & (points2D_target[1] < height_expanded))]

            # take the min of all values
            dep_map_target = np.full((height_expanded, width_expanded), np.inf)
            np.minimum.at(dep_map_target, (points2D_target_valid[1], points2D_target_valid[0]), depth_target_valid)
            dep_map_target[dep_map_target == np.inf] = 0.0

            dep_map_target = fill_in_fast(dep_map_target, max_depth=np.max(dep_map_target))
            dep_map_target = dep_map_target[None] # 1 x H x W

            # mask out boundaries
            # dep_map_target_mask = np.zeros_like(dep_map_target)
            # dep_map_target_mask[:, (points2D_target_valid[1].min()):(points2D_target_valid[1].max()+1),
            #         (points2D_target_valid[0].min()):(points2D_target_valid[0].max()+1)] = 1.0
            # dep_map_target = dep_map_target * dep_map_target_mask

            # return torch.tensor(dep_map_target).unsqueeze(0)

            # sample the lidar patterns
            pitch_max = np.random.uniform(0.25, 0.30)
            pitch_min = np.random.uniform(-0.15, -0.20)
            num_lines = np.random.randint(8, 64)
            num_horizontal_points = np.random.randint(400, 1000)

            tgt_pitch = np.linspace(pitch_min, pitch_max, num_lines)
            tgt_yaw = np.linspace(-np.pi/2.1, np.pi/2.1, num_horizontal_points)

            pitch_grid, yaw_grid = np.meshgrid(tgt_pitch, tgt_yaw)
            y, x = np.sin(pitch_grid), np.cos(pitch_grid) * np.sin(yaw_grid) # assume the distace is unit
            z = np.sqrt(1. - x**2 - y**2)
            points_3D = np.stack([x, y, z], axis=0).reshape(3, -1) # 3 x (num_horizontal_points * num_lines)
            points_2D = Km @ points_3D
            points_2D = points_2D[0:2] / (points_2D[2:3] + 1e-8) # 2 x (num_horizontal_points * num_lines)

            points_2D = np.round(points_2D).astype(int)
            points_2D_valid = points_2D[:, ((points_2D[0]>=0) & (points_2D[0]<width_expanded) & (points_2D[1]>=0) & (points_2D[1]<height_expanded))]

            mask = np.zeros([channel, height_expanded, width_expanded])
            mask[:, points_2D_valid[1], points_2D_valid[0]] = 1.0

            dep_map_target_sampled = dep_map_target * mask

            # project it back to source
            _, v, u = np.nonzero(dep_map_target_sampled)
            if len(v) == 0:
                return self.get_sparse_depth(dep, "1000", match_density=match_density, rgb_np=rgb_np, input_noise=input_noise)
            
            z = dep_map_target_sampled[0, v, u]
            points3D_target = np.linalg.inv(Km_target) @ (np.vstack([u, v, np.ones_like(u)]) * z)  # 3 x N
            points3D_source = np.copy(points3D_target)
            points3D_source[0] += baseline_horizontal  # move in the x direction
            points3D_source[1] += baseline_vertical  # move in the y direction

            points2D_source = Km @ points3D_source
            depth_source = points2D_source[2]
            points2D_source = points2D_source[0:2] / (points2D_source[2:3] + 1e-8)  # 2 x N

            # 2 x N_valid
            points2D_source = np.round(points2D_source).astype(int)
            points2D_source_valid = points2D_source[:, ((points2D_source[0] >= 0) & (points2D_source[0] < width) &
                                                        (points2D_source[1] >= 0) & (points2D_source[1] < height))]

            # N_valid
            depth_source_valid = depth_source[((points2D_source[0] >= 0) & (points2D_source[0] < width) &
                                                        (points2D_source[1] >= 0) & (points2D_source[1] < height))]

            # take the min of all values
            dep_map_source = np.full((height, width), np.inf)
            np.minimum.at(dep_map_source, (points2D_source_valid[1], points2D_source_valid[0]), depth_source_valid)
            dep_map_source[dep_map_source == np.inf] = 0.0
            mask = np.zeros([channel, height, width])
            mask[:, points2D_source_valid[1], points2D_source_valid[0]] = 1.0

            # only keep the orginal valid regions
            dep_map_source = dep_map_source * ((dep_np > 0.0).astype(float))

            # only allow deeper value to appear in shallower region
            dep_map_source[dep_map_source < dep_np] = 0.0

            dep_sp = torch.tensor(dep_map_source).float()
            pattern_id = PATTERN_IDS['velodyne']

        elif pattern == "sift" or pattern == "orb":
            assert rgb_np is not None
            gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)

            if pattern == "sift":
                detector = cv2.SIFT.create()
            elif pattern == "orb":
                detector = cv2.ORB.create(nfeatures=100000, scoreType=cv2.ORB_FAST_SCORE)
            else:
                raise NotImplementedError

            keypoints = detector.detect(gray)
            mask = torch.zeros([1, height, width])

            if len(keypoints) < 20:
                return self.get_sparse_depth(dep, "1000", match_density=match_density, rgb_np=rgb_np, input_noise=input_noise)
            
            for keypoint in keypoints:
                x = round(keypoint.pt[1])
                y = round(keypoint.pt[0])
                mask[:, x, y] = 1.0

            #%#
            train_sfm_max_dropout_rate = 0.0
            if train_sfm_max_dropout_rate > 0.0:
                keep_prob = 1.0 - np.random.uniform(0.0, train_sfm_max_dropout_rate)
                mask_keep = keep_prob * torch.ones_like(mask)
                mask_keep = torch.bernoulli(mask_keep)

                mask = mask * mask_keep

            dep = add_noise(dep, input_noise)
            dep_sp = dep * mask.type_as(dep)
            pattern_id = PATTERN_IDS['sfm']

        elif pattern == "LiDAR_64" or pattern == "LiDAR_32" or pattern == "LiDAR_16" or pattern == "LiDAR_8":
            baseline_horizontal = 0.0
            baseline_vertical = 0.0

            w_c = 0.5 * width
            h_c = 0.5 * height
            focal = height

            Km = np.eye(3)
            Km[0, 0] = focal
            Km[1, 1] = focal
            Km[0, 2] = w_c
            Km[1, 2] = h_c

            Km_target = np.copy(Km)

            dep_np = dep.numpy()

            # sample the lidar patterns
            pitch_max = 0.5
            pitch_min = -0.5
            num_lines = int(pattern.split('_')[1])
            num_horizontal_points = 200

            tgt_pitch = np.linspace(pitch_min, pitch_max, num_lines)
            tgt_yaw = np.linspace(-np.pi / 2.1, np.pi / 2.1, num_horizontal_points)

            pitch_grid, yaw_grid = np.meshgrid(tgt_pitch, tgt_yaw)
            y, x = np.sin(pitch_grid), np.cos(pitch_grid) * np.sin(yaw_grid)  # assume the distace is unit
            z = np.sqrt(1. - x ** 2 - y ** 2)
            points_3D = np.stack([x, y, z], axis=0).reshape(3, -1)  # 3 x (num_horizontal_points * num_lines)
            points_2D = Km @ points_3D
            points_2D = points_2D[0:2] / (points_2D[2:3] + 1e-8)  # 2 x (num_horizontal_points * num_lines)

            points_2D = np.round(points_2D).astype(int)
            points_2D_valid = points_2D[:, ((points_2D[0] >= 0) & (points_2D[0] < width) & (
                        points_2D[1] >= 0) & (points_2D[1] < height))]

            mask = np.zeros([channel, height, width])
            mask[:, points_2D_valid[1], points_2D_valid[0]] = 1.0

            dep_map_target_sampled = dep_np * mask

            # only keep the orginal valid regions
            dep_map_target_sampled = dep_map_target_sampled * ((dep_np > 0.0).astype(float))

            dep_sp = torch.tensor(dep_map_target_sampled).float()
            pattern_id = PATTERN_IDS['velodyne']
            
        elif 'cubic' in pattern:
            factor = pattern.split("cubic")[-1]
            
            if '~' in factor:
                sbegin, send = factor.split('~')
                s_start, s_end = int(sbegin), int(send)
                clen = np.random.randint(s_start, s_end)
            else:
                clen = int(factor)
                
            mask = torch.ones_like(dep, dtype=torch.bool)
            H, W = mask.shape[-2:]
            max_h, max_w = H - clen, W - clen
            h = np.random.randint(0, max_h)
            w = np.random.randint(0, max_w)
            mask[:, h : h+clen, w : w+clen] = False
            dep_sp = dep * mask.type_as(dep)
            
            pattern_id = PATTERN_IDS['cubic']
            
        else:
            raise NotImplementedError

        dep_sp = torch.nan_to_num(dep_sp)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        return dep_sp, torch.tensor(pattern_id), mask.to(torch.bool)


def is_good_type(key, v):
    """returns (is_good, err_msg)"""
    if isinstance(v, (str, int, tuple)):
        return True, None
    if v.dtype not in (np.float32, torch.float32, bool, np.int32, np.int64, np.uint8):
        return False, f"bad {v.dtype=}"
    return True, None


def view_name(view, batch_index=None):
    def sel(x):
        return x[batch_index] if batch_index not in (None, slice(None)) else x

    db = sel(view["dataset"])
    label = sel(view["label"])
    instance = sel(view["instance"])
    return f"{db}/{label}/{instance}"


def transpose_to_landscape(view):
    height, width = view["true_shape"]

    if width < height:
        # rectify portrait to landscape
        assert view["img"].shape == (3, height, width)
        view["img"] = view["img"].swapaxes(1, 2)

        assert view["valid_mask"].shape == (height, width)
        view["valid_mask"] = view["valid_mask"].swapaxes(0, 1)

        assert view["depthmap"].shape == (height, width)
        view["depthmap"] = view["depthmap"].swapaxes(0, 1)

        assert view["pts3d"].shape == (height, width, 3)
        view["pts3d"] = view["pts3d"].swapaxes(0, 1)

        # transpose x and y pixels
        view["camera_intrinsics"] = view["camera_intrinsics"][[1, 0, 2]]

        assert view["ray_map"].shape == (height, width, 6)
        view["ray_map"] = view["ray_map"].swapaxes(0, 1)

        assert view["sky_mask"].shape == (height, width)
        view["sky_mask"] = view["sky_mask"].swapaxes(0, 1)

        if "corres" in view:
            # transpose correspondences x and y
            view["corres"][0] = view["corres"][0][:, [1, 0]]
            view["corres"][1] = view["corres"][1][:, [1, 0]]
