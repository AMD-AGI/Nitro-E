# Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Dict, List, Optional, Sequence, Union
from torch.utils.data import DataLoader
from torchvision import transforms
from streaming import Stream, StreamingDataset
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

#Reference https://github.com/SonyResearch/micro_diffusion/blob/main/micro_diffusion/datasets/prepare/sa1b/base.py
class StreamingTotalDatasetForPreCompute(StreamingDataset):
    """Streaming dataset that resizes images to user-provided resolutions and tokenizes captions."""

    def __init__(
        self,
        streams: Sequence[Stream],
        transforms_list: List[Callable],
        batch_size: int,
        shuffle: bool = False,
    ):
        super().__init__(
            streams=streams,
            shuffle=shuffle,
            batch_size=batch_size,
        )

        self.transforms_list = transforms_list
        assert self.transforms_list is not None, 'Must provide transforms to resize and center crop images'

    def __getitem__(self, index: int) -> Dict:
        sample = super().__getitem__(index)
        ret = {}

        for i, tr in enumerate(self.transforms_list):
            img = sample['jpg']
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = tr(img)
            ret[f'image_re{i}'] = img

        ret['sample'] = sample
        return ret


def build_streaming_total_precompute_dataloader(
    datadir: Union[List[str], str],
    batch_size: int,
    resize_sizes: Optional[List[int]] = None,
    drop_last: bool = False,
    shuffle: bool = True,
    **dataloader_kwargs,
) -> DataLoader:
    """Builds a streaming mds dataloader returning multiple image sizes and text captions."""
    assert resize_sizes is not None, 'Must provide target resolution for image resizing'
    datadir = [datadir] if isinstance(datadir, str) else datadir
    streams = [Stream(remote=None, local=l) for l in datadir]

    transforms_list = []
    
    transforms_list.append(
        transforms.Compose([
            transforms.Resize(
                resize_sizes[0],interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(resize_sizes[0]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]) )

    transforms_list.append(
        transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]) )


    dataset = StreamingTotalDatasetForPreCompute(
        streams=streams,
        shuffle=shuffle,
        transforms_list=transforms_list,
        batch_size=batch_size,
    )

    def custom_collate(list_of_dict: List[Dict]) -> Dict:
        out = {k: [] for k in list_of_dict[0].keys()}
        for d in list_of_dict:
            for k, v in d.items():
                out[k].append(v)
        return out

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        collate_fn=custom_collate,
        **dataloader_kwargs,
    )

    return dataloader
