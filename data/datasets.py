import os
import random
from torch.utils.data import Dataset
from data.utils import _crop_and_pad, _normalise_intensity, _to_tensor, _load2d, _create_edge_map

import numpy as np

class _BaseDataset(Dataset):
    """Base dataset class"""
    def __init__(self, data_dir_path):
        super(_BaseDataset, self).__init__()
        self.data_dir = data_dir_path
        assert os.path.exists(data_dir_path), f"Data dir does not exist: {data_dir_path}"
        self.subject_list = sorted(os.listdir(self.data_dir))

        self.data_path_dict = dict()

    def _set_path(self, index):
        """ Set the paths of data files to load and the keys in data_dict"""
        raise NotImplementedError

    def __getitem__(self, index):
        """ Load data and pre-process """
        raise NotImplementedError

    def __len__(self):
        return len(self.subject_list)


class BrainMRInterSubj3D(_BaseDataset):
    def __init__(self,
                 data_dir_path,
                 crop_size,
                 evaluate=False,
                 modality='t1t1',
                 atlas_path=None,
                 patch_w=64,
                 patch_h=64,
                 transform=True):
        super(BrainMRInterSubj3D, self).__init__(data_dir_path)
        self.crop_size = crop_size
        self.atlas_path = atlas_path
        self.modality = modality
        self.evaluate = evaluate
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.transform = transform

    def _set_path(self, index):
        # choose the target and source subjects/paths
        if self.atlas_path is None:
            self.tar_subj_id = self.subject_list[index]
            self.tar_subj_path = f'{self.data_dir}/{self.tar_subj_id}'
        else:
            self.tar_subj_path = self.atlas_path

        # TODO change to intra patient just to make the problem a bit easier
        self.src_subj_id = random.choice(self.subject_list)
        # self.src_subj_id = self.tar_subj_id
        self.src_subj_path = f'{self.data_dir}/{self.src_subj_id}'

        self.data_path_dict['target'] = f'{self.tar_subj_path}/T1_brain.nii.gz'
        self.data_path_dict['target_edges'] = f'{self.tar_subj_path}/T1_brain.nii.gz'

        # modality
        if self.modality == 't1t1':
            self.data_path_dict['source'] = f'{self.src_subj_path}/T1_brain.nii.gz'
            self.data_path_dict['source_edges'] = f'{self.src_subj_path}/T1_brain.nii.gz'
        elif self.modality == 't1t2':
            self.data_path_dict['source'] = f'{self.src_subj_path}/T2_brain.nii.gz'
            self.data_path_dict['source_edges'] = f'{self.src_subj_path}/T2_brain.nii.gz'
        else:
            raise ValueError(f'Modality ({self.modality}) not recognised.')

        if self.modality == 't1t1':
            self.data_path_dict['source_mask'] = f'{self.tar_subj_path}/T1_brain_mask.nii.gz'
            self.data_path_dict['target_mask'] = f'{self.src_subj_path}/T1_brain_mask.nii.gz'
        elif self.modality == 't1t2':
            self.data_path_dict['source_mask'] = f'{self.src_subj_path}/T1_brain_mask.nii.gz'
            self.data_path_dict['target_mask'] = f'{self.tar_subj_path}/T2_brain_mask.nii.gz'

        # eval data
        if self.evaluate:
            # T1w image of source subject for visualisation
            self.data_path_dict['target_original'] = f'{self.src_subj_path}/T1_brain.nii.gz'
            self.data_path_dict['target_original_edges'] = f'{self.src_subj_path}/T1_brain.nii.gz'

        # segmentation
        self.data_path_dict['target_seg'] = f'{self.tar_subj_path}/T1_brain_MALPEM_tissues.nii.gz'
        self.data_path_dict['source_seg'] = f'{self.src_subj_path}/T1_brain_MALPEM_tissues.nii.gz'

    def __getitem__(self, index):
        self._set_path(index)
        data_dict = _load2d(self.data_path_dict)
        data_dict = dict((key, np.array(data)) for key, data in data_dict.items())

        data_dict = _crop_and_pad(data_dict, self.crop_size)
        data_dict = _normalise_intensity(data_dict, keys={'target', 'source', 'target_original', 'target_edges', 'source_edges', 'target_original_edges'})


        data_dict = _create_edge_map(data_dict, N=6)
        data_dict = _normalise_intensity(data_dict, keys={'target_edges', 'source_edges', 'target_original_edges'})

        return _to_tensor(data_dict)
