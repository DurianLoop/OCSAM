r""" Dataloader builder for few-shot semantic segmentation dataset  """
from torchvision import transforms
from torch.utils.data import DataLoader

from .coco import DatasetCOCO
from .pascal import DatasetPASCAL
from .fss import DatasetFSS
from .pascal_part import DatasetPASCALPart
from .dr import DatasetDR
from .MonuSeg import DatasetMonuSeg

try:
    from .paco_part import DatasetPACOPart
except ModuleNotFoundError:
    DatasetPACOPart = None

try:
    from .lvis import DatasetLVIS
except ModuleNotFoundError:
    DatasetLVIS = None

class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize, target_color, interactive_mode=False):

        cls.datasets = {
            'coco': DatasetCOCO,
            'pascal': DatasetPASCAL,
            'fss': DatasetFSS,
            'pascal_part': DatasetPASCALPart,
            'dr': DatasetDR,
            'MonuSeg': DatasetMonuSeg,
        }
        if DatasetPACOPart is not None:
            cls.datasets['paco_part'] = DatasetPACOPart
        if DatasetLVIS is not None:
            cls.datasets['lvis'] = DatasetLVIS

        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize
        # cls.interactive_mode = interactive_mode

        cls.target_color = target_color
        print(f"The Target Color Mask is: {target_color}")

        cls.transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor()
        ])

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        shuffle = split == 'trn'
        nworker = nworker if split == 'trn' else 0
        if benchmark == 'MonuSeg':
            dataset = DatasetMonuSeg(
                datapath=cls.datapath,
                transform=cls.transform,
                use_original_imgsize=cls.use_original_imgsize
            )
        
        elif benchmark == 'dr':
            dataset = cls.datasets[benchmark](
            cls.datapath, 
            transform=cls.transform, 
            use_original_imgsize=cls.use_original_imgsize,
            target_color=cls.target_color,
            # interactive_mode = cls.interactive_mode
        )

        else:
            dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=cls.transform, split=split, shot=shot, use_original_imgsize=cls.use_original_imgsize)

        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker)

        return dataloader
