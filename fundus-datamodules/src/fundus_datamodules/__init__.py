from .aptos import AptosClassificationDataModule
from .ddr import DDRClassificationDataModule, DDRSegmentationDataModule
from .maples import MaplesClassificationDataModule, MaplesSegmentationDataModule

__all__ = [
    "AptosClassificationDataModule",
    "DDRClassificationDataModule",
    "DDRSegmentationDataModule",
    "MaplesClassificationDataModule",
    "MaplesSegmentationDataModule",
]
