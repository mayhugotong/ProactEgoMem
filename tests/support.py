from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REAL_YAML_AVAILABLE = importlib.util.find_spec("yaml") is not None


def _module_missing(name: str) -> bool:
    if name in sys.modules:
        return False
    return importlib.util.find_spec(name) is None


def install_common_stubs() -> None:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    if _module_missing("cv2"):
        cv2 = types.ModuleType("cv2")
        cv2.__spec__ = importlib.machinery.ModuleSpec("cv2", None)
        cv2.INTER_AREA = 1
        cv2.INTER_LINEAR = 2
        cv2.resize = lambda frame, size, interpolation=None: frame
        sys.modules["cv2"] = cv2

    if _module_missing("av"):
        av = types.ModuleType("av")
        av.__spec__ = importlib.machinery.ModuleSpec("av", None)
        av.time_base = 1_000_000
        av.AVError = Exception
        av.open = lambda path: None
        sys.modules["av"] = av

    if not REAL_YAML_AVAILABLE and "yaml" not in sys.modules:
        yaml = types.ModuleType("yaml")

        class YAMLError(Exception):
            pass

        yaml.YAMLError = YAMLError
        yaml.safe_load = lambda text: json.loads(text)
        yaml.safe_dump = lambda value, **kwargs: json.dumps(value)
        sys.modules["yaml"] = yaml

    if _module_missing("numpy"):
        numpy = types.ModuleType("numpy")

        class ndarray:
            pass

        numpy.ndarray = ndarray
        numpy.float32 = float
        numpy.float64 = float
        numpy.uint8 = int
        sys.modules["numpy"] = numpy

    if _module_missing("PIL"):
        pil = types.ModuleType("PIL")
        image_mod = types.ModuleType("PIL.Image")

        class FakeImage:
            mode = "RGB"

            def convert(self, mode: str):
                return self

            def save(self, *args, **kwargs) -> None:
                return None

        image_mod.Image = FakeImage
        image_mod.open = lambda *args, **kwargs: FakeImage()
        image_mod.fromarray = lambda *args, **kwargs: FakeImage()
        pil.Image = image_mod
        sys.modules["PIL"] = pil
        sys.modules["PIL.Image"] = image_mod

    if _module_missing("torch"):
        torch = types.ModuleType("torch")
        torch.bfloat16 = object()
        torch.float16 = object()
        torch.float32 = object()
        torch.manual_seed = lambda seed: None
        torch.tensor = lambda value, *args, **kwargs: value

        class _Cuda:
            @staticmethod
            def is_available() -> bool:
                return False

            @staticmethod
            def empty_cache() -> None:
                return None

            @staticmethod
            def manual_seed_all(seed: int) -> None:
                return None

        torch.cuda = _Cuda()

        utils = types.ModuleType("torch.utils")
        data = types.ModuleType("torch.utils.data")

        class Dataset:
            pass

        class DataLoader(list):
            pass

        data.Dataset = Dataset
        data.DataLoader = DataLoader
        utils.data = data
        torch.utils = utils
        sys.modules["torch"] = torch
        sys.modules["torch.utils"] = utils
        sys.modules["torch.utils.data"] = data


def install_pipeline_stubs() -> None:
    install_common_stubs()

    if "proact.data.datamodule" not in sys.modules:
        datamodule = types.ModuleType("proact.data.datamodule")
        datamodule.build_dataloaders = lambda cfg: []
        sys.modules["proact.data.datamodule"] = datamodule

    if "proact.utils.workflow_functions" not in sys.modules:
        sys.modules["proact.utils.workflow_functions"] = types.ModuleType(
            "proact.utils.workflow_functions"
        )
