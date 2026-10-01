"""Guard the public pip install contract that failed in mixed notebook environments."""
from pathlib import Path
import tomllib

from packaging.requirements import Requirement


def test_default_install_has_no_training_gpu_or_server_dependencies():
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["project"]
    requirements = {Requirement(value).name for value in project["dependencies"]}
    assert {"onnxruntime", "tokenizers", "numpy"} <= requirements
    assert not requirements.intersection({
        "torch", "torchvision", "transformers", "vllm", "pydantic-ai",
        "pydantic-ai-slim", "fastapi", "granian", "cuda-toolkit",
    })
    extras = project["optional-dependencies"]
    assert "torch" in {Requirement(value).name for value in extras["torch"]}
    assert "pydantic-ai-slim" in {Requirement(value).name for value in extras["pydantic-ai"]}
    assert "granian" in {Requirement(value).name for value in extras["serve"]}
