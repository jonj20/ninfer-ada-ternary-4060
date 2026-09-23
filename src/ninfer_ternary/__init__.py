"""ninfer-ternary：ninfer-4090 的三元移植工具链。

仓库形态下它把补丁落到 ninfer 检出、编译引擎、打包三元制品；安装形态下它只留下两个引擎
可执行文件与一个 GGUF 转换器（见 assets 模块对两种形态的说明）。
"""

from .engine import BuildRequest, BuildResult, EngineError, build_engine
from .manifest import FileEntry, ManifestError, PatchManifest
from .patchset import FileState, PatchError, PatchReport, apply_patch_set, inspect

__version__ = "0.3.0"

__all__ = [
    "BuildRequest",
    "BuildResult",
    "EngineError",
    "FileEntry",
    "FileState",
    "ManifestError",
    "PatchError",
    "PatchManifest",
    "PatchReport",
    "__version__",
    "apply_patch_set",
    "build_engine",
    "inspect",
    "main",
]


def main() -> int:
    """包级入口：转交命令行主函数。

    Returns:
        进程退出码。
    """
    from .cli import main as _main

    return _main()
