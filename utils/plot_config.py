"""
Matplotlib 绘图配置工具。

用于在无 sudo 权限的 Linux / Conda 环境中尽量自动启用可用中文字体，
并支持通过环境变量显式指定字体文件路径。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import os
import sys

import matplotlib.pyplot as plt
from matplotlib import font_manager


_CANDIDATE_FONT_NAMES = [
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
    "Sarasa Gothic SC",
    "Sarasa UI SC",
    "LXGW WenKai",
    "AR PL UKai CN",
    "AR PL UMing CN",
    "SimHei",
    "Microsoft YaHei",
    "PingFang SC",
    "Heiti SC",
    "Arial Unicode MS",
]

_FONT_FILE_HINTS = (
    "notosanscjk",
    "notosanssc",
    "sourcehansans",
    "sourcehan",
    "wenquanyi",
    "sarasa",
    "lxgwwenkai",
    "simhei",
    "yahei",
    "pingfang",
    "heiti",
    "arialuni",
    "ukai",
    "uming",
    "cjk",
)

_FONT_EXTENSIONS = {".ttf", ".otf", ".ttc"}
_FONT_ENV_VAR = "SPEAKER_PLOT_FONT_PATH"

_cached_font_name: Optional[str] = None
_font_search_completed = False
_warned_missing_font = False


def _dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if not item:
            continue
        lowered = item.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered.append(item)
    return ordered


def _iter_candidate_font_files(directory: Path, include_all: bool = False) -> Iterable[Path]:
    if not directory.exists() or not directory.is_dir():
        return

    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in _FONT_EXTENSIONS:
            continue

        stem = path.stem.lower().replace(" ", "")
        if include_all or any(hint in stem for hint in _FONT_FILE_HINTS):
            yield path


def _iter_env_font_paths() -> Iterable[Path]:
    raw_value = os.environ.get(_FONT_ENV_VAR, "").strip()
    if not raw_value:
        return

    for raw_path in raw_value.split(os.pathsep):
        candidate = Path(raw_path).expanduser()
        if candidate.is_file():
            yield candidate
        elif candidate.is_dir():
            yield from _iter_candidate_font_files(candidate, include_all=True)


def _iter_search_dirs() -> Iterable[Path]:
    project_root = Path(__file__).resolve().parents[1]
    py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"

    search_dirs = [
        project_root / "fonts",
        project_root / "assets" / "fonts",
        project_root / "data" / "fonts",
        Path.home() / ".fonts",
        Path.home() / ".local" / "share" / "fonts",
        Path(sys.prefix) / "fonts",
        Path(sys.prefix) / "share" / "fonts",
        Path(sys.prefix) / "share" / "fonts" / "truetype",
        Path(sys.prefix) / "lib" / py_version / "site-packages" / "matplotlib" / "mpl-data" / "fonts",
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
    ]

    yielded: set[Path] = set()
    for directory in search_dirs:
        resolved = directory.resolve() if directory.exists() else directory
        if resolved in yielded:
            continue
        yielded.add(resolved)
        yield directory


def _register_font_file(font_path: Path) -> Optional[str]:
    try:
        font_manager.fontManager.addfont(str(font_path))
        return font_manager.FontProperties(fname=str(font_path)).get_name()
    except Exception:
        return None


def _find_installed_font_name() -> Optional[str]:
    available_names = [font.name for font in font_manager.fontManager.ttflist]
    lower_to_name: dict[str, str] = {}
    for name in available_names:
        lower_to_name.setdefault(name.lower(), name)

    for candidate in _CANDIDATE_FONT_NAMES:
        matched = lower_to_name.get(candidate.lower())
        if matched:
            return matched

    for candidate in _CANDIDATE_FONT_NAMES:
        candidate_lower = candidate.lower()
        for available in available_names:
            if candidate_lower in available.lower():
                return available

    return None


def _discover_chinese_font() -> Optional[str]:
    for font_path in _iter_env_font_paths():
        font_name = _register_font_file(font_path)
        if font_name:
            return font_name

    installed_font = _find_installed_font_name()
    if installed_font:
        return installed_font

    for directory in _iter_search_dirs():
        for font_path in _iter_candidate_font_files(directory):
            font_name = _register_font_file(font_path)
            if font_name:
                return font_name

    return None


def setup_matplotlib_for_chinese() -> Optional[str]:
    """
    为 Matplotlib 配置中文字体。

    优先级：
    1. 环境变量 `SPEAKER_PLOT_FONT_PATH` 指定的字体文件或目录
    2. 当前环境里已安装/已注册的常见中文字体
    3. 项目目录、用户目录、conda 目录中的常见字体文件

    Returns:
        成功命中的字体名称；若未找到则返回 `None`。
    """
    global _cached_font_name, _font_search_completed, _warned_missing_font

    if not _font_search_completed:
        _cached_font_name = _discover_chinese_font()
        _font_search_completed = True

    fallback_fonts = _dedupe_keep_order(
        ([_cached_font_name] if _cached_font_name else [])
        + _CANDIDATE_FONT_NAMES
        + list(plt.rcParams.get("font.sans-serif", []))
    )

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = fallback_fonts
    plt.rcParams["axes.unicode_minus"] = False

    if not _cached_font_name and not _warned_missing_font:
        print(
            "警告: 未检测到可用中文字体，图中的中文可能无法正常显示。"
            "可将字体文件放到项目的 `fonts/` 目录，"
            f"或设置环境变量 `{_FONT_ENV_VAR}` 指向字体文件。"
        )
        _warned_missing_font = True

    return _cached_font_name
