# sse_env.py
# -----------------------------------------------------------------------------
# Minimal dependency/setup helpers for Colab, RunPod, LightningAI, LocalPC
# Converted from notebook magics to pure-Python so it can be imported anywhere.
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
import sys
import subprocess
import shutil
from typing import Tuple

# -------------------- Small helpers --------------------
def _run(cmd: list[str], check: bool = True) -> int:
    """Run a command (list form). Returns exit code. Raises if check and nonzero."""
    # Print the command for transparency
    print("$", " ".join(cmd))
    try:
        return subprocess.run(cmd, check=check).returncode
    except subprocess.CalledProcessError as e:
        print(f"Command failed with code {e.returncode}: {' '.join(cmd)}")
        if check:
            raise
        return e.returncode

def _pip_install(*pkgs: str, extra_args: list[str] | None = None, check: bool = True) -> int:
    args = [sys.executable, "-m", "pip", "install"]
    if extra_args:
        args += extra_args
    args += list(pkgs)
    return _run(args, check=check)

def _apt_available() -> bool:
    return shutil.which("apt-get") is not None

def _sudo_prefix() -> list[str]:
    return ["sudo"] if shutil.which("sudo") else []

def _is_colab() -> bool:
    # Colab usually has /content and a specific runtime
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return "content" in os.getcwd()

# -------------------- Platform Handling --------------------
class PlatformManager:
    def __init__(self):
        self.platform, self.local_path = self.detect_platform()

    @staticmethod
    def mount_gdrive():
        """Mount Google Drive in Colab (noop elsewhere)."""
        if _is_colab():
            try:
                from google.colab import drive  # type: ignore
                drive.mount('/content/drive')
            except Exception as e:
                print(f"Failed to mount Google Drive: {e}")
        else:
            print("Google Drive mount is only applicable in Colab.")

    @staticmethod
    def detect_platform() -> Tuple[str, str]:
        """Detect platform and return its name and the local path"""
        computing_platform = 'LocalPC'

        if os.getenv('RUNPOD_POD_ID'):
            computing_platform = "RunPod"
            print("Running on RunPod.")
            local_path = "/workspace/"
        elif 'content' in str(os.getcwd()):
            computing_platform = "Colab"
            print("Running on Colab.")
            local_path = "/content/"
        elif os.getenv("LIGHTNING_ARTIFACTS_DIR"):
            computing_platform = "LightningAI"
            print("Running on Lightning AI Studio")
            local_path = os.getenv("LIGHTNING_ARTIFACTS_DIR") + '/'
        else:
            local_path = os.getcwd() + '/'

        return computing_platform, local_path

# -------------------- Installation Helpers --------------------
class DependencyInstaller:
    @staticmethod
    def install_glut(computing_platform: str):
        _pip_install("--upgrade", "pip", extra_args=[], check=False)

        # Only try apt on Debian/Ubuntu-based images
        if not _apt_available():
            print("apt-get not available on this system; skipping GL/GLUT dev packages.")
            return

        apt = _sudo_prefix() + ["apt-get"]
        if computing_platform in {"LightningAI", "RunPod", "Colab", "LocalPC"}:
            _run(apt + ["-qq", "update"], check=False)
            # Quiet, noninteractive install
            _run(apt + ["install", "-y",
                        "freeglut3-dev", "libglew-dev", "libsdl2-dev"], check=False)

    @staticmethod
    def install_opengl():
        try:
            _pip_install("PyOpenGL", "PyOpenGL_accelerate", extra_args=[], check=True)
        except Exception as e:
            print(f"Failed to install OpenGL: {e}")

    @staticmethod
    def get_pytorch3d_version_string() -> str | None:
        """
        Compose the official PyTorch3D wheel tag used in the FB packaging index.
        Returns None if CUDA version is unknown (CPU-only), in which case we should
        build from source instead of wheels.
        """
        import torch
        pyt_version_str = torch.__version__.split("+")[0].replace(".", "")
        cuda_ver = getattr(torch.version, "cuda", None)
        if not cuda_ver:
            # CPU-only builds usually require source install
            return None
        version_str = "".join([
            f"py3{sys.version_info.minor}_cu",
            cuda_ver.replace(".", ""),
            f"_pyt{pyt_version_str}"
        ])
        return version_str

# -------------------- PyTorch3D Installer --------------------
class PyTorch3DInstaller:
    def __init__(self, computing_platform: str, local_path: str):
        self.platform = computing_platform
        self.local_path = local_path

    def install(self):
        import sys

        need_pytorch3d = False

        _pip_install("--upgrade", "pip", extra_args=[], check=False)
        DependencyInstaller.install_glut(self.platform)
        DependencyInstaller.install_opengl()

        version_str = DependencyInstaller.get_pytorch3d_version_string()
        print(f"\nPyTorch3D target wheel tag: {version_str}\n")

        # Always need iopath first
        _pip_install("iopath", extra_args=[], check=False)

        # Try wheels (Linux only)
        if sys.platform.startswith("linux") and version_str:
            print(f"Trying to install PyTorch3D wheel on {self.platform} (Linux).")
            try:
                if self.platform in {"RunPod", "LightningAI"}:
                    # Official wheel index
                    index_url = f"https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html"
                    _pip_install("pytorch3d",
                                 extra_args=["--no-index", "--no-cache-dir", "-f", index_url],
                                 check=True)

                elif self.platform == "Colab":
                    # Your prebuilt wheel from Dropbox (install directly from URL)
                    wheel_url = ("https://www.dropbox.com/scl/fi/fqvlnyponcbekjd01omhj/"
                                 "pytorch3d-0.7.8-cp312-cp312-linux_x86_64.whl"
                                 "?rlkey=563mfx35rog42z1c8y7qn31sk&dl=1")
                    _pip_install(wheel_url, extra_args=[], check=True)

                else:
                    # Try the official wheel index as a best-effort default
                    index_url = f"https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html"
                    _pip_install("pytorch3d",
                                 extra_args=["--no-index", "--no-cache-dir", "-f", index_url],
                                 check=True)
            except Exception as e:
                print(f"Wheel install failed: {e}")
                need_pytorch3d = True
        else:
            # Non-Linux or no CUDA tag known -> source install
            need_pytorch3d = True

        # Verify or fall back to source
        try:
            import pytorch3d  # noqa: F401
            print("✅ PyTorch3D successfully installed!")
            need_pytorch3d = False
        except Exception:
            need_pytorch3d = True

        if need_pytorch3d:
            print("Falling back to source install for PyTorch3D (this may take a while).")
            # ninja speeds up the build; ignore root-user action warnings where relevant
            _pip_install("ninja", extra_args=["--root-user-action", "ignore"], check=False)
            # Use stable branch
            _pip_install("git+https://github.com/facebookresearch/pytorch3d.git@stable",
                         extra_args=[], check=True)
            try:
                import pytorch3d  # noqa: F401
            except Exception:
                print("❌ PyTorch3D failed to install from source.")
            else:
                print("✅ PyTorch3D successfully installed from source.")
