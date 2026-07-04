"""
PyInstaller runtime hook — fixes cv2 import recursion in frozen bundles.

cv2's bootstrap() does:
    py_module = sys.modules.pop("cv2")   # removes itself from modules
    native_module = importlib.import_module("cv2")  # re-imports native ext
    sys.modules["cv2"] = py_module       # puts wrapper back

PyInstaller's frozen importer intercepts the second import_module("cv2") call
because cv2/__init__.py is in the frozen module table. That triggers bootstrap()
a second time while sys.OpenCV_LOADER is already set → recursion error.

Fix: insert a meta_path finder at position 0 that detects the re-import (via
sys.OpenCV_LOADER) and loads the native cv2.abi3.so directly, bypassing the
frozen importer entirely.
"""
import sys
import os
import glob
import importlib.util


class _CV2BootstrapFix:
    def find_module(self, fullname, path=None):
        # Only intercept the re-import that happens inside cv2's bootstrap()
        if fullname == 'cv2' and hasattr(sys, 'OpenCV_LOADER'):
            return self
        return None

    def load_module(self, fullname):
        _base = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
        # Search for the cv2 native extension in the bundle directory
        _candidates = (
            glob.glob(os.path.join(_base, 'cv2', 'cv2.abi3.so'))   +
            glob.glob(os.path.join(_base, 'cv2', 'cv2.cpython*.so')) +
            glob.glob(os.path.join(_base, 'cv2.abi3.so'))           +
            glob.glob(os.path.join(_base, 'cv2.cpython*.so'))
        )
        if not _candidates:
            raise ImportError(
                f'cv2 native extension not found in bundle at {_base!r}. '
                'Rebuild with: bash build_app.sh'
            )
        _spec = importlib.util.spec_from_file_location('cv2', _candidates[0])
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[fullname] = _mod
        _spec.loader.exec_module(_mod)
        return _mod


sys.meta_path.insert(0, _CV2BootstrapFix())
