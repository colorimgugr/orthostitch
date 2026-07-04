"""Modern PyQt6 GUI for the lossless ortho-anchored HSI stitcher.

Layout is a flexible 2x2 grid (draggable splitters, scrollable):
  top-left  = inputs        top-right = progress
  bottom    = two image viewers (pseudo-RGB | seam map), given the most room.

Each result viewer supports zoom (wheel / buttons), pan (drag), rotate, fit and
1:1. An "Advanced..." dialog exposes the registration thresholds.

Run:  python hsi_stitcher/gui.py
"""

import os
import sys
import traceback
from pathlib import Path

# make sibling modules importable whether launched as a script or a module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRectF, QSize
from PyQt6.QtGui import QPixmap, QFont, QTransform, QPainter, QColor, QImageReader
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QToolButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QProgressBar, QPlainTextEdit, QSplitter, QFrame, QMessageBox, QSizePolicy,
    QScrollArea, QDialog, QDialogButtonBox, QSpinBox, QDoubleSpinBox,
    QGraphicsView, QGraphicsScene,
)

import stitcher
import modality

ACCENT = "#5b8def"
QSS = f"""
QWidget {{
    background: #14161c; color: #e6e8ee;
    font-family: -apple-system, "SF Pro Text", "Segoe UI", sans-serif;
    font-size: 13px;
}}
#Title {{ font-size: 18px; font-weight: 600; }}
#Subtitle {{ color: #8a90a0; }}
#Card {{ background: #1b1e26; border: 1px solid #262a35; border-radius: 12px; }}
#CardTitle {{ color: #9aa1b2; font-size: 11px; font-weight: 600; }}
QLabel#FieldLabel {{ color: #aab0c0; }}
QLineEdit, QSpinBox, QDoubleSpinBox {{
    background: #0f1116; border: 1px solid #2b303c; border-radius: 8px;
    padding: 7px 10px; selection-background-color: {ACCENT};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {ACCENT}; }}
QPushButton {{
    background: #262a35; border: 1px solid #313643; border-radius: 8px; padding: 8px 14px;
}}
QPushButton:hover {{ background: #2d323f; }}
QPushButton:pressed {{ background: #21252e; }}
QPushButton#Primary {{
    background: {ACCENT}; border: 1px solid {ACCENT}; color: white;
    font-weight: 600; padding: 10px 18px;
}}
QPushButton#Primary:hover {{ background: #6f9bf2; }}
QPushButton#Primary:disabled {{ background: #2a2e38; border: 1px solid #2a2e38; color: #6b7180; }}
QToolButton {{
    background: #262a35; border: 1px solid #313643; border-radius: 6px;
    padding: 3px 8px; color: #d6dae4; font-size: 13px;
}}
QToolButton:hover {{ background: #2d323f; }}
QProgressBar {{
    background: #0f1116; border: 1px solid #2b303c; border-radius: 8px;
    text-align: center; height: 18px;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 7px; }}
QPlainTextEdit {{
    background: #0f1116; border: 1px solid #2b303c; border-radius: 8px;
    font-family: "SF Mono", "Menlo", "Consolas", monospace; font-size: 12px; color: #c2c7d4;
}}
QGraphicsView {{ background: #0f1116; border: 1px solid #2b303c; border-radius: 8px; }}
QSplitter::handle {{ background: #262a35; }}
QSplitter::handle:horizontal {{ width: 6px; margin: 2px; border-radius: 3px; }}
QSplitter::handle:vertical {{ height: 6px; margin: 2px; border-radius: 3px; }}
QScrollArea {{ border: none; }}
QDialog {{ background: #14161c; }}
"""


def card(title=None):
    frame = QFrame()
    frame.setObjectName("Card")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(16, 12, 16, 14)
    lay.setSpacing(9)
    if title:
        lab = QLabel(title.upper())
        lab.setObjectName("CardTitle")
        lay.addWidget(lab)
    return frame, lay


# ---------------------------------------------------------------------------
# advanced options dialog
# ---------------------------------------------------------------------------
# (key, label, kind, min, max, step, decimals, tooltip)
PARAM_SPECS = [
    ("SIFT_CONTRAST", "SIFT contrast threshold", "double", 0.001, 0.200, 0.001, 3,
     "Lower keeps more low-contrast keypoints (needed for dark paintings); "
     "too low admits spurious matches. OpenCV default is 0.04."),
    ("SIFT_EDGE", "SIFT edge threshold", "double", 1.0, 100.0, 1.0, 0,
     "Higher keeps more edge-like keypoints."),
    ("RATIO", "Lowe ratio test", "double", 0.50, 0.95, 0.01, 2,
     "Higher accepts looser (more) descriptor matches."),
    ("MIN_INLIERS", "Min pair inliers", "int", 4, 1000, 1, 0,
     "Minimum RANSAC inliers to accept a tile-to-tile pair."),
    ("MIN_ANCHOR_INLIERS", "Min ortho-anchor inliers", "int", 4, 1000, 1, 0,
     "Minimum inliers to anchor a tile to the ortho image."),
    ("RANSAC_THRESH", "RANSAC reproj. threshold (px)", "double", 0.5, 20.0, 0.5, 1,
     "Maximum reprojection error for tile-pair RANSAC."),
    ("ANCHOR_WEIGHT", "Ortho anchor weight", "double", 0.0, 5.0, 0.05, 2,
     "Weight of ortho-anchor residuals relative to tile-pair residuals."),
    ("MAX_CANVAS_PX", "Max output long side (px)", "int", 1000, 60000, 500, 0,
     "Caps the mosaic resolution. Essential for high-res photographic tiles "
     "(NEF/large TIFF): native resolution there is gigapixel and exhausts RAM. "
     "Output stays bit-exact (nearest-neighbor), just subsampled."),
]


class AdvancedDialog(QDialog):
    """Edit the registration thresholds for the next run."""

    def __init__(self, current, defaults, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced options")
        self.setMinimumWidth(440)
        self._defaults = defaults
        self._spins = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)
        intro = QLabel("Registration parameters applied on the next Stitch.")
        intro.setObjectName("Subtitle")
        root.addWidget(intro)

        form = QFormLayout()
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)
        for key, label, kind, lo, hi, step, dec, tip in PARAM_SPECS:
            spin = QSpinBox() if kind == "int" else QDoubleSpinBox()
            if kind == "double":
                spin.setDecimals(dec)
            spin.setRange(lo, hi)
            spin.setSingleStep(step)
            spin.setValue(current.get(key, defaults[key]))
            spin.setToolTip(tip)
            lab = QLabel(label)
            lab.setObjectName("FieldLabel")
            lab.setToolTip(tip)
            self._spins[key] = spin
            form.addRow(lab, spin)
        root.addLayout(form)

        btns = QDialogButtonBox()
        reset = btns.addButton("Reset to defaults", QDialogButtonBox.ButtonRole.ResetRole)
        reset.clicked.connect(self._reset)
        btns.addButton(QDialogButtonBox.StandardButton.Cancel)
        ok = btns.addButton(QDialogButtonBox.StandardButton.Ok)
        ok.setObjectName("Primary")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    def _reset(self):
        for key, spin in self._spins.items():
            spin.setValue(self._defaults[key])

    def values(self):
        return {key: spin.value() for key, spin in self._spins.items()}


# ---------------------------------------------------------------------------
# interactive image viewer (zoom / pan / rotate / fit / 1:1)
# ---------------------------------------------------------------------------
class _GraphicsView(QGraphicsView):
    """Graphics view with wheel-zoom about the cursor and drag-to-pan."""

    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._has = False
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setRenderHints(QPainter.RenderHint.SmoothPixmapTransform
                            | QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(QColor("#0f1116"))
        self.setMinimumHeight(220)
        self._placeholder("")

    def _placeholder(self, text):
        self._scene.clear()
        self._has = False
        if text:
            t = self._scene.addText(text)
            t.setDefaultTextColor(QColor("#6b7180"))

    def show_pixmap(self, pix):
        self._scene.clear()
        self._scene.addPixmap(pix)
        self._scene.setSceneRect(QRectF(pix.rect()))
        self._has = True

    def fit(self):
        if self._has:
            self.resetTransform()
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def actual_size(self):
        if self._has:
            self.resetTransform()

    def zoom(self, factor):
        if self._has:
            self.scale(factor, factor)

    def wheelEvent(self, e):
        if self._has:
            self.zoom(1.25 if e.angleDelta().y() > 0 else 0.8)
        else:
            super().wheelEvent(e)


class ImageViewer(QWidget):
    """A captioned, interactive viewer: keeps the source pixmap + rotation."""

    def __init__(self, title, placeholder):
        super().__init__()
        self._orig = None
        self._angle = 0
        self._placeholder = placeholder

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        bar = QHBoxLayout()
        bar.setSpacing(6)
        cap = QLabel(title.upper())
        cap.setObjectName("CardTitle")
        bar.addWidget(cap)
        bar.addStretch(1)
        for sym, tip, slot in [
            ("-", "Zoom out", lambda: self.view.zoom(0.8)),
            ("+", "Zoom in", lambda: self.view.zoom(1.25)),
            ("Fit", "Fit to window", self.view_fit),
            ("1:1", "Actual size", lambda: self.view.actual_size()),
            ("↶", "Rotate left", lambda: self.rotate(-90)),
            ("↷", "Rotate right", lambda: self.rotate(90)),
        ]:
            b = QToolButton()
            b.setText(sym)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            bar.addWidget(b)
        lay.addLayout(bar)

        self.view = _GraphicsView()
        self.view._placeholder(placeholder)
        lay.addWidget(self.view, 1)

    def view_fit(self):
        self.view.fit()

    DISPLAY_MAX = 6000          # cap display resolution (big mosaics are gigapixel)

    def set_image(self, path):
        # load downscaled for display: result mosaics can be ~100 MP, which both
        # exceeds Qt's image allocation limit and is needless detail for a preview
        reader = QImageReader(str(path))
        reader.setAutoTransform(True)
        sz = reader.size()
        if sz.isValid():
            ls = max(sz.width(), sz.height())
            if ls > self.DISPLAY_MAX:
                f = self.DISPLAY_MAX / ls
                reader.setScaledSize(QSize(max(1, int(sz.width() * f)),
                                           max(1, int(sz.height() * f))))
        img = reader.read()
        pix = QPixmap.fromImage(img) if not img.isNull() else QPixmap()
        self._orig = None if pix.isNull() else pix
        self._angle = 0
        self._render()

    def _render(self):
        if self._orig is None:
            self.view._placeholder(self._placeholder)
            return
        pix = (self._orig if self._angle == 0
               else self._orig.transformed(QTransform().rotate(self._angle),
                                           Qt.TransformationMode.SmoothTransformation))
        self.view.show_pixmap(pix)
        self.view.fit()

    def rotate(self, deg):
        if self._orig is not None:
            self._angle = (self._angle + deg) % 360
            self._render()


# ---------------------------------------------------------------------------
# worker thread
# ---------------------------------------------------------------------------
class _Stream:
    def __init__(self, cb):
        self._cb = cb

    def write(self, s):
        if s:
            self._cb(s)

    def flush(self):
        pass


class Worker(QThread):
    progress = pyqtSignal(int, int, str)
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(str, str)
    failed = pyqtSignal(str)

    def __init__(self, bil, orth, out, name, params):
        super().__init__()
        self._args = (bil, orth, out, name)
        self._params = params

    def run(self):
        bil, orth, out, name = self._args
        old = sys.stdout
        sys.stdout = _Stream(self.log.emit)
        try:
            stitcher.stitch(bil, orth, out, name,
                            progress_cb=lambda i, t, m: self.progress.emit(i, t, m),
                            params=self._params)
            self.finished_ok.emit(out, name.strip() or "pano")
        except Exception as exc:
            self.log.emit("\n" + traceback.format_exc())
            self.failed.emit(str(exc))
        finally:
            sys.stdout = old


# ---------------------------------------------------------------------------
# main window
# ---------------------------------------------------------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HSI Stitcher")
        self.resize(1180, 880)
        self.worker = None
        self._params = stitcher.get_default_params()
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # compact header
        header = QHBoxLayout()
        header.setContentsMargins(22, 14, 22, 6)
        title = QLabel("HSI Stitcher")
        title.setObjectName("Title")
        sub = QLabel("   Lossless ortho-anchored hyperspectral panorama stitching")
        sub.setObjectName("Subtitle")
        header.addWidget(title)
        header.addWidget(sub)
        header.addStretch(1)
        outer.addLayout(header)

        # scrollable body holding the flexible 2x2 splitter grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(18, 6, 18, 18)

        vsplit = QSplitter(Qt.Orientation.Vertical)
        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(self._inputs_card())
        top.addWidget(self._progress_card())
        top.setSizes([560, 560])
        bottom = QSplitter(Qt.Orientation.Horizontal)
        self.rgb_viewer = ImageViewer("pseudo-RGB", "pseudo-RGB will appear here")
        self.seam_viewer = ImageViewer("seam map", "seam map will appear here")
        bottom.addWidget(self.rgb_viewer)
        bottom.addWidget(self.seam_viewer)
        bottom.setSizes([560, 560])

        vsplit.addWidget(top)
        vsplit.addWidget(bottom)
        vsplit.setStretchFactor(0, 0)        # inputs/progress: minimal growth
        vsplit.setStretchFactor(1, 1)        # results: take the extra space
        vsplit.setSizes([250, 620])
        body_lay.addWidget(vsplit)

        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

    # -- cards --------------------------------------------------------------
    def _inputs_card(self):
        cardw, lay = card("Inputs")
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(9)
        grid.setColumnStretch(1, 1)

        def row(r, label, edit, button):
            lab = QLabel(label)
            lab.setObjectName("FieldLabel")
            grid.addWidget(lab, r, 0)
            grid.addWidget(edit, r, 1)
            if button:
                grid.addWidget(button, r, 2)

        self.bil_edit = DropLineEdit(want_dir=True, on_drop=self._bil_changed)
        self.bil_edit.setPlaceholderText("Folder with tiles (.bil/.png/.jpg/.tif/.nef)")
        self.bil_edit.textChanged.connect(self._bil_changed)
        bb = QPushButton("Browse…"); bb.clicked.connect(self._pick_bil)
        row(0, "Tiles folder", self.bil_edit, bb)

        self.orth_edit = DropLineEdit(want_dir=False)
        self.orth_edit.setPlaceholderText("Ortho reference (.tif/.png/.jpg)")
        self.orth_edit.textChanged.connect(self._validate)
        ob = QPushButton("Browse…"); ob.clicked.connect(self._pick_ortho)
        row(1, "Ortho image", self.orth_edit, ob)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Artwork name (output prefix)")
        self.name_edit.textChanged.connect(self._validate)
        row(2, "Artwork name", self.name_edit, None)

        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Output folder")
        self.out_edit.textChanged.connect(self._validate)
        sb = QPushButton("Browse…"); sb.clicked.connect(self._pick_out)
        row(3, "Output folder", self.out_edit, sb)
        lay.addLayout(grid)

        self.detect_lbl = QLabel("")
        self.detect_lbl.setObjectName("Subtitle")
        lay.addWidget(self.detect_lbl)

        btn_row = QHBoxLayout()
        adv = QPushButton("Advanced…"); adv.clicked.connect(self._open_advanced)
        self.open_btn = QPushButton("Open output folder")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._open_out)
        btn_row.addWidget(adv)
        btn_row.addWidget(self.open_btn)
        btn_row.addStretch(1)
        self.run_btn = QPushButton("Stitch")
        self.run_btn.setObjectName("Primary")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._run)
        btn_row.addWidget(self.run_btn)
        lay.addLayout(btn_row)
        lay.addStretch(1)
        return cardw

    def _progress_card(self):
        cardw, lay = card("Progress")
        self.bar = QProgressBar()
        self.bar.setRange(0, 6)
        self.bar.setValue(0)
        self.bar.setFormat("idle")
        lay.addWidget(self.bar)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(120)
        lay.addWidget(self.log, 1)
        return cardw

    # -- pickers / fields ---------------------------------------------------
    def _pick_bil(self):
        d = QFileDialog.getExistingDirectory(self, "Select BIL folder",
                                             self.bil_edit.text() or str(Path.home()))
        if d:
            self.bil_edit.setText(d)

    def _pick_ortho(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select ortho image", self.orth_edit.text() or str(Path.home()),
            "Images (*.tif *.tiff *.png *.jpg *.jpeg);;All files (*)")
        if f:
            self.orth_edit.setText(f)

    def _pick_out(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder",
                                             self.out_edit.text() or str(Path.home()))
        if d:
            self.out_edit.setText(d)

    def _bil_changed(self, *_):
        p = self.bil_edit.text().strip()
        if p and os.path.isdir(p):
            folder = Path(p)
            if not self.name_edit.text().strip():
                self.name_edit.setText(folder.name)
            if not self.out_edit.text().strip():
                self.out_edit.setText(str(folder / "stitched"))
            self.detect_lbl.setText(modality.describe(p))
        else:
            self.detect_lbl.setText("")
        self._validate()

    def _validate(self, *_):
        ok = (os.path.isdir(self.bil_edit.text().strip())
              and os.path.isfile(self.orth_edit.text().strip())
              and bool(self.name_edit.text().strip())
              and bool(self.out_edit.text().strip()))
        self.run_btn.setEnabled(ok and self.worker is None)

    def _open_advanced(self):
        dlg = AdvancedDialog(self._params, stitcher.get_default_params(), self)
        if dlg.exec():
            self._params = dlg.values()

    # -- run ----------------------------------------------------------------
    def _run(self):
        self.log.clear()
        self.bar.setValue(0)
        self.bar.setFormat("starting…")
        self.run_btn.setEnabled(False)
        self.open_btn.setEnabled(False)
        self._set_inputs_enabled(False)
        self.worker = Worker(self.bil_edit.text().strip(),
                             self.orth_edit.text().strip(),
                             self.out_edit.text().strip(),
                             self.name_edit.text().strip(),
                             dict(self._params))
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._on_log)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_fail)
        self.worker.start()

    def _set_inputs_enabled(self, on):
        for w in (self.bil_edit, self.orth_edit, self.name_edit, self.out_edit):
            w.setEnabled(on)

    def _on_progress(self, i, total, msg):
        self.bar.setRange(0, total)
        self.bar.setValue(i)
        self.bar.setFormat(f"[{i}/{total}] {msg}")

    def _on_log(self, s):
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)
        self.log.insertPlainText(s)
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)

    def _on_done(self, out_dir, name):
        self.bar.setValue(self.bar.maximum())
        self.bar.setFormat("done")
        self.rgb_viewer.set_image(Path(out_dir) / f"{name}_pano_rgb.png")
        self.seam_viewer.set_image(Path(out_dir) / "seam_map.png")
        self._out_dir = out_dir
        self.open_btn.setEnabled(True)
        self._finish_worker()

    def _on_fail(self, msg):
        self.bar.setFormat("failed")
        QMessageBox.critical(self, "Stitching failed", msg)
        self._finish_worker()

    def _finish_worker(self):
        self.worker = None
        self._set_inputs_enabled(True)
        self._validate()

    def _open_out(self):
        d = getattr(self, "_out_dir", None)
        if d:
            if sys.platform == "darwin":
                os.system(f'open "{d}"')
            elif os.name == "nt":
                os.startfile(d)             # type: ignore[attr-defined]
            else:
                os.system(f'xdg-open "{d}"')


class DropLineEdit(QLineEdit):
    """Line edit that accepts a dropped file or folder path."""

    def __init__(self, want_dir, on_drop=None, **kw):
        super().__init__(**kw)
        self._want_dir = want_dir
        self._on_drop = on_drop
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if not p:
                continue
            if self._want_dir and not os.path.isdir(p):
                p = os.path.dirname(p)
            self.setText(p)
            if self._on_drop:
                self._on_drop(p)
            break


def main():
    QImageReader.setAllocationLimit(0)          # allow large result PNGs (we downscale on read)
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    app.setFont(QFont("-apple-system", 13))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
