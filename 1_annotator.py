import sys
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QGroupBox,
    QMessageBox,
    QAbstractItemView,
)
from PyQt5.QtCore import Qt
from src.pipeline.preprocessing.loader import DataLoader as DataLoaderETL
# -------- Global Matplotlib Styling --------
# Set Arial as the default font
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
matplotlib.rcParams["font.size"] = 12
matplotlib.rcParams["axes.titlesize"] = 12
matplotlib.rcParams["axes.labelsize"] = 12
matplotlib.rcParams["xtick.labelsize"] = 12
matplotlib.rcParams["ytick.labelsize"] = 12
matplotlib.rcParams["legend.fontsize"] = 12
LABELS = ["Clamping", "Bending", "Mandrel Extraction", "De-Clamping"]
LABEL_COLORS = {
    "Clamping": "#1f77b4", # Blue
    "Bending": "#ff7f0e", # Orange
    "Mandrel Extraction": "#2ca02c", # Green
    "De-Clamping": "#d62728", # Red
}
class Annotator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MAR Annotator")
        # ✅ Wider, shorter window
        self.setGeometry(100, 100, 1700, 620)
        # Always white background
        self.setStyleSheet("background-color: white; color: black;")
        self.df = None
        self.labels = []
        self.current_exp = None
        self.click_stage = "start"
        self.temp_label = {"start": None, "end": None, "label": None}
        # line objects for current plot
        self.lines_by_col = {}
        # global per-column visibility (persist across experiments)
        self.col_visibility = {}
        # guard to avoid recursion when programmatically checking boxes
        self._block_collist_signal = False
        # track press state so row clicks can toggle without double-flipping checkbox clicks
        self._pressed_col_item = None
        # legend reference
        self.legend = None
        self.annotation_legend_items = []
        self._setup_gui()
        loader = DataLoaderETL("data/processed/tube_geometry.db")
        dataframes = loader.load_all_data_from_sqlite()
        self.df = dataframes["machine_and_movement"]
        self.df = self.df.set_index(self.df.columns[0])
        all_experiments = list(self.df["Experiment_ID"].unique())
        self.exp_selector.clear()
        self.exp_selector.addItems([str(e) for e in all_experiments])
        self.exp_selector.setCurrentIndex(0)
        # Initialize visibility for all columns (except Experiment_ID)
        all_cols = [c for c in self.df.columns if c != "Experiment_ID"]
        for c in all_cols:
            self.col_visibility.setdefault(c, True)
        self._plot_experiment()
        self._set_status(f"Data loaded with {len(all_experiments)} experiments.")
    # ---------------- GUI ----------------
    def _setup_gui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(10)
        # ---- Plot frame ----
        plot_frame = QVBoxLayout()
        plot_frame.setContentsMargins(0, 0, 0, 0)
        plot_frame.setSpacing(6)
        # ✅ Figure shape (secondary; Qt will stretch unless canvas height is constrained)
        self.figure = plt.Figure(figsize=(18, 3.2), constrained_layout=False)
        self.figure.set_facecolor("white")
        # Single subplot for sensors
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor("white")
        self.ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.8)
        self.ax.set_axisbelow(True)
        # Remove top and right spines
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["left"].set_linewidth(1.2)
        self.ax.spines["bottom"].set_linewidth(1.2)
        self.ax.spines["left"].set_color("#333333")
        self.ax.spines["bottom"].set_color("#333333")
        # Keep title/labels visible while reserving space for the external legend.
        self.figure.subplots_adjust(left=0.07, right=0.82, top=0.88, bottom=0.18)
        self.canvas = FigureCanvas(self.figure)
        # ✅ KEY FIX: force the plot area to be about half height
        self.canvas.setMinimumHeight(360)
        self.canvas.setMaximumHeight(420)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        plot_frame.addWidget(self.canvas, 1)
        self.hide_btn = QPushButton("Toggle All Sensors")
        self.hide_btn.setFixedHeight(28)
        self.hide_btn.clicked.connect(self._toggle_all_lines)
        plot_frame.addWidget(self.hide_btn, 0)
        # ✅ Give plot much more width than controls
        top_layout.addLayout(plot_frame, 6)
        # ---- Control frame ----
        control_frame = QVBoxLayout()
        control_frame.setSpacing(8)
        # Experiment selector
        exp_group = QGroupBox("Experiment")
        exp_layout = QVBoxLayout()
        exp_layout.setContentsMargins(8, 8, 8, 8)
        self.exp_selector = QComboBox()
        self.exp_selector.currentIndexChanged.connect(self._plot_experiment)
        exp_layout.addWidget(self.exp_selector)
        exp_group.setLayout(exp_layout)
        control_frame.addWidget(exp_group)
        # Label selector
        label_group = QGroupBox("Annotation Label")
        label_layout = QVBoxLayout()
        label_layout.setContentsMargins(8, 8, 8, 8)
        self.label_selector = QComboBox()
        self.label_selector.addItems(LABELS)
        label_layout.addWidget(self.label_selector)
        label_group.setLayout(label_layout)
        control_frame.addWidget(label_group)
        # Column checkboxes list
        col_group = QGroupBox("Sensors (check to show)")
        col_layout = QVBoxLayout()
        col_layout.setContentsMargins(8, 8, 8, 8)
        self.col_list = QListWidget()
        self.col_list.setMaximumHeight(200)
        self.col_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.col_list.itemPressed.connect(self._on_col_item_pressed)
        self.col_list.itemChanged.connect(self._on_col_item_changed)
        self.col_list.itemClicked.connect(self._on_col_item_clicked)
        col_layout.addWidget(self.col_list)
        col_group.setLayout(col_layout)
        control_frame.addWidget(col_group)
        # Annotation management
        annot_group = QGroupBox("Annotations")
        annot_layout = QVBoxLayout()
        annot_layout.setContentsMargins(8, 8, 8, 8)
        self.delete_btn = QPushButton("Delete Annotation")
        self.delete_btn.clicked.connect(self._delete_annotation)
        annot_layout.addWidget(self.delete_btn)
        annot_group.setLayout(annot_layout)
        control_frame.addWidget(annot_group)
        # File operations
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout()
        file_layout.setContentsMargins(8, 8, 8, 8)
        self.save_json_btn = QPushButton("Save JSON")
        self.save_json_btn.clicked.connect(self._save_json)
        file_layout.addWidget(self.save_json_btn)
        self.load_json_btn = QPushButton("Load JSON")
        self.load_json_btn.clicked.connect(self._load_json)
        file_layout.addWidget(self.load_json_btn)
        self.save_pdf_btn = QPushButton("Save Plot (PDF)")
        self.save_pdf_btn.clicked.connect(self._save_plot_pdf)
        file_layout.addWidget(self.save_pdf_btn)
        file_group.setLayout(file_layout)
        control_frame.addWidget(file_group)
        # Keyboard shortcuts
        instr_group = QGroupBox("Keyboard Shortcuts")
        instr_layout = QVBoxLayout()
        instr_layout.setContentsMargins(8, 8, 8, 8)
        instr_label = QLabel("Spacebar: Next experiment\nLeft Alt: Next label type")
        instr_label.setWordWrap(True)
        instr_layout.addWidget(instr_label)
        instr_group.setLayout(instr_layout)
        control_frame.addWidget(instr_group)
        control_frame.addStretch()
        top_layout.addLayout(control_frame, 1)
        main_layout.addLayout(top_layout, 1)
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(
            "background-color: #f2f2f2; padding: 4px; font-size: 10px;"
        )
        main_layout.addWidget(self.status_label)
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self._next_experiment()
        elif event.key() == Qt.Key_Alt:
            self._next_label()
    def _set_status(self, message: str):
        self.status_label.setText(message)
    # ---------------- Plotting ----------------
    def _plot_experiment(self):
        if self.df is None or self.exp_selector.currentText() == "":
            return
        self.current_exp = int(self.exp_selector.currentText())
        exp_df = self.df[self.df["Experiment_ID"] == self.current_exp]
        self.click_stage = "start"
        self.temp_label = {"start": None, "end": None, "label": None}
        self.ax.clear()
        self.ax.set_facecolor("white")
        self.ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.8)
        self.ax.set_axisbelow(True)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["left"].set_linewidth(1.2)
        self.ax.spines["bottom"].set_linewidth(1.2)
        self.lines_by_col = {}
        cols = [c for c in exp_df.columns if c != "Experiment_ID"]
        cmap = plt.get_cmap("tab20")
        colors_sensors = cmap(np.linspace(0, 1, len(cols)))
        for col, color in zip(cols, colors_sensors):
            (line,) = self.ax.plot(
                exp_df.index,
                exp_df[col].values,
                color=color,
                label=col,
                linewidth=1.8,
                alpha=0.85,
                marker="o",
                markersize=2,
                markevery=max(1, len(exp_df) // 30),
            )
            visible = self.col_visibility.get(col, True)
            line.set_visible(visible)
            self.lines_by_col[col] = line
        self.annotation_legend_items = []
        for lbl in self.labels:
            if lbl["Experiment_ID"] == self.current_exp:
                start, end = sorted([lbl["start"], lbl["end"]])
                color = LABEL_COLORS.get(lbl["label"], "#888888")
                self.ax.axvspan(start, end, color=color, alpha=0.15, zorder=0)
                duration = lbl.get("duration", end - start)
                self.annotation_legend_items.append(
                    (
                        Patch(facecolor=color, edgecolor=color, alpha=0.15),
                        f"{lbl['label']} ({duration:.1f})",
                    )
                )
        self.ax.set_title(
            f"Experiment {self.current_exp} - Sensor Data",
            fontsize=11,
            fontweight="semibold",
            pad=8,
            y=1.01,
        )
        self.ax.set_xlabel("Time Step", fontsize=11, fontweight="semibold", labelpad=6)
        self.ax.set_ylabel("Sensor Values", fontsize=11, fontweight="semibold", labelpad=6)
        if len(exp_df.index) > 0:
            self.ax.set_xlim(exp_df.index.min(), exp_df.index.max())
        self.figure.subplots_adjust(left=0.07, right=0.75, top=0.88, bottom=0.18)
        self._update_legend()
        self._populate_column_checks(cols)
        self._update_y_axis_limits()
        self.canvas.draw_idle()
    def _update_legend(self):
        if self.legend is not None:
            self.legend.remove()
        legend_handles = []
        legend_labels = []
        for col, line in self.lines_by_col.items():
            if line.get_visible():
                legend_handles.append(line)
                legend_labels.append(col)
        for handle, label in self.annotation_legend_items:
            legend_handles.append(handle)
            legend_labels.append(label)
        if legend_handles:
            self.legend = self.ax.legend(
                legend_handles,
                legend_labels,
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                borderaxespad=0.0,
                frameon=True,
                fancybox=True,
                shadow=False,
                fontsize=8,
                framealpha=1.0,
                edgecolor="#cccccc",
                ncol=1,
                labelspacing=0.35,
                borderpad=0.4,
                handlelength=1.6,
                handletextpad=0.5,
                columnspacing=1.1,
            )
            self.legend.get_frame().set_facecolor("white")
            self.legend.get_frame().set_linewidth(0.8)
        else:
            self.legend = None
    def _populate_column_checks(self, cols):
        self._block_collist_signal = True
        try:
            self.col_list.clear()
            for col in sorted(cols):
                item = QListWidgetItem(col)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                checked = Qt.Checked if self.col_visibility.get(col, True) else Qt.Unchecked
                item.setCheckState(checked)
                self.col_list.addItem(item)
        finally:
            self._block_collist_signal = False
    def _on_col_item_changed(self, item: QListWidgetItem):
        if self._block_collist_signal:
            return
        col = item.text()
        visible = item.checkState() == Qt.Checked
        self.col_visibility[col] = visible
        line = self.lines_by_col.get(col)
        if line is not None:
            line.set_visible(visible)
        self._refresh_plot_after_visibility_change()
    def _on_col_item_clicked(self, item: QListWidgetItem):
        if self._block_collist_signal:
            return
        if self._pressed_col_item is None:
            return
        pressed_col, pressed_state = self._pressed_col_item
        self._pressed_col_item = None
        if item.text() != pressed_col or item.checkState() != pressed_state:
            return
        new_state = Qt.Unchecked if pressed_state == Qt.Checked else Qt.Checked
        item.setCheckState(new_state)
    def _on_col_item_pressed(self, item: QListWidgetItem):
        self._pressed_col_item = (item.text(), item.checkState())
    def _refresh_plot_after_visibility_change(self):
        self._update_legend()
        self._update_y_axis_limits()
        self.canvas.draw_idle()
    def _update_y_axis_limits(self):
        if not self.lines_by_col:
            return
        visible_lines = [ln for ln in self.lines_by_col.values() if ln.get_visible()]
        if not visible_lines:
            return
        ymin = float("inf")
        ymax = float("-inf")
        for ln in visible_lines:
            y = ln.get_ydata()
            if len(y) == 0:
                continue
            ymin = min(ymin, float(np.min(y)))
            ymax = max(ymax, float(np.max(y)))
        if ymin == float("inf") or ymax == float("-inf"):
            return
        span = (ymax - ymin) if ymax != ymin else 1.0
        pad = 0.05 * span
        self.ax.set_ylim(ymin - pad, ymax + pad)
    def _toggle_all_lines(self):
        if not self.lines_by_col:
            return
        any_visible = any(ln.get_visible() for ln in self.lines_by_col.values())
        target_state = Qt.Unchecked if any_visible else Qt.Checked
        self._block_collist_signal = True
        try:
            for i in range(self.col_list.count()):
                item = self.col_list.item(i)
                item.setCheckState(target_state)
                col = item.text()
                visible = (target_state == Qt.Checked)
                self.col_visibility[col] = visible
                if col in self.lines_by_col:
                    self.lines_by_col[col].set_visible(visible)
        finally:
            self._block_collist_signal = False
        self._refresh_plot_after_visibility_change()
    # ---------------- Annotation click logic ----------------
    def on_click(self, event):
        if self.current_exp is None or event.xdata is None:
            return
        if self.click_stage == "start":
            self.temp_label["start"] = float(event.xdata)
            self.temp_label["label"] = self.label_selector.currentText()
            self.click_stage = "end"
            self._set_status(f"Start {self.temp_label['label']} at {event.xdata:.2f}")
        else:
            self.temp_label["end"] = float(event.xdata)
            start, end = sorted([self.temp_label["start"], self.temp_label["end"]])
            duration = end - start
            self.labels.append(
                {
                    "Experiment_ID": self.current_exp,
                    "label": self.temp_label["label"],
                    "start": start,
                    "end": end,
                    "duration": duration,
                }
            )
            self.click_stage = "start"
            self.temp_label = {"start": None, "end": None, "label": None}
            self._set_status(f"End label at {event.xdata:.2f} (duration: {duration:.2f})")
            self._plot_experiment()
    # ---------------- Navigation shortcuts ----------------
    def _next_experiment(self):
        if self.df is None:
            return
        idx = self.exp_selector.currentIndex()
        next_idx = (idx + 1) % self.exp_selector.count()
        self.exp_selector.setCurrentIndex(next_idx)
        self._plot_experiment()
    def _next_label(self):
        idx = self.label_selector.currentIndex()
        next_idx = (idx + 1) % self.label_selector.count()
        self.label_selector.setCurrentIndex(next_idx)
        self._set_status(f"Current Label: {self.label_selector.currentText()}")
    # ---------------- Save plot to PDF ----------------
    def _save_plot_pdf(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot as PDF", filter="PDF Files (*.pdf)"
        )
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        try:
            # Save the current figure without tight-bbox recomputation.
            # The embedded Qt figure plus external legend can become unstable
            # during PDF export when Matplotlib recomputes a tight layout.
            self.canvas.draw()
            self.figure.savefig(
                path,
                format="pdf",
                facecolor="white",
            )
            self._set_status(f"Plot saved as PDF: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save PDF:\n{e}")
    # ---------------- Save/Load JSON ----------------
    def _save_json(self):
        if not self.labels:
            QMessageBox.warning(self, "No Labels", "No labels to save!")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save JSON", filter="JSON Files (*.json)"
        )
        if not path:
            return
        grouped = {}
        for lbl in self.labels:
            exp_id = str(lbl["Experiment_ID"])
            grouped.setdefault(exp_id, []).append(
                {
                    "label": lbl["label"],
                    "start": lbl["start"],
                    "end": lbl["end"],
                    "duration": lbl.get("duration", lbl["end"] - lbl["start"]),
                }
            )
        with open(path, "w") as f:
            json.dump(grouped, f, indent=4)
        self._set_status(f"Labels saved to {path}")
    def _parse_annotations_payload(self, payload):
        annotations = []
        if isinstance(payload, dict):
            for exp_id, lbl_list in payload.items():
                if not isinstance(lbl_list, list):
                    raise ValueError(
                        f"Expected a list of annotations for experiment '{exp_id}'."
                    )
                for lbl in lbl_list:
                    start, end = lbl["start"], lbl["end"]
                    duration = lbl.get("duration", end - start)
                    annotations.append(
                        {
                            "Experiment_ID": int(exp_id),
                            "label": lbl["label"],
                            "start": start,
                            "end": end,
                            "duration": duration,
                        }
                    )
            return annotations
        if isinstance(payload, list):
            for lbl in payload:
                exp_id = lbl.get("Experiment_ID", lbl.get("experiment_id"))
                if exp_id is None:
                    raise ValueError(
                        "Each annotation entry must include 'Experiment_ID'."
                    )
                start, end = lbl["start"], lbl["end"]
                duration = lbl.get("duration", end - start)
                annotations.append(
                    {
                        "Experiment_ID": int(exp_id),
                        "label": lbl["label"],
                        "start": start,
                        "end": end,
                        "duration": duration,
                    }
                )
            return annotations
        raise ValueError("Unsupported annotations format in JSON file.")
    def _apply_loaded_config(self, loaded):
        if not isinstance(loaded, dict):
            return
        experiment_value = loaded.get("selected_experiment", loaded.get("current_experiment"))
        if experiment_value is not None:
            idx = self.exp_selector.findText(str(experiment_value))
            if idx >= 0:
                self.exp_selector.setCurrentIndex(idx)
        label_value = loaded.get("selected_label", loaded.get("current_label"))
        if label_value is not None:
            idx = self.label_selector.findText(str(label_value))
            if idx >= 0:
                self.label_selector.setCurrentIndex(idx)
        sensor_visibility = loaded.get("sensor_visibility")
        if isinstance(sensor_visibility, dict):
            for col, visible in sensor_visibility.items():
                if col in self.col_visibility:
                    self.col_visibility[col] = bool(visible)
        visible_sensors = loaded.get("visible_sensors")
        if isinstance(visible_sensors, list):
            visible_set = {str(sensor) for sensor in visible_sensors}
            for col in self.col_visibility:
                self.col_visibility[col] = col in visible_set
        hidden_sensors = loaded.get("hidden_sensors")
        if isinstance(hidden_sensors, list):
            hidden_set = {str(sensor) for sensor in hidden_sensors}
            for col in self.col_visibility:
                if col in hidden_set:
                    self.col_visibility[col] = False
        self._plot_experiment()
    def _load_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load JSON", filter="JSON Files (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "r") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                has_explicit_annotations = ("annotations" in loaded) or ("labels" in loaded)
                has_config_keys = any(
                    key in loaded
                    for key in (
                        "selected_experiment",
                        "current_experiment",
                        "selected_label",
                        "current_label",
                        "sensor_visibility",
                        "visible_sensors",
                        "hidden_sensors",
                    )
                )
                loaded_labels = self.labels
                if has_explicit_annotations:
                    annotations_payload = loaded.get("annotations", loaded.get("labels"))
                    loaded_labels = self._parse_annotations_payload(annotations_payload)
                elif has_config_keys:
                    loaded_labels = self.labels
                else:
                    loaded_labels = self._parse_annotations_payload(loaded)
                self.labels = loaded_labels
                self._apply_loaded_config(loaded)
                self._set_status(
                    f"Loaded {len(self.labels)} annotations from {path}"
                )
            else:
                raise ValueError("Top-level JSON must be an object.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load JSON:\n{e}")
            self._set_status(f"Failed to load JSON: {path}")
    # ---------------- Delete annotation ----------------
    def _delete_annotation(self):
        if self.current_exp is None:
            QMessageBox.warning(self, "No Experiment", "Select an experiment first!")
            return
        exp_labels = [lbl for lbl in self.labels if lbl["Experiment_ID"] == self.current_exp]
        if not exp_labels:
            QMessageBox.information(
                self, "No Annotations", "No annotations to delete for this experiment."
            )
            return
        dlg = QListWidget()
        dlg.setWindowTitle("Delete Annotation (double-click to delete)")
        for i, lbl in enumerate(exp_labels):
            duration = lbl.get("duration", lbl["end"] - lbl["start"])
            dlg.addItem(
                f"{i}: {lbl['label']} [{lbl['start']:.2f}, {lbl['end']:.2f}] "
                f"(duration: {duration:.2f})"
            )
        dlg.itemDoubleClicked.connect(lambda item: self._remove_selected(item, dlg, exp_labels))
        dlg.setMinimumWidth(500)
        dlg.setMinimumHeight(250)
        dlg.show()
        self._delete_dialog = dlg
    def _remove_selected(self, item, dlg, exp_labels):
        idx = dlg.row(item)
        self.labels.remove(exp_labels[idx])
        dlg.close()
        self._plot_experiment()
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Annotator()
    window.show()
    sys.exit(app.exec_())