import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    QGroupBox,
)
from PyQt5.QtCore import Qt

from src.pipeline.preprocessing.loader import DataLoader as DataLoaderETL

LABELS = ["Clamping", "Bending", "Mandrel Extraction", "De-Clamping"]
LABEL_COLORS = {
    "Clamping": "#1f77b4",
    "Bending": "#ff7f0e",
    "Mandrel Extraction": "#2ca02c",
    "De-Clamping": "#d62728",
}


class Annotator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MAR Annotator")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: #2b2b2b; color: white;")

        self.df = None
        self.labels = []
        self.current_exp = None
        self.click_stage = "start"
        self.temp_label = {"start": None, "end": None, "label": None}
        self.legend_map = {}
        self.pick_cid = None
        self._setup_gui()

        loader = DataLoaderETL("data/processed/tube_geometry.db")
        dataframes = loader.load_all_data_from_sqlite()
        self.df = dataframes["machine_and_movement"]
        self.df = self.df.set_index(self.df.columns[0])
        all_experiments = list(self.df["Experiment_ID"].unique())
        self.exp_selector.clear()
        self.exp_selector.addItems([str(e) for e in all_experiments])
        self.exp_selector.setCurrentIndex(0)
        self._plot_experiment()
        self._set_status(f"Data loaded with {len(all_experiments)} experiments.")

    def _setup_gui(self):
        main_layout = QVBoxLayout(self)  

        top_layout = QHBoxLayout()

        plot_frame = QVBoxLayout()
        self.figure = plt.Figure(figsize=(8, 6))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        plot_frame.addWidget(self.canvas)
        self.hide_btn = QPushButton("Hide All Lines")
        self.hide_btn.clicked.connect(self._toggle_all_lines)
        plot_frame.addWidget(self.hide_btn)
        top_layout.addLayout(plot_frame, 4)

        control_frame = QVBoxLayout()

        self.exp_selector = QComboBox()
        self.exp_selector.currentIndexChanged.connect(self._plot_experiment)
        control_frame.addWidget(self.exp_selector)

        self.label_selector = QComboBox()
        self.label_selector.addItems(LABELS)
        control_frame.addWidget(self.label_selector)

        self.delete_btn = QPushButton("Delete Annotation")
        self.delete_btn.clicked.connect(self._delete_annotation)
        control_frame.addWidget(self.delete_btn)

        self.save_json_btn = QPushButton("Save JSON")
        self.save_json_btn.clicked.connect(self._save_json)
        control_frame.addWidget(self.save_json_btn)

        self.load_json_btn = QPushButton("Load JSON")
        self.load_json_btn.clicked.connect(self._load_json)
        control_frame.addWidget(self.load_json_btn)

        instr_group = QGroupBox("Keyboard Shortcuts")
        instr_layout = QVBoxLayout()
        instr_label = QLabel(
            "Keyboard Controls:\n"
            "Spacebar: Next experiment\n"
            "Left Alt: Next label type"
        )
        instr_label.setWordWrap(True)
        instr_layout.addWidget(instr_label)
        instr_group.setLayout(instr_layout)
        control_frame.addWidget(instr_group)
        control_frame.addStretch()

        top_layout.addLayout(control_frame, 1)

        main_layout.addLayout(top_layout)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("background-color: #1f1f1f; padding: 4px;")
        main_layout.addWidget(self.status_label)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self._next_experiment()
        elif event.key() == Qt.Key_Alt:
            self._next_label()

    def _set_status(self, message):
        self.status_label.setText(message)

    def _plot_experiment(self):
        if self.df is None or self.exp_selector.currentText() == "":
            return
        self.current_exp = int(self.exp_selector.currentText())
        exp_df = self.df[self.df["Experiment_ID"] == self.current_exp]
        self.click_stage = "start"
        self.temp_label = {"start": None, "end": None, "label": None}
        self.ax.clear()
        self.lines_by_col = {}
        lines = []
        for col in exp_df.columns:
            if col != "Experiment_ID":
                (line,) = self.ax.plot(exp_df[col], label=col)
                self.lines_by_col[col] = line
                lines.append(line)
        for lbl in self.labels:
            if lbl["Experiment_ID"] == self.current_exp:
                start, end = lbl["start"], lbl["end"]
                color = LABEL_COLORS.get(lbl["label"], "#888888")
                self.ax.axvspan(start, end, color=color, alpha=0.3, zorder=0)
                y_pos = exp_df.max().max() * 1.05
                self.ax.text(
                    (start + end) / 2,
                    y_pos,
                    f"{lbl['label']}\n({lbl['duration']:.2f})",
                    color=color,
                    ha="center",
                    va="bottom",
                )
        self.ax.set_title(f"Experiment {self.current_exp}")
        self.ax.set_xlabel("Time Step / Index")
        self.ax.set_ylabel("Values")
        self._draw_legends(lines)
        self.canvas.draw_idle()

    def _draw_legends(self, lines):
        """Draw the legends by selecting them"""
        sensor_handles, sensor_labels = self.ax.get_legend_handles_labels()
        annotation_handles = [
            mpatches.Patch(color=color, alpha=0.3, label=lbl)
            for lbl, color in LABEL_COLORS.items()
        ]

        handles = sensor_handles + annotation_handles
        labels = sensor_labels + list(LABEL_COLORS.keys())
        num_items = len(labels)
        ncol = (num_items + 2) // 5

        self.figure.subplots_adjust(
            bottom=0.25
        )  
        leg = self.ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=ncol,
            frameon=True,
            fontsize=8,
        )

        self.legend_map.clear()
        if self.pick_cid:
            self.figure.canvas.mpl_disconnect(self.pick_cid)
            self.pick_cid = None

        leg_lines = list(leg.get_lines())
        for legline, origline in zip(leg_lines, lines):
            legline.set_picker(5)
            legline.set_alpha(1.0 if origline.get_visible() else 0.3)
            self.legend_map[legline] = origline

        self.pick_cid = self.figure.canvas.mpl_connect(
            "pick_event", self.on_pick_legend
        )

    def on_pick_legend(self, event):
        """Toggle original line visibility when legend line is clicked."""
        origline = self.legend_map.get(event.artist)
        if origline is None:
            return
        visible = not origline.get_visible()
        origline.set_visible(visible)
        event.artist.set_alpha(1.0 if visible else 0.2)
        self.canvas.draw_idle()

    def on_click(self, event):
        if self.current_exp is None or event.xdata is None:
            return
        if self.click_stage == "start":
            self.temp_label["start"] = event.xdata
            self.temp_label["label"] = self.label_selector.currentText()
            self.click_stage = "end"
            self._set_status(f"Start {self.temp_label['label']} at {event.xdata:.2f}")
        else:
            self.temp_label["end"] = event.xdata
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
            self._set_status(
                f"End label at {event.xdata:.2f} (duration: {duration:.2f})"
            )
            self._plot_experiment()

    def _toggle_all_lines(self):
        if not hasattr(self, "lines_by_col"):
            return
        any_visible = any(line.get_visible() for line in self.lines_by_col.values())
        for line in self.lines_by_col.values():
            line.set_visible(not any_visible)
        self.canvas.draw_idle()

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
                self.labels = []
                for exp_id, lbl_list in loaded.items():
                    for lbl in lbl_list:
                        start, end = lbl["start"], lbl["end"]
                        duration = lbl.get("duration", end - start)
                        self.labels.append(
                            {
                                "Experiment_ID": int(exp_id),
                                "label": lbl["label"],
                                "start": start,
                                "end": end,
                                "duration": duration,
                            }
                        )
                self._set_status(f"Loaded {len(self.labels)} annotations from JSON.")
                self._plot_experiment()
            else:
                QMessageBox.critical(self, "Error", "Invalid JSON format!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load JSON: {e}")

    def _delete_annotation(self):
        if self.current_exp is None:
            QMessageBox.warning(self, "No Experiment", "Select an experiment first!")
            return
        exp_labels = [
            lbl for lbl in self.labels if lbl["Experiment_ID"] == self.current_exp
        ]
        if not exp_labels:
            QMessageBox.information(
                self, "No Annotations", "No annotations to delete for this experiment."
            )
            return
        dlg = QListWidget()
        for i, lbl in enumerate(exp_labels):
            duration = lbl.get("duration", lbl["end"] - lbl["start"])
            dlg.addItem(
                f"{i}: {lbl['label']} [{lbl['start']:.2f}, {lbl['end']:.2f}] (duration: {duration:.2f})"
            )
        dlg.setWindowTitle("Delete Annotation")
        dlg.show()
        dlg.itemDoubleClicked.connect(
            lambda item: self._remove_selected(item, dlg, exp_labels)
        )

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
