import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import tkinter.font as tkFont
import json
import os

LABELS = ["Clamping", "Bending", "Mandrel Extraction", "De-Clamping"]
LABEL_COLORS = {
    "Clamping":"#1f77b4",
    "Bending":"#ff7f0e",
    "Mandrel Extraction":"#2ca02c",
    "De-Clamping":"#d62728"
}

class ExperimentAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Experiment Annotator")
        self.root.geometry("1200x800")

        # Make button font a bit smaller globally
        default_font = tkFont.nametofont("TkDefaultFont")
        default_font.configure(size=9)

        self.df = None
        self.labels = []
        self.current_exp = None
        self.click_stage = "start"
        self.temp_label = {"start": None, "end": None, "label": None}

        # Legend pick mapping and pick connection id
        self.legend_map = {}   # maps legend artist -> original line
        self.pick_cid = None   # connection id for pick_event

        # GUI widgets
        top_frame = tk.Frame(root)
        top_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        tk.Button(top_frame, text="Load CSV", command=self.load_csv).pack(side=tk.LEFT, padx=5)
        self.exp_selector = ttk.Combobox(top_frame, state="readonly")
        self.exp_selector.pack(side=tk.LEFT, padx=5)
        self.exp_selector.bind("<<ComboboxSelected>>", self.plot_experiment)

        self.label_var = tk.StringVar()
        self.label_var.set(LABELS[0])
        tk.OptionMenu(top_frame, self.label_var, *LABELS).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Delete Annotation", command=self.delete_annotation).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Save JSON", command=self.save_json).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Load JSON", command=self.load_json).pack(side=tk.LEFT, padx=5)

        self.figure = plt.Figure(figsize=(10,6))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_click)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        self.df = pd.read_csv(file_path, index_col=0)
        self.exp_selector['values'] = list(self.df['Experiment_ID'].unique())
        messagebox.showinfo("Loaded", f"CSV loaded with {len(self.df['Experiment_ID'].unique())} experiments.")

    def load_json(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not file_path:
            return
        try:
            with open(file_path, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    if "labels" in loaded and isinstance(loaded["labels"], list):
                        self.labels = loaded["labels"]
                    else:
                        messagebox.showerror("Error", "JSON format invalid: expected a list of labels")
                        return
                elif isinstance(loaded, list):
                    self.labels = loaded
                else:
                    messagebox.showerror("Error", "JSON format invalid: expected list or dict")
                    return
            messagebox.showinfo("Loaded", f"Loaded {len(self.labels)} annotations from JSON.")
            self.plot_experiment()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load JSON: {e}")

    def plot_experiment(self, event=None):
        exp_id = self.exp_selector.get()
        if not exp_id or self.df is None:
            return
        # Reset current selection state
        try:
            self.current_exp = int(exp_id)
        except Exception:
            self.current_exp = exp_id
        self.click_stage = "start"
        self.temp_label = {"start": None, "end": None, "label": None}

        exp_df = self.df[self.df['Experiment_ID'] == self.current_exp]
        if exp_df.empty:
            return

        # --- Save old visibility states ---
        old_vis = {}
        if hasattr(self, "lines_by_col"):
            for col, line in self.lines_by_col.items():
                old_vis[col] = line.get_visible()

        self.ax.clear()
        self.lines_by_col = {}

        # --- Replot signals, restore visibility ---
        lines = []
        for col in exp_df.columns:
            if col != 'Experiment_ID':
                line, = self.ax.plot(exp_df[col], label=col, picker=False)
                if col in old_vis:
                    line.set_visible(old_vis[col])   # restore old visibility
                self.lines_by_col[col] = line
                lines.append(line)

        # Draw annotation labels (shaded regions + text)
        for lbl in self.labels:
            if lbl["Experiment_ID"] == self.current_exp:
                start = lbl["start"]
                end = lbl["end"]
                color = LABEL_COLORS.get(lbl["label"], "#888888")
                self.ax.axvspan(start, end, color=color, alpha=0.3, zorder=0)
                # place text above the plotted lines
                y_pos = exp_df.max().max() * 1.05
                self.ax.text((start + end) / 2, y_pos, lbl["label"],
                             color=color, ha='center', va='bottom')

        self.ax.set_title(f"Experiment {self.current_exp}")
        self.ax.set_xlabel("Time Step / Index")
        self.ax.set_ylabel("Values")

        # SENSOR legend entries (line handles)
        sensor_handles, sensor_labels = self.ax.get_legend_handles_labels()

        # ANNOTATION legend entries (patch handles)
        annotation_handles = []
        annotation_labels = []
        for lbl_name, color in LABEL_COLORS.items():
            patch = mpatches.Patch(color=color, alpha=0.3, label=lbl_name)
            annotation_handles.append(patch)
            annotation_labels.append(lbl_name)

        # Combine sensors then annotation patches (sensors first so order is predictable)
        handles = sensor_handles + annotation_handles
        labels = sensor_labels + annotation_labels

        # Draw the legend outside the plot
        leg = self.ax.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, frameon=True)

        # -- Prepare interactive mapping for sensor legend lines only --
        # Clear any previous mapping and disconnect previous pick handler to avoid duplicates
        self.legend_map.clear()
        if self.pick_cid is not None:
            try:
                self.figure.canvas.mpl_disconnect(self.pick_cid)
            except Exception:
                pass
            self.pick_cid = None

        # leg.get_lines() returns legend line artists (for the line handles we provided)
        leg_lines = list(leg.get_lines())
        # If there are N sensor lines, leg_lines should be of length N (since annotation are patches)
        # Map these legend line-artist objects to the original plotted Line2D objects.
        for legline, origline in zip(leg_lines, lines):
            legline.set_picker(5)  # 5 points tolerance
            # set initial alpha to reflect visibility
            legline.set_alpha(1.0 if origline.get_visible() else 0.3)
            self.legend_map[legline] = origline

        # Connect a single pick_event handler
        self.pick_cid = self.figure.canvas.mpl_connect("pick_event", self.on_pick_legend)

        # Draw canvas
        self.canvas.draw_idle()

    def on_pick_legend(self, event):
        """Pick event handler for legend lines -> toggles corresponding original line."""
        artist = event.artist
        origline = self.legend_map.get(artist, None)
        if origline is None:
            return  # clicked something we don't manage (e.g. an annotation patch)
        # Toggle visibility
        new_vis = not origline.get_visible()
        origline.set_visible(new_vis)
        # Update legend artist alpha so user can see it's toggled
        artist.set_alpha(1.0 if new_vis else 0.2)
        # Also optionally dim the line itself when hidden (not necessary, set_visible handles it)
        # Redraw
        self.canvas.draw_idle()

    def on_click(self, event):
        if self.current_exp is None or event.xdata is None:
            return

        if self.click_stage == "start":
            self.temp_label["start"] = event.xdata
            self.temp_label["label"] = self.label_var.get()
            self.click_stage = "end"
            print(f"Start {self.temp_label['label']} at {event.xdata:.2f}")
        elif self.click_stage == "end":
            self.temp_label["end"] = event.xdata
            start, end = sorted([self.temp_label["start"], self.temp_label["end"]])
            self.labels.append({
                "Experiment_ID": self.current_exp,
                "label": self.temp_label["label"],
                "start": start,
                "end": end
            })
            self.click_stage = "start"
            self.temp_label = {"start": None, "end": None, "label": None}
            print(f"End label at {event.xdata:.2f}")
            self.plot_experiment()

    def delete_annotation(self):
        if self.current_exp is None:
            messagebox.showwarning("No Experiment", "Select an experiment first!")
            return
        exp_labels = [lbl for lbl in self.labels if lbl["Experiment_ID"] == self.current_exp]
        if not exp_labels:
            messagebox.showinfo("No Annotations", "No annotations to delete for this experiment.")
            return

        dlg = tk.Toplevel(self.root)
        dlg.title("Delete Annotation")
        tk.Label(dlg, text="Select annotation to delete:").pack()
        listbox = tk.Listbox(dlg, width=80)
        listbox.pack()
        for i, lbl in enumerate(exp_labels):
            listbox.insert(tk.END, f"{i}: {lbl['label']} [{lbl['start']:.2f}, {lbl['end']:.2f}]")

        def delete_selected():
            sel = listbox.curselection()
            if sel:
                idx = sel[0]
                actual = exp_labels[idx]
                self.labels.remove(actual)
                dlg.destroy()
                self.plot_experiment()

        tk.Button(dlg, text="Delete", command=delete_selected).pack(pady=5)
        tk.Button(dlg, text="Cancel", command=dlg.destroy).pack(pady=5)

    def save_json(self):
        if not self.labels:
            messagebox.showwarning("No Labels", "No labels to save!")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "w") as f:
                json.dump(self.labels, f, indent=4)
            messagebox.showinfo("Saved", f"Labels saved to {file_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ExperimentAnnotator(root)
    root.mainloop()
