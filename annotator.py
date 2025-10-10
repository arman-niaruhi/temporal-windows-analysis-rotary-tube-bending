import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import tkinter.font as tkFont

# Constants
LABELS = ["Clamping", "Bending", "Mandrel Extraction", "De-Clamping"]
LABEL_COLORS = {
    "Clamping": "#1f77b4",
    "Bending": "#ff7f0e",
    "Mandrel Extraction": "#2ca02c",
    "De-Clamping": "#d62728"
}

class Annotator:
    def __init__(self, root:tk.Tk) -> None:
        # Window setup and style
        self.root = root 
        self.root.title("MAR Annotator")
        self.root.geometry("1200x800")
        style = ttk.Style()
        style.theme_use("clam")

        default_font = tkFont.nametofont("TkDefaultFont")
        default_font.configure(family="Segoe UI", size=11)
        root.option_add("*Font", default_font)

        root.configure(bg="#2b2b2b")  # dark gray background
        style.configure("TFrame", background="#2b2b2b")
        style.configure("TLabel", background="#2b2b2b", foreground="white")
        style.configure("TCombobox", 
                        fieldbackground="#3c3c3c", 
                        background="#3c3c3c", 
                        foreground="white")
        style.configure("TButton", 
                        background="#4a4a4a", 
                        foreground="white", 
                        font=("Segoe UI", 11, "bold"), 
                        padding=6)
        style.map("TButton",
                background=[("active", "#5c5c5c")])

        self.root.option_add("*TButton*foreground", "white")
        self.root.option_add("*TButton*background", "#4a4a4a")
        self.root.option_add("*TLabel*foreground", "white")
        self.root.option_add("*TLabel*background", "#2b2b2b")
        self.root.option_add("*Entry*background", "#3c3c3c")
        self.root.option_add("*Entry*foreground", "white")
        self.root.option_add("*Listbox*background", "#3c3c3c")
        self.root.option_add("*Listbox*foreground", "white")
            
        # Define useful variables for funtions    
        self.df = None
        self.labels = []
        self.current_exp = None
        self.click_stage = "start"
        self.temp_label = {"start": None, "end": None, "label": None}
        
        self.legend_map = {} # Legend mapping for interactive toggling
        self.pick_cid = None
        
        # Update GUI and status-bar
        self._setup_gui()
        self._set_status("Ready")
    
    def _setup_gui(self):
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Right frame
        right_frame = tk.Frame(main_frame, width=250)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        tk.Button(right_frame, text="Load CSV", command=self._load_csv).pack(fill=tk.X, pady=2)
        self.exp_selector = ttk.Combobox(right_frame, state="readonly")
        self.exp_selector.pack(fill=tk.X, pady=2)
        self.exp_selector.bind("<<ComboboxSelected>>", self._plot_experiment)
        self.label_var = tk.StringVar(value=LABELS[0])
        tk.OptionMenu(right_frame, self.label_var, *LABELS).pack(fill=tk.X, pady=2)
        tk.Button(right_frame, text="Delete Annotation", command=self._delete_annotation).pack(fill=tk.X, pady=2)
        tk.Button(right_frame, text="Save JSON", command=self._save_json).pack(fill=tk.X, pady=2)
        tk.Button(right_frame, text="Load JSON", command=self._load_json).pack(fill=tk.X, pady=2)
        
        instruction_frame = tk.LabelFrame(right_frame, text="Keyboard Shortcuts", padx=10, pady=10)
        instruction_frame.pack(fill=tk.X, pady=10)
        instructions = (
            "Keyboard Controls:\n"
            "Spacebar:\nNext experiment\n"
            "Left Alt:\nNext label type\n"
        )
        tk.Label(instruction_frame, text=instructions, justify="left", wraplength=220).pack()

        # Left frame
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.figure = plt.Figure(figsize=(8, 6))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_click)

        self.root.bind("<space>", self._next_experiment)
        self.root.bind("<Alt_L>", self._next_label)

        # Status bar
        self.status_var = tk.StringVar()
        status_frame = tk.Frame(self.root, relief=tk.SUNKEN, bd=1)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_label = tk.Label(
            status_frame, textvariable=self.status_var, anchor="w", padx=10
        )
        self.status_label.pack(fill=tk.X)


        tk.Button(left_frame, text="Hide All Lines", command=self._toggle_all_lines).pack(side=tk.TOP, pady=5)

    def _set_status(self, message: str):
        """Update status bar text."""
        self.status_var.set(message)
        self.root.update_idletasks()
   
    def _plot_experiment(self, event= None):
        """Update the experiment plot"""
        exp_id = self.exp_selector.get()
        try:
            self.current_exp = int(exp_id)
        except ValueError:
            self.current_exp = exp_id

        self.click_stage = "start"
        self.temp_label = {"start": None, "end": None, "label": None}

        exp_df = self.df[self.df['Experiment_ID'] == self.current_exp]
        self._set_status(f"{self.current_exp} Experiment is selected")
        if exp_df.empty:
            return

        # Preserve old visibility
        old_vis = getattr(self, "lines_by_col", {})

        self.ax.clear()
        self.lines_by_col = {}

        # Plot all signals
        lines = []
        for col in exp_df.columns:
            if col != 'Experiment_ID':
                line, = self.ax.plot(exp_df[col], label=col)
                line.set_visible(old_vis.get(col, line).get_visible() if col in old_vis else True)
                self.lines_by_col[col] = line
                lines.append(line)

        # Draw existing annotations
        for lbl in self.labels:
            if lbl["Experiment_ID"] == self.current_exp:
                start, end = lbl["start"], lbl["end"]
                color = LABEL_COLORS.get(lbl["label"], "#888888")
                self.ax.axvspan(start, end, color=color, alpha=0.3, zorder=0)
                y_pos = exp_df.max().max() * 1.05
                self.ax.text((start + end) / 2, y_pos, f"{lbl['label']}\n({lbl['duration']:.2f})", color=color,
                             ha='center', va='bottom')

        # Axes
        self.ax.set_title(f"Experiment {self.current_exp}")
        self.ax.set_xlabel("Time Step / Index")
        self.ax.set_ylabel("Values")

        # Legends
        self._draw_legends(lines)

        # Redraw
        self.canvas.draw_idle()
    
    def _draw_legends(self, lines):
        """Draw the legends by selecting them"""
        sensor_handles, sensor_labels = self.ax.get_legend_handles_labels()
        annotation_handles = [mpatches.Patch(color=color, alpha=0.3, label=lbl) for lbl, color in LABEL_COLORS.items()]

        handles = sensor_handles + annotation_handles
        labels = sensor_labels + list(LABEL_COLORS.keys())
        num_items = len(labels)
        ncol = (num_items + 2) // 5
        
        self.figure.subplots_adjust(bottom=0.25)  # leave more space for multi-line legend
        leg = self.ax.legend(
            handles, labels,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.2),
            ncol=ncol,
            frameon=True,
            fontsize=8
        )
        
        # Connect pick events for interactivity
        self.legend_map.clear()
        if self.pick_cid:
            self.figure.canvas.mpl_disconnect(self.pick_cid)
            self.pick_cid = None

        leg_lines = list(leg.get_lines())
        for legline, origline in zip(leg_lines, lines):
            legline.set_picker(5)
            legline.set_alpha(1.0 if origline.get_visible() else 0.3)
            self.legend_map[legline] = origline

        self.pick_cid = self.figure.canvas.mpl_connect("pick_event", self.on_pick_legend)
    
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
            self.temp_label["label"] = self.label_var.get()
            self.click_stage = "end"
            self._set_status(f"Start {self.temp_label['label']} at {event.xdata:.2f}")
        else:
            self.temp_label["end"] = event.xdata
            start, end = sorted([self.temp_label["start"], self.temp_label["end"]])
            duration = end - start
            self.labels.append({
                "Experiment_ID": self.current_exp,
                "label": self.temp_label["label"],
                "start": start,
                "end": end,
                "duration": duration
            })
            self.click_stage = "start"
            self.temp_label = {"start": None, "end": None, "label": None}
            self._set_status(f"End label at {event.xdata:.2f} (duration: {duration:.2f})")
            self._plot_experiment()
  
    def _toggle_all_lines(self):
        """Toggle visibility of all plotted lines."""
        if not hasattr(self, 'lines_by_col'):
            return
        # Check if at least one line is visible
        any_visible = any(line.get_visible() for line in self.lines_by_col.values())
        for line in self.lines_by_col.values():
            line.set_visible(not any_visible)  # hide if any visible, else show all
        # Update legend alpha accordingly
        for legline, origline in self.legend_map.items():
            legline.set_alpha(1.0 if origline.get_visible() else 0.2)
        self.canvas.draw_idle()

    def _delete_annotation(self):
        if self.current_exp is None:
            messagebox.showwarning("No Experiment", "Select an experiment first!")
            return

        exp_labels = [lbl for lbl in self.labels if lbl["Experiment_ID"] == self.current_exp]
        if not exp_labels:
            messagebox.showinfo("No Annotations", "No annotations to delete for this experiment.")
            return

        dlg = tk.Toplevel(self.root)
        dlg.title("Delete Annotation")
        tk.Label(dlg, text="Select annotation to delete:").pack(pady=5)

        listbox = tk.Listbox(dlg, width=80)
        listbox.pack(padx=5, pady=5)
        for i, lbl in enumerate(exp_labels):
            duration = lbl.get("duration", lbl["end"] - lbl["start"])
            listbox.insert(tk.END, f"{i}: {lbl['label']} [{lbl['start']:.2f}, {lbl['end']:.2f}] (duration: {duration:.2f})")

        def delete_selected():
            sel = listbox.curselection()
            if sel:
                self.labels.remove(exp_labels[sel[0]])
                dlg.destroy()
                self._plot_experiment()

        tk.Button(dlg, text="Delete", command=delete_selected).pack(side=tk.LEFT, padx=10, pady=5)
        tk.Button(dlg, text="Cancel", command=dlg.destroy).pack(side=tk.RIGHT, padx=10, pady=5)
    
    def _save_json(self):
        if not self.labels:
            messagebox.showwarning("No Labels", "No labels to save!")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            grouped = {}
            for lbl in self.labels:
                exp_id = str(lbl["Experiment_ID"])
                grouped.setdefault(exp_id, []).append({
                    "label": lbl["label"],
                    "start": lbl["start"],
                    "end": lbl["end"],
                    "duration": lbl.get("duration", lbl["end"] - lbl["start"])
                })
            with open(file_path, "w") as f:
                json.dump(grouped, f, indent=4)
            self._set_status(f"Labels saved to {file_path}")
    
    def _load_json(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not file_path:
            return

        try:
            with open(file_path, "r") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                self.labels = []
                for exp_id, lbl_list in loaded.items():
                    for lbl in lbl_list:
                        start, end = lbl["start"], lbl["end"]
                        duration = lbl.get("duration", end - start)
                        self.labels.append({
                            "Experiment_ID": int(exp_id),
                            "label": lbl["label"],
                            "start": start,
                            "end": end,
                            "duration": duration
                        })
                self._set_status(f"Loaded {len(self.labels)} annotations from JSON.")
                self._plot_experiment()
            else:
                messagebox.showerror("Error", "JSON format invalid: expected dict of experiments")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load JSON: {e}")
    
    def _load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        self.df = pd.read_csv(file_path, index_col=0)
        all_experiment_ids = list(self.df['Experiment_ID'].unique())
        self.exp_selector['values'] = all_experiment_ids
        self.exp_selector.set(all_experiment_ids[0])
        self._plot_experiment()
        message = f"CSV loaded with {len(self.df['Experiment_ID'].unique())} experiments."
        self._set_status(message=message)

    def _next_experiment(self, event=None):
        """Go to the next experiment using spacebar."""
        if not self.df is None and len(self.exp_selector['values']) > 0:
            current = self.exp_selector.get()
            values = list(self.exp_selector['values'])
            if current in values:
                idx = values.index(current)
                next_idx = (idx + 1) % len(values)
            else:
                next_idx = 0
            self.exp_selector.set(values[next_idx])
            self._plot_experiment()
    
    def _next_label(self, event=None):
        """Cycle to the next label type using Left Alt."""
        current_label = self.label_var.get()
        idx = LABELS.index(current_label)
        next_idx = (idx + 1) % len(LABELS)
        self.label_var.set(LABELS[next_idx])
        self._set_status(f"Current Label: {LABELS[next_idx]}")
    
if __name__ == "__main__":
    root = tk.Tk()
    app = Annotator(root)
    root.mainloop()