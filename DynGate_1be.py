import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Rectangle
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from dataclasses import dataclass
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import FlowCytometryTools as fct
    HAS_FCT = True
except ImportError:
    HAS_FCT = False

try:
    import fcsparser
    HAS_FCSPARSER = True
except ImportError:
    HAS_FCSPARSER = False


@dataclass
class TimePoint:
    time: float
    filename: str
    data: pd.DataFrame


class KolmogorovSmirnovAnalyzer:
    @staticmethod
    def two_sample_ks_statistic(sample1, sample2):
        n, m = len(sample1), len(sample2)
        if n == 0 or m == 0:
            return 1.0, 0, 0, 0, 0
        combined = np.sort(np.concatenate([sample1, sample2]))
        cdf1 = np.searchsorted(np.sort(sample1), combined, side='right') / n
        cdf2 = np.searchsorted(np.sort(sample2), combined, side='right') / m
        differences = np.abs(cdf1 - cdf2)
        D = np.max(differences)
        return D, 0, 0, 0, 0


class ExponentialKinetics:
    """Fit exponential kinetics: A + B*(1 - exp(-t/tau))"""
    
    @staticmethod
    def model(t, A, B, tau_inv):
        """A + B * (1 - exp(-tau_inv * t))"""
        return A + B * (1 - np.exp(-tau_inv * t))
    
    @staticmethod
    def fit(times, values, sigmas=None):
        if len(times) < 3:
            return np.array([0, 0, 0.01]), np.array([0, 0, 0])
        
        A_init = np.min(values) * 0.9 if np.min(values) > 0 else np.min(values) * 1.1
        B_init = np.max(values) - A_init
        if B_init <= 0:
            B_init = abs(np.max(values) - np.min(values))
        tau_init = 1.0 / (np.max(times) * 0.3) if np.max(times) > 0 else 0.05
        
        try:
            if sigmas is not None and not np.any(np.isnan(sigmas)) and np.all(sigmas > 0):
                popt, pcov = curve_fit(
                    ExponentialKinetics.model, times, values,
                    p0=[A_init, B_init, tau_init],
                    sigma=sigmas,
                    bounds=([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf]),
                    maxfev=10000
                )
            else:
                popt, pcov = curve_fit(
                    ExponentialKinetics.model, times, values,
                    p0=[A_init, B_init, tau_init],
                    bounds=([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf]),
                    maxfev=10000
                )
            perr = np.sqrt(np.diag(pcov))
            return popt, perr
        except:
            try:
                popt, pcov = curve_fit(
                    ExponentialKinetics.model, times, values,
                    p0=[np.min(values), np.max(values)-np.min(values), 0.01],
                    bounds=([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf]),
                    maxfev=10000
                )
                perr = np.sqrt(np.diag(pcov))
                return popt, perr
            except:
                return np.array([A_init, B_init, tau_init]), np.array([0, 0, 0])


class LMDReader:
    @staticmethod
    def read_lmd_file(filename):
        if HAS_FCSPARSER:
            try:
                meta, data = fcsparser.parse(filename, reformat_meta=True)
                return data
            except:
                pass
        if HAS_FCT:
            try:
                sample = fct.FCMeasurement(ID='test', datafile=filename)
                return sample.data
            except:
                pass
        try:
            return LMDReader._read_fcs_basic(filename)
        except:
            raise ValueError(f"Could not read file: {filename}")
    
    @staticmethod
    def _read_fcs_basic(filename):
        with open(filename, 'rb') as f:
            header = f.read(58)
            text_start = int(header[10:18].decode().strip())
            text_end = int(header[18:26].decode().strip())
            f.seek(text_start)
            text_segment = f.read(text_end - text_start).decode()
            data_start = int(header[26:34].decode().strip())
            data_end = int(header[34:42].decode().strip())
            params = {}
            for line in text_segment.split('/'):
                if '$P' in line and 'N' in line:
                    parts = line.split('$')
                    for part in parts:
                        if 'P' in part and 'N' in part:
                            key, value = part.split('N', 1)
                            params[key] = value.strip()
            f.seek(data_start)
            n_params = int(params.get('$PAR', '1'))
            n_events = (data_end - data_start) // (4 * n_params)
            dtype = np.float32
            raw_data = np.fromfile(f, dtype=dtype, count=n_events * n_params)
            data = raw_data.reshape(n_events, n_params)
            columns = [params.get(f'$P{i}N', f'Channel_{i}') for i in range(1, n_params+1)]
            df = pd.DataFrame(data, columns=columns)
            return df


class DynamicGatingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Dynamic Gating - 2D Cytogram with Kinetics Fit")
        self.root.geometry("1400x950")
        self.time_points: List[TimePoint] = []
        self.results = {}
        self.reference_gate = None
        self.gate_stats = {}
        self.setup_gui()
        
    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        left_panel = ttk.Frame(main_frame, padding="5")
        left_panel.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        row = 0
        ttk.Label(left_panel, text="File Loading", font=('Arial', 12, 'bold')).grid(row=row, column=0, pady=5); row += 1
        ttk.Button(left_panel, text="Add Time Point Files", command=self.add_time_points).grid(row=row, column=0, pady=5); row += 1
        ttk.Label(left_panel, text="Loaded Time Points:").grid(row=row, column=0, pady=5); row += 1
        self.time_listbox = tk.Listbox(left_panel, width=45, height=4)
        self.time_listbox.grid(row=row, column=0, pady=5); row += 1
        ttk.Button(left_panel, text="Remove Selected", command=self.remove_time_point).grid(row=row, column=0, pady=5); row += 1
        
        ttk.Label(left_panel, text="Channel Selection", font=('Arial', 12, 'bold')).grid(row=row, column=0, pady=5); row += 1
        ttk.Label(left_panel, text="Y-axis channel (FIXED gate):").grid(row=row, column=0, pady=2); row += 1
        self.y_channel_var = tk.StringVar(value="FSC-A")
        self.y_channel_combo = ttk.Combobox(left_panel, textvariable=self.y_channel_var, width=42)
        self.y_channel_combo.grid(row=row, column=0, pady=2); row += 1
        ttk.Label(left_panel, text="X-axis channel (OPTIMIZED position):").grid(row=row, column=0, pady=2); row += 1
        self.x_channel_var = tk.StringVar(value="FL1-A")
        self.x_channel_combo = ttk.Combobox(left_panel, textvariable=self.x_channel_var, width=42)
        self.x_channel_combo.grid(row=row, column=0, pady=2); row += 1
        
        ttk.Label(left_panel, text="Gate Settings", font=('Arial', 12, 'bold')).grid(row=row, column=0, pady=5); row += 1
        ttk.Label(left_panel, text="Reference time point:").grid(row=row, column=0, pady=2); row += 1
        rf = ttk.Frame(left_panel)
        rf.grid(row=row, column=0, pady=2); row += 1
        self.ref_index_var = tk.StringVar(value="last")
        self.ref_index_combo = ttk.Combobox(rf, textvariable=self.ref_index_var, width=35, state='readonly')
        self.ref_index_combo.grid(row=0, column=0, padx=2)
        self.ref_index_info = ttk.Label(left_panel, text="", foreground="gray")
        self.ref_index_info.grid(row=row, column=0, pady=2); row += 1
        
        ttk.Button(left_panel, text="Set Reference Gate (2D cytogram)", 
                  command=self.open_manual_gating).grid(row=row, column=0, pady=5); row += 1
        self.gate_info_label = ttk.Label(left_panel, text="Reference gate: not set", foreground="red", wraplength=380)
        self.gate_info_label.grid(row=row, column=0, pady=2); row += 1
        
        ttk.Label(left_panel, text="X-axis Search Parameters", font=('Arial', 12, 'bold')).grid(row=row, column=0, pady=5); row += 1
        ttk.Label(left_panel, text="X search range (± units):").grid(row=row, column=0, pady=2); row += 1
        sf = ttk.Frame(left_panel)
        sf.grid(row=row, column=0, pady=2); row += 1
        self.search_range_var = tk.StringVar(value="100000")
        ttk.Label(sf, text="±").grid(row=0, column=0)
        ttk.Entry(sf, textvariable=self.search_range_var, width=10).grid(row=0, column=1)
        ttk.Label(sf, text="units").grid(row=0, column=2)
        
        ttk.Label(left_panel, text="Number of search steps:").grid(row=row, column=0, pady=2); row += 1
        self.n_steps_var = tk.StringVar(value="200")
        ttk.Entry(left_panel, textvariable=self.n_steps_var, width=10).grid(row=row, column=0); row += 1
        self.step_size_label = ttk.Label(left_panel, text="Step size: --")
        self.step_size_label.grid(row=row, column=0, pady=2); row += 1
        
        ttk.Label(left_panel, text="Optimization Criteria:", font=('Arial', 10, 'bold')).grid(row=row, column=0, pady=5); row += 1
        self.use_fraction_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left_panel, text="Match gate fraction", variable=self.use_fraction_var,
                       command=self.on_fraction_checkbox_changed).grid(row=row, column=0, pady=2); row += 1
        ttk.Label(left_panel, text="Fraction penalty weight:").grid(row=row, column=0, pady=2); row += 1
        self.fraction_weight_var = tk.StringVar(value="10.0")
        self.fraction_weight_entry = ttk.Entry(left_panel, textvariable=self.fraction_weight_var, width=10, state='disabled')
        self.fraction_weight_entry.grid(row=row, column=0); row += 1
        
        ttk.Label(left_panel, text="Kinetics Fitting:", font=('Arial', 10, 'bold')).grid(row=row, column=0, pady=5); row += 1
        self.fit_mean_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left_panel, text="Fit Mean X: A + B*(1 - exp(-t/τ))", 
                       variable=self.fit_mean_var).grid(row=row, column=0, pady=2); row += 1
        self.fit_position_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left_panel, text="Fit Gate Position: A + B*(1 - exp(-t/τ))", 
                       variable=self.fit_position_var).grid(row=row, column=0, pady=2); row += 1
        
        ttk.Button(left_panel, text="Run Dynamic Gating Analysis", 
                  command=self.run_analysis, style='Large.TButton').grid(row=row, column=0, pady=20); row += 1
        ttk.Button(left_panel, text="Export Results", command=self.export_results).grid(row=row, column=0, pady=5); row += 1
        
        ttk.Label(left_panel, text="Gate Statistics", font=('Arial', 10, 'bold')).grid(row=row, column=0, pady=5); row += 1
        self.gate_stats_text = tk.Text(left_panel, height=10, width=45)
        self.gate_stats_text.grid(row=row, column=0, pady=5); row += 1
        
        right_panel = ttk.Frame(main_frame, padding="5")
        right_panel.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=1)
        
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.plot_tabs = {}
        for name in ["2D Cytograms", "Histograms (X)", "CDF Shape Comparison", 
                     "Gate Search", "Kinetics (Mean X)", "Kinetics (Gate Position)", "Results Table"]:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=name)
            self.plot_tabs[name] = frame
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN).grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))
    
    def on_fraction_checkbox_changed(self):
        if self.use_fraction_var.get():
            self.fraction_weight_entry.config(state='normal')
        else:
            self.fraction_weight_entry.config(state='disabled')
    
    def update_channel_combos(self):
        if not self.time_points: return
        tp = self.time_points[0]
        if isinstance(tp.data, pd.DataFrame):
            channels = list(tp.data.columns)
            self.x_channel_combo['values'] = channels
            self.y_channel_combo['values'] = channels
    
    def update_ref_index_combo(self):
        if not self.time_points:
            self.ref_index_combo['values'] = []
            return
        sorted_points = sorted(self.time_points, key=lambda x: x.time)
        values = [f"{i+1}: t={tp.time:.1f} min" for i, tp in enumerate(sorted_points)]
        values.append("last")
        self.ref_index_combo['values'] = values
    
    def get_reference_time_point(self):
        if not self.time_points: return None, None
        sorted_points = sorted(self.time_points, key=lambda x: x.time)
        sel = self.ref_index_var.get()
        if sel == "last": idx = len(sorted_points) - 1
        else:
            try: idx = int(sel.split(":")[0]) - 1
            except: idx = len(sorted_points) - 1
        return idx, sorted_points[idx]
    
    def add_time_points(self):
        files = filedialog.askopenfilenames(
            title="Select LMD/FCS files",
            filetypes=[("Flow cytometry files", "*.lmd *.fcs"), ("All files", "*.*")]
        )
        if not files: return
        td = tk.Toplevel(self.root)
        td.title("Enter Time Points"); td.geometry("400x300")
        ttk.Label(td, text="Enter time (minutes) for each file:").grid(row=0, column=0, columnspan=2, pady=10)
        entries = []
        for i, f in enumerate(files):
            ttk.Label(td, text=os.path.basename(f)[:30]+"...").grid(row=i+1, column=0, padx=5, pady=2)
            tv = tk.StringVar(value=str(i*5))
            ttk.Entry(td, textvariable=tv, width=10).grid(row=i+1, column=1, padx=5, pady=2)
            entries.append((f, tv))
        def load():
            for f, tv in entries:
                try:
                    t = float(tv.get())
                    self.status_var.set(f"Loading {os.path.basename(f)}..."); self.root.update()
                    data = LMDReader.read_lmd_file(f)
                    self.time_points.append(TimePoint(time=t, filename=f, data=data))
                    self.time_listbox.insert(tk.END, f"t={t:.1f} min - {os.path.basename(f)}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed: {e}")
            self.status_var.set(f"Loaded {len(self.time_points)} points")
            self.update_channel_combos(); self.update_ref_index_combo()
            td.destroy()
        ttk.Button(td, text="Load Files", command=load).grid(row=len(files)+1, column=0, columnspan=2, pady=20)
    
    def remove_time_point(self):
        sel = self.time_listbox.curselection()
        if sel:
            self.time_listbox.delete(sel[0])
            del self.time_points[sel[0]]
            self.update_ref_index_combo()
            self.status_var.set(f"Removed. {len(self.time_points)} left.")
    
    def get_channel_data(self, tp, channel):
        if isinstance(tp.data, pd.DataFrame):
            data = tp.data[channel].values.copy() if channel in tp.data.columns else tp.data.iloc[:, 0].values.copy()
        else:
            data = tp.data.flatten().copy()
        data = data[~np.isnan(data)]
        return data
    
    def open_manual_gating(self):
        if len(self.time_points) < 1:
            messagebox.showerror("Error", "Need at least 1 time point"); return
        ref_idx, ref_tp = self.get_reference_time_point()
        if ref_tp is None: return
        
        x_ch = self.x_channel_var.get()
        y_ch = self.y_channel_var.get()
        
        x_data = self.get_channel_data(ref_tp, x_ch)
        y_data = self.get_channel_data(ref_tp, y_ch)
        ml = min(len(x_data), len(y_data))
        x_data, y_data = x_data[:ml], y_data[:ml]
        total_n = ml
        
        gw = tk.Toplevel(self.root)
        gw.title(f"Set 2D Gate - t={ref_tp.time:.1f} min"); gw.geometry("1000x750")
        
        ttk.Label(gw, text=f"SET 2D GATE (t={ref_tp.time:.1f} min, #{ref_idx+1})",
                 font=('Arial', 14, 'bold'), foreground='red').pack(pady=5)
        ttk.Label(gw, text=f"Y-axis: {y_ch} → FIXED | X-axis: {x_ch} → OPTIMIZED",
                 font=('Arial', 11, 'bold')).pack()
        ttk.Label(gw, text="Click and drag to select rectangular gate region",
                 font=('Arial', 10)).pack()
        
        pf = ttk.Frame(gw); pf.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(x_data, y_data, alpha=0.3, s=2, c='blue')
        ax.set_title(f"2D Cytogram - t={ref_tp.time:.1f} min (N={total_n})", fontsize=14, fontweight='bold')
        ax.set_xlabel(x_ch); ax.set_ylabel(y_ch); ax.grid(True, alpha=0.3)
        
        rect = Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='red', facecolor='red', alpha=0.2, visible=False)
        ax.add_patch(rect)
        gl = {'x_min': None, 'x_max': None, 'y_min': None, 'y_max': None}
        pd_data = {'p': False, 'xs': None, 'ys': None}
        
        def on_press(event):
            if event.inaxes != ax: return
            pd_data['p'] = True; pd_data['xs'] = event.xdata; pd_data['ys'] = event.ydata
        
        def on_motion(event):
            if not pd_data['p'] or event.inaxes != ax: return
            if pd_data['xs'] is not None:
                x1, x2 = pd_data['xs'], event.xdata
                y1, y2 = pd_data['ys'], event.ydata
                rect.set_visible(True)
                rect.set_bounds(min(x1,x2), min(y1,y2), abs(x2-x1), abs(y2-y1))
                canvas.draw()
        
        def on_release(event):
            if not pd_data['p']: return
            if event.inaxes == ax:
                x1, x2 = pd_data['xs'], event.xdata
                y1, y2 = pd_data['ys'], event.ydata
                gl['x_min'], gl['x_max'] = min(x1,x2), max(x1,x2)
                gl['y_min'], gl['y_max'] = min(y1,y2), max(y1,y2)
                rect.set_visible(True)
                rect.set_bounds(gl['x_min'], gl['y_min'], gl['x_max']-gl['x_min'], gl['y_max']-gl['y_min'])
                canvas.draw()
                mask = (x_data >= gl['x_min']) & (x_data <= gl['x_max']) & (y_data >= gl['y_min']) & (y_data <= gl['y_max'])
                gx = x_data[mask]
                for t in ax.texts[:]: t.remove()
                if len(gx) > 0:
                    frac = len(gx)/total_n*100
                    pos = (gl['x_min'] + gl['x_max']) / 2
                    txt = f"Y-gate (FIXED): [{gl['y_min']:.1f}, {gl['y_max']:.1f}]\n"
                    txt += f"X-gate: [{gl['x_min']:.1f}, {gl['x_max']:.1f}]\n"
                    txt += f"Position: {pos:.1f}\n"
                    txt += f"Gated: {len(gx)}/{total_n} ({frac:.1f}%)\n"
                    txt += f"X mean: {np.mean(gx):.1f}, sigma: {np.std(gx):.1f}"
                    ax.text(0.02, 0.98, txt, transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=9)
                canvas.draw()
            pd_data['p'] = False; pd_data['xs'] = None; pd_data['ys'] = None
        
        canvas = FigureCanvasTkAgg(fig, pf)
        canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.mpl_connect('button_press_event', on_press)
        canvas.mpl_connect('motion_notify_event', on_motion)
        canvas.mpl_connect('button_release_event', on_release)
        NavigationToolbar2Tk(canvas, pf).update()
        
        bf = ttk.Frame(gw); bf.pack(pady=10)
        def save():
            if gl['x_min'] is None: messagebox.showerror("Error", "Set gate first"); return
            self.reference_gate = {
                'time': ref_tp.time, 'x_min': gl['x_min'], 'x_max': gl['x_max'],
                'y_min': gl['y_min'], 'y_max': gl['y_max'],
                'total_n': total_n, 'index': ref_idx,
                'x_channel': x_ch, 'y_channel': y_ch
            }
            self.gate_info_label.config(
                text=f"Ref t={ref_tp.time:.1f} Y:[{gl['y_min']:.1f},{gl['y_max']:.1f}] X:[{gl['x_min']:.1f},{gl['x_max']:.1f}]",
                foreground="green")
            messagebox.showinfo("Success", "Gate saved!"); gw.destroy()
        ttk.Button(bf, text="Save Gate", command=save).pack(side=tk.LEFT, padx=10)
        ttk.Button(bf, text="Cancel", command=gw.destroy).pack(side=tk.LEFT, padx=10)
    
    def find_optimal_gate(self, ref_x_gated, x_data, y_data,
                         ref_x_min, ref_x_max, ref_y_min, ref_y_max,
                         ref_total_n, target_total_n, search_range, n_steps,
                         use_fraction=False, fraction_weight=10.0):
        ref_gate_width = ref_x_max - ref_x_min
        ref_center = (ref_x_min + ref_x_max) / 2
        
        y_mask = (y_data >= ref_y_min) & (y_data <= ref_y_max)
        x_in_y = x_data[y_mask]
        
        n_ref_gated = len(ref_x_gated)
        ref_fraction = n_ref_gated / ref_total_n
        
        ref_sorted = np.sort(ref_x_gated)
        ref_cdf_norm = np.arange(1, n_ref_gated + 1) / ref_total_n
        ref_xc = ref_sorted - ref_sorted[0]
        
        shifts = np.linspace(-search_range, search_range, n_steps)
        shape_scores = np.full(n_steps, np.inf)
        combined_scores = np.full(n_steps, np.inf)
        ks_distances = np.full(n_steps, np.inf)
        fraction_diffs = np.full(n_steps, np.inf)
        n_in_gate = np.zeros(n_steps, dtype=int)
        
        y_max = max(ref_fraction * 1.2, 0.1)
        common_y = np.linspace(0, y_max, 200)
        ref_xy = np.interp(common_y, ref_cdf_norm, ref_xc, left=ref_xc[0], right=ref_xc[-1])
        ref_w = ref_xc[-1] - ref_xc[0] if len(ref_xc) > 1 else 1.0
        
        for i, shift in enumerate(shifts):
            gc = ref_center + shift
            gmin, gmax = gc - ref_gate_width/2, gc + ref_gate_width/2
            mask = (x_in_y >= gmin) & (x_in_y <= gmax)
            gated = x_in_y[mask]
            n_target = len(gated)
            n_in_gate[i] = n_target
            
            if n_target >= 20:
                tfrac = n_target / target_total_n
                fraction_diffs[i] = abs(tfrac - ref_fraction)
                
                ts = np.sort(gated)
                tcn = np.arange(1, n_target+1) / target_total_n
                txc = ts - ts[0]
                txy = np.interp(common_y, tcn, txc, left=txc[0], right=txc[-1])
                
                shape_scores[i] = np.sum((ref_xy - txy)**2) / (len(common_y) * max(ref_w**2, 1e-10))
                
                if use_fraction:
                    combined_scores[i] = shape_scores[i] + fraction_weight * fraction_diffs[i]
                else:
                    combined_scores[i] = shape_scores[i]
                
                D, _, _, _, _ = KolmogorovSmirnovAnalyzer.two_sample_ks_statistic(ref_x_gated, gated)
                ks_distances[i] = D
        
        score = combined_scores if use_fraction else shape_scores
        vm = np.isfinite(score)
        
        if np.sum(vm) == 0:
            opt_shift = 0.0; min_shape = np.inf; min_comb = np.inf; min_ks = np.inf; min_fd = np.inf
        else:
            bi = np.where(vm)[0][np.argmin(score[vm])]
            opt_shift = shifts[bi]
            min_shape = shape_scores[bi]; min_comb = combined_scores[bi]
            min_ks = ks_distances[bi]; min_fd = fraction_diffs[bi]
        
        gc = ref_center + opt_shift
        gmin, gmax = gc - ref_gate_width/2, gc + ref_gate_width/2
        mask = (x_in_y >= gmin) & (x_in_y <= gmax)
        opt_gated = x_in_y[mask]
        opt_frac = len(opt_gated)/target_total_n if target_total_n>0 else 0
        
        return {
            'gate_x_min': gmin, 'gate_x_max': gmax,
            'gate_y_min': ref_y_min, 'gate_y_max': ref_y_max,
            'gate_position': (gmin + gmax) / 2,
            'optimal_shift': opt_shift, 'min_shape_score': min_shape,
            'min_combined_score': min_comb, 'min_ks_distance': min_ks,
            'min_fraction_diff': min_fd,
            'shifts': shifts, 'shape_scores': shape_scores,
            'combined_scores': combined_scores, 'ks_distances': ks_distances,
            'fraction_diffs': fraction_diffs, 'n_events_in_gate': n_in_gate,
            'optimal_n_events': len(opt_gated), 'optimal_fraction': opt_frac,
            'ref_fraction': ref_fraction,
            'optimal_gated_data': opt_gated, 'ref_gated_data': ref_x_gated,
            'use_fraction': use_fraction, 'fraction_weight': fraction_weight
        }
    
    def _fit_kinetics_data(self, times_list, values_list, sigmas_list, label):
        """Fit A + B*(1 - exp(-t/tau)) to kinetics data"""
        kin_times = []
        kin_vals = []
        kin_sigs = []
        for t, v, s in zip(times_list, values_list, sigmas_list):
            if not np.isnan(v):
                kin_times.append(t)
                kin_vals.append(v)
                kin_sigs.append(s)
        
        if len(kin_times) < 3:
            return None
        
        kin_times = np.array(kin_times)
        kin_vals = np.array(kin_vals)
        kin_sigs = np.array(kin_sigs)
        
        popt, perr = ExponentialKinetics.fit(kin_times, kin_vals, kin_sigs)
        A_fit, B_fit, tau_inv_fit = popt
        A_err, B_err, tau_inv_err = perr
        tau_fit = 1.0 / tau_inv_fit if tau_inv_fit > 0 else np.inf
        tau_err = tau_inv_err / (tau_inv_fit**2) if tau_inv_fit > 0 else np.inf
        
        print(f"\n{label} Fit: A + B*(1 - exp(-t/τ))")
        print(f"  A = {A_fit:.2f} ± {A_err:.2f}")
        print(f"  B = {B_fit:.2f} ± {B_err:.2f}")
        print(f"  τ = {tau_fit:.2f} ± {tau_err:.2f} min")
        print(f"  1/τ = {tau_inv_fit:.4f} ± {tau_inv_err:.4f} min⁻¹")
        
        return {
            'A': A_fit, 'A_err': A_err,
            'B': B_fit, 'B_err': B_err,
            'tau': tau_fit, 'tau_err': tau_err,
            'tau_inv': tau_inv_fit, 'tau_inv_err': tau_inv_err,
            'times': kin_times, 'values': kin_vals, 'sigmas': kin_sigs
        }
    
    def run_analysis(self):
        if len(self.time_points) < 2:
            messagebox.showerror("Error", "Need at least 2 time points"); return
        if self.reference_gate is None:
            messagebox.showerror("Error", "Set reference gate first"); return
        
        try:
            sr = float(self.search_range_var.get())
            ns = int(self.n_steps_var.get())
            use_fraction = self.use_fraction_var.get()
            fraction_weight = float(self.fraction_weight_var.get()) if use_fraction else 0.0
            fit_mean = self.fit_mean_var.get()
            fit_position = self.fit_position_var.get()
            
            if ns < 10: messagebox.showerror("Error", "Steps >= 10"); return
            
            self.status_var.set("Running..."); self.root.update()
            
            x_ch = self.reference_gate['x_channel']
            y_ch = self.reference_gate['y_channel']
            
            time_data = []
            for tp in self.time_points:
                xd = self.get_channel_data(tp, x_ch)
                yd = self.get_channel_data(tp, y_ch)
                ml = min(len(xd), len(yd))
                if ml == 0: continue
                time_data.append({'time': float(tp.time), 'x': xd[:ml], 'y': yd[:ml], 'n': ml})
            
            if len(time_data) < 2:
                messagebox.showerror("Error", "Need >= 2 points"); return
            
            time_data.sort(key=lambda x: x['time'])
            times_list = [td['time'] for td in time_data]
            x_list = [td['x'] for td in time_data]
            y_list = [td['y'] for td in time_data]
            n_list = [td['n'] for td in time_data]
            
            ref_tv = self.reference_gate['time']
            ref_si = next((i for i, td in enumerate(time_data) if abs(td['time']-ref_tv)<1e-10), len(time_data)-1)
            
            ref_x, ref_y = x_list[ref_si], y_list[ref_si]
            ref_time, ref_n = times_list[ref_si], n_list[ref_si]
            
            ry0, ry1 = self.reference_gate['y_min'], self.reference_gate['y_max']
            rx0, rx1 = self.reference_gate['x_min'], self.reference_gate['x_max']
            
            ym = (ref_y >= ry0) & (ref_y <= ry1)
            xiy = ref_x[ym]
            xm = (xiy >= rx0) & (xiy <= rx1)
            ref_gated = xiy[xm]
            
            if len(ref_gated) == 0:
                messagebox.showerror("Error", "No events in gate"); return
            
            step_sz = 2*sr/(ns-1)
            self.step_size_label.config(text=f"Step size: {step_sz:.1f} units")
            
            ref_frac = len(ref_gated)/ref_n
            ref_position = (rx0 + rx1) / 2
            
            self.gate_stats = {}
            all_sr = []
            
            for i, (xd, yd, time, tn) in enumerate(zip(x_list, y_list, times_list, n_list)):
                time = float(time)
                if i == ref_si:
                    self.gate_stats[time] = {
                        'mean': float(np.mean(ref_gated)), 'sigma': float(np.std(ref_gated)),
                        'n': len(ref_gated), 'total_n': ref_n, 'fraction': ref_frac,
                        'gate_x_min': rx0, 'gate_x_max': rx1,
                        'gate_y_min': ry0, 'gate_y_max': ry1,
                        'position': ref_position, 'shift': 0.0,
                        'shape_score': 0.0, 'combined_score': 0.0,
                        'ks_distance': 0.0, 'frac_diff': 0.0,
                        'is_reference': True, 'gated_data': ref_gated,
                        'x_data': ref_x, 'y_data': ref_y
                    }
                    continue
                
                result = self.find_optimal_gate(
                    ref_gated, xd, yd, rx0, rx1, ry0, ry1, ref_n, tn, sr, ns,
                    use_fraction, fraction_weight)
                
                og = result['optimal_gated_data']
                if len(og) > 0:
                    self.gate_stats[time] = {
                        'mean': float(np.mean(og)), 'sigma': float(np.std(og)),
                        'n': len(og), 'total_n': tn, 'fraction': result['optimal_fraction'],
                        'ref_fraction': ref_frac,
                        'gate_x_min': result['gate_x_min'], 'gate_x_max': result['gate_x_max'],
                        'gate_y_min': result['gate_y_min'], 'gate_y_max': result['gate_y_max'],
                        'position': result['gate_position'], 'shift': result['optimal_shift'],
                        'shape_score': result['min_shape_score'],
                        'combined_score': result['min_combined_score'],
                        'ks_distance': result['min_ks_distance'],
                        'frac_diff': result['min_fraction_diff'],
                        'is_reference': False, 'gated_data': og,
                        'x_data': xd, 'y_data': yd
                    }
                else:
                    self.gate_stats[time] = {
                        'mean': float('nan'), 'sigma': float('nan'), 'n': 0, 'total_n': tn,
                        'fraction': 0, 'ref_fraction': ref_frac,
                        'gate_x_min': result['gate_x_min'], 'gate_x_max': result['gate_x_max'],
                        'gate_y_min': result['gate_y_min'], 'gate_y_max': result['gate_y_max'],
                        'position': result['gate_position'], 'shift': result['optimal_shift'],
                        'shape_score': result['min_shape_score'],
                        'combined_score': result['min_combined_score'],
                        'ks_distance': result['min_ks_distance'], 'frac_diff': result['min_fraction_diff'],
                        'is_reference': False, 'gated_data': np.array([]),
                        'x_data': xd, 'y_data': yd
                    }
                all_sr.append({'time': time, **result})
            
            # Fit kinetics for Mean X
            fit_mean_results = None
            if fit_mean:
                mean_vals = [self.gate_stats.get(float(t), {}).get('mean', float('nan')) for t in times_list]
                sigma_vals = [self.gate_stats.get(float(t), {}).get('sigma', 0) for t in times_list]
                fit_mean_results = self._fit_kinetics_data(times_list, mean_vals, sigma_vals, "Mean X")
            
            # Fit kinetics for Gate Position
            fit_position_results = None
            if fit_position:
                position_vals = [self.gate_stats.get(float(t), {}).get('position', float('nan')) for t in times_list]
                zero_sigmas = [0.0 for _ in times_list]
                fit_position_results = self._fit_kinetics_data(times_list, position_vals, zero_sigmas, "Gate Position")
            
            self.results = {
                'times': times_list, 'gate_stats': self.gate_stats,
                'gate_search_results': all_sr,
                'x_data_list': x_list, 'y_data_list': y_list,
                'fit_mean_results': fit_mean_results,
                'fit_position_results': fit_position_results,
                'ref_position': ref_position,
                'search_params': {'search_range': sr, 'n_steps': ns,
                                 'x_channel': x_ch, 'y_channel': y_ch, 'ref_idx': ref_si,
                                 'use_fraction': use_fraction, 'fraction_weight': fraction_weight,
                                 'fit_mean': fit_mean, 'fit_position': fit_position},
            }
            
            self.update_plots()
            self.update_gate_stats_display()
            self.status_var.set(f"Done. {len(self.gate_stats)} points processed.")
            
        except Exception as e:
            import traceback; traceback.print_exc()
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
    
    def update_gate_stats_display(self):
        if not self.gate_stats: return
        sp = self.results.get('search_params', {})
        txt = f"X: {sp.get('x_channel','?')} | Y: {sp.get('y_channel','?')}\n"
        txt += f"Search: ±{sp.get('search_range',1):.1f}, {sp.get('n_steps',200)} steps"
        if sp.get('use_fraction'): txt += f"\nFraction matching: ON"
        txt += "\n\n"
        
        fit_m = self.results.get('fit_mean_results')
        if fit_m:
            txt += "MEAN X FIT: A + B*(1 - exp(-t/τ))\n"
            txt += f"  A={fit_m['A']:.1f}±{fit_m['A_err']:.1f}, B={fit_m['B']:.1f}±{fit_m['B_err']:.1f}\n"
            txt += f"  τ={fit_m['tau']:.2f}±{fit_m['tau_err']:.2f} min\n\n"
        
        fit_p = self.results.get('fit_position_results')
        if fit_p:
            txt += "GATE POSITION FIT: A + B*(1 - exp(-t/τ))\n"
            txt += f"  A={fit_p['A']:.1f}±{fit_p['A_err']:.1f}, B={fit_p['B']:.1f}±{fit_p['B_err']:.1f}\n"
            txt += f"  τ={fit_p['tau']:.2f}±{fit_p['tau_err']:.2f} min\n\n"
        
        for time in sorted(self.gate_stats.keys()):
            s = self.gate_stats[time]
            if s.get('is_reference'):
                txt += f"t={time:.1f} [REF]: pos={s['position']:.1f}, mean={s['mean']:.1f}\n"
            else:
                txt += f"t={time:.1f}: pos={s['position']:.1f}, mean={s['mean']:.1f}, shift={s['shift']:.1f}\n"
        self.gate_stats_text.delete(1.0, tk.END)
        self.gate_stats_text.insert(tk.END, txt)
    
    def update_plots(self):
        if not self.results: return
        for tab_name in self.plot_tabs:
            for w in self.plot_tabs[tab_name].winfo_children(): w.destroy()
        self.plot_2d_cytograms()
        self.plot_histograms()
        self.plot_cdf_shape_comparison()
        self.plot_gate_search()
        self.plot_kinetics_mean()
        self.plot_kinetics_position()
        self.plot_results_table()
    
    def plot_2d_cytograms(self):
        tab = self.plot_tabs["2D Cytograms"]
        gs = self.results.get('gate_stats', {})
        times = self.results['times']
        n = len(times)
        if n == 0: return
        nc, nr = min(3, n), (n+2)//3
        fig, axes = plt.subplots(nr, nc, figsize=(5*nc, 4*nr))
        if nr*nc == 1: axes = np.array([axes])
        axes = axes.flatten()
        for i in range(n):
            time = float(times[i]); ax = axes[i]
            if time in gs:
                s = gs[time]; xd, yd = s.get('x_data',[]), s.get('y_data',[])
                if len(xd) > 0:
                    ax.scatter(xd, yd, alpha=0.3, s=1, c='blue')
                    c = 'green' if s.get('is_reference') else 'red'
                    rect = Rectangle((s['gate_x_min'], s['gate_y_min']),
                                   s['gate_x_max']-s['gate_x_min'], s['gate_y_max']-s['gate_y_min'],
                                   linewidth=2, edgecolor=c, facecolor=c, alpha=0.2)
                    ax.add_patch(rect)
            ax.set_title(f"t={time:.1f}", fontweight='bold')
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.grid(True, alpha=0.3)
        for i in range(n, len(axes)): axes[i].set_visible(False)
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, tab); canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, tab).update()
    
    def plot_histograms(self):
        tab = self.plot_tabs["Histograms (X)"]
        fig, ax = plt.subplots(figsize=(10, 6))
        gs = self.results.get('gate_stats', {})
        times = self.results['times']
        if len(times) == 0: return
        colors = plt.cm.viridis(np.linspace(0, 1, len(times)))
        for i in range(len(times)):
            time = float(times[i])
            if time in gs:
                s = gs[time]; xd, yd = s.get('x_data',[]), s.get('y_data',[])
                if len(xd) > 0:
                    ym = (yd >= s['gate_y_min']) & (yd <= s['gate_y_max'])
                    xiy = xd[ym]
                    if len(xiy) > 0:
                        ax.hist(xiy, bins=50, alpha=0.3, color=colors[i], label=f"t={time:.1f}", density=True)
                        ls = '-' if s.get('is_reference') else '--'
                        ax.axvline(x=s['gate_x_min'], color=colors[i], linestyle=ls, alpha=0.7, lw=2)
                        ax.axvline(x=s['gate_x_max'], color=colors[i], linestyle=ls, alpha=0.7, lw=2)
        ax.set_xlabel("X channel"); ax.set_ylabel("Density")
        ax.set_title("X distributions (Y-gated)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        canvas = FigureCanvasTkAgg(fig, tab); canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, tab).update()
    
    def plot_cdf_shape_comparison(self):
        tab = self.plot_tabs["CDF Shape Comparison"]
        gs = self.results.get('gate_stats', {})
        times = self.results['times']
        if len(times) == 0: return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(times)))
        for i, time in enumerate(sorted(times)):
            time = float(time)
            if time in gs:
                s = gs[time]; d = s['gated_data']
                if len(d) > 0:
                    sd = np.sort(d); cn = np.arange(1, len(d)+1)/s['total_n']
                    if s.get('is_reference'):
                        ax1.plot(sd-sd[0], cn, 'k-', lw=3, label=f"REF t={time:.1f}")
                        ax2.plot(sd, cn, 'k-', lw=3, label=f"REF")
                    else:
                        ax1.plot(sd-sd[0], cn, color=colors[i], lw=1.5, ls='--', label=f"t={time:.1f}")
                        ax2.plot(sd, cn, color=colors[i], lw=1.5, ls='--', label=f"t={time:.1f}")
        ax1.set_xlabel("Centered X"); ax1.set_ylabel("CDF"); ax1.set_title("CDF SHAPES"); ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)
        ax2.set_xlabel("X"); ax2.set_ylabel("CDF"); ax2.set_title("Original CDFs"); ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, tab); canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, tab).update()
    
    def plot_gate_search(self):
        tab = self.plot_tabs["Gate Search"]
        results = self.results.get('gate_search_results', [])
        if not results: ttk.Label(tab, text="No data").pack(pady=20); return
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        (ax1, ax2), (ax3, ax4) = axes
        for r in results:
            vm = np.isfinite(r['shape_scores'])
            if np.sum(vm)>0:
                ax1.plot(r['shifts'][vm], r['shape_scores'][vm], alpha=0.7, lw=1.5, label=f"t={r['time']:.1f}")
                ax1.axvline(x=r['optimal_shift'], alpha=0.3, ls='--')
        ax1.set_xlabel("X-shift"); ax1.set_ylabel("Shape score"); ax1.set_title("Shape"); ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)
        for r in results:
            vm = np.isfinite(r['ks_distances'])
            if np.sum(vm)>0:
                ax2.plot(r['shifts'][vm], r['ks_distances'][vm], alpha=0.7, lw=1.5, label=f"t={r['time']:.1f}")
                ax2.axvline(x=r['optimal_shift'], alpha=0.3, ls='--')
        ax2.set_xlabel("X-shift"); ax2.set_ylabel("KS"); ax2.set_title("KS"); ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)
        ts = [r['time'] for r in results]
        ax3.plot(ts, [r['optimal_shift'] for r in results], 'bo-', ms=8)
        ax3.axhline(y=0, color='r', ls='--'); ax3.set_xlabel("Time"); ax3.set_ylabel("Optimal X-shift"); ax3.grid(True, alpha=0.3)
        if results:
            r = results[0]; vm = np.isfinite(r['shape_scores'])
            if np.sum(vm)>0:
                ax4.plot(r['shifts'][vm], r['shape_scores'][vm], 'b-', lw=2)
                ax4.axvline(x=r['optimal_shift'], color='r', ls='--')
                ax4.set_xlabel("X-shift"); ax4.set_ylabel("Shape"); ax4.set_title(f"t={r['time']:.1f}"); ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, tab); canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, tab).update()
    
    def plot_kinetics_mean(self):
        tab = self.plot_tabs["Kinetics (Mean X)"]
        gs = self.results.get('gate_stats', {})
        if not gs: ttk.Label(tab, text="No data").pack(pady=20); return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        times_all = sorted(gs.keys())
        times_plot = [t for t in times_all if not np.isnan(gs[t]['mean'])]
        means_plot = [gs[t]['mean'] for t in times_plot]
        sigmas_plot = [gs[t]['sigma'] for t in times_plot]
        is_ref = [gs[t].get('is_reference', False) for t in times_plot]
        
        t_ref = [t for t, r in zip(times_plot, is_ref) if r]
        m_ref = [m for m, r in zip(means_plot, is_ref) if r]
        s_ref = [s for s, r in zip(sigmas_plot, is_ref) if r]
        t_opt = [t for t, r in zip(times_plot, is_ref) if not r]
        m_opt = [m for m, r in zip(means_plot, is_ref) if not r]
        s_opt = [s for s, r in zip(sigmas_plot, is_ref) if not r]
        
        if t_ref:
            ax.errorbar(t_ref, m_ref, yerr=s_ref, fmt='gs', ms=10, capsize=5, label='Reference', zorder=5)
        if t_opt:
            ax.errorbar(t_opt, m_opt, yerr=s_opt, fmt='ro', ms=8, capsize=5, label='Optimized')
        
        fit = self.results.get('fit_mean_results')
        if fit:
            tf = np.linspace(min(times_plot), max(times_plot), 200)
            yf = ExponentialKinetics.model(tf, fit['A'], fit['B'], fit['tau_inv'])
            label = f"A={fit['A']:.1f}±{fit['A_err']:.1f}\nB={fit['B']:.1f}±{fit['B_err']:.1f}\nτ={fit['tau']:.2f}±{fit['tau_err']:.2f} min"
            ax.plot(tf, yf, 'b-', lw=2, label=label)
            ax.axhline(y=fit['A'], color='gray', ls=':', alpha=0.5)
            ax.axhline(y=fit['A']+fit['B'], color='gray', ls=':', alpha=0.5)
        
        ax.set_xlabel("Time (min)", fontsize=12)
        ax.set_ylabel("Mean X (gated)", fontsize=12)
        ax.set_title("Kinetics: Mean X vs Time", fontsize=14)
        ax.legend(fontsize=9, loc='lower right'); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, tab); canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, tab).update()
    
    def plot_kinetics_position(self):
        tab = self.plot_tabs["Kinetics (Gate Position)"]
        gs = self.results.get('gate_stats', {})
        if not gs: ttk.Label(tab, text="No data").pack(pady=20); return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        times_all = sorted(gs.keys())
        times_plot = [t for t in times_all if not np.isnan(gs[t]['position'])]
        positions_plot = [gs[t]['position'] for t in times_plot]
        is_ref = [gs[t].get('is_reference', False) for t in times_plot]
        
        t_ref = [t for t, r in zip(times_plot, is_ref) if r]
        p_ref = [p for p, r in zip(positions_plot, is_ref) if r]
        t_opt = [t for t, r in zip(times_plot, is_ref) if not r]
        p_opt = [p for p, r in zip(positions_plot, is_ref) if not r]
        
        # Reference point as green square (no line)
        if t_ref:
            ax.plot(t_ref, p_ref, 'gs', ms=12, label='Reference', zorder=5)
        
        # Optimized points as red circles (NO LINE - just markers)
        if t_opt:
            ax.plot(t_opt, p_opt, 'ro', ms=10, label='Optimized gates')
        
        # Add reference position line
        ref_pos = self.results.get('ref_position')
        if ref_pos is not None:
            ax.axhline(y=ref_pos, color='green', ls=':', alpha=0.5, label=f'Ref position: {ref_pos:.1f}')
        
        fit = self.results.get('fit_position_results')
        if fit:
            tf = np.linspace(min(times_plot), max(times_plot), 200)
            yf = ExponentialKinetics.model(tf, fit['A'], fit['B'], fit['tau_inv'])
            label = f"A={fit['A']:.1f}±{fit['A_err']:.1f}\nB={fit['B']:.1f}±{fit['B_err']:.1f}\nτ={fit['tau']:.2f}±{fit['tau_err']:.2f} min"
            ax.plot(tf, yf, 'b-', lw=2, label=label)
            ax.axhline(y=fit['A'], color='gray', ls=':', alpha=0.5)
            ax.axhline(y=fit['A']+fit['B'], color='gray', ls=':', alpha=0.5)
        
        ax.set_xlabel("Time (min)", fontsize=12)
        ax.set_ylabel("Gate Position (X center)", fontsize=12)
        ax.set_title("Kinetics: Gate Position vs Time", fontsize=14)
        ax.legend(fontsize=9, loc='lower right'); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, tab); canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, tab).update()
    
    def plot_results_table(self):
        tab = self.plot_tabs["Results Table"]
        tree = ttk.Treeview(tab, columns=("P","V"), show='headings', height=10)
        tree.heading("P", text="Parameter"); tree.heading("V", text="Value")
        tree.column("P", width=280); tree.column("V", width=220)
        sp = self.results.get('search_params', {})
        tree.insert("", tk.END, values=("CHANNELS", ""))
        tree.insert("", tk.END, values=("  X", sp.get('x_channel','?')))
        tree.insert("", tk.END, values=("  Y", sp.get('y_channel','?')))
        tree.insert("", tk.END, values=("", ""))
        
        fit_m = self.results.get('fit_mean_results')
        if fit_m:
            tree.insert("", tk.END, values=("MEAN X FIT: A+B*(1-exp(-t/τ))", ""))
            tree.insert("", tk.END, values=("  A", f"{fit_m['A']:.1f} ± {fit_m['A_err']:.1f}"))
            tree.insert("", tk.END, values=("  B", f"{fit_m['B']:.1f} ± {fit_m['B_err']:.1f}"))
            tree.insert("", tk.END, values=("  A+B", f"{fit_m['A']+fit_m['B']:.1f}"))
            tree.insert("", tk.END, values=("  τ (min)", f"{fit_m['tau']:.2f} ± {fit_m['tau_err']:.2f}"))
            tree.insert("", tk.END, values=("", ""))
        
        fit_p = self.results.get('fit_position_results')
        if fit_p:
            tree.insert("", tk.END, values=("GATE POSITION FIT: A+B*(1-exp(-t/τ))", ""))
            tree.insert("", tk.END, values=("  A", f"{fit_p['A']:.1f} ± {fit_p['A_err']:.1f}"))
            tree.insert("", tk.END, values=("  B", f"{fit_p['B']:.1f} ± {fit_p['B_err']:.1f}"))
            tree.insert("", tk.END, values=("  A+B", f"{fit_p['A']+fit_p['B']:.1f}"))
            tree.insert("", tk.END, values=("  τ (min)", f"{fit_p['tau']:.2f} ± {fit_p['tau_err']:.2f}"))
            tree.insert("", tk.END, values=("", ""))
        
        tree.insert("", tk.END, values=("GATE RESULTS", ""))
        for time in sorted(self.results['gate_stats'].keys()):
            s = self.results['gate_stats'][time]
            label = f"t={time:.1f} [REF]" if s.get('is_reference') else f"t={time:.1f}"
            tree.insert("", tk.END, values=(label, f"pos={s['position']:.1f}, mean={s['mean']:.1f}, n={s['n']}"))
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=scrollbar.set)
    
    def export_results(self):
        if not self.results: messagebox.showerror("Error", "No results"); return
        d = filedialog.askdirectory(title="Export")
        if not d: return
        try:
            gd = []
            for time, s in sorted(self.results['gate_stats'].items()):
                gd.append({
                    'Time': time, 'Is_Ref': s.get('is_reference', False),
                    'Gate_X_Min': s['gate_x_min'], 'Gate_X_Max': s['gate_x_max'],
                    'Gate_Y_Min': s['gate_y_min'], 'Gate_Y_Max': s['gate_y_max'],
                    'Position': s['position'], 'Shift': s['shift'],
                    'Mean': s['mean'], 'Sigma': s['sigma'],
                    'N_gated': s['n'], 'N_total': s['total_n'], 'Fraction': s['fraction']
                })
            pd.DataFrame(gd).to_csv(os.path.join(d, 'gate_results.csv'), index=False)
            
            for name, fit in [('mean_kinetics_fit', self.results.get('fit_mean_results')),
                             ('position_kinetics_fit', self.results.get('fit_position_results'))]:
                if fit:
                    pd.DataFrame({
                        'Parameter': ['A', 'B', 'tau', 'tau_inv', 'A+B'],
                        'Value': [fit['A'], fit['B'], fit['tau'], fit['tau_inv'], fit['A']+fit['B']],
                        'Error': [fit['A_err'], fit['B_err'], fit['tau_err'], fit['tau_inv_err'], 0]
                    }).to_csv(os.path.join(d, f'{name}.csv'), index=False)
            
            for r in self.results.get('gate_search_results', []):
                pd.DataFrame({'Shift': r['shifts'], 'Shape': r['shape_scores'],
                             'KS': r['ks_distances'], 'N': r['n_events_in_gate']}
                           ).to_csv(os.path.join(d, f'search_t{r["time"]:.1f}.csv'), index=False)
            
            messagebox.showinfo("Success", f"Exported to {d}")
        except Exception as e:
            messagebox.showerror("Error", f"Export: {e}")


def main():
    root = tk.Tk()
    app = DynamicGatingGUI(root)
    style = ttk.Style()
    style.configure('Large.TButton', font=('Arial', 11, 'bold'))
    root.mainloop()


if __name__ == "__main__":
    print("Dynamic Gating - 2D Cytogram with Kinetics Fit")
    print("Y-axis: FIXED gate | X-axis: OPTIMIZED position")
    print("Fits: A + B*(1 - exp(-t/τ)) for Mean X and Gate Position")
    main()