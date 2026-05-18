import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

# Try to import LMD file reader
try:
    import FlowCytometryTools as fct
    HAS_FCT = True
except ImportError:
    HAS_FCT = False
    print("FlowCytometryTools not installed. Using basic FCS reader.")

try:
    import fcsparser
    HAS_FCSPARSER = True
except ImportError:
    HAS_FCSPARSER = False


@dataclass
class TimePoint:
    """Data class for storing time point measurements"""
    time: float  # Time in minutes
    filename: str
    data: np.ndarray  # Fluorescence data
    gate_data: Optional[np.ndarray] = None  # Gated data


class KolmogorovSmirnovAnalyzer:
    """Implementation of KS test-based dynamic gating"""
    
    @staticmethod
    def T0_approximation(z):
        """Analytical approximation of Kolmogorov distribution function (Eq. 26)"""
        return 0.5 * np.exp(-4 * z**6 / (2 * z**4 + 1))
    
    @staticmethod
    def T0_derivative(z):
        """Derivative of T0 approximation (Eq. 27)"""
        exp_term = np.exp(-4 * z**6 / (2 * z**4 + 1))
        numerator = -4 * z**5 * (2 * z**4 + 3)
        denominator = (2 * z**4 + 1)**2
        return numerator / denominator * exp_term
    
    @staticmethod
    def calculate_z_tilde(n, D):
        """Calculate z-tilde for improved KS statistic (Eq. 17)"""
        sqrt_n = np.sqrt(n)
        z_tilde = sqrt_n * D + 1/(6*sqrt_n) + (sqrt_n*D - 1)/(4*n)
        return z_tilde
    
    @staticmethod
    def calculate_n_star(n, m):
        """Calculate effective sample size for two-sample KS test (Eq. 21)"""
        return (n * m) / (n + m)
    
    @staticmethod
    def empirical_cdf(data):
        """Calculate empirical CDF"""
        sorted_data = np.sort(data)
        n = len(sorted_data)
        y = np.arange(1, n + 1) / n
        return sorted_data, y
    
    @staticmethod
    def two_sample_ks_statistic(sample1, sample2):
        """Calculate two-sample KS statistic D and related parameters"""
        n, m = len(sample1), len(sample2)
        
        # Calculate empirical CDFs
        combined = np.sort(np.concatenate([sample1, sample2]))
        cdf1 = np.searchsorted(np.sort(sample1), combined, side='right') / n
        cdf2 = np.searchsorted(np.sort(sample2), combined, side='right') / m
        
        # Find maximum distance
        differences = np.abs(cdf1 - cdf2)
        D = np.max(differences)
        x0_index = np.argmax(differences)
        
        # Calculate parameters
        n_star = (n * m) / (n + m)
        F1_at_x0 = cdf1[x0_index]
        F2_at_x0 = cdf2[x0_index]
        
        # Calculate epsilon (Eq. 18 extended to two-sample)
        epsilon = np.sqrt(F1_at_x0/n + F2_at_x0/m)
        
        return D, n_star, epsilon, F1_at_x0, F2_at_x0
    
    @staticmethod
    def probability_H0_given_D(D, n_star, epsilon, N_D=1):
        """Calculate Pr(H0|D) using Bayes theorem (Eq. 32)"""
        # Calculate z-tilde
        z_tilde = KolmogorovSmirnovAnalyzer.calculate_z_tilde(n_star, D)
        
        # Calculate Pr(D|H0) using Eq. 25
        T0_prime = KolmogorovSmirnovAnalyzer.T0_derivative(z_tilde)
        sqrt_n_star = np.sqrt(n_star)
        Pr_D_given_H0 = -2 * T0_prime * (sqrt_n_star + 1/(4*sqrt_n_star)) * epsilon
        
        # Apply Bayes theorem (Eq. 32)
        Pr_H0 = 1.0
        Pr_H0_given_D = Pr_D_given_H0 * Pr_H0 / max(N_D, 1)
        
        return max(0, min(1, Pr_H0_given_D))
    
    def find_optimal_shift(self, reference_sample, time_sample, 
                          alpha_range=(-2, 2), n_alphas=100):
        """Find optimal scaling coefficient using KS test (Eq. 33)"""
        alphas = np.linspace(alpha_range[0], alpha_range[1], n_alphas)
        probabilities = []
        D_values = []
        
        for alpha in alphas:
            # Shift time sample by alpha
            shifted_sample = time_sample + alpha
            
            # Calculate KS statistic
            D, n_star, epsilon, _, _ = self.two_sample_ks_statistic(
                reference_sample, shifted_sample
            )
            
            # Calculate probability
            prob = self.probability_H0_given_D(D, n_star, epsilon)
            
            D_values.append(D)
            probabilities.append(prob)
        
        # Find optimal alpha (maximum probability)
        probabilities = np.array(probabilities)
        optimal_idx = np.argmax(probabilities)
        optimal_alpha = alphas[optimal_idx]
        
        # Convert alpha to phi (Eq. 28, 34)
        phi = 10 ** (-optimal_alpha)
        
        return phi, optimal_alpha, alphas, probabilities, D_values


class ExponentialKinetics:
    """Fit exponential kinetics model (Eq. 7)"""
    
    @staticmethod
    def exponential_model(t, y_max, tau_inv):
        """Exponential binding kinetics: y(t) = y_max * (1 - exp(-t/tau))"""
        return y_max * (1 - np.exp(-tau_inv * t))
    
    @staticmethod
    def fit_kinetics(times, values, initial_guess=None):
        """Fit exponential model to kinetics data"""
        if initial_guess is None:
            initial_guess = [np.max(values), 0.1]
        
        try:
            popt, pcov = curve_fit(
                ExponentialKinetics.exponential_model, 
                times, values, 
                p0=initial_guess,
                bounds=([0, 0], [np.inf, np.inf])
            )
            perr = np.sqrt(np.diag(pcov))
            return popt, perr
        except:
            return initial_guess, [0, 0]
    
    @staticmethod
    def calculate_rate_constant(y_max, tau_inv, A0, n_bar=1.0):
        """Calculate association rate constant k+ (Eq. 8)"""
        tau = 1.0 / tau_inv if tau_inv > 0 else np.inf
        k_plus = (y_max * tau_inv) / (A0 * n_bar - y_max) if A0 * n_bar > y_max else 0
        return k_plus


class LMDReader:
    """Reader for LMD/FCS flow cytometry files"""
    
    @staticmethod
    def read_lmd_file(filename):
        """Read LMD/FCS file and extract data"""
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
        
        # Fallback: try reading as numpy binary
        try:
            # Basic FCS 3.0/3.1 reader
            return LMDReader._read_fcs_basic(filename)
        except:
            raise ValueError(f"Could not read file: {filename}")
    
    @staticmethod
    def _read_fcs_basic(filename):
        """Basic FCS file parser"""
        with open(filename, 'rb') as f:
            # Read header
            header = f.read(58)
            
            # Extract text segment offsets
            text_start = int(header[10:18].decode().strip())
            text_end = int(header[18:26].decode().strip())
            
            # Read text segment
            f.seek(text_start)
            text_segment = f.read(text_end - text_start).decode()
            
            # Extract data segment offsets
            data_start = int(header[26:34].decode().strip())
            data_end = int(header[34:42].decode().strip())
            
            # Parse parameters
            params = {}
            for line in text_segment.split('/'):
                if '$P' in line and 'N' in line:
                    parts = line.split('$')
                    for part in parts:
                        if 'P' in part and 'N' in part:
                            key, value = part.split('N', 1)
                            params[key] = value.strip()
            
            # Read data
            f.seek(data_start)
            n_params = int(params.get('$PAR', '1'))
            n_events = (data_end - data_start) // (4 * n_params)
            
            dtype = np.float32
            
            # Read raw data
            raw_data = np.fromfile(f, dtype=dtype, count=n_events * n_params)
            data = raw_data.reshape(n_events, n_params)
            
            # Create DataFrame
            columns = [params.get(f'$P{i}N', f'Channel_{i}') for i in range(1, n_params+1)]
            df = pd.DataFrame(data, columns=columns)
            
            return df


class DynamicGatingGUI:
    """Main GUI application for dynamic gating analysis"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Dynamic Gating Based on KS Test - Flow Cytometry Analysis")
        self.root.geometry("1400x900")
        
        # Data storage
        self.time_points: List[TimePoint] = []
        self.kinetics_data = {}
        self.results = {}
        
        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the graphical user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Left panel - Controls
        left_panel = ttk.Frame(main_frame, padding="5")
        left_panel.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # File loading section
        ttk.Label(left_panel, text="File Loading", font=('Arial', 12, 'bold')).grid(row=0, column=0, pady=5)
        
        ttk.Button(left_panel, text="Add Time Point Files", command=self.add_time_points).grid(row=1, column=0, pady=5)
        
        # Time points list
        ttk.Label(left_panel, text="Loaded Time Points:").grid(row=2, column=0, pady=5)
        self.time_listbox = tk.Listbox(left_panel, width=40, height=10)
        self.time_listbox.grid(row=3, column=0, pady=5)
        
        ttk.Button(left_panel, text="Remove Selected", command=self.remove_time_point).grid(row=4, column=0, pady=5)
        
        # Channel selection
        ttk.Label(left_panel, text="Fluorescence Channel:", font=('Arial', 10)).grid(row=5, column=0, pady=5)
        self.channel_var = tk.StringVar(value="FL1")
        self.channel_combo = ttk.Combobox(left_panel, textvariable=self.channel_var, width=38)
        self.channel_combo.grid(row=6, column=0, pady=5)
        
        # Gate settings
        ttk.Label(left_panel, text="Gate Settings", font=('Arial', 12, 'bold')).grid(row=7, column=0, pady=5)
        
        ttk.Label(left_panel, text="Gate Range (percentiles):").grid(row=8, column=0)
        gate_frame = ttk.Frame(left_panel)
        gate_frame.grid(row=9, column=0, pady=5)
        self.gate_min_var = tk.StringVar(value="0")
        self.gate_max_var = tk.StringVar(value="100")
        ttk.Entry(gate_frame, textvariable=self.gate_min_var, width=10).grid(row=0, column=0)
        ttk.Label(gate_frame, text="to").grid(row=0, column=1)
        ttk.Entry(gate_frame, textvariable=self.gate_max_var, width=10).grid(row=0, column=2)
        ttk.Label(gate_frame, text="%").grid(row=0, column=3)
        
        # Predefined gates
        ttk.Label(left_panel, text="Predefined Gates:").grid(row=10, column=0, pady=5)
        gates_frame = ttk.Frame(left_panel)
        gates_frame.grid(row=11, column=0, pady=5)
        
        predefined_gates = [
            "[-2σ; 2σ]", "[-σ; σ]", "[mean; 2σ]", "[mean; σ]",
            "[σ; 2σ]", "[-2σ; mean]", "[-σ; mean]", "[-2σ; -σ]"
        ]
        
        self.gate_var = tk.StringVar(value="[-2σ; 2σ]")
        for i, gate in enumerate(predefined_gates):
            ttk.Radiobutton(gates_frame, text=gate, variable=self.gate_var, 
                          value=gate).grid(row=i//2, column=i%2, sticky=tk.W, padx=5)
        
        # Analysis settings
        ttk.Label(left_panel, text="Analysis Settings", font=('Arial', 12, 'bold')).grid(row=13, column=0, pady=5)
        
        ttk.Label(left_panel, text="Alpha range:").grid(row=14, column=0)
        alpha_frame = ttk.Frame(left_panel)
        alpha_frame.grid(row=15, column=0, pady=5)
        self.alpha_min_var = tk.StringVar(value="-2")
        self.alpha_max_var = tk.StringVar(value="2")
        ttk.Entry(alpha_frame, textvariable=self.alpha_min_var, width=10).grid(row=0, column=0)
        ttk.Label(alpha_frame, text="to").grid(row=0, column=1)
        ttk.Entry(alpha_frame, textvariable=self.alpha_max_var, width=10).grid(row=0, column=2)
        
        ttk.Label(left_panel, text="Number of alpha steps:").grid(row=16, column=0)
        self.alpha_steps_var = tk.StringVar(value="100")
        ttk.Entry(left_panel, textvariable=self.alpha_steps_var, width=10).grid(row=17, column=0)
        
        # Ligand concentration
        ttk.Label(left_panel, text="Ligand concentration (M):").grid(row=18, column=0, pady=5)
        self.A0_var = tk.StringVar(value="1e-8")
        ttk.Entry(left_panel, textvariable=self.A0_var, width=20).grid(row=19, column=0)
        
        # Run analysis button
        ttk.Button(left_panel, text="Run Dynamic Gating Analysis", 
                  command=self.run_analysis, style='Large.TButton').grid(row=20, column=0, pady=20)
        
        # Export results
        ttk.Button(left_panel, text="Export Results", 
                  command=self.export_results).grid(row=21, column=0, pady=5)
        
        # Right panel - Plots
        right_panel = ttk.Frame(main_frame, padding="5")
        right_panel.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=1)
        
        # Notebook for different plots
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create plot tabs
        self.plot_tabs = {}
        plot_names = ["Cytograms", "Histograms", "CDFs", "Scaling Kinetics", "KS Analysis", "Results Table"]
        for name in plot_names:
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=name)
            self.plot_tabs[name] = frame
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
    def add_time_points(self):
        """Add LMD files with time point specification"""
        files = filedialog.askopenfilenames(
            title="Select LMD/FCS files",
            filetypes=[("Flow cytometry files", "*.lmd *.fcs"), ("All files", "*.*")]
        )
        
        if not files:
            return
        
        # Create dialog for time input
        time_dialog = tk.Toplevel(self.root)
        time_dialog.title("Enter Time Points")
        time_dialog.geometry("400x300")
        
        ttk.Label(time_dialog, text="Enter time (minutes) for each file:").grid(row=0, column=0, columnspan=2, pady=10)
        
        time_entries = []
        for i, file in enumerate(files):
            ttk.Label(time_dialog, text=os.path.basename(file)[:30] + "...").grid(row=i+1, column=0, padx=5, pady=2)
            time_var = tk.StringVar(value=str(i*5))  # Default: 5 minute intervals
            entry = ttk.Entry(time_dialog, textvariable=time_var, width=10)
            entry.grid(row=i+1, column=1, padx=5, pady=2)
            time_entries.append((file, time_var))
        
        def load_files():
            for file, time_var in time_entries:
                try:
                    time = float(time_var.get())
                    self.status_var.set(f"Loading {os.path.basename(file)}...")
                    self.root.update()
                    
                    # Read LMD file
                    data = LMDReader.read_lmd_file(file)
                    
                    # Store time point
                    time_point = TimePoint(time=time, filename=file, data=data)
                    self.time_points.append(time_point)
                    
                    # Update listbox
                    self.time_listbox.insert(tk.END, f"t={time:.1f} min - {os.path.basename(file)}")
                    
                    # Update channel combo if first file
                    if len(self.time_points) == 1 and isinstance(data, pd.DataFrame):
                        self.channel_combo['values'] = list(data.columns)
                        if len(data.columns) > 0:
                            self.channel_var.set(data.columns[0])
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load {file}: {str(e)}")
            
            self.status_var.set(f"Loaded {len(self.time_points)} time points")
            time_dialog.destroy()
        
        ttk.Button(time_dialog, text="Load Files", command=load_files).grid(
            row=len(files)+1, column=0, columnspan=2, pady=20)
    
    def remove_time_point(self):
        """Remove selected time point"""
        selection = self.time_listbox.curselection()
        if selection:
            idx = selection[0]
            self.time_listbox.delete(idx)
            del self.time_points[idx]
            self.status_var.set(f"Removed time point. {len(self.time_points)} remaining.")
    
    def get_gate_mask(self, data, gate_type=None):
        """Get boolean mask for gating based on selected gate type"""
        if data is None or len(data) == 0:
            return np.ones(len(data), dtype=bool) if data is not None else None
        
        if gate_type is None:
            gate_type = self.gate_var.get()
        
        mean = np.mean(data)
        std = np.std(data)
        
        gate_map = {
            "[-2σ; 2σ]": (mean - 2*std, mean + 2*std),
            "[-σ; σ]": (mean - std, mean + std),
            "[mean; 2σ]": (mean, mean + 2*std),
            "[mean; σ]": (mean, mean + std),
            "[σ; 2σ]": (mean + std, mean + 2*std),
            "[-2σ; mean]": (mean - 2*std, mean),
            "[-σ; mean]": (mean - std, mean),
            "[-2σ; -σ]": (mean - 2*std, mean - std),
        }
        
        if gate_type in gate_map:
            gmin, gmax = gate_map[gate_type]
            return (data >= gmin) & (data <= gmax)
        else:
            # Custom percentile gate
            try:
                pmin = float(self.gate_min_var.get())
                pmax = float(self.gate_max_var.get())
                gmin = np.percentile(data, pmin)
                gmax = np.percentile(data, pmax)
                return (data >= gmin) & (data <= gmax)
            except:
                return np.ones(len(data), dtype=bool)
    
    def run_analysis(self):
        """Run the complete dynamic gating analysis"""
        if len(self.time_points) < 2:
            messagebox.showerror("Error", "Need at least 2 time points for analysis")
            return
        
        try:
            self.status_var.set("Running analysis...")
            self.root.update()
            
            # Get parameters
            channel = self.channel_var.get()
            alpha_min = float(self.alpha_min_var.get())
            alpha_max = float(self.alpha_max_var.get())
            alpha_steps = int(self.alpha_steps_var.get())
            A0 = float(self.A0_var.get())
            
            # Extract fluorescence data for each time point
            fluo_data = []
            times = []
            
            for tp in self.time_points:
                if isinstance(tp.data, pd.DataFrame):
                    if channel in tp.data.columns:
                        fluo = tp.data[channel].values
                    else:
                        # Use first fluorescence channel
                        fluo = tp.data.iloc[:, 0].values
                else:
                    fluo = tp.data.flatten()
                
                # Remove negative and zero values for log transform
                fluo = fluo[fluo > 0]
                
                # Log transform
                log_fluo = np.log10(fluo)
                
                # Apply gate
                gate_type = self.gate_var.get()
                mask = self.get_gate_mask(log_fluo, gate_type)
                gate_fluo = log_fluo[mask]
                
                fluo_data.append(gate_fluo)
                times.append(tp.time)
            
            # Sort by time
            sort_idx = np.argsort(times)
            times = np.array(times)[sort_idx]
            fluo_data = [fluo_data[i] for i in sort_idx]
            
            # Reference sample (last time point)
            ref_fluo = fluo_data[-1]
            ref_time = times[-1]
            
            # Initialize analyzer
            ks_analyzer = KolmogorovSmirnovAnalyzer()
            
            # Calculate scaling coefficients for each time point
            phis = []
            phi_times = []
            alpha_results = []
            
            for i, (fluo, time) in enumerate(zip(fluo_data[:-1], times[:-1])):
                self.status_var.set(f"Processing time point {time:.1f} min...")
                self.root.update()
                
                phi, alpha, alphas, probs, D_vals = ks_analyzer.find_optimal_shift(
                    ref_fluo, fluo, 
                    alpha_range=(alpha_min, alpha_max),
                    n_alphas=alpha_steps
                )
                
                phis.append(phi)
                phi_times.append(time)
                alpha_results.append({
                    'time': time,
                    'phi': phi,
                    'alpha': alpha,
                    'alphas': alphas,
                    'probabilities': probs,
                    'D_values': D_vals
                })
            
            # Add reference point (phi = 1)
            phis.append(1.0)
            phi_times.append(ref_time)
            
            # Sort by time
            sort_idx = np.argsort(phi_times)
            phi_times_arr = np.array(phi_times)[sort_idx]
            phis_arr = np.array(phis)[sort_idx]
            
            # Calculate average fluorescence kinetics
            y_max_ref = np.mean(ref_fluo)
            fluo_kinetics = y_max_ref / phis_arr
            
            # Fit exponential model
            kinetics_fitter = ExponentialKinetics()
            popt, perr = kinetics_fitter.fit_kinetics(phi_times_arr, fluo_kinetics)
            
            y_max_fit, tau_inv_fit = popt
            y_max_err, tau_inv_err = perr
            
            # Calculate rate constant
            k_plus = kinetics_fitter.calculate_rate_constant(y_max_fit, tau_inv_fit, A0)
            
            # Store results
            self.results = {
                'times': phi_times_arr,
                'phis': phis_arr,
                'fluoro_kinetics': fluo_kinetics,
                'y_max': y_max_fit,
                'y_max_err': y_max_err,
                'tau_inv': tau_inv_fit,
                'tau_inv_err': tau_inv_err,
                'k_plus': k_plus,
                'alpha_results': alpha_results,
                'fluo_data': fluo_data,
                'times_all': times,
            }
            
            # Update plots
            self.update_plots()
            
            self.status_var.set(f"Analysis complete. k+ = {k_plus:.2e} M⁻¹s⁻¹")
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.status_var.set("Analysis failed")
    
    def update_plots(self):
        """Update all plot tabs with results"""
        if not self.results:
            return
        
        # Clear existing plots
        for tab_name in self.plot_tabs:
            for widget in self.plot_tabs[tab_name].winfo_children():
                widget.destroy()
        
        # Plot 1: Cytograms
        self.plot_cytograms()
        
        # Plot 2: Histograms
        self.plot_histograms()
        
        # Plot 3: CDFs
        self.plot_cdfs()
        
        # Plot 4: Scaling Kinetics
        self.plot_scaling_kinetics()
        
        # Plot 5: KS Analysis
        self.plot_ks_analysis()
        
        # Plot 6: Results Table
        self.plot_results_table()
    
    def plot_cytograms(self):
        """Plot cytograms (scatter plots of fluorescence)"""
        tab = self.plot_tabs["Cytograms"]
        fig, axes = plt.subplots(2, 4, figsize=(12, 8))
        fig.suptitle("Temporal Evolution of Cell Distributions (Cytograms)")
        
        times = self.results['times_all']
        fluo_data = self.results['fluo_data']
        
        for i, (time, fluo) in enumerate(zip(times, fluo_data)):
            if i < 8:
                ax = axes[i // 4, i % 4]
                ax.scatter(range(len(fluo)), fluo, alpha=0.1, s=1)
                ax.set_title(f"t = {time:.1f} min")
                ax.set_xlabel("Event")
                ax.set_ylabel("log10(Fluorescence)")
        
        for i in range(len(times), 8):
            axes[i // 4, i % 4].set_visible(False)
        
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(canvas, tab)
        toolbar.update()
    
    def plot_histograms(self):
        """Plot fluorescence histograms over time"""
        tab = self.plot_tabs["Histograms"]
        fig, ax = plt.subplots(figsize=(10, 6))
        
        times = self.results['times_all']
        fluo_data = self.results['fluo_data']
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(times)))
        
        for time, fluo, color in zip(times, fluo_data, colors):
            ax.hist(fluo, bins=50, alpha=0.3, color=color, label=f"t={time:.1f} min", density=True)
        
        ax.set_xlabel("log10(Fluorescence)")
        ax.set_ylabel("Density")
        ax.set_title("Fluorescence Distribution Evolution")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, tab)
        toolbar.update()
    
    def plot_cdfs(self):
        """Plot cumulative distribution functions"""
        tab = self.plot_tabs["CDFs"]
        fig, ax = plt.subplots(figsize=(10, 6))
        
        times = self.results['times_all']
        fluo_data = self.results['fluo_data']
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(times)))
        
        for time, fluo, color in zip(times, fluo_data, colors):
            sorted_fluo = np.sort(fluo)
            cdf = np.arange(1, len(sorted_fluo) + 1) / len(sorted_fluo)
            ax.plot(sorted_fluo, cdf, color=color, label=f"t={time:.1f} min", linewidth=2)
        
        ax.set_xlabel("log10(Fluorescence)")
        ax.set_ylabel("Cumulative Probability")
        ax.set_title("Cumulative Distribution Functions Evolution")
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, tab)
        toolbar.update()
    
    def plot_scaling_kinetics(self):
        """Plot scaling coefficient kinetics and fit"""
        tab = self.plot_tabs["Scaling Kinetics"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        times = self.results['times']
        phis = self.results['phis']
        fluo_kinetics = self.results['fluoro_kinetics']
        
        # Plot phi kinetics
        ax1.plot(times, phis, 'bo-', markersize=8, label='φ(t) from KS test')
        ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Reference (φ=1)')
        ax1.set_xlabel("Time (min)")
        ax1.set_ylabel("Scaling coefficient φ(t)")
        ax1.set_title("Scaling Coefficient Kinetics")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot fluorescence kinetics
        ax2.plot(times, fluo_kinetics, 'ro-', markersize=8, label='MFI from KS method')
        
        # Plot fit
        t_fit = np.linspace(min(times), max(times), 100)
        y_fit = ExponentialKinetics.exponential_model(
            t_fit, 
            self.results['y_max'], 
            self.results['tau_inv']
        )
        ax2.plot(t_fit, y_fit, 'b-', linewidth=2, 
                label=f"Fit: y_max={self.results['y_max']:.3f}, 1/τ={self.results['tau_inv']:.3f}")
        
        ax2.set_xlabel("Time (min)")
        ax2.set_ylabel("Average Fluorescence")
        ax2.set_title("Fluorescence Kinetics")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, tab)
        toolbar.update()
    
    def plot_ks_analysis(self):
        """Plot KS test analysis for each time point"""
        tab = self.plot_tabs["KS Analysis"]
        
        alpha_results = self.results.get('alpha_results', [])
        if not alpha_results:
            return
        
        n_plots = len(alpha_results)
        n_cols = min(4, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, result in enumerate(alpha_results):
            ax = axes[i]
            
            # Plot probability vs alpha
            color = 'tab:blue'
            ax.plot(result['alphas'], result['probabilities'], 'b-', linewidth=2)
            ax.axvline(x=result['alpha'], color='r', linestyle='--', 
                      label=f"Optimal α={result['alpha']:.3f}")
            
            ax.set_xlabel("Shift α")
            ax.set_ylabel("Pr(H0|D)", color=color)
            ax.set_title(f"t = {result['time']:.1f} min")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Add secondary axis for D values
            ax2 = ax.twinx()
            ax2.plot(result['alphas'], result['D_values'], 'r-', alpha=0.3, linewidth=1)
            ax2.set_ylabel("D statistic", color='tab:red')
            ax2.tick_params(axis='y', labelcolor='tab:red')
        
        # Hide unused axes
        for i in range(len(alpha_results), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, tab)
        toolbar.update()
    
    def plot_results_table(self):
        """Display results in a table"""
        tab = self.plot_tabs["Results Table"]
        
        # Create treeview
        columns = ("Parameter", "Value", "Error")
        tree = ttk.Treeview(tab, columns=columns, show='headings', height=10)
        
        tree.heading("Parameter", text="Parameter")
        tree.heading("Value", text="Value")
        tree.heading("Error", text="Error")
        
        tree.column("Parameter", width=200)
        tree.column("Value", width=150)
        tree.column("Error", width=150)
        
        # Add data
        tree.insert("", tk.END, values=("y_max", f"{self.results['y_max']:.4f}", f"±{self.results['y_max_err']:.4f}"))
        tree.insert("", tk.END, values=("1/τ", f"{self.results['tau_inv']:.4f}", f"±{self.results['tau_inv_err']:.4f}"))
        tree.insert("", tk.END, values=("τ (min)", f"{1/self.results['tau_inv']:.2f}", ""))
        
        k_plus = self.results['k_plus']
        if k_plus > 0:
            tree.insert("", tk.END, values=("k+ (M⁻¹s⁻¹)", f"{k_plus:.2e}", ""))
        else:
            tree.insert("", tk.END, values=("k+ (M⁻¹s⁻¹)", "N/A", ""))
        
        tree.insert("", tk.END, values=("", "", ""))
        tree.insert("", tk.END, values=("Time Points", f"{len(self.results['times'])}", ""))
        
        # Pack tree
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Add text with summary
        summary_text = f"""
        Dynamic Gating Analysis Results
        ===============================
        
        Method: Kolmogorov-Smirnov Test-based Dynamic Gating
        
        Fitted Parameters:
        • y_max = {self.results['y_max']:.4f} ± {self.results['y_max_err']:.4f}
        • 1/τ = {self.results['tau_inv']:.4f} ± {self.results['tau_inv_err']:.4f} min⁻¹
        • τ = {1/self.results['tau_inv']:.2f} min
        
        Association rate constant:
        • k+ = {k_plus:.2e} M⁻¹s⁻¹ (based on A₀ = {float(self.A0_var.get()):.2e} M)
        
        The scaling coefficient φ(t) was determined using the two-sample KS test
        to find the optimal shift that maximizes the probability that two samples
        come from the same underlying distribution.
        """
        
        text_widget = tk.Text(tab, height=12, wrap=tk.WORD)
        text_widget.insert(tk.END, summary_text)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def export_results(self):
        """Export results to CSV files"""
        if not self.results:
            messagebox.showerror("Error", "No results to export")
            return
        
        # Select directory
        export_dir = filedialog.askdirectory(title="Select Export Directory")
        if not export_dir:
            return
        
        try:
            # Export kinetics data
            kinetics_df = pd.DataFrame({
                'Time_min': self.results['times'],
                'Phi': self.results['phis'],
                'Average_Fluorescence': self.results['fluoro_kinetics']
            })
            kinetics_df.to_csv(os.path.join(export_dir, 'kinetics.csv'), index=False)
            
            # Export fitted parameters
            params_df = pd.DataFrame({
                'Parameter': ['y_max', '1/tau', 'tau', 'k_plus'],
                'Value': [
                    self.results['y_max'],
                    self.results['tau_inv'],
                    1/self.results['tau_inv'],
                    self.results['k_plus']
                ],
                'Error': [
                    self.results['y_max_err'],
                    self.results['tau_inv_err'],
                    0,
                    0
                ]
            })
            params_df.to_csv(os.path.join(export_dir, 'parameters.csv'), index=False)
            
            # Export KS analysis for each time point
            for i, result in enumerate(self.results['alpha_results']):
                ks_df = pd.DataFrame({
                    'Alpha': result['alphas'],
                    'Probability_H0_given_D': result['probabilities'],
                    'D_statistic': result['D_values']
                })
                ks_df.to_csv(
                    os.path.join(export_dir, f'ks_analysis_t{result["time"]:.1f}min.csv'),
                    index=False
                )
            
            messagebox.showinfo("Success", f"Results exported to {export_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")


def main():
    root = tk.Tk()
    app = DynamicGatingGUI(root)
    
    # Configure style
    style = ttk.Style()
    style.configure('Large.TButton', font=('Arial', 11, 'bold'))
    
    root.mainloop()


if __name__ == "__main__":
    print("Dynamic Gating Analysis Tool")
    print("Based on Kolmogorov-Smirnov test of cell distributions")
    print("=" * 60)
    print()
    print("Required packages:")
    print("  - numpy, scipy, pandas")
    print("  - matplotlib, tkinter")
    print("  - Optional: FlowCytometryTools, fcsparser")
    print()
    print("Starting GUI...")
    
    main()