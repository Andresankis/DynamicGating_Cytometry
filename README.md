Dynamic Gating Analysis Tool
Summary
A Python-based GUI application for flow cytometry data analysis that performs 2D dynamic gating with CDF shape matching. The program optimizes gate position on one fluorescence channel (X-axis) while keeping the gate fixed on another channel (Y-axis) across time-series measurements. The optimization finds the gate position that minimizes the difference between CDF shapes (centered, position-independent comparison) of the gated populations.
Key Features
•	2D cytogram visualization with interactive rectangular gate selection
•	CDF shape-based optimization - finds optimal X-gate position by minimizing L2 norm of centered CDF differences
•	Fixed Y-gate across all time points
•	Selectable reference time point (not necessarily the last one)
•	Optional fraction matching - penalizes deviations in the percentage of events in the gate
•	Kinetics fitting with A + B•(1 - exp(-t/τ)) model for both Mean X and Gate Position
•	Comprehensive visualization: cytograms, histograms, CDF comparisons, search curves, kinetics plots
•	Results export to CSV files
User Guide
1. Load Data Files
1.	Click "Add Time Point Files" button
2.	Select your FCS/LMD files (multiple selection allowed)
3.	In the dialog, enter the time (in minutes) for each file
4.	Files are sorted by time automatically
2. Select Channels
•	Y-axis channel (FIXED gate): Select the channel for the fixed gate (e.g., FSC-A, SSC-A)
•	X-axis channel (OPTIMIZED position): Select the channel whose gate position will be optimized (e.g., FL1-A)
Log transform options are not applied - raw data is used directly
3. Set Reference Gate
5.	Select the reference time point from the dropdown list
6.	Click "Set Reference Gate (2D cytogram)"
7.	In the new window, click and drag with mouse to draw a rectangular gate
8.	Use the toolbar (zoom/pan) if needed for precise selection
9.	The statistics panel shows gate boundaries, event count, fraction, mean, and sigma
10.	Click "Save Gate" to confirm
4. Configure Search Parameters
•	X search range (± units): Range to search for optimal X-gate position (default: 100000)
•	Number of search steps: More steps = finer search but slower (default: 200)
Step size is automatically calculated and displayed
5. Optional Settings
•	Match gate fraction: When checked, the optimization penalizes deviations in the percentage of events in the gate compared to the reference
•	Fraction penalty weight: Adjustable weight for this penalty (default: 10.0)
•	Fit Mean X: Fits A + B•(1 - exp(-t/τ)) to mean X values vs time
•	Fit Gate Position: Fits A + B•(1 - exp(-t/τ)) to gate center position vs time
6. Run Analysis
Click "Run Dynamic Gating Analysis" to start the optimization.
The program will:
11.	Apply the fixed Y-gate to all time points
12.	For each non-reference time point, search for the X-gate position that minimizes CDF shape difference
13.	Calculate mean and sigma for each optimized gate
14.	Fit kinetics models (if enabled)
15.	Display results in the plots and tables
7. View Results
The results are displayed in 7 tabs:
Tab	Description
2D Cytograms	Scatter plots with gate rectangles (green=reference, red=optimized)
Histograms (X)	X-channel distributions after Y-gating with gate boundaries
CDF Shape Comparison	Centered CDFs (shape comparison) and original CDFs
Gate Search	Shape score and KS distance vs X-shift curves
Kinetics (Mean X)	Mean X values vs time with exponential fit
Kinetics (Gate Position)	Gate center position vs time with exponential fit
Results Table	Summary of all parameters and results
8. Export Results
Click "Export Results" to save CSV files:
•	gate_results.csv - Gate positions, statistics for all time points
•	mean_kinetics_fit.csv - Fitted parameters for Mean X kinetics (if enabled)
•	position_kinetics_fit.csv - Fitted parameters for Gate Position kinetics (if enabled)
•	search_t*.csv - Search curves for each time point
Algorithm Details
16.	Reference gate is defined on the selected time point (X-range and Y-range)
17.	For each other time point: Y-gate is applied as-is (same absolute values), X-gate position is searched by trying many small shifts
18.	For each shift, the CDF of gated X-values is computed and centered (shifted to start at 0)
19.	Shape score = L2 norm of difference between reference and target centered CDFs
20.	Optional fraction penalty = weight × |target_fraction - reference_fraction|
21.	Optimal shift = position with minimum combined score
22.	Gate width is preserved throughout the optimization
Requirements
•	Python 3.7+
•	numpy, scipy, pandas, matplotlib, tkinter
•	Optional: FlowCytometryTools, fcsparser (for FCS file reading)
Tips
•	For linear-scale data (e.g., FSC values), use a larger search range (e.g., 100000)
•	For log-scale data, convert before loading or use smaller search range
•	More search steps give smoother optimization curves but take longer
•	The reference time point can be any point, not necessarily the last one
•	Gate fraction matching is useful when you expect similar percentages of gated events across time points
