import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from colorsys import rgb_to_hls

# ==========================================
# Configuration & Global Styles
# ==========================================
st.set_page_config(
    page_title="NYU-ViscoMOD v1.1.4 (Web)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom plot styles to match the original
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['font.family'] = 'Times New Roman'

# ==========================================
# Helper Functions
# ==========================================

# Compatibility for NumPy 2.0+ where trapz is renamed to trapezoid
try:
    trapz = np.trapezoid
except AttributeError:
    trapz = np.trapz

def is_dark_color(hex_color):
    """Check if a color is dark to adjust text contrast if needed."""
    rgb = mcolors.hex2color(hex_color)
    h, l, s = rgb_to_hls(*rgb)
    return l < 0.5

def add_watermark(ax, text="NYU-ViscoMOD"):
    """Adds the watermark to a matplotlib axis."""
    ax.text(
        0.99, 0.01,
        text,
        fontsize=14,
        fontname="Times New Roman",
        color="purple",
        ha="right", va="bottom", alpha=0.5,
        transform=ax.transAxes
    )

def storage_modulus_model(log_omega, a, b, c, d):
    """The hyperbolic tangent model for E'(w)."""
    return a * np.tanh(b * (log_omega + c)) + d

def Etime_time_cycle(time, cycle, a, b, c, d):
    """Calculates E(t) based on the fitted parameters using the integration method."""
    Etime = np.zeros_like(time)
    N1, N2, N3 = 240, 74, 24
    
    def E_prime(w, a, b, c, d):
        # Model expects log(w)
        return a * np.tanh(b * (np.log(w) + c)) + d
        
    def integrand(t_val, E_prime_w, w_val):
        return (2/np.pi) * (E_prime_w / w_val) * np.sin(w_val * t_val)
        
    for i, t_val in enumerate(time):
        if t_val == 0: continue
        # Integration ranges based on cycle and time
        w1 = np.linspace(1e-6 / t_val, cycle * 0.1 * 2 * np.pi / t_val, int(cycle * 0.1 * N1) + 1)
        w2 = np.linspace(cycle * 0.1 * 2 * np.pi / t_val, cycle * 0.4 * 2 * np.pi / t_val, int(cycle * 0.3 * N2) + 1)
        w3 = np.linspace(cycle * 0.4 * 2 * np.pi / t_val, cycle * 2 * np.pi / t_val, int(cycle * 0.6 * N3) + 1)
        
        # Combine ranges, skipping duplicates at boundaries
        all_w = np.concatenate([w1, w2[1:], w3[1:]])
        
        y = integrand(t_val, E_prime(all_w, a, b, c, d), all_w)
        
        # FIX: Use the compatible trapz function defined above
        Etime[i] = trapz(y, all_w)
        
    return Etime

# ==========================================
# Session State Initialization
# ==========================================
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analysis_shift_factors' not in st.session_state:
    st.session_state.analysis_shift_factors = {}
if 'fitted_params' not in st.session_state:
    st.session_state.fitted_params = {}
if 'master_curve_data' not in st.session_state:
    st.session_state.master_curve_data = None

# ==========================================
# Page Logic
# ==========================================

def page_load_data():
    st.title("Step 1: Load Data")
    st.markdown("Upload your Viscoelastic Data CSV. The file must contain **Frequency**, **Storage Modulus**, and **Temperature** columns.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Ensure numeric types
                cols_to_check = ['Frequency', 'Storage Modulus', 'Temperature']
                missing_cols = [c for c in cols_to_check if c not in df.columns]
                
                if missing_cols:
                    st.error(f"Missing columns: {missing_cols}")
                else:
                    df['Frequency'] = pd.to_numeric(df['Frequency'], errors='coerce')
                    df['Storage Modulus'] = pd.to_numeric(df['Storage Modulus'], errors='coerce')
                    df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
                    
                    df.dropna(subset=cols_to_check, inplace=True)
                    
                    st.session_state.data = df
                    st.success(f"Loaded {len(df)} rows successfully.")
                    st.dataframe(df.head())
            except Exception as e:
                st.error(f"Error loading file: {e}")

    with col2:
        st.info("**Instructions:**\n1. Prepare a CSV file with columns: `Frequency`, `Storage Modulus`, `Temperature`.\n2. Upload it on the left.\n3. Navigate to 'Raw Data' to visualize.")
        st.markdown("---")
        st.markdown("*Software Patent: US Patent #10,345,210*")
        st.markdown("*Developed at NYU*")


def page_raw_data():
    st.title("Step 2: Raw Data Visualization")
    
    if st.session_state.data is None:
        st.warning("Please upload data in 'Load Data' first.")
        return

    data = st.session_state.data
    
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    temps = sorted(data['Temperature'].unique())
    all_colors = list(mcolors.CSS4_COLORS.values())
    # Filter for darker colors for better visibility
    darker = [c for c in all_colors if is_dark_color(c)]
    
    for i, temp in enumerate(temps):
        subdata = data[data['Temperature'] == temp].sort_values('Frequency')
        # Use a cyclical color picker
        color = darker[i % len(darker)]
        ax.semilogx(subdata['Frequency'], subdata['Storage Modulus'], 
                    color=color, label=f"{temp} °C", linewidth=1.5, marker='o', markersize=4)

    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)', fontsize=14)
    ax.set_ylabel("Storage Modulus (MPa)", fontsize=14)
    ax.set_title('Raw Data Plot', fontsize=16)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    add_watermark(ax)
    
    st.pyplot(fig)


def page_3d_surface():
    st.title("Step 3: 3D Surface Plot")
    
    if st.session_state.data is None:
        st.warning("Please upload data first.")
        return

    data = st.session_state.data
    
    # Prepare data
    X = data['Temperature']
    Y = data['Frequency']
    Z = data['Storage Modulus']

    # Grid for interpolation
    x_grid, y_grid = np.meshgrid(
        np.linspace(X.max(), X.min(), 100),
        np.linspace(Y.min(), Y.max(), 100)
    )
    
    try:
        Z_grid = griddata((X, Y), Z, (x_grid, y_grid), method='cubic')
    except Exception as e:
        st.error(f"Error generating surface (data might be too sparse): {e}")
        return

    fig = Figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(x_grid, y_grid, Z_grid, cmap='rainbow', edgecolor='none', alpha=0.8)
    
    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    ax.set_zlabel('Storage Modulus (MPa)', fontsize=12)
    
    # Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Storage Modulus (MPa)')
    
    # Standard watermark call
    add_watermark(ax)
    st.pyplot(fig)


def page_analysis_tts():
    st.title("Step 4: Time-Temperature Superposition (TTS)")
    
    if st.session_state.data is None:
        st.warning("Please upload data first.")
        return

    data = st.session_state.data
    
    st.markdown("""
    This step automatically shifts the curves horizontally to form a **Master Curve**.
    It selects the lowest temperature as the reference and shifts higher temperatures to overlap.
    """)
    
    if st.button("Run Auto-Shift Analysis"):
        reference_temp = min(data["Temperature"].unique())
        df_ref = data[data["Temperature"] == reference_temp].sort_values('Frequency')
        
        extended_data = df_ref.copy()
        temperatures = sorted(data["Temperature"].unique())
        temperatures = [t for t in temperatures if t > reference_temp]
        
        shift_factors = {reference_temp: 1.0}
        
        # Auto-shift logic
        for temp in temperatures:
            df_temp = data[data["Temperature"] == temp].sort_values('Frequency')
            
            # Find overlap: Max freq of current master curve vs Data
            # Heuristic: Match the modulus value at the boundary
            max_freq = df_temp["Frequency"].max()
            modulus_at_max = df_temp.iloc[-1]["Storage Modulus"] 
            
            # Find closest modulus in the accumulated data
            idx = (extended_data["Storage Modulus"] - modulus_at_max).abs().idxmin()
            closest_match_freq = extended_data.loc[idx, "Frequency"]
            
            # Calculate shift factor aT
            shift_factor = closest_match_freq / max_freq
            shift_factors[temp] = shift_factor
            
            # Shift the new segment
            shifted_df = df_temp.copy()
            shifted_df["Frequency"] = shifted_df["Frequency"] * shift_factor
            
            # Add to master curve data
            extended_data = pd.concat([extended_data, shifted_df], ignore_index=True)
            extended_data = extended_data.sort_values("Frequency")
        
        st.session_state.analysis_shift_factors = shift_factors
        st.session_state.master_curve_data = extended_data
        st.success("Analysis Complete!")

    # Display results
    if st.session_state.master_curve_data is not None:
        shift_factors = st.session_state.analysis_shift_factors
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            
            ref_temp = min(shift_factors.keys())
            temps = sorted(st.session_state.data['Temperature'].unique())
            
            for temp in temps:
                sf = shift_factors.get(temp, 1.0)
                sub = st.session_state.data[st.session_state.data['Temperature'] == temp]
                ax.plot(sub['Frequency'] * sf, sub['Storage Modulus'], 'o', markersize=3, label=f"{temp}°C (aT={sf:.2e})")

            ax.set_xscale("log")
            ax.set_xlabel("Reduced Frequency (Hz)", fontsize=14)
            ax.set_ylabel("Storage Modulus (MPa)", fontsize=14)
            ax.set_title(f"Master Curve (Ref: {ref_temp}°C)", fontsize=16)
            ax.legend()
            ax.grid(True, which="both", ls="--")
            add_watermark(ax)
            st.pyplot(fig)
            
        with col2:
            st.write("### Shift Factors")
            sf_df = pd.DataFrame(list(shift_factors.items()), columns=['Temp (°C)', 'Shift Factor'])
            st.dataframe(sf_df)


def page_curve_fitting():
    st.title("Step 5: Master Curve Fitting")
    
    if not st.session_state.analysis_shift_factors:
        st.warning("Please run the TTS Analysis first.")
        return

    st.markdown("Fit the Master Curve to the hyperbolic tangent model:  \n$E'(\\omega) = a \\tanh(b(\\log(\\omega) + c)) + d$")

    col_ctrl, col_plot = st.columns([1, 3])
    
    with col_ctrl:
        st.subheader("Bounds")
        a_upper = st.slider("Upper Bound for 'a'", 10.0, 5000.0, 2000.0, 10.0)
        d_upper = st.slider("Upper Bound for 'd'", 10.0, 5000.0, 2000.0, 10.0)
        
        if st.button("Fit Curve"):
            # Prepare data
            data = st.session_state.data
            shift_factors = st.session_state.analysis_shift_factors
            
            combined_log_freq = []
            combined_modulus = []
            
            for t, sf in shift_factors.items():
                sub = data[data['Temperature'] == t]
                # Filter > 0 for log
                valid_sub = sub[sub['Frequency'] > 0]
                # Use Log10 of reduced frequency for X
                combined_log_freq.extend(np.log10(valid_sub['Frequency'] * sf))
                combined_modulus.extend(valid_sub['Storage Modulus'])
            
            X_fit = np.array(combined_log_freq)
            Y_fit = np.array(combined_modulus)
            
            # Bounds: [a, b, c, d]
            lower_bounds = [1e-6, -100.0, -100.0, 1e-6]
            upper_bounds = [a_upper, 100.0, 100.0, d_upper]
            
            try:
                popt, _ = curve_fit(storage_modulus_model, X_fit, Y_fit, bounds=(lower_bounds, upper_bounds), maxfev=100000)
                st.session_state.fitted_params = {
                    'a': popt[0], 'b': popt[1], 'c': popt[2], 'd': popt[3],
                    'r2': r2_score(Y_fit, storage_modulus_model(X_fit, *popt))
                }
                st.success("Fitting successful!")
            except Exception as e:
                st.error(f"Fitting failed: {e}")

    with col_plot:
        if st.session_state.fitted_params:
            params = st.session_state.fitted_params
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            
            data = st.session_state.data
            shift_factors = st.session_state.analysis_shift_factors
            all_x_vals = []
            
            # Plot Scatter
            for t in sorted(data['Temperature'].unique()):
                sf = shift_factors.get(t, 1.0)
                sub = data[data['Temperature'] == t]
                reduced_freq = sub['Frequency'] * sf
                # Avoid log(0)
                mask = reduced_freq > 0
                x_vals = np.log10(reduced_freq[mask])
                y_vals = sub['Storage Modulus'][mask]
                
                all_x_vals.extend(x_vals)
                ax.scatter(x_vals, y_vals, s=10, alpha=0.5, label=f"{t}C")

            # Plot Line
            if all_x_vals:
                x_min, x_max = min(all_x_vals), max(all_x_vals)
                # Range with padding
                x_range = np.linspace(x_min - 0.5, x_max + 0.5, 500)
                y_fit = storage_modulus_model(x_range, params['a'], params['b'], params['c'], params['d'])
                ax.plot(x_range, y_fit, 'r-', linewidth=3, label="Fitted Model")
            
            ax.set_xlabel("Log(Frequency)", fontsize=14)
            ax.set_ylabel("Storage Modulus (MPa)", fontsize=14)
            ax.set_title(f"Curve Fit (R² = {params['r2']:.4f})", fontsize=16)
            ax.legend()
            ax.grid(True)
            add_watermark(ax)
            st.pyplot(fig)
            
            st.write("### Fitted Parameters")
            st.json(params)
        else:
            st.info("Click 'Fit Curve' to generate the model.")

def page_etime():
    st.title("Step 6: Time Domain E(t)")
    
    if not st.session_state.fitted_params:
        st.warning("Please fit the curve in Step 5 first.")
        return

    st.markdown("Calculate **E(t)** using the fitted parameters from the Master Curve.")
    
    cycle = st.number_input("Cycle Parameter", value=10.0)
    
    if st.button("Calculate E(t)"):
        params = st.session_state.fitted_params
        # Logspace time
        time = np.logspace(-5, 5, 100)
        
        with st.spinner("Calculating integral..."):
            E_t = Etime_time_cycle(time, cycle, params['a'], params['b'], params['c'], params['d'])
        
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.loglog(time, E_t, 'b-', linewidth=2)
        ax.set_xlabel("Time (s)", fontsize=14)
        ax.set_ylabel("Relaxation Modulus E(t) (MPa)", fontsize=14)
        ax.set_title("Relaxation Modulus vs Time", fontsize=16)
        ax.grid(True, which="both", ls="--")
        add_watermark(ax)
        
        st.pyplot(fig)


# ==========================================
# Main App Structure
# ==========================================
def main():
    st.sidebar.title("NYU-ViscoMOD")
    
    pages = {
        "1. Load CSV": page_load_data,
        "2. Raw Data": page_raw_data,
        "3. 3D Surface": page_3d_surface,
        "4. TTS Analysis": page_analysis_tts,
        "5. Curve Fitting": page_curve_fitting,
        "6. E(t) Prediction": page_etime
    }
    
    selection = st.sidebar.radio("Navigation", list(pages.keys()))
    
    # Run the selected page function
    pages[selection]()

if __name__ == "__main__":
    main()
