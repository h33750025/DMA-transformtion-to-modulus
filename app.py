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
import io

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
    """Calculates E(t) based on the fitted parameters."""
    Etime = np.zeros_like(time)
    N1, N2, N3 = 240, 74, 24
    
    def E_prime(w, a, b, c, d):
        return a * np.tanh(b * (np.log(w) + c)) + d
        
    def integrand(t_val, E_prime_w, w_val):
        return (2/np.pi) * (E_prime_w / w_val) * np.sin(w_val * t_val)
        
    for i, t_val in enumerate(time):
        if t_val == 0: continue
        # Integration ranges
        w1 = np.linspace(1e-6 / t_val, cycle * 0.1 * 2 * np.pi / t_val, int(cycle * 0.1 * N1) + 1)
        w2 = np.linspace(cycle * 0.1 * 2 * np.pi / t_val, cycle * 0.4 * 2 * np.pi / t_val, int(cycle * 0.3 * N2) + 1)
        w3 = np.linspace(cycle * 0.4 * 2 * np.pi / t_val, cycle * 2 * np.pi / t_val, int(cycle * 0.6 * N3) + 1)
        all_w = np.concatenate([w1, w2[1:], w3[1:]])
        
        y = integrand(t_val, E_prime(all_w, a, b, c, d), all_w)
        Etime[i] = np.trapz(y, all_w)
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
                df['Frequency'] = pd.to_numeric(df['Frequency'], errors='coerce')
                df['Storage Modulus'] = pd.to_numeric(df['Storage Modulus'], errors='coerce')
                df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
                df.dropna(subset=['Frequency', 'Storage Modulus', 'Temperature'], inplace=True)
                
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
        ax.semilogx(subdata['Frequency'], subdata['Storage Modulus'], 
                    color=darker[i % len(darker)], label=f"{temp} °C", linewidth=1.5, marker='o', markersize=4)

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
    
    X = data['Temperature']
    Y = np.log10(data['Frequency']) # Log scale for frequency in mesh usually looks better or linear freq axis
    # The original code uses linear Frequency for meshgrid but data is often log. 
    # Let's stick to original code logic: raw values.
    Y = data['Frequency']
    Z = data['Storage Modulus']

    # Interpolation for surface
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
    
    add_watermark(ax, on_axes=True)
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
            
            # Find overlap point: max freq of current vs existing master curve
            # Simple heuristic from original code: match modulus at boundaries
            max_freq = df_temp["Frequency"].max()
            modulus_at_max = df_temp.iloc[-1]["Storage Modulus"] # Assuming sorted
            
            # Find closest modulus in the extended data constructed so far
            idx = (extended_data["Storage Modulus"] - modulus_at_max).abs().idxmin()
            closest_match_freq = extended_data.loc[idx, "Frequency"]
            
            # Calculate shift factor
            # Shift factor aT = f_ref / f_temp. 
            # Original code: shift_factor = closest_match["Frequency"] / max_freq
            # This aligns the rightmost point of the current temp curve to the matching modulus point on the master curve
            shift_factor = closest_match_freq / max_freq
            
            shift_factors[temp] = shift_factor
            
            # Create shifted dataframe segment
            shifted_df = df_temp.copy()
            shifted_df["Frequency"] = shifted_df["Frequency"] * shift_factor
            
            extended_data = pd.concat([extended_data, shifted_df], ignore_index=True)
            extended_data = extended_data.sort_values("Frequency")
        
        st.session_state.analysis_shift_factors = shift_factors
        st.session_state.master_curve_data = extended_data
        st.success("Analysis Complete!")

    # Display results if available
    if st.session_state.master_curve_data is not None:
        master_data = st.session_state.master_curve_data
        shift_factors = st.session_state.analysis_shift_factors
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            
            # Plot reference
            ref_temp = min(shift_factors.keys())
            
            # Plot all shifted segments
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
            st.write("### Shift Factors (aT)")
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
        a_upper = st.slider("Upper Bound for 'a'", 10.0, 2000.0, 500.0, 10.0)
        d_upper = st.slider("Upper Bound for 'd'", 10.0, 2000.0, 500.0, 10.0)
        
        if st.button("Fit Curve"):
            # Prepare data
            data = st.session_state.data
            shift_factors = st.session_state.analysis_shift_factors
            
            combined_log_freq = []
            combined_modulus = []
            
            for t, sf in shift_factors.items():
                sub = data[data['Temperature'] == t]
                combined_log_freq.extend(np.log10(sub['Frequency'] * sf))
                combined_modulus.extend(sub['Storage Modulus'])
            
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
            
            # Plot scatter data
            data = st.session_state.data
            shift_factors = st.session_state.analysis_shift_factors
            for t in sorted(data['Temperature'].unique()):
                sf = shift_factors.get(t, 1.0)
                sub = data[data['Temperature'] == t]
                ax.scatter(np.log10(sub['Frequency'] * sf), sub['Storage Modulus'], s=10, alpha=0.5, label=f"{t}C data")

            # Plot Fit
            x_range = np.linspace(min(np.log10(data['Frequency'].min())), max(np.log10(data['Frequency'].max())) + 5, 500)
            # Adjust range to cover master curve width
            x_min_all = min([np.min(np.log10(data[data['Temperature']==t]['Frequency']*shift_factors[t])) for t in shift_factors])
            x_max_all = max([np.max(np.log10(data[data['Temperature']==t]['Frequency']*shift_factors[t])) for t in shift_factors])
            x_range = np.linspace(x_min_all, x_max_all, 500)
            
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
        # Time range: usually inverse of frequency range
        time = np.logspace(-5, 5, 100)
        
        with st.spinner("Calculating integral... this may take a moment."):
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
