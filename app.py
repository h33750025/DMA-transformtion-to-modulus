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

# Custom plot styles
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['font.family'] = 'Times New Roman'

# ==========================================
# Helper Functions
# ==========================================

def is_dark_color(hex_color):
    """Check if a color is dark to adjust text contrast."""
    rgb = mcolors.hex2color(hex_color)
    h, l, s = rgb_to_hls(*rgb)
    return l < 0.5

def add_watermark(ax, text="NYU-ViscoMOD"):
    """Adds the watermark to a matplotlib axis."""
    ax.text(
        0.99, 0.01, text,
        fontsize=14, fontname="Times New Roman", color="purple",
        ha="right", va="bottom", alpha=0.5, transform=ax.transAxes
    )

def storage_modulus_model(log_omega, a, b, c, d):
    """The hyperbolic tangent model for E'(w)."""
    return a * np.tanh(b * (log_omega + c)) + d

# ==========================================
# Session State
# ==========================================
if 'data' not in st.session_state: st.session_state.data = None
if 'analysis_shift_factors' not in st.session_state: st.session_state.analysis_shift_factors = {}
if 'fitted_params' not in st.session_state: st.session_state.fitted_params = {}
if 'master_curve_data' not in st.session_state: st.session_state.master_curve_data = None
if 'param_per_temp' not in st.session_state: st.session_state.param_per_temp = None

# ==========================================
# Pages
# ==========================================

def page_load_data():
    st.title("Step 1: Load Data")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            cols = ['Frequency', 'Storage Modulus', 'Temperature']
            # Basic validation
            if all(c in df.columns for c in cols):
                df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
                df.dropna(subset=cols, inplace=True)
                st.session_state.data = df
                st.success(f"Loaded {len(df)} rows.")
                st.dataframe(df.head())
            else:
                st.error(f"CSV must contain columns: {cols}")
        except Exception as e:
            st.error(f"Error: {e}")

def page_raw_data():
    st.title("Step 2: Raw Data")
    if st.session_state.data is None: return st.warning("No data loaded.")
    
    data = st.session_state.data
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    temps = sorted(data['Temperature'].unique())
    darker = [c for c in list(mcolors.CSS4_COLORS.values()) if is_dark_color(c)]
    
    for i, temp in enumerate(temps):
        sub = data[data['Temperature'] == temp].sort_values('Frequency')
        ax.semilogx(sub['Frequency'], sub['Storage Modulus'], 
                   color=darker[i%len(darker)], label=f"{temp} °C", marker='o')
    
    ax.set_xlabel('Frequency (Hz)', fontsize=14)
    ax.set_ylabel("Storage Modulus (MPa)", fontsize=14)
    ax.set_title("Raw Data Plot", fontsize=16)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, ls="--")
    add_watermark(ax)
    st.pyplot(fig)

def page_3d_surface():
    st.title("Step 3: 3D Surface")
    if st.session_state.data is None: return st.warning("No data loaded.")
    
    data = st.session_state.data
    X, Y, Z = data['Temperature'], data['Frequency'], data['Storage Modulus']
    x_grid, y_grid = np.meshgrid(np.linspace(X.max(), X.min(), 100), np.linspace(Y.min(), Y.max(), 100))
    
    try:
        Z_grid = griddata((X, Y), Z, (x_grid, y_grid), method='cubic')
        fig = Figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(x_grid, y_grid, Z_grid, cmap='rainbow', edgecolor='none', alpha=0.8)
        
        ax.set_xlabel('Temperature (°C)'); ax.set_ylabel('Frequency (Hz)'); ax.set_zlabel('Modulus (MPa)')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10).set_label('Storage Modulus (MPa)')
        add_watermark(ax)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"3D Plot Error: {e}")

def page_tts():
    st.title("Step 4: TTS Analysis")
    if st.session_state.data is None: return st.warning("No data loaded.")
    
    if st.button("Run Auto-Shift"):
        data = st.session_state.data
        ref_temp = min(data["Temperature"].unique())
        extended = data[data["Temperature"] == ref_temp].sort_values('Frequency')
        shift_factors = {ref_temp: 1.0}
        
        for temp in sorted([t for t in data["Temperature"].unique() if t > ref_temp]):
            df_temp = data[data["Temperature"] == temp].sort_values('Frequency')
            max_freq = df_temp["Frequency"].max()
            mod_at_max = df_temp.iloc[-1]["Storage Modulus"]
            
            # Find overlap match
            idx = (extended["Storage Modulus"] - mod_at_max).abs().idxmin()
            match_freq = extended.loc[idx, "Frequency"]
            
            sf = match_freq / max_freq
            shift_factors[temp] = sf
            
            shifted = df_temp.copy()
            shifted["Frequency"] *= sf
            extended = pd.concat([extended, shifted], ignore_index=True).sort_values("Frequency")
            
        st.session_state.analysis_shift_factors = shift_factors
        st.session_state.master_curve_data = extended
        st.success("TTS Complete")
        
    if st.session_state.master_curve_data is not None:
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        shifts = st.session_state.analysis_shift_factors
        
        for t in sorted(shifts.keys()):
            sf = shifts[t]
            sub = st.session_state.data[st.session_state.data['Temperature'] == t]
            ax.semilogx(sub['Frequency']*sf, sub['Storage Modulus'], 'o', label=f"{t}C (aT={sf:.2e})")
            
        ax.set_xlabel("Reduced Frequency (Hz)"); ax.set_ylabel("Storage Modulus (MPa)")
        ax.set_title("Master Curve")
        ax.legend()
        add_watermark(ax)
        st.pyplot(fig)

def page_fitting():
    st.title("Step 5: Curve Fitting")
    if not st.session_state.analysis_shift_factors: return st.warning("Run Step 4 first.")
    
    a_high = st.slider("Max 'a'", 10.0, 5000.0, 2000.0)
    d_high = st.slider("Max 'd'", 10.0, 5000.0, 2000.0)
    
    if st.button("Fit Model"):
        data, shifts = st.session_state.data, st.session_state.analysis_shift_factors
        x_all, y_all = [], []
        
        for t, sf in shifts.items():
            sub = data[data['Temperature'] == t]
            valid = sub[sub['Frequency'] > 0]
            x_all.extend(np.log10(valid['Frequency'] * sf))
            y_all.extend(valid['Storage Modulus'])
            
        try:
            popt, _ = curve_fit(storage_modulus_model, x_all, y_all, 
                              bounds=([1e-6, -100, -100, 1e-6], [a_high, 100, 100, d_high]), maxfev=100000)
            
            params = {'a': popt[0], 'b': popt[1], 'c': popt[2], 'd': popt[3]}
            params['r2'] = r2_score(y_all, storage_modulus_model(x_all, *popt))
            st.session_state.fitted_params = params
            st.success(f"Fit Complete. R2 = {params['r2']:.4f}")
        except Exception as e:
            st.error(f"Fit Failed: {e}")

    if st.session_state.fitted_params:
        params = st.session_state.fitted_params
        st.json(params)
        
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Plot data
        all_x = []
        shifts = st.session_state.analysis_shift_factors
        for t in sorted(shifts):
            sub = st.session_state.data[st.session_state.data['Temperature'] == t]
            freq = sub['Frequency'] * shifts[t]
            mask = freq > 0
            x_log = np.log10(freq[mask])
            ax.scatter(x_log, sub.loc[mask, 'Storage Modulus'], alpha=0.5, label=f"{t}C")
            all_x.extend(x_log)
            
        # Plot fit
        if all_x:
            x_rng = np.linspace(min(all_x)-0.5, max(all_x)+0.5, 500)
            y_fit = storage_modulus_model(x_rng, params['a'], params['b'], params['c'], params['d'])
            ax.plot(x_rng, y_fit, 'r-', lw=3, label="Model")
            
        ax.set_xlabel("Log(Frequency)"); ax.set_ylabel("Modulus (MPa)")
        ax.legend()
        add_watermark(ax)
        st.pyplot(fig)

def page_params_per_temp():
    st.title("Step 6: Master Curve Parameters")
    if not st.session_state.fitted_params: return st.warning("Run Step 5 first.")
    
    st.markdown("Calculates the model parameters for each temperature based on the shift factors.")
    
    shifts = st.session_state.analysis_shift_factors
    master = st.session_state.fitted_params
    
    results = []
    for t in sorted(shifts.keys()):
        log_aT = np.log10(shifts[t])
        # The shift affects the 'c' parameter: c_new = c_master + log(aT)
        c_temp = master['c'] + log_aT
        results.append({
            "Temperature": t,
            "Shift Factor": shifts[t],
            "a": master['a'],
            "b": master['b'],
            "c": c_temp,
            "d": master['d']
        })
    
    df_res = pd.DataFrame(results)
    st.session_state.param_per_temp = df_res
    st.dataframe(df_res)
    
    # Download
    csv = df_res.to_csv(index=False)
    st.download_button("Download Parameters CSV", csv, "parameters_per_temp.csv", "text/csv")

def page_elastic_modulus():
    st.title("Step 7: Elastic Modulus vs Strain Rate")
    if st.session_state.fitted_params is None: return st.warning("Run Step 5 first.")
    
    st.markdown("Predicts Elastic Modulus ($E$) as a function of Strain Rate ($\dot{\epsilon}$).")
    
    strain_rates = st.text_input("Enter Strain Rates (comma separated)", "0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1")
    
    if st.button("Calculate Elastic Modulus"):
        try:
            rates = [float(x.strip()) for x in strain_rates.split(',')]
            rates = np.array(sorted(rates))
            
            # Using the Master Curve model parameters
            # Input to model is log10(frequency), assuming Frequency ~ Strain Rate
            
            params = st.session_state.fitted_params
            log_rates = np.log10(rates)
            E_values = storage_modulus_model(log_rates, params['a'], params['b'], params['c'], params['d'])
            
            # Display Result Table
            res_df = pd.DataFrame({"Strain Rate (1/s)": rates, "Predicted Elastic Modulus (MPa)": E_values})
            st.dataframe(res_df)
            
            # Plot
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            ax.semilogx(rates, E_values, 'bs-', lw=2)
            ax.set_xlabel("Strain Rate (1/s)"); ax.set_ylabel("Elastic Modulus (MPa)")
            ax.set_title("Elastic Modulus vs Strain Rate")
            ax.grid(True, which="both", ls="--")
            add_watermark(ax)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Calculation Error: {e}")

# ==========================================
# Main Navigation
# ==========================================
def main():
    st.sidebar.title("NYU-ViscoMOD Web")
    
    pages = {
        "1. Load Data": page_load_data,
        "2. Raw Data": page_raw_data,
        "3. 3D Surface": page_3d_surface,
        "4. TTS Analysis": page_tts,
        "5. Curve Fitting": page_fitting,
        "6. Params per Temp": page_params_per_temp,
        "7. Elastic Modulus": page_elastic_modulus
    }
    
    selection = st.sidebar.radio("Go to Step:", list(pages.keys()))
    pages[selection]()

if __name__ == "__main__":
    main()
