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
from matplotlib.lines import Line2D
import io
# ==========================================
# Configuration & Global Styles
# ==========================================
st.set_page_config(
    page_title="NYU-ViscoMOD v1.1.4 (Web)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CSS: Force Times New Roman for Web UI Elements
# ---------------------------------------------------------
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Times New Roman', Times, serif !important;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Times New Roman', Times, serif !important;
    }
    .stButton button {
        font-family: 'Times New Roman', Times, serif !important;
    }
    .stTextInput input {
        font-family: 'Times New Roman', Times, serif !important;
    }
    div[data-baseweb="slider"] {
        font-family: 'Times New Roman', Times, serif !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Matplotlib: ROBUST Times New Roman Configuration
# ---------------------------------------------------------
# 'stix' is a built-in matplotlib font that looks exactly like Times New Roman.
# It is much more reliable than 'custom' when the OS lacks specific font files.
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Liberation Serif', 'serif']
plt.rcParams['mathtext.fontset'] = 'stix' 
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# ==========================================
# Helper Functions
# ==========================================

def is_dark_color(hex_color):
    """Check if a color is dark to adjust text contrast."""
    rgb = mcolors.hex2color(hex_color)
    h, l, s = rgb_to_hls(*rgb)
    return l < 0.5

# def add_watermark(ax, text="NYU-ViscoMOD"):
#     """Adds the watermark to a matplotlib axis, handling both 2D and 3D plots."""
#     props = dict(
#         fontsize=14, fontname="Times New Roman", color="purple",
#         ha="right", va="bottom", alpha=0.5, transform=ax.transAxes
#     )
    
#     # Check if the axis is a 3D axis
#     if hasattr(ax, 'text2D'):
#         ax.text2D(0.99, 0.01, text, **props)
#     else:
#         ax.text(0.99, 0.01, text, **props)
#################################################        
def add_watermark(ax, text="NYU-ViscoMOD"):
    """Adds the watermark to a matplotlib axis, placing it outside the plot area."""
    props = dict(
        fontsize=14, fontname="Times New Roman", color="purple",
        ha="right", va="top", alpha=0.5, transform=ax.transAxes
    )
    
    # Check if the axis is a 3D axis
    if hasattr(ax, 'text2D'):
        # For 3D plots, placing it outside can be tricky due to rotation.
        # It is often safer to keep it slightly inside or anchor it to the figure corner.
        # This setting puts it at the bottom right corner of the drawing canvas.
        ax.text2D(1.1, -0.001, text, **props)
    else:
        # For 2D plots:
        # x=1.0 aligns with the right edge of the axis
        # y=-0.1 places it below the x-axis labels
        ax.text(1.0, -0.07, text, **props)
##########################################################       
def storage_modulus_model(log_omega, a, b, c, d):
    """The hyperbolic tangent model for E'(w)."""
    return a * np.tanh(b * (log_omega + c)) + d

# ==========================================
# Session State Initialization
# ==========================================
if 'data' not in st.session_state: st.session_state.data = None
if 'analysis_shift_factors' not in st.session_state: st.session_state.analysis_shift_factors = {}
if 'fitted_params' not in st.session_state: st.session_state.fitted_params = {}
if 'master_curve_data' not in st.session_state: st.session_state.master_curve_data = None
if 'param_per_temp' not in st.session_state: st.session_state.param_per_temp = None

# Initialize Global Bounds if they don't exist
if 'global_a_upper' not in st.session_state: st.session_state.global_a_upper = 500.0
if 'global_d_upper' not in st.session_state: st.session_state.global_d_upper = 500.0

# ==========================================
# Pages
# ==========================================

def page_load_and_visualize():
    st.title("Step 1: Load & Visualize Data")
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
                
                # ==========================================
                # PLOTTING SECTION
                # ==========================================
                st.markdown("---")
                st.subheader("Data Visualization")
                
                tab1, tab2 = st.tabs(["2D Raw Data Plot", "3D Surface Plot"])
                
                # --- 2D Plot Logic ---
                with tab1:
                    data = st.session_state.data
                    fig = Figure(figsize=(10, 6))
                    ax = fig.add_subplot(111)
                    
                    temps = sorted(data['Temperature'].unique())
                    darker = [c for c in list(mcolors.CSS4_COLORS.values()) if is_dark_color(c)]
                    
                    for i, temp in enumerate(temps):
                        sub = data[data['Temperature'] == temp].sort_values('Frequency')
                        ax.semilogx(sub['Frequency'], sub['Storage Modulus'], 
                                   color=darker[i%len(darker)], label=f"{temp} °C")
                    
                    ax.set_xlabel('Frequency (Hz)')
                    ax.set_ylabel("Storage Modulus (MPa)")
                    ax.set_title("Raw Data Plot")
                    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
                    add_watermark(ax)
                    st.pyplot(fig)

                    # 2. Save plot to a temporary buffer
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches='tight', dpi=500)
                    buf.seek(0)
        
                    # 3. Layout: Spacer on left, Button on right
                    # [5, 2] ratio gives 5 parts empty space, 2 parts for the button
                    buff_col, button_col = st.columns([5, 2]) 
                    
                    with button_col:
                        st.download_button(
                            label="💾 Download Graph",
                            data=buf,
                            file_name="Raw data plot.png",
                            mime="image/png",
                            use_container_width=True # Makes the button fill the column width
                        )

                # --- 3D Plot Logic ---
                with tab2:
                    data = st.session_state.data
                    X, Y, Z = data['Temperature'], data['Frequency'], data['Storage Modulus']
                    
                    x_grid, y_grid = np.meshgrid(np.linspace(X.max(), X.min(), 100), np.linspace(Y.min(), Y.max(), 100))
                    
                    try:
                        Z_grid = griddata((X, Y), Z, (x_grid, y_grid), method='cubic')
                        fig = Figure(figsize=(10, 7))
                        ax = fig.add_subplot(111, projection='3d')
                        surf = ax.plot_surface(x_grid, y_grid, Z_grid, cmap='rainbow', edgecolor='none', alpha=0.8)
                        
                        ax.set_xlabel('Temperature (°C)')
                        ax.set_ylabel('Frequency (Hz)')
                        ax.set_zlabel('Modulus (MPa)')
                        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10).set_label('Storage Modulus (MPa)')
                        add_watermark(ax)
                        st.pyplot(fig)
                        # 2. Save plot to a temporary buffer
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches='tight', dpi=500)
                        buf.seek(0)
            
                        # 3. Layout: Spacer on left, Button on right
                        # [5, 2] ratio gives 5 parts empty space, 2 parts for the button
                        buff_col, button_col = st.columns([5, 2]) 
                        
                        with button_col:
                            st.download_button(
                                label="💾 Download Graph",
                                data=buf,
                                file_name="3D Raw data plot.png",
                                mime="image/png",
                                use_container_width=True # Makes the button fill the column width
                            )
                    except Exception as e:
                        st.error(f"3D Plot Error: {e}")

            else:
                st.error(f"CSV must contain columns: {cols}")
        except Exception as e:
            st.error(f"Error: {e}")
    elif st.session_state.data is not None:
        # If data is already loaded but file uploader is empty (e.g. navigation back)
        st.info("Data already loaded. Upload a new file to replace it.")
        
        tab1, tab2 = st.tabs(["2D Raw Data Plot", "3D Surface Plot"])
        with tab1:
            data = st.session_state.data
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            temps = sorted(data['Temperature'].unique())
            darker = [c for c in list(mcolors.CSS4_COLORS.values()) if is_dark_color(c)]
            for i, temp in enumerate(temps):
                sub = data[data['Temperature'] == temp].sort_values('Frequency')
                ax.semilogx(sub['Frequency'], sub['Storage Modulus'], 
                           color=darker[i%len(darker)], label=f"{temp} °C")
            ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel("Storage Modulus (MPa)"); ax.set_title("Raw Data Plot")
            ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left'); add_watermark(ax)
            st.pyplot(fig)
            # 2. Save plot to a temporary buffer
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight', dpi=500)
            buf.seek(0)

            # 3. Layout: Spacer on left, Button on right
            # [5, 2] ratio gives 5 parts empty space, 2 parts for the button
            buff_col, button_col = st.columns([5, 2]) 
            
            with button_col:
                st.download_button(
                    label="💾 Download Graph",
                    data=buf,
                    file_name="Raw data plot.png",
                    mime="image/png",
                    use_container_width=True # Makes the button fill the column width
                )
        with tab2:
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
                # 2. Save plot to a temporary buffer
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches='tight', dpi=500)
                buf.seek(0)
    
                # 3. Layout: Spacer on left, Button on right
                # [5, 2] ratio gives 5 parts empty space, 2 parts for the button
                buff_col, button_col = st.columns([5, 2]) 
                
                with button_col:
                    st.download_button(
                        label="💾 Download Graph",
                        data=buf,
                        file_name="3D Raw data plot.png",
                        mime="image/png",
                        use_container_width=True # Makes the button fill the column width
                    )
            except: pass


def page_tts():
    st.title("Step 2: TTS Analysis")
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
        shifts = st.session_state.analysis_shift_factors
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.write("### Shift Factors")
            sf_list = [{"Temp (°C)": t, "aT": s} for t, s in shifts.items()]
            df_sf = pd.DataFrame(sf_list)
            st.dataframe(df_sf, height=300, hide_index=True)
            
        with col2:
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            
            for t in sorted(shifts.keys()):
                sf = shifts[t]
                sub = st.session_state.data[st.session_state.data['Temperature'] == t]
                ax.semilogx(sub['Frequency']*sf, sub['Storage Modulus'], 'o', label=f"{t} °C")
                
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Storage Modulus (MPa)")
            ax.set_title("Master Curve")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            add_watermark(ax)
            
            # 1. Show the plot
            st.pyplot(fig)

            # 2. Save plot to a temporary buffer
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight', dpi=500)
            buf.seek(0)

            # 3. Layout: Spacer on left, Button on right
            # [5, 2] ratio gives 5 parts empty space, 2 parts for the button
            buff_col, button_col = st.columns([5, 2]) 
            
            with button_col:
                st.download_button(
                    label="💾 Download Graph",
                    data=buf,
                    file_name="master_curve.png",
                    mime="image/png",
                    use_container_width=True # Makes the button fill the column width
                )
    
def page_fitting():
    st.title("Step 3: Curve Fitting")
    if not st.session_state.analysis_shift_factors: return st.warning("Run Step 2 first.")
    
    # Create Layout: Left for controls (smaller), Right for graph (larger)
    col_controls, col_graph = st.columns([1, 3])
    
    # --- Left Column: Sliders ---
    with col_controls:
        st.markdown("### Fit Bounds")
        # Real-time fitting sliders using Global Defaults
        a_high = st.slider("Upper Bound 'a'", 10.0, 5000.0, st.session_state.global_a_upper, key="s5_a")
        d_high = st.slider("Upper Bound 'd'", 10.0, 5000.0, st.session_state.global_d_upper, key="s5_d")
        
        # Update Global State
        st.session_state.global_a_upper = a_high
        st.session_state.global_d_upper = d_high
    
    # --- Data Processing & Fitting (Runs in background) ---
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
    except Exception as e:
        with col_graph: # Show error where graph would be
            st.error(f"Fit Failed: {e}")
        return

    # --- Right Column: Graph ---
    with col_graph:
        if st.session_state.fitted_params:
            params = st.session_state.fitted_params
            
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            
            all_x_log = [] # Store log values to calculate range
            shifts = st.session_state.analysis_shift_factors
            
            for t in sorted(shifts):
                sub = st.session_state.data[st.session_state.data['Temperature'] == t]
                freq = sub['Frequency'] * shifts[t]
                mask = freq > 0
                
                # --- CHANGE 1: Use semilogx with raw frequency ---
                # We plot 'freq' (Hz) directly, not the log of it.
                ax.semilogx(freq[mask], sub.loc[mask, 'Storage Modulus'], 'o', alpha=0.5, label=f"{t} °C")
                
                # Keep tracking log values just to determine the min/max range for the red line
                all_x_log.extend(np.log10(freq[mask]))
                
            if all_x_log:
                # Create range in log space (for the math model)
                x_rng_log = np.linspace(min(all_x_log)-0.5, max(all_x_log)+0.5, 500)
                
                # Calculate Y values using the model (model expects log inputs)
                y_fit = storage_modulus_model(x_rng_log, params['a'], params['b'], params['c'], params['d'])
                
                # --- CHANGE 2: Convert X range back to Linear Hz for plotting ---
                # We raise 10 to the power of the log-range to match the semilog axis
                ax.semilogx(10**x_rng_log, y_fit, 'r-', lw=3, label="Model")
                
            # --- CHANGE 3: Update Labels ---
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Modulus (MPa)")
            ax.set_title("Model Fitting")
            
            ax.text(0.05, 0.95, f"$R^2 = {params['r2']:.4f}$", 
                    transform=ax.transAxes, 
                    verticalalignment='top', fontsize = 16)
                    
            ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
            add_watermark(ax)
            st.pyplot(fig)

            # 2. Save plot to a temporary buffer
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight', dpi=500)
            buf.seek(0)

            # 3. Layout: Spacer on left, Button on right
            buff_col, button_col = st.columns([5, 2]) 
            
            with button_col:
                st.download_button(
                    label="💾 Download Graph",
                    data=buf,
                    file_name="Curve Fitting plot.png",
                    mime="image/png",
                    use_container_width=True
                )

def page_params_per_temp():
    st.title("Step 4: Master Curve Parameters")
    if not st.session_state.analysis_shift_factors: return st.warning("Run Step 2 first.")
    
    st.markdown("Please check each fittd curves, if they are not align properly, adjust the fitting bounds for parameters a and b until the fitted curves align with the corresponding master curves at each temperature.")
  
    col_ctrl, col_graph = st.columns([1, 3])
    with col_ctrl:
        st.subheader("Fitting Bounds")
        # Initialize with Global State
        a_high = st.slider("Upper Bound for 'a'", 10.0, 5000.0, st.session_state.global_a_upper, key="s6_a")
        d_high = st.slider("Upper Bound for 'd'", 10.0, 5000.0, st.session_state.global_d_upper, key="s6_d")
        
        # Update Global State (Sync with Step 3)
        st.session_state.global_a_upper = a_high
        st.session_state.global_d_upper = d_high

    # Prepare logic
    data = st.session_state.data
    original_shifts = st.session_state.analysis_shift_factors
    temps = sorted(original_shifts.keys())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(temps)))
    
    results = []
    
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Loop through each temp as the REFERENCE temp
    for i, ref_temp in enumerate(temps):
        # 1. Calculate shift factors for this new reference
        # Shift(new_ref -> T) = Shift(old_ref -> T) / Shift(old_ref -> new_ref)
        scale_factor = 1.0 / original_shifts[ref_temp]
        current_shift_factors = {t: s * scale_factor for t, s in original_shifts.items()}
        
        combined_log_freq = []
        combined_storage_modulus = []
        
        # 2. Gather shifted data for this configuration
        for t in temps:
            if t in current_shift_factors:
                sub = data[data['Temperature'] == t]
                shifted_freq = sub['Frequency'] * current_shift_factors[t]
                valid = shifted_freq > 0
                
                log_freq = np.log10(shifted_freq[valid])
                modulus = sub.loc[valid, 'Storage Modulus']
                
                combined_log_freq.extend(log_freq)
                combined_storage_modulus.extend(modulus)
                
        # 3. Plot Data Points for this Ref Temp
        ax.scatter(combined_log_freq, combined_storage_modulus, color=colors[i], s=10, alpha=0.5, label=f"{ref_temp} °C")
        
        # 4. Fit Curve
        try:
            combined_log_freq = np.array(combined_log_freq)
            combined_storage_modulus = np.array(combined_storage_modulus)
            
            popt, _ = curve_fit(storage_modulus_model, combined_log_freq, combined_storage_modulus, 
                              bounds=([1e-6, -100, -100, 1e-6], [a_high, 100, 100, d_high]), maxfev=100000)
            
            results.append({
                "Temperature": ref_temp,
                "a": popt[0], "b": popt[1], "c": popt[2], "d": popt[3],
                "r2": r2_score(combined_storage_modulus, storage_modulus_model(combined_log_freq, *popt))
            })
            
            # 5. Plot Fit Line (Black Dashed)
            x_rng = np.linspace(min(combined_log_freq), max(combined_log_freq), 100)
            y_fit = storage_modulus_model(x_rng, *popt)
            ax.plot(x_rng, y_fit, color='black', linestyle='--', linewidth=1)
            
        except Exception as e:
            pass

    ax.set_xlabel('Reduced Frequency (Hz)')
    ax.set_ylabel('Storage Modulus (MPa)')
    ax.set_title('Master Curve for All Reference Temperatures')
    
    handles, labels = ax.get_legend_handles_labels()
    dummy_line = Line2D([], [], color='black', linestyle='--', linewidth=1, label='Fit Line')
    
    # Filter duplicates in legend
    by_label = dict(zip(labels, handles))
    final_handles = [dummy_line] + list(by_label.values())
    final_labels = ['Fit Line'] + list(by_label.keys())
    
    ax.legend(final_handles, final_labels, bbox_to_anchor=(1.01, 1), loc='upper left')
    add_watermark(ax)
    
    with col_graph:
        st.pyplot(fig)
        # 2. Save plot to a temporary buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', dpi=500)
        buf.seek(0)

        # 3. Layout: Spacer on left, Button on right
        # [5, 2] ratio gives 5 parts empty space, 2 parts for the button
        buff_col, button_col = st.columns([5, 2]) 
        
        with button_col:
            st.download_button(
                label="💾 Download Graph",
                data=buf,
                file_name="Curve Fitting All Temperatures plot.png",
                mime="image/png",
                use_container_width=True # Makes the button fill the column width
            )
        
        
    df_res = pd.DataFrame(results)
    st.session_state.param_per_temp = df_res
    
def page_elastic_modulus():
    st.title("Step 5: Elastic Modulus vs Strain Rate")
    st.markdown("Predicts quasi-static Elastic Modulus ($E$) as a function of Strain Rate ($\dot{\epsilon}$).")
    #strain_rates = st.text_input("Enter Strain Rates (comma separated) for Table", "0.00001, 0.0001, 0.001, 0.01")
    
    if st.session_state.param_per_temp is None and st.session_state.fitted_params is None: 
        return st.warning("Run Step 3 or 4 first.")
    
    # --- Logic Pre-calculation ---
    if st.session_state.fitted_params:
        params = st.session_state.fitted_params
    elif st.session_state.param_per_temp is not None:
         params = st.session_state.param_per_temp.iloc[0].to_dict()
    
    col_left, col_right = st.columns([1, 3])

    # --- Left Column: Inputs & Table ---
    with col_left:
        try:
            rates_table = [float(x.strip()) for x in strain_rates.split(',')]
            rates_table = np.array(sorted(rates_table))
        except:
            rates_table = np.array([1e-5, 1e-4, 1e-3, 0.01])

        log_rates = np.log10(rates_table)
        E_values = storage_modulus_model(log_rates, params['a'], params['b'], params['c'], params['d'])
        
        res_df = pd.DataFrame({"Strain rate (s⁻¹)": rates_table, "E (MPa)": E_values})
        #st.dataframe(res_df.style.set_properties(**{'text-align': 'center'}), hide_index=True)
        st.dataframe(res_df, hide_index=True)
    # --- Right Column: Graph & Download ---
    with col_right:
        # --- NEW CODE: Set Font to Times New Roman ---
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        # ---------------------------------------------

        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        plot_rates = np.logspace(-5, -2, 500)
        plot_log_rates = np.log10(plot_rates)
        
        darker = [c for c in list(mcolors.CSS4_COLORS.values()) if is_dark_color(c)]
        
        if st.session_state.param_per_temp is not None:
            df_params = st.session_state.param_per_temp
            for i, row in df_params.iterrows():
                t = row['Temperature']
                E_curve = storage_modulus_model(plot_log_rates, row['a'], row['b'], row['c'], row['d'])
                
                color = darker[i % len(darker)]
                ax.semilogx(plot_rates, E_curve, '-', color=color, linewidth=2, label=f"{t} °C")
        else:
            shifts = st.session_state.analysis_shift_factors
            for i, t in enumerate(sorted(shifts.keys())):
                sf = shifts[t]
                c_temp = params['c'] + np.log10(sf)
                E_curve = storage_modulus_model(plot_log_rates, params['a'], params['b'], c_temp, params['d'])
                color = darker[i % len(darker)]
                ax.semilogx(plot_rates, E_curve, '-', color=color, linewidth=2, label=f"{t} °C")

        ax.set_xlabel("Strain Rate ($s^{-1}$)")
        ax.set_ylabel("Elastic Modulus (MPa)")
        ax.set_title("Elastic Modulus vs Strain Rate (All Temperatures)")
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        add_watermark(ax)
        st.pyplot(fig)
        
        # 2. Save plot to a temporary buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', dpi=500)
        buf.seek(0)

        # 3. Layout: Spacer on left, Button on right
        # [5, 2] ratio gives 5 parts empty space, 2 parts for the button
        buff_col, button_col = st.columns([5, 2]) 
        
        with button_col:
            st.download_button(
                label="💾 Download Graph",
                data=buf,
                file_name="Elastic Modulus plot.png",
                mime="image/png",
                use_container_width=True # Makes the button fill the column width
            )
            

# ==========================================
# Main Navigation
# ==========================================
def main():
    st.sidebar.title("NYU-ViscoMOD")
    
    pages = {
        "1. Load & Visualize": page_load_and_visualize,
        "2. TTS Analysis": page_tts,
        "3. Curve Fitting": page_fitting,
        "4. Params per Temp": page_params_per_temp,
        "5. Elastic Modulus": page_elastic_modulus
    }
    
    selection = st.sidebar.radio("Go to Step:", list(pages.keys()))
    pages[selection]()

if __name__ == "__main__":
    main()

















































