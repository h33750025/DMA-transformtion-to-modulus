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
from scipy.integrate import simpson
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

###========================= Master Curve only ==================================####
# def page_tts():
#     st.title("Step 2: Master Curve")
#     if st.session_state.data is None: return st.warning("No data loaded.")
    
#     if st.button("Run Analysis"):
#         data = st.session_state.data
#         ref_temp = min(data["Temperature"].unique())
#         extended = data[data["Temperature"] == ref_temp].sort_values('Frequency')
#         shift_factors = {ref_temp: 1.0}
        
#         for temp in sorted([t for t in data["Temperature"].unique() if t > ref_temp]):
#             df_temp = data[data["Temperature"] == temp].sort_values('Frequency')
#             max_freq = df_temp["Frequency"].max()
#             mod_at_max = df_temp.iloc[-1]["Storage Modulus"]
            
#             # Find overlap match
#             idx = (extended["Storage Modulus"] - mod_at_max).abs().idxmin()
#             match_freq = extended.loc[idx, "Frequency"]
            
#             sf = match_freq / max_freq
#             shift_factors[temp] = sf
            
#             shifted = df_temp.copy()
#             shifted["Frequency"] *= sf
#             extended = pd.concat([extended, shifted], ignore_index=True).sort_values("Frequency")
            
#         st.session_state.analysis_shift_factors = shift_factors
#         st.session_state.master_curve_data = extended
#         st.success("Complete")
        
#     if st.session_state.master_curve_data is not None:
#         shifts = st.session_state.analysis_shift_factors
        
#         #col1, col2 = st.columns([1, 3])
        
#         # with col1:
#         #     st.write("### Shift Factors")
#         #     sf_list = [{"Temp (°C)": t, "aT": s} for t, s in shifts.items()]
#         #     df_sf = pd.DataFrame(sf_list)
#         #     st.dataframe(df_sf, height=300, hide_index=True)
            
#         #with col2:
#         fig = Figure(figsize=(10, 6))
#         ax = fig.add_subplot(111)
        
#         for t in sorted(shifts.keys()):
#             sf = shifts[t]
#             sub = st.session_state.data[st.session_state.data['Temperature'] == t]
#             ax.semilogx(sub['Frequency']*sf, sub['Storage Modulus'], 'o', label=f"{t} °C")
            
#         ax.set_xlabel("Frequency (Hz)")
#         ax.set_ylabel("Storage Modulus (MPa)")
#         ax.set_title("Master Curve")
#         ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
#         add_watermark(ax)
        
#         # 1. Show the plot
#         st.pyplot(fig)

#         # 2. Save plot to a temporary buffer
#         buf = io.BytesIO()
#         fig.savefig(buf, format="png", bbox_inches='tight', dpi=500)
#         buf.seek(0)

#         # 3. Layout: Spacer on left, Button on right
#         # [5, 2] ratio gives 5 parts empty space, 2 parts for the button
#         buff_col, button_col = st.columns([5, 2]) 
        
#         with button_col:
#             st.download_button(
#                 label="💾 Download Graph",
#                 data=buf,
#                 file_name="master_curve.png",
#                 mime="image/png",
#                 use_container_width=True # Makes the button fill the column width
#             )
###############################################################
def page_fitting():
    st.title("Step 2: Curve Fitting")
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
    
    #if not st.session_state.analysis_shift_factors: return st.warning("Run Step 2 first.")
    # Use a single st.markdown call with inline LaTeX ($...$)
    st.markdown(r"Curve fitting function used:  $E'(\omega) = A \tanh(B \log(\omega) + C) + D$")
    st.info(r"Adjust the fitting bounds A and D to fit the master curve, check the $R^2$ score for reference.")
 
    # --- AUTOMATED ANALYSIS (Previously Button) ---
    # Runs automatically if master_curve_data is missing, or recalculates if needed.
    # We check session state to avoid re-calculating on every slider movement (performance fix).
    if "master_curve_data" not in st.session_state or st.session_state.master_curve_data is None:
        with st.spinner("Processing Master Curve..."):
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
            
            # Update Session State
            st.session_state.analysis_shift_factors = shift_factors
            st.session_state.master_curve_data = extended
    if st.session_state.master_curve_data is not None:
        shifts = st.session_state.analysis_shift_factors

    # Create Layout: Left for controls (smaller), Right for graph (larger)
    col_controls, col_graph = st.columns([1, 3])
    
    # --- Left Column: Sliders ---
    with col_controls:
        st.markdown("### Fitting Bounds")

        # Real-time fitting sliders using Global Defaults
        a_high = st.slider("Upper Bound 'A'", 10.0, 5000.0, st.session_state.global_a_upper, key="s5_a")
        d_high = st.slider("Upper Bound 'D'", 10.0, 5000.0, st.session_state.global_d_upper, key="s5_d")
        
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

#########################################################
def page_params_per_temp():
    st.title("Step 3: Master Curve Parameters")
    if not st.session_state.analysis_shift_factors: return st.warning("Run Step 2 first.")
    st.info(r"Please check each fitted curves, if they are not align properly, adjust the fitting bounds.")
  
    col_ctrl, col_graph = st.columns([1, 3])
    with col_ctrl:
        st.subheader("Fitting Bounds")
        # Initialize with Global State
        a_high = st.slider("Upper Bound for 'A'", 10.0, 5000.0, st.session_state.global_a_upper, key="s6_a")
        d_high = st.slider("Upper Bound for 'D'", 10.0, 5000.0, st.session_state.global_d_upper, key="s6_d")
        
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


#=================================================================
def page_elastic_modulus():
    st.title("Step 4: Elastic Modulus Prediction")
    #st.markdown(r"**Elastic Modulus ($E$)** .")

    # --- Check Prerequisites ---
    if st.session_state.fitted_params is None: 
        return st.warning("Please Run Step 3 (Curve Fitting) first to get model parameters.")

    # --- 0. Robust Imports (Fixes for SciPy 1.13+ and NumPy 2.0+) ---
    try:
        from scipy.integrate import simpson
    except ImportError:
        from scipy.integrate import simps as simpson
        
    if hasattr(np, 'trapezoid'):
        trapz_func = np.trapezoid
    else:
        trapz_func = np.trapz

    # --- 1. Simulation Controls ---
    strain_rates_to_plot = [1e-5, 1e-4, 1e-3, 1e-2]
    strain_min = 1e-25
    strain_max = 0.0025
    num_steps = 500
    
    # Initialize Session State for Results if not present
    if 'modulus_results_df' not in st.session_state:
        st.session_state.modulus_results_df = None

    # Run Button
    run_sim = st.button("Run Prediction", type="primary")

    # --- 2. Calculation Logic (Runs ONLY if Button Clicked) ---
    if run_sim:
        # Define internal functions
        def E_prime(w, a, b, c, d):
            return a * np.tanh(b * ((np.log(w)) + c)) + d

        def Etime_time_cycle(time, cycle, a, b, c, d):
            N1, N2, N3 = 240, 74, 24
            Etime = np.zeros_like(time)
            def integrand(t, E_prime_w, w):
                return (2/np.pi)*(E_prime_w/w)*np.sin(w*t)
            
            for i, t in enumerate(time):
                if t == 0: continue
                w1 = np.linspace((1e-6 / t), (cycle * 0.1 * 2 * np.pi / t), int(cycle * 0.1 * N1 + 1))
                w2 = np.linspace((cycle * 0.1 * 2 * np.pi) / t, (cycle * 0.4 * 2 * np.pi) / t, int(cycle * 0.3 * N2 + 1))
                w3 = np.linspace((cycle * 0.4 * 2 * np.pi) / t, (cycle * 2 * np.pi) / t, int(cycle * 0.6 * N3 + 1))
                all_w = np.concatenate([w1, w2[1:], w3[1:]])
                y = integrand(t, E_prime(all_w, a, b, c, d), all_w)
                Etime[i] = trapz_func(y, all_w)
            return Etime

        # Prepare Params
        abcd_parameters = []
        avail_temps = sorted(st.session_state.analysis_shift_factors.keys())
        master = st.session_state.fitted_params

        for temp in avail_temps:
            if st.session_state.param_per_temp is not None:
                row = st.session_state.param_per_temp.loc[st.session_state.param_per_temp['Temperature'] == temp]
                if not row.empty:
                    p = row.iloc[0].to_dict()
                    abcd_parameters.append([temp, p['a'], p['b'], p['c'], p['d']])
                    continue
            sf = st.session_state.analysis_shift_factors.get(temp, 1.0)
            c_shifted = master['c'] + np.log10(sf) 
            abcd_parameters.append([temp, master['a'], master['b'], c_shifted, master['d']])

        # Main Calculation Loop
        results = []
        total_ops = len(abcd_parameters) * len(strain_rates_to_plot)
        progress_bar = st.progress(0, text="Predicting Modulus...")
        current_op = 0

        with st.spinner("Estimating... this may take a moment."):
            for params in abcd_parameters:
                ref_temp, a, b, c, d = params
                final_cumulative_integrals = []

                for rate in strain_rates_to_plot:
                    current_op += 1
                    progress_bar.progress(int(current_op / total_ops * 100), text=f"Processing {ref_temp}°C | Rate {rate}")

                    time_min_rate = strain_min / rate
                    time_max_rate = strain_max / rate
                    time_range_rate = np.linspace(time_min_rate, time_max_rate, num_steps)

                    E_t_time_range_rate = Etime_time_cycle(time_range_rate, 500, a, b, c, d)
                    Stress_history_rate = E_t_time_range_rate * rate

                    cumulative_integral = np.array([
                        simpson(Stress_history_rate[:i+1], x=time_range_rate[:i+1]) 
                        for i in range(len(time_range_rate))
                    ])
                    
                    final_modulus = cumulative_integral[-1] / strain_max
                    final_cumulative_integrals.append(final_modulus)

                results.append([ref_temp] + final_cumulative_integrals)

        progress_bar.empty()
        
        # Save to Session State
        cols = ['Ref Temp (°C)'] + [f'Strain Rate {rate} (1/s)' for rate in strain_rates_to_plot]
        st.session_state.modulus_results_df = pd.DataFrame(results, columns=cols)

    # --- 3. Visualization (Runs if Data Exists in Session State) ---
    if st.session_state.modulus_results_df is not None:
        df = st.session_state.modulus_results_df

        # Check if we have negative values in the calculated Modulus columns (indices 1 to end)
        has_negative_values = df.iloc[:, 1:].min().min() < 0
        
        # Global Plot Settings
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        
        tab1, tab2, tab3 = st.tabs(["📈 Modulus vs Strain Rate", "🌡️ Modulus vs Temp", "📄 Data Table"])

        with tab1:
            fig1 = Figure(figsize=(10, 6))
            ax1 = fig1.add_subplot(111)
            for i, row in df.iterrows():
                ax1.plot(strain_rates_to_plot, row[1:], label=f"{row['Ref Temp (°C)']}°C")

            ax1.set_xscale('log')
            ax1.set_xlabel(r'Strain Rate ($s^{-1}$)')
            ax1.set_ylabel('Elastic Modulus (MPa)')
            
            # --- CONDITIONAL LIMIT ---
            if has_negative_values:
                ax1.set_ylim(bottom=0)
            # -------------------------
            
            ax1.tick_params(axis='both', which='major')
            ax1.set_xlim(strain_rates_to_plot[0], strain_rates_to_plot[-1])
            ax1.set_title('Modulus vs Strain Rate for Different Temperatures')
            ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
            add_watermark(ax1)
            st.pyplot(fig1)
            
            buf1 = io.BytesIO()
            fig1.savefig(buf1, format="png", bbox_inches='tight', dpi=500)
            buf1.seek(0)
            st.download_button("💾 Download Graph", buf1, "Modulus_vs_Strain_Rate.png", "image/png")

        with tab2:
            fig2 = Figure(figsize=(10, 6))
            ax2 = fig2.add_subplot(111)
            for j, rate in enumerate(strain_rates_to_plot):
                ax2.plot(df['Ref Temp (°C)'], df.iloc[:, j+1], label=f"Rate {rate} (1/s)")

            ax2.set_xlabel('Temperature (°C)')
            ax2.set_ylabel('Elastic Modulus (MPa)')
            # --- CONDITIONAL LIMIT ---
            if has_negative_values:
                ax2.set_ylim(bottom=0)
            # -------------------------
            ax2.set_title('Modulus vs Temperature for Different Strain Rates')
            ax2.tick_params(axis='both', which='major')
            ax2.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
            add_watermark(ax2)
            st.pyplot(fig2)
            
            buf2 = io.BytesIO()
            fig2.savefig(buf2, format="png", bbox_inches='tight', dpi=500)
            buf2.seek(0)
            st.download_button("💾 Download Graph", buf2, "Modulus_vs_Temp.png", "image/png")

        with tab3:
            st.dataframe(df)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📄 Download Results (CSV)", csv, "Modulus_Simulation_Results.csv", "text/csv")
            
    else:
        st.info("Click 'Run Prediction' to start calculations.")

# =================== strain stress curve ==========================================
# def page_stress_strain_curve():
#     st.title("Step 5: Stress vs Strain Prediction")
#     st.markdown(r"Predicts **Stress ($\sigma$) vs Strain ($\epsilon$)** response for a **selected Temperature** and **Strain Rate**.")

#     # --- Check Prerequisites ---
#     if st.session_state.fitted_params is None: 
#         return st.warning("Please Run Step 3 (Curve Fitting) first to get model parameters.")

#     # --- 0. Robust Import for SciPy ---
#     try:
#         from scipy.integrate import simpson
#     except ImportError:
#         from scipy.integrate import simps as simpson

#     # --- 1. Initialize Session State for this Page ---
#     # We store the last calculated data here so it persists during navigation
#     if 'stress_strain_data' not in st.session_state:
#         st.session_state.stress_strain_data = None

#     # --- 2. Simulation Controls ---
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         avail_temps = sorted(st.session_state.analysis_shift_factors.keys())
#         selected_temp = st.selectbox("Select Temperature (°C)", avail_temps)

#     with col2:
#         rates_options = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
#         selected_rate = st.selectbox(
#             "Select Strain Rate (1/s)", 
#             rates_options, 
#             index=3, 
#             format_func=lambda x: f"{x:.0e}"
#         )

#     with col3:
#         strain_max = st.number_input("Max Strain", value=0.0025, format="%.4f")
    
#     run_sim = st.button("Run Simulation", type="primary")

#     # --- 3. Calculation Logic (Runs ONLY if Button Clicked) ---
#     if run_sim:
#         # Define internal functions
#         def E_prime(w, A, B, C, D):
#             return A * np.tanh(B * (np.log10(w) + C)) + D

#         def Etime_time_cycle(time_arr, params, cycle=500):
#             A, B, C, D = params['a'], params['b'], params['c'], params['d']
#             N1, N2, N3 = 240, 74, 24
#             Etime = np.zeros_like(time_arr)

#             def integrand(t, E_prime_w, w):
#                 return (2/np.pi)*(E_prime_w/w)*np.sin(w*t)

#             for i, t in enumerate(time_arr):
#                 if t == 0: continue
#                 w1 = np.linspace((1e-6 / t), (cycle * 0.1 * 2 * np.pi / t), int(cycle * 0.1 * N1 + 1))
#                 w2 = np.linspace((cycle * 0.1 * 2 * np.pi) / t, (cycle * 0.4 * 2 * np.pi) / t, int(cycle * 0.3 * N2 + 1))
#                 w3 = np.linspace((cycle * 0.4 * 2 * np.pi) / t, (cycle * 2 * np.pi) / t, int(cycle * 0.6 * N3 + 1))
#                 all_w = np.concatenate([w1, w2[1:], w3[1:]])
#                 y = integrand(t, E_prime(all_w, A, B, C, D), all_w)
#                 Etime[i] = simpson(y, x=all_w)
#             return Etime

#         # Prepare Params
#         params = {}
#         if st.session_state.param_per_temp is not None:
#             row = st.session_state.param_per_temp.loc[st.session_state.param_per_temp['Temperature'] == selected_temp]
#             if not row.empty:
#                 params = row.iloc[0].to_dict()
        
#         if not params:
#             master = st.session_state.fitted_params
#             sf = st.session_state.analysis_shift_factors.get(selected_temp, 1.0)
#             params = master.copy()
#             params['c'] = master['c'] + np.log10(sf)

#         # Calculation
#         with st.spinner(f"Calculating for {selected_temp}°C..."):
#             num_steps = 500
#             strain_min = 1e-25
#             time_min_rate = strain_min / selected_rate
#             time_max_rate = strain_max / selected_rate
#             time_range_rate = np.linspace(time_min_rate, time_max_rate, num_steps)
            
#             E_t = Etime_time_cycle(time_range_rate, params, cycle=500)
#             Stress_history_rate = E_t / 2 * selected_rate
            
#             stress_vals = []
#             for i in range(len(time_range_rate)):
#                 if i == 0:
#                     stress_vals.append(0)
#                 else:
#                     val = simpson(Stress_history_rate[:i+1], x=time_range_rate[:i+1])
#                     stress_vals.append(val)
            
#             cumulative_integral = np.array(stress_vals)
#             strain_range_rate = time_range_rate * selected_rate
            
#             # Save results to Session State
#             st.session_state.stress_strain_data = {
#                 "strain": strain_range_rate,
#                 "stress": cumulative_integral,
#                 "temp": selected_temp,
#                 "rate": selected_rate
#             }

#     # --- 4. Visualization (Runs if Data Exists in Session State) ---
#     if st.session_state.stress_strain_data is not None:
#         data = st.session_state.stress_strain_data
        
#         # Setup Plot
#         plt.rcParams["font.family"] = "serif"
#         plt.rcParams["font.serif"] = ["Times New Roman"]
        
#         fig = Figure(figsize=(10, 6))
#         ax = fig.add_subplot(111)
        
#         label_txt = f"{data['temp']}°C, {data['rate']:.0e}/s"
#         ax.plot(data['strain'], data['stress'], label=label_txt, color='black', linewidth=2)

#         # --- Conditional Y-Limit Logic ---
#         # "if there are y values that below 0, make the graph show only bottom 0"
#         if np.min(data['stress']) < 0:
#             ax.set_ylim(bottom=0)
#         # ---------------------------------

#         ax.set_xlabel("Strain (1)", fontsize=16)
#         ax.set_ylabel("Stress (MPa)", fontsize=16)
#         ax.set_title(f"Stress vs Strain Prediction", fontsize=18)
#         ax.tick_params(axis='both', which='major', labelsize=12)
#         ax.set_xlim(left=0)
        
#         ax.legend(loc='upper left', fontsize=12)
#         ax.grid(True, which="both", ls="--", alpha=0.4)
#         add_watermark(ax)
        
#         st.pyplot(fig)
        
#         # Buffers for Download
#         buf = io.BytesIO()
#         fig.savefig(buf, format="png", bbox_inches='tight', dpi=500)
#         buf.seek(0)
        
#         df_export = pd.DataFrame({"Strain": data['strain'], "Stress (MPa)": data['stress']})
#         csv_buf = df_export.to_csv(index=False).encode('utf-8')

#         buff_col, csv_col, img_col = st.columns([4, 2, 2]) 
        
#         with csv_col:
#             st.download_button(
#                 label="📄 Download CSV",
#                 data=csv_buf,
#                 file_name=f"Stress_Strain_{data['temp']}C.csv",
#                 mime="text/csv",
#                 use_container_width=True
#             )
            
#         with img_col:
#             st.download_button(
#                 label="💾 Download Graph",
#                 data=buf,
#                 file_name=f"Stress_Strain_{data['temp']}C.png",
#                 mime="image/png",
#                 use_container_width=True
#             )
#     else:
#         st.info("Select parameters and click 'Run Simulation'.")

# ==========================================
# Main Navigation
# ==========================================
def main():
    st.sidebar.title("NYU-ViscoMOD")
    
    # --- 1. Navigation Menu ---
    pages = {
        "1. Load & Visualize": page_load_and_visualize,
        #"2. Master Curve": page_tts,#
        "2. Curve Fitting": page_fitting,
        "3. Master Curves Fitting": page_params_per_temp,
        #"4. Stress-Strain Curve": page_stress_strain_curve,#
        "4. Elastic Modulus": page_elastic_modulus
    }
    
    selection = st.sidebar.radio("Go to Step:", list(pages.keys()))
    
    # --- 2. Sidebar Resources (Manual & Template) ---
    st.sidebar.markdown("---") 
    st.sidebar.markdown("### Help & Resources")
    
    # Try/Except block ensures the app doesn't crash if the file is missing
    try:
        with open("manual.pdf", "rb") as f:
            st.sidebar.download_button(
                label="📄User Manual",
                data=f,
                file_name="NYU_ViscoMOD_Manual.pdf",
                mime="application/pdf"
            )
    except FileNotFoundError:
        st.sidebar.warning("Manual file not found.")
        
    # B. Template Download
    # (Ensure 'data_template.csv' exists in your folder)
    try:
        with open("data_template.csv", "rb") as f:
            st.sidebar.download_button(
                label="📊Data Template",
                data=f,
                file_name="data_template.csv",
                mime="text/csv"
            )
    except FileNotFoundError:
        st.sidebar.caption("⚠️ Template not found")
    # --- 3. Render Selected Page ---
    pages[selection]()

if __name__ == "__main__":
    main()














