import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit, minimize
from sklearn.metrics import r2_score
import io

# --- Page Config ---
st.set_page_config(page_title="NYU ViscoMOD Web", layout="wide")

# --- Session State Management ---
if 'data' not in st.session_state:
    st.session_state.data = None
if 'master_curve_data' not in st.session_state:
    st.session_state.master_curve_data = None
if 'shift_params' not in st.session_state:
    st.session_state.shift_params = {"C1": 17.44, "C2": 51.6, "Tref": 25.0}
if 'prony_results' not in st.session_state:
    st.session_state.prony_results = None

# --- Helper Functions ---

def wlf_shift(temp, t_ref, c1, c2):
    """Calculate log10(aT) using WLF equation."""
    # Prevent division by zero
    denom = c2 + (temp - t_ref)
    if abs(denom) < 1e-9: 
        return 0
    return (-c1 * (temp - t_ref)) / denom

def arrhenius_shift(temp, t_ref, ea):
    """Calculate log10(aT) using Arrhenius equation."""
    R = 8.314  # Gas constant J/(mol K)
    # Convert to Kelvin
    T_k = temp + 273.15
    Tref_k = t_ref + 273.15
    # Natural log shift
    ln_at = (ea * 1000 / R) * (1/T_k - 1/Tref_k)
    # Convert to log10
    return ln_at / np.log(10)

def prony_model_storage(omega, g_terms, tau_terms, g_e):
    """Calculate Storage Modulus G'(omega) from Prony Series."""
    # G'(w) = Ge + Sum [ Gi * (w*tau_i)^2 / (1 + (w*tau_i)^2) ]
    g_prime = np.full_like(omega, g_e)
    for g_i, tau_i in zip(g_terms, tau_terms):
        term = (g_i * (omega * tau_i)**2) / (1 + (omega * tau_i)**2)
        g_prime += term
    return g_prime

# --- Sidebar ---
st.sidebar.title("ViscoMOD Navigation")
page = st.sidebar.radio("Go to:", [
    "1. Load Data", 
    "2. Master Curve (TTS)", 
    "3. Polynomial Fit", 
    "4. Prony Series Fit", 
    "5. 3D Visualization"
])

# ================= PAGE 1: LOAD DATA =================
if page == "1. Load Data":
    st.title("Step 1: Upload Rheology Data")
    st.info("Ensure CSV has columns: `Frequency`, `Storage Modulus`, `Temperature`")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            # Normalize column names for easier access
            df.columns = [c.strip() for c in df.columns]
            
            # Simple column mapping (case insensitive check)
            col_map = {}
            for c in df.columns:
                lower_c = c.lower()
                if "freq" in lower_c: col_map["Frequency"] = c
                if "storage" in lower_c or "modulus" in lower_c: col_map["Storage Modulus"] = c
                if "temp" in lower_c: col_map["Temperature"] = c
            
            if len(col_map) >= 3:
                # Rename columns standardly
                df = df.rename(columns={
                    col_map["Frequency"]: "Frequency",
                    col_map["Storage Modulus"]: "Storage Modulus",
                    col_map["Temperature"]: "Temperature"
                })
                # Drop N/As
                df = df.dropna()
                st.session_state.data = df
                st.success(f"Loaded {len(df)} rows.")
                st.dataframe(df.head())
            else:
                st.error(f"Could not auto-identify columns. Found: {list(df.columns)}")
                
        except Exception as e:
            st.error(f"Error reading file: {e}")

# ================= PAGE 2: MASTER CURVE (TTS) =================
elif page == "2. Master Curve (TTS)":
    st.title("Step 2: Time-Temperature Superposition")
    
    if st.session_state.data is None:
        st.warning("Please upload data in Step 1.")
    else:
        df = st.session_state.data
        temps = sorted(df["Temperature"].unique())
        
        # --- Parameters ---
        col1, col2, col3 = st.columns(3)
        with col1:
            t_ref = st.selectbox("Reference Temperature (°C)", temps, index=len(temps)//2)
        with col2:
            model_type = st.selectbox("Shift Model", ["WLF", "Arrhenius"])
        
        if model_type == "WLF":
            with col3:
                c1 = st.number_input("C1", value=17.44)
                c2 = st.number_input("C2", value=51.6)
        else:
            with col3:
                ea = st.number_input("Activation Energy (kJ/mol)", value=50.0)

        # --- Calculation ---
        # Calculate Shift Factors
        shifted_data = []
        shift_factors = [] # Store for plotting
        
        for t in temps:
            sub = df[df["Temperature"] == t].copy()
            
            if model_type == "WLF":
                log_at = wlf_shift(t, t_ref, c1, c2)
            else:
                log_at = arrhenius_shift(t, t_ref, ea)
                
            at = 10**log_at
            
            # Shift Frequency: w_red = w * at
            sub["Reduced Frequency"] = sub["Frequency"] * at
            sub["Shift Factor"] = at
            sub["Log_aT"] = log_at
            
            shifted_data.append(sub)
            shift_factors.append({"Temperature": t, "Log_aT": log_at})
        
        master_df = pd.concat(shifted_data).sort_values(by="Reduced Frequency")
        st.session_state.master_curve_data = master_df
        
        # --- Plotting ---
        tab1, tab2 = st.tabs(["Master Curve", "Shift Factors"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(8, 6))
            # Plot by temperature to show colors
            scatter = ax.scatter(master_df["Reduced Frequency"], master_df["Storage Modulus"], 
                       c=master_df["Temperature"], cmap="viridis", alpha=0.6, s=15)
            
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Reduced Frequency (Hz)")
            ax.set_ylabel("Storage Modulus (Pa)")
            ax.set_title(f"Master Curve at T_ref={t_ref}°C")
            plt.colorbar(scatter, label="Temperature (°C)")
            ax.grid(True, which="both", ls="-", alpha=0.3)
            st.pyplot(fig)
            
        with tab2:
            sf_df = pd.DataFrame(shift_factors)
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(sf_df["Temperature"], sf_df["Log_aT"], 'o-', color='teal')
            ax2.set_xlabel("Temperature (°C)")
            ax2.set_ylabel("Log(aT)")
            ax2.set_title("Shift Factors vs Temperature")
            ax2.grid(True)
            st.pyplot(fig2)

# ================= PAGE 3: POLYNOMIAL FIT =================
elif page == "3. Polynomial Fit":
    st.title("Step 3: Polynomial Fit of Master Curve")
    
    if st.session_state.master_curve_data is None:
        st.warning("Please generate the Master Curve in Step 2 first.")
    else:
        m_df = st.session_state.master_curve_data
        
        degree = st.slider("Polynomial Degree", 1, 9, 5)
        
        # Fit Log-Log
        x_log = np.log10(m_df["Reduced Frequency"])
        y_log = np.log10(m_df["Storage Modulus"])
        
        # Polyfit
        coeffs = np.polyfit(x_log, y_log, degree)
        poly_func = np.poly1d(coeffs)
        
        y_pred_log = poly_func(x_log)
        r2 = r2_score(y_log, y_pred_log)
        
        st.metric("R² Score", f"{r2:.5f}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(m_df["Reduced Frequency"], m_df["Storage Modulus"], 
                   color='lightgray', label='Master Curve Data', s=10)
        
        # Sort for clean line
        sort_idx = np.argsort(x_log)
        x_sorted = 10**x_log.iloc[sort_idx]
        y_sorted = 10**y_pred_log.iloc[sort_idx]
        
        ax.plot(x_sorted, y_sorted, 'r-', linewidth=2, label=f'Poly Fit (Deg {degree})')
        
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Reduced Frequency (Hz)")
        ax.set_ylabel("Storage Modulus (Pa)")
        ax.legend()
        st.pyplot(fig)
        
        st.write(" **Polynomial Coefficients (Highest power first):**")
        st.write(coeffs)

# ================= PAGE 4: PRONY SERIES =================
elif page == "4. Prony Series Fit":
    st.title("Step 4: Generalized Maxwell (Prony) Fit")
    
    if st.session_state.master_curve_data is None:
        st.warning("Please generate the Master Curve in Step 2 first.")
    else:
        m_df = st.session_state.master_curve_data
        
        # Filter negative or zero values for log scale
        m_df = m_df[(m_df["Reduced Frequency"] > 0) & (m_df["Storage Modulus"] > 0)]
        
        n_elements = st.number_input("Number of Maxwell Elements", 1, 20, 5)
        
        if st.button("Run Optimization (Minimize Error)"):
            with st.spinner("Optimizing Prony parameters... this may take a moment."):
                
                omega = m_df["Reduced Frequency"].values
                G_exp = m_df["Storage Modulus"].values
                
                # Initial Guess
                # Spread taus logarithmically across frequency range
                min_w, max_w = np.min(omega), np.max(omega)
                taus_init = np.logspace(np.log10(1/max_w), np.log10(1/min_w), n_elements)
                # Guess G_i roughly as mean G / N
                g_init = np.full(n_elements, np.mean(G_exp)/n_elements)
                ge_init = 0.0
                
                initial_guess = np.concatenate([g_init, taus_init, [ge_init]])
                
                # Objective Function
                def objective(params):
                    gs = params[:n_elements]
                    ts = params[n_elements:2*n_elements]
                    ge = params[-1]
                    
                    # Log-Log MSE is usually better for Rheology
                    g_model = prony_model_storage(omega, gs, ts, ge)
                    
                    # Avoid log(0)
                    g_model = np.maximum(g_model, 1e-9)
                    
                    return np.sum((np.log10(G_exp) - np.log10(g_model))**2)
                
                # Constraints: all params > 0
                bounds = [(0, np.inf)] * len(initial_guess)
                
                res = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
                
                if res.success:
                    st.success(f"Optimization Successful! Error: {res.fun:.4f}")
                    
                    g_opt = res.x[:n_elements]
                    tau_opt = res.x[n_elements:2*n_elements]
                    ge_opt = res.x[-1]
                    
                    # Generate smooth line for plotting
                    w_smooth = np.logspace(np.log10(min_w), np.log10(max_w), 100)
                    g_smooth = prony_model_storage(w_smooth, g_opt, tau_opt, ge_opt)
                    
                    # Save results
                    st.session_state.prony_results = {
                        "G_i": g_opt, "Tau_i": tau_opt, "Ge": ge_opt
                    }
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(omega, G_exp, color='lightgray', label='Exp Data', s=15)
                    ax.plot(w_smooth, g_smooth, 'b-', label='Prony Fit')
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    ax.set_xlabel("Reduced Frequency (Hz)")
                    ax.set_ylabel("Storage Modulus (Pa)")
                    ax.legend()
                    st.pyplot(fig)
                    
                    # Table
                    res_df = pd.DataFrame({
                        "Element": range(1, n_elements+1),
                        "G_i (Pa)": g_opt,
                        "Tau_i (s)": tau_opt
                    })
                    st.write(f"**Ge (Equilibrium Modulus):** {ge_opt:.4e} Pa")
                    st.dataframe(res_df)
                    
                else:
                    st.error(f"Optimization failed: {res.message}")

# ================= PAGE 5: 3D Visualization =================
elif page == "5. 3D Visualization":
    st.title("Step 5: 3D Surface View")
    
    if st.session_state.data is None:
        st.warning("Please upload data in Step 1.")
    else:
        df = st.session_state.data
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot Trisurf
        surf = ax.plot_trisurf(df["Frequency"], df["Temperature"], df["Storage Modulus"], 
                        cmap="viridis", edgecolor="none", alpha=0.9)
        
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Temperature (°C)")
        ax.set_zlabel("Storage Modulus (Pa)")
        ax.set_title("3D View of Raw Data")
        
        # Rotate view (180 deg from standard -60 => 120)
        ax.view_init(elev=30, azim=120)
        
        fig.colorbar(surf, shrink=0.5, aspect=10)
        st.pyplot(fig)