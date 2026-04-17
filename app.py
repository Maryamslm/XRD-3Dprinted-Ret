# TAB 6 — PUBLICATION-QUALITY PLOT (ENHANCED)
with tabs[6]:
    st.subheader("🖼️ Publication-Quality Plot (matplotlib)")
    st.caption("Generate journal-ready figures with customizable phase markers, legend control & spacing")
    
    if "last_result" not in st.session_state:
        st.info("Run the Rietveld refinement first (Tab 3) to enable publication plotting.")
    else:
        result = st.session_state["last_result"]
        phases = st.session_state["last_phases"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            fig_width = st.slider("Figure width (inches)", 6.0, 14.0, 10.0, 0.5)
            offset_factor = st.slider("Difference curve offset", 0.05, 0.25, 0.12, 0.01)
            font_size = st.slider("Global Font Size", 6, 22, 11)
        with col2:
            fig_height = st.slider("Figure height (inches)", 5.0, 12.0, 7.0, 0.5)
            show_hkl = st.checkbox("Show hkl labels", value=True)
            legend_pos = st.selectbox("Legend Position", 
                                      ["best", "upper right", "upper left", "lower left", "lower right", 
                                       "center right", "center left", "lower center", "upper center", "center", "off"])
        with col3:
            export_format = st.selectbox("Export format", ["PDF", "PNG", "EPS"], index=0)
            # ✅ NEW: Marker row spacing control
            marker_spacing = st.slider("Marker row spacing", 0.8, 2.5, 1.3, 0.1, 
                                      help="Vertical distance between phase marker rows")
            st.markdown("**🎨 Phase Customization**")
            
        # ✅ NEW: Legend control section
        st.markdown("### 📋 Legend Control")
        st.caption("Select which phases to include in the plot legend")
        legend_cols = st.columns(len(phases) if len(phases) <= 4 else 4)
        legend_phases_selected = []
        for idx, ph in enumerate(phases):
            col_idx = idx % (len(phases) if len(phases) <= 4 else 4)
            with legend_cols[col_idx]:
                # Default: include all phases in legend
                if st.checkbox(f"✓ {ph}", value=True, key=f"leg_{ph}"):
                    legend_phases_selected.append(ph)
        
        phase_data = []
        for i, ph in enumerate(phases):
            pk_df = generate_theoretical_peaks(ph, wavelength, tt_min, tt_max)
            
            # ✅ PER-PHASE CUSTOMIZATION UI (with square & diamond options)
            with st.expander(f"⚙️ Settings for {ph}", expanded=True):
                c_col, c_shape = st.columns(2)
                custom_color = c_col.color_picker(f"Color", value=PHASE_LIBRARY[ph]["color"], key=f"col_{ph}")
                # ✅ UPDATED: Added square (s) and diamond (d) to shape options
                custom_shape = c_shape.selectbox(f"Marker Shape", 
                                                 ["|", "_", "s", "^", "v", "d", "x", "+", "*"], 
                                                 index=["|", "_", "s", "^", "v", "d", "x", "+", "*"].index(PHASE_LIBRARY[ph].get("marker_shape", "|")),
                                                 key=f"shp_{ph}",
                                                 help="| = vertical bar, _ = horizontal, s = square, d = diamond")
            
            phase_data.append({
                "name": ph,
                "positions": pk_df["two_theta"].values if len(pk_df) > 0 else [],
                "color": custom_color,
                "marker_shape": custom_shape,
                "hkl": [hkl.strip("()").split(",") if hkl else None for hkl in pk_df["hkl_label"].values] if show_hkl and len(pk_df) > 0 else None
            })
            
        # ✅ PASS NEW PARAMETERS to plotting function
        fig, ax = plot_rietveld_publication(
            active_df["two_theta"].values,
            active_df["intensity"].values,
            result["y_calc"],
            active_df["intensity"].values - result["y_calc"],
            phase_data,
            offset_factor=offset_factor,
            figsize=(fig_width, fig_height),
            font_size=font_size,
            legend_pos=legend_pos,
            marker_row_spacing=marker_spacing,  # ✅ NEW: controlled spacing
            legend_phases=legend_phases_selected if legend_phases_selected else None  # ✅ NEW: legend filter
        )
        st.pyplot(fig, dpi=150)
        
        st.markdown("#### 📥 Export Options")
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            buf = io.BytesIO()
            fig.savefig(buf, format='pdf', bbox_inches='tight')
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()
            href = f'<a href="application/pdf;base64,{b64}" download="rietveld_publication.pdf" style="display:inline-block;padding:8px 16px;background:#1f77b4;color:white;border-radius:4px;text-decoration:none;font-weight:500">📄 Download PDF</a>'
            st.markdown(href, unsafe_allow_html=True)
        with col_e2:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()
            href = f'<a href="image/png;base64,{b64}" download="rietveld_publication_300dpi.png" style="display:inline-block;padding:8px 16px;background:#2ca02c;color:white;border-radius:4px;text-decoration:none;font-weight:500">🖼️ Download PNG (300 DPI)</a>'
            st.markdown(href, unsafe_allow_html=True)
        with col_e3:
            buf = io.BytesIO()
            fig.savefig(buf, format='eps', bbox_inches='tight')
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()
            href = f'<a href="application/postscript;base64,{b64}" download="rietveld_publication.eps" style="display:inline-block;padding:8px 16px;background:#ff7f0e;color:white;border-radius:4px;text-decoration:none;font-weight:500">📐 Download EPS</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        with st.expander("🎨 Marker Shape Reference"):
            st.markdown("""
            | Shape | Code | Visual | Best For |
            |-------|------|--------|----------|
            | Vertical bar | `|` | │ | Primary FCC-Co matrix |
            | Horizontal bar | `_` | ─ | Secondary HCP-Co |
            | **Square** ✅ | `s` | ■ | Carbide phases (M₂₃C₆) |
            | Triangle up | `^` | ▲ | Sigma phase |
            | Triangle down | `v` | ▼ | Additional phases |
            | **Diamond** ✅ | `d` | ◆ | Trace/intermetallic phases |
            | Cross | `x` | × | Reference markers |
            | Plus | `+` | + | Calibration peaks |
            | Star | `*` | ✦ | Special annotations |
            """)
        plt.close(fig)
