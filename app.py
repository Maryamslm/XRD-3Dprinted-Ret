# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — PUBLICATION-QUALITY PLOT (ENHANCED & BUG-FIXED)
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[6]:  # ✅ Ensure this is inside the main app flow, after tabs = st.tabs([...])
    st.subheader("🖼️ Publication-Quality Plot (matplotlib)")
    st.caption("Generate journal-ready figures with customizable phase markers, legend control & spacing")
    
    # Check if refinement results exist
    if "last_result" not in st.session_state or "last_phases" not in st.session_state:
        st.info("🔬 Run the Rietveld refinement first (Tab 3: 🧮 Rietveld Fit) to enable publication plotting.")
        st.markdown("""
        **Quick steps:**
        1. Select a sample in the sidebar
        2. Choose phases to refine (e.g., FCC-Co, M23C6)
        3. Click **▶ Run Rietveld Refinement**
        4. Return here to generate your publication plot
        """)
    else:
        result = st.session_state["last_result"]
        phases = st.session_state["last_phases"]
        
        # ── Layout Controls ─────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        with col1:
            fig_width = st.slider("Figure width (inches)", 6.0, 14.0, 10.0, 0.5, key="pub_width")
            offset_factor = st.slider("Difference curve offset", 0.05, 0.25, 0.12, 0.01, key="pub_offset")
            font_size = st.slider("Global Font Size", 6, 22, 11, key="pub_font")
        with col2:
            fig_height = st.slider("Figure height (inches)", 5.0, 12.0, 7.0, 0.5, key="pub_height")
            show_hkl = st.checkbox("Show hkl labels", value=True, key="pub_hkl")
            legend_pos = st.selectbox("Legend Position", 
                                      ["best", "upper right", "upper left", "lower left", "lower right", 
                                       "center right", "center left", "lower center", "upper center", "center", "off"],
                                      index=0, key="pub_legend_pos")
        with col3:
            export_format = st.selectbox("Export format", ["PDF", "PNG", "EPS"], index=0, key="pub_format")
            # ✅ NEW: Marker row spacing control
            marker_spacing = st.slider("Marker row spacing", 0.8, 2.5, 1.3, 0.1, 
                                      help="Vertical distance between phase marker rows",
                                      key="pub_spacing")
            st.markdown("**🎨 Phase Customization**")
            
        # ✅ NEW: Legend control section
        st.markdown("### 📋 Legend Control")
        st.caption("Select which phases to include in the plot legend (uncheck to hide from legend)")
        n_cols = min(4, len(phases))
        legend_cols = st.columns(n_cols)
        legend_phases_selected = []
        for idx, ph in enumerate(phases):
            col_idx = idx % n_cols
            with legend_cols[col_idx]:
                if st.checkbox(f"✓ {ph}", value=True, key=f"leg_{ph}"):
                    legend_phases_selected.append(ph)
        
        # Build phase_data list with user customizations
        phase_data = []
        for i, ph in enumerate(phases):
            pk_df = generate_theoretical_peaks(ph, wavelength, tt_min, tt_max)
            
            with st.expander(f"⚙️ Settings for **{ph}**", expanded=(i==0)):
                c_col, c_shape = st.columns(2)
                custom_color = c_col.color_picker("Color", value=PHASE_LIBRARY[ph]["color"], key=f"col_{ph}")
                # ✅ UPDATED: Square (s) and Diamond (d) now available
                shape_options = ["|", "_", "s", "^", "v", "d", "x", "+", "*"]
                default_idx = shape_options.index(PHASE_LIBRARY[ph].get("marker_shape", "|"))
                custom_shape = c_shape.selectbox("Marker Shape", shape_options, 
                                                 index=default_idx, key=f"shp_{ph}",
                                                 help="| = vertical bar, _ = horizontal, s = square ■, d = diamond ◆")
            
            phase_data.append({
                "name": ph,
                "positions": pk_df["two_theta"].values if len(pk_df) > 0 else np.array([]),
                "color": custom_color,
                "marker_shape": custom_shape,
                "hkl": [hkl.strip("()").split(",") if hkl else None for hkl in pk_df["hkl_label"].values] if show_hkl and len(pk_df) > 0 else None
            })
            
        # Generate the plot
        try:
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
                marker_row_spacing=marker_spacing,
                legend_phases=legend_phases_selected if legend_phases_selected else None
            )
            st.pyplot(fig, dpi=150, use_container_width=True)
            
            # Export buttons
            st.markdown("#### 📥 Export Options")
            col_e1, col_e2, col_e3 = st.columns(3)
            
            with col_e1:
                buf = io.BytesIO()
                fig.savefig(buf, format='pdf', bbox_inches='tight')
                buf.seek(0)
                st.download_button("📄 PDF", buf.read(), 
                                 file_name=f"rietveld_pub_{selected_key}.pdf",
                                 mime="application/pdf", use_container_width=True)
            with col_e2:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                st.download_button("🖼️ PNG (300 DPI)", buf.read(), 
                                 file_name=f"rietveld_pub_{selected_key}.png",
                                 mime="image/png", use_container_width=True)
            with col_e3:
                buf = io.BytesIO()
                fig.savefig(buf, format='eps', bbox_inches='tight')
                buf.seek(0)
                st.download_button("📐 EPS", buf.read(), 
                                 file_name=f"rietveld_pub_{selected_key}.eps",
                                 mime="application/postscript", use_container_width=True)
            
            # Marker reference
            with st.expander("🎨 Marker Shape Reference"):
                st.markdown("""
                | Shape | Code | Visual | Recommended Use |
                |-------|------|--------|----------------|
                | Vertical bar | `|` | │ | FCC-Co matrix (primary) |
                | Horizontal bar | `_` | ─ | HCP-Co (secondary) |
                | **Square** ✨ | `s` | ■ | M₂₃C₆ carbides |
                | Triangle up | `^` | ▲ | Sigma phase |
                | Triangle down | `v` | ▼ | Additional precipitates |
                | **Diamond** ✨ | `d` | ◆ | Trace intermetallics |
                | Cross | `x` | × | Reference peaks |
                | Plus | `+` | + | Calibration markers |
                | Star | `*` | ✦ | Special annotations |
                """)
            plt.close(fig)  # Free memory
            
        except Exception as e:
            st.error(f"❌ Plot generation failed: {str(e)}")
            st.code("Tip: Try reducing the number of phases or resetting font size to default.")
