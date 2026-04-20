import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.title("🔬 XRD Analysis - Test Version")
st.write("✅ Streamlit is working!")

# Simple test plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=[30, 40, 50], y=[100, 200, 150], mode='lines+markers'))
st.plotly_chart(fig, use_container_width=True)

st.success("🎉 Basic app is working!")
