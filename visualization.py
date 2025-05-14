import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="ML Profiler Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load memory data from CSV files
def load_memory_data(results_dir):
    memory_data = {}
    for exp_dir in glob.glob(os.path.join(results_dir, "*")):
        if os.path.isdir(exp_dir):
            exp_name = os.path.basename(exp_dir)
            memory_csv = os.path.join(exp_dir, "memory_data.csv")
            if os.path.exists(memory_csv):
                try:
                    df = pd.read_csv(memory_csv)
                    memory_data[exp_name] = df
                except Exception as e:
                    st.warning(f"Could not load {memory_csv}: {e}")
    return memory_data

# Function to load metadata from JSON files
def load_metadata(results_dir):
    metadata = {}
    for exp_dir in glob.glob(os.path.join(results_dir, "*")):
        if os.path.isdir(exp_dir):
            exp_name = os.path.basename(exp_dir)
            metadata_json = os.path.join(exp_dir, "metadata.json")
            if os.path.exists(metadata_json):
                try:
                    with open(metadata_json, 'r') as f:
                        metadata[exp_name] = json.load(f)
                except Exception as e:
                    st.warning(f"Could not load {metadata_json}: {e}")
    return metadata

# Function to parse experiment parameters from experiment name
def parse_exp_params(exp_name):
    parts = exp_name.split('_')
    if len(parts) >= 5:
        framework = parts[0]
        model_size = parts[1]
        batch_info = parts[2]  # Like 'b16'
        mode = parts[3]
        device = parts[4]
        
        # Extract batch size from batch_info (remove 'b' prefix)
        batch_size = int(batch_info[1:]) if batch_info.startswith('b') else None
        
        return {
            'framework': framework,
            'model_size': model_size,
            'batch_size': batch_size,
            'mode': mode,
            'device': device
        }
    return None

# Function to extract runtime from metadata
def extract_runtime(metadata_dict):
    if metadata_dict and 'runtime_seconds' in metadata_dict:
        return metadata_dict['runtime_seconds']
    return None

# Function to create summary dataframe
def create_summary_df(memory_data, metadata):
    summary_data = []
    
    for exp_name, mem_df in memory_data.items():
        params = parse_exp_params(exp_name)
        if params:
            max_memory = mem_df['memory_mb'].max() if not mem_df.empty else None
            avg_memory = mem_df['memory_mb'].mean() if not mem_df.empty else None
            runtime = extract_runtime(metadata.get(exp_name, {}))
            
            summary_data.append({
                'Experiment': exp_name,
                'Framework': params['framework'],
                'Model Size': params['model_size'],
                'Batch Size': params['batch_size'],
                'Mode': params['mode'],
                'Device': params['device'],
                'Max Memory (MB)': max_memory,
                'Avg Memory (MB)': avg_memory,
                'Runtime (s)': runtime
            })
    
    return pd.DataFrame(summary_data)

# Main function
def main():
    st.title("ML Profiler Dashboard")
    
    # Sidebar for data loading
    st.sidebar.header("Data Source")
    results_dir = st.sidebar.text_input("Results Directory Path", value="results")
    load_button = st.sidebar.button("Load Data")
    
    if load_button or 'summary_df' not in st.session_state:
        with st.spinner("Loading data..."):
            memory_data = load_memory_data(results_dir)
            metadata = load_metadata(results_dir)
            
            if not memory_data:
                st.error(f"No data found in {results_dir}. Please check the path.")
                return
                
            summary_df = create_summary_df(memory_data, metadata)
            st.session_state.memory_data = memory_data
            st.session_state.metadata = metadata
            st.session_state.summary_df = summary_df
    
    if 'summary_df' in st.session_state:
        summary_df = st.session_state.summary_df
        memory_data = st.session_state.memory_data
        
        # Filters
        st.sidebar.header("Filters")
        
        # Get unique values for filters
        frameworks = sorted(summary_df['Framework'].unique())
        model_sizes = sorted(summary_df['Model Size'].unique())
        batch_sizes = sorted(summary_df['Batch Size'].unique())
        modes = sorted(summary_df['Mode'].unique())
        devices = sorted(summary_df['Device'].unique())
        
        # Add filters to sidebar
        selected_frameworks = st.sidebar.multiselect("Frameworks", frameworks, default=frameworks)
        selected_model_sizes = st.sidebar.multiselect("Model Sizes", model_sizes, default=model_sizes)
        selected_batch_sizes = st.sidebar.multiselect("Batch Sizes", batch_sizes, default=batch_sizes)
        selected_modes = st.sidebar.multiselect("Modes", modes, default=modes)
        selected_devices = st.sidebar.multiselect("Devices", devices, default=devices)
        
        # Apply filters
        filtered_df = summary_df[
            (summary_df['Framework'].isin(selected_frameworks)) &
            (summary_df['Model Size'].isin(selected_model_sizes)) &
            (summary_df['Batch Size'].isin(selected_batch_sizes)) &
            (summary_df['Mode'].isin(selected_modes)) &
            (summary_df['Device'].isin(selected_devices))
        ]
        
        # Overview section
        st.header("Experiment Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Experiments", len(filtered_df))
        
        with col2:
            if not filtered_df.empty and 'Runtime (s)' in filtered_df.columns:
                avg_runtime = filtered_df['Runtime (s)'].mean()
                st.metric("Average Runtime (s)", f"{avg_runtime:.2f}")
        
        # Display filtered dataframe
        st.subheader("Experiment Summary")
        st.dataframe(filtered_df.style.highlight_max(subset=['Max Memory (MB)', 'Runtime (s)'], color='#ffcccb'))
        
        # Visualizations
        st.header("Performance Visualizations")
        
        # Ensure we have data after filtering
        if not filtered_df.empty:
            # Performance Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["Memory Usage", "Runtime Analysis", "Comparative Analysis", "Memory Traces"])
            
            with tab1:
                st.subheader("Memory Usage by Configuration")
                
                # Memory usage chart
                fig1 = px.bar(
                    filtered_df,
                    x='Experiment',
                    y='Max Memory (MB)',
                    color='Framework',
                    pattern_shape='Model Size',
                    facet_col='Device',
                    facet_row='Mode',
                    hover_data=['Batch Size'],
                    title='Maximum Memory Usage by Experiment',
                    height=600
                )
                st.plotly_chart(fig1, use_container_width=True)
                
                # Memory usage by model size and batch size
                col1, col2 = st.columns(2)
                
                with col1:
                    # Memory vs. Model Size
                    fig = px.box(
                        filtered_df,
                        x='Model Size',
                        y='Max Memory (MB)',
                        color='Framework',
                        facet_col='Device',
                        title='Memory Usage by Model Size',
                        points="all"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Memory vs. Batch Size
                    fig = px.box(
                        filtered_df,
                        x='Batch Size',
                        y='Max Memory (MB)',
                        color='Framework',
                        facet_col='Device',
                        title='Memory Usage by Batch Size',
                        points="all"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("Runtime Analysis")
                
                # Runtime chart
                fig2 = px.bar(
                    filtered_df,
                    x='Experiment',
                    y='Runtime (s)',
                    color='Framework',
                    pattern_shape='Model Size',
                    facet_col='Device',
                    facet_row='Mode',
                    hover_data=['Batch Size'],
                    title='Runtime by Experiment',
                    height=600
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                # Runtime comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    # Runtime vs. Model Size
                    fig = px.box(
                        filtered_df,
                        x='Model Size',
                        y='Runtime (s)',
                        color='Framework',
                        facet_col='Device',
                        title='Runtime by Model Size',
                        points="all"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Runtime vs. Batch Size
                    fig = px.box(
                        filtered_df,
                        x='Batch Size',
                        y='Runtime (s)',
                        color='Framework',
                        facet_col='Device',
                        title='Runtime by Batch Size',
                        points="all"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("Comparative Analysis")
                
                # CPU vs GPU comparison
                if 'cpu' in devices and 'gpu' in devices:
                    st.subheader("CPU vs GPU Performance")
                    
                    # Group by relevant factors and compute average metrics
                    grouped_df = filtered_df.groupby(['Framework', 'Model Size', 'Batch Size', 'Mode', 'Device']).agg({
                        'Max Memory (MB)': 'mean',
                        'Runtime (s)': 'mean'
                    }).reset_index()
                    
                    # Create a pivot table for CPU vs GPU comparison
                    pivot_runtime = pd.pivot_table(
                        grouped_df,
                        values='Runtime (s)',
                        index=['Framework', 'Model Size', 'Batch Size', 'Mode'],
                        columns='Device'
                    ).reset_index()
                    
                    # Calculate speedup
                    if 'cpu' in pivot_runtime.columns and 'gpu' in pivot_runtime.columns:
                        pivot_runtime['Speedup'] = pivot_runtime['cpu'] / pivot_runtime['gpu']
                        
                        # Create bar chart for speedup
                        fig = px.bar(
                            pivot_runtime,
                            x=pivot_runtime.index,
                            y='Speedup',
                            color='Framework',
                            facet_col='Mode',
                            title='GPU Speedup Factor (CPU Time / GPU Time)',
                            labels={'index': 'Configuration', 'Speedup': 'Speedup Factor'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Train vs Inference comparison
                if 'train' in modes and 'inference' in modes:
                    st.subheader("Training vs Inference Performance")
                    
                    # Group by relevant factors
                    grouped_df = filtered_df.groupby(['Framework', 'Model Size', 'Batch Size', 'Device', 'Mode']).agg({
                        'Max Memory (MB)': 'mean',
                        'Runtime (s)': 'mean'
                    }).reset_index()
                    
                    # Create pivot table for train vs inference
                    pivot_mode = pd.pivot_table(
                        grouped_df,
                        values=['Runtime (s)', 'Max Memory (MB)'],
                        index=['Framework', 'Model Size', 'Batch Size', 'Device'],
                        columns='Mode'
                    ).reset_index()
                    
                    # Flatten the multi-index columns
                    pivot_mode.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in pivot_mode.columns]
                    
                    # Create comparison visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'Runtime (s)_train' in pivot_mode.columns and 'Runtime (s)_inference' in pivot_mode.columns:
                            pivot_mode['Train/Inference Runtime Ratio'] = pivot_mode['Runtime (s)_train'] / pivot_mode['Runtime (s)_inference']
                            
                            fig = px.bar(
                                pivot_mode,
                                x=pivot_mode.index,
                                y='Train/Inference Runtime Ratio',
                                color='Framework',
                                facet_col='Device',
                                title='Training/Inference Runtime Ratio',
                                labels={'index': 'Configuration', 'Train/Inference Runtime Ratio': 'Ratio'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        if 'Max Memory (MB)_train' in pivot_mode.columns and 'Max Memory (MB)_inference' in pivot_mode.columns:
                            pivot_mode['Train/Inference Memory Ratio'] = pivot_mode['Max Memory (MB)_train'] / pivot_mode['Max Memory (MB)_inference']
                            
                            fig = px.bar(
                                pivot_mode,
                                x=pivot_mode.index,
                                y='Train/Inference Memory Ratio',
                                color='Framework',
                                facet_col='Device',
                                title='Training/Inference Memory Ratio',
                                labels={'index': 'Configuration', 'Train/Inference Memory Ratio': 'Ratio'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.subheader("Memory Traces")
                
                # Experiment selector for memory traces
                selected_exp = st.selectbox(
                    "Select experiment to view memory trace", 
                    options=filtered_df['Experiment'].tolist()
                )
                
                if selected_exp in memory_data:
                    mem_trace = memory_data[selected_exp]
                    
                    # Plot memory trace
                    fig = px.line(
                        mem_trace,
                        x='timestamp',
                        y='memory_mb',
                        title=f'Memory Trace for {selected_exp}',
                        labels={'timestamp': 'Time (s)', 'memory_mb': 'Memory Usage (MB)'}
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display memory trace statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Max Memory (MB)", f"{mem_trace['memory_mb'].max():.2f}")
                    
                    with col2:
                        st.metric("Min Memory (MB)", f"{mem_trace['memory_mb'].min():.2f}")
                    
                    with col3:
                        st.metric("Avg Memory (MB)", f"{mem_trace['memory_mb'].mean():.2f}")
                    
                    with col4:
                        st.metric("Memory Range (MB)", f"{mem_trace['memory_mb'].max() - mem_trace['memory_mb'].min():.2f}")
                    
                    # Show raw data
                    if st.checkbox("Show raw memory data"):
                        st.dataframe(mem_trace)
        else:
            st.warning("No data available after applying filters. Please adjust your filter criteria.")
    
    # Footer
    st.markdown("---")
    st.markdown("ML Profiler Dashboard - A tool for analyzing ML framework performance")

if __name__ == '__main__':
    main()