import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import json
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Vitual ML Memory Profiler Dashboard - Yaxita Amin",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0083B8;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #0083B8;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .subsection-header {
        font-size: 1.4rem;
        font-weight: 500;
        color: #0083B8;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stPlotlyChart {
        background-color: white;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">Vitual ML Memory Profiler Dashboard - Yaxita Amin</div>', unsafe_allow_html=True)

# Function to load the data
@st.cache_data
def load_data():
    # Look for the CSV files in the expected directories
    data_frames = []
    
    # If running in production with specific directory structure
    try:
        # Directory of results
        directories = [d for d in os.listdir("results") if os.path.isdir(os.path.join("results", d))]
        
        for directory in directories:
            csv_path = os.path.join("results", directory, "memory_data.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # Add source directory as a column
                df['source_directory'] = directory
                data_frames.append(df)
    except:
        # Fallback: Look in current directory
        if os.path.exists("memory_data.csv"):
            df = pd.read_csv("memory_data.csv")
            data_frames.append(df)
    
    if not data_frames:
        # If no data found, create sample data for UI development
        st.warning("❗ No data files found. Using sample data instead. Please place your data files in the correct location.")
        return create_sample_data()
    
    # Combine all dataframes
    df = pd.concat(data_frames, ignore_index=True)
    
    # Extract components from the source directory name if available
    if 'source_directory' in df.columns:
        # Expected format: {framework}_{model_size}_{batch_size}_{mode}_{device}
        df[['framework', 'model_size', 'batch_size_str', 'mode', 'device']] = df['source_directory'].str.split('_', expand=True)
        # Extract numeric batch size
        df['batch_size'] = df['batch_size_str'].str.extract('b(\d+)').astype(int)
    
    # Ensure proper data types
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp if available
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
        
    # Set the primary memory metric - use 'rss' as the main memory usage metric
    if 'rss' in df.columns:
        # Convert RSS to MB (assuming it's in KB)
        df['memory_used_mb'] = df['rss'] / 1024
    
    # If no memory columns exist, create sample data
    if 'memory_used_mb' not in df.columns:
        st.warning("Memory usage columns not found in the CSV. Please ensure your data contains memory metrics.")
        # Create a default memory column using whatever we have available
        if 'vms' in df.columns:
            df['memory_used_mb'] = df['vms'] / 1024
        else:
            # Create dummy data as a last resort
            df['memory_used_mb'] = np.random.uniform(low=100, high=1000, size=len(df))
            st.error("No valid memory metrics found. Using random data for visualization purposes.")
    
    return df

def create_sample_data():
    """Create sample data for UI development"""
    frameworks = ['pytorch', 'tensorflow']
    model_sizes = ['small', 'medium', 'large']
    batch_sizes = [16, 32, 64]
    modes = ['train', 'inference']
    devices = ['cpu', 'gpu']
    
    rows = []
    timestamp_base = pd.Timestamp('2023-01-01')
    
    for framework in frameworks:
        for model_size in model_sizes:
            for batch_size in batch_sizes:
                for mode in modes:
                    for device in devices:
                        # Generate 10 measurement points for each configuration
                        for i in range(10):
                            # Generate memory metrics
                            rss = np.random.uniform(
                                low=100_000 if model_size == 'small' else (500_000 if model_size == 'medium' else 1_000_000),
                                high=500_000 if model_size == 'small' else (2_000_000 if model_size == 'medium' else 5_000_000)
                            )
                            
                            vms = rss * np.random.uniform(1.5, 2.5)
                            
                            # GPU memory is typically higher
                            if device == 'gpu':
                                rss *= 1.5
                                vms *= 1.5
                            
                            # Create a sample row
                            row = {
                                'timestamp': timestamp_base + pd.Timedelta(minutes=i*10),
                                'rss': rss,
                                'vms': vms,
                                'page_faults': np.random.randint(1000, 10000),
                                'major_faults': np.random.randint(0, 100),
                                'minor_faults': np.random.randint(100, 5000),
                                'swap': np.random.uniform(0, 1000),
                                'cpu_percent': np.random.uniform(0, 100),
                                'source_directory': f"{framework}_{model_size}_b{batch_size}_{mode}_{device}",
                                'framework': framework,
                                'model_size': model_size,
                                'batch_size': batch_size,
                                'batch_size_str': f"b{batch_size}",
                                'mode': mode,
                                'device': device
                            }
                            rows.append(row)
    
    df = pd.DataFrame(rows)
    df['memory_used_mb'] = df['rss'] / 1024  # Convert to MB
    return df

# Load the data
df = load_data()

# ----------------- SIDEBAR FILTERS -----------------
st.sidebar.markdown("## Filters")

# Unique values for each categorical column
frameworks = sorted(df['framework'].unique())
model_sizes = sorted(df['model_size'].unique())
batch_sizes = sorted(df['batch_size'].unique())
modes = sorted(df['mode'].unique())
devices = sorted(df['device'].unique())

# Create filters in the sidebar
selected_frameworks = st.sidebar.multiselect("Frameworks", frameworks, default=frameworks)
selected_model_sizes = st.sidebar.multiselect("Model Sizes", model_sizes, default=model_sizes)
selected_batch_sizes = st.sidebar.multiselect("Batch Sizes", batch_sizes, default=batch_sizes)
selected_modes = st.sidebar.multiselect("Modes", modes, default=modes)
selected_devices = st.sidebar.multiselect("Devices", devices, default=devices)

# Apply filters
filtered_df = df[
    df['framework'].isin(selected_frameworks) &
    df['model_size'].isin(selected_model_sizes) &
    df['batch_size'].isin(selected_batch_sizes) &
    df['mode'].isin(selected_modes) &
    df['device'].isin(selected_devices)
]

# Aggregation options
st.sidebar.markdown("## Aggregation")
agg_method = st.sidebar.selectbox("Aggregation Method", 
                                 ["Mean", "Max", "Min", "Median"], 
                                 index=0)

agg_function = {
    "Mean": np.mean,
    "Max": np.max,
    "Min": np.min,
    "Median": np.median
}[agg_method]

# ----------------- METRIC SELECTION -----------------
st.sidebar.markdown("## Metrics")
available_metrics = ['memory_used_mb']

# Add other metrics if available
if 'vms' in df.columns:
    if 'memory_used_mb' not in df.columns or not np.array_equal(df['memory_used_mb'], df['rss'] / 1024):
        available_metrics.append('vms (KB)')
if 'cpu_percent' in df.columns:
    available_metrics.append('cpu_percent')
if 'page_faults' in df.columns:
    available_metrics.append('page_faults')
if 'major_faults' in df.columns:
    available_metrics.append('major_faults')
if 'minor_faults' in df.columns:
    available_metrics.append('minor_faults')
if 'swap' in df.columns:
    available_metrics.append('swap')

selected_metric = st.sidebar.selectbox("Primary Metric", available_metrics, index=0)

# Function to get the actual column name and scaling
def get_metric_info(metric_name):
    if metric_name == 'memory_used_mb':
        return 'memory_used_mb', 1, 'MB'
    elif metric_name == 'vms (KB)':
        return 'vms', 1, 'KB'
    else:
        return metric_name, 1, ''

metric_col, scale_factor, unit = get_metric_info(selected_metric)

# ----------------- OVERVIEW SECTION -----------------
st.markdown('<div class="section-header">📊 Memory Usage Overview</div>', unsafe_allow_html=True)

# Key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Experiments", 
        f"{len(filtered_df['source_directory'].unique())}",
        delta=f"{len(filtered_df['source_directory'].unique()) - len(df['source_directory'].unique())}" if len(filtered_df['source_directory'].unique()) != len(df['source_directory'].unique()) else None
    )

with col2:
    avg_value = filtered_df[metric_col].mean() / scale_factor
    metric_name = "Memory Usage" if "memory" in selected_metric.lower() else selected_metric
    st.metric(
        f"{agg_method} {metric_name}", 
        f"{avg_value:.2f} {unit}"
    )

with col3:
    peak_value = filtered_df[metric_col].max() / scale_factor
    st.metric(
        f"Peak {metric_name}", 
        f"{peak_value:.2f} {unit}"
    )

with col4:
    baseline = filtered_df[
        (filtered_df['model_size'] == 'small') & 
        (filtered_df['batch_size'] == min(batch_sizes)) & 
        (filtered_df['mode'] == 'inference')
    ][metric_col].mean()
    
    largest_config = filtered_df[
        (filtered_df['model_size'] == 'large') & 
        (filtered_df['batch_size'] == max(batch_sizes)) & 
        (filtered_df['mode'] == 'train')
    ][metric_col].mean()
    
    if not np.isnan(baseline) and not np.isnan(largest_config) and baseline > 0:
        growth_factor = largest_config / baseline
        st.metric(
            f"{metric_name} Growth Factor",
            f"{growth_factor:.2f}x"
        )
    else:
        st.metric(
            f"{metric_name} Growth Factor",
            "N/A"
        )

# ----------------- FRAMEWORK COMPARISON SECTION -----------------
st.markdown('<div class="section-header">🔄 Framework Comparison</div>', unsafe_allow_html=True)

# Framework comparison
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    # Group by framework and compute aggregates
    framework_summary = filtered_df.groupby('framework')[metric_col].agg([
        ('mean', 'mean'), 
        ('max', 'max'),
        ('min', 'min'),
        ('median', 'median')
    ]).reset_index()
    
    fig_framework_bar = px.bar(
        framework_summary,
        x='framework',
        y=agg_method.lower(),
        color='framework',
        title=f'{agg_method} {metric_name} by Framework',
        labels={'framework': 'Framework', agg_method.lower(): f'{metric_name} ({unit})'},
        template='plotly_white'
    )
    fig_framework_bar.update_layout(height=400)
    st.plotly_chart(fig_framework_bar, use_container_width=True)

with row1_col2:
    # Framework comparison across devices
    framework_device_summary = filtered_df.groupby(['framework', 'device'])[metric_col].agg(agg_function).reset_index()
    
    fig_framework_device = px.bar(
        framework_device_summary,
        x='framework',
        y=metric_col,
        color='device',
        barmode='group',
        title=f'{agg_method} {metric_name} by Framework and Device',
        labels={'framework': 'Framework', metric_col: f'{metric_name} ({unit})', 'device': 'Device'},
        template='plotly_white'
    )
    fig_framework_device.update_layout(height=400)
    st.plotly_chart(fig_framework_device, use_container_width=True)

# ----------------- DEVICE COMPARISON SECTION -----------------
st.markdown('<div class="section-header">💻 CPU vs GPU Analysis</div>', unsafe_allow_html=True)

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    # Memory usage difference between CPU and GPU across different dimensions
    pivot_df = filtered_df.pivot_table(
        index=['framework', 'model_size', 'batch_size', 'mode'],
        columns='device',
        values=metric_col,
        aggfunc=agg_function
    ).reset_index()
    
    if 'cpu' in pivot_df.columns and 'gpu' in pivot_df.columns:
        pivot_df['gpu_to_cpu_ratio'] = pivot_df['gpu'] / pivot_df['cpu']
        
        fig_gpu_cpu_ratio = px.bar(
            pivot_df,
            x='framework',
            y='gpu_to_cpu_ratio',
            color='model_size',
            facet_col='mode',
            facet_row='batch_size',
            title=f'GPU to CPU {metric_name} Ratio',
            labels={
                'framework': 'Framework',
                'gpu_to_cpu_ratio': f'GPU/CPU {metric_name} Ratio',
                'model_size': 'Model Size',
                'batch_size': 'Batch Size',
                'mode': 'Mode'
            },
            template='plotly_white',
            barmode='group'
        )
        fig_gpu_cpu_ratio.update_layout(height=600)
        st.plotly_chart(fig_gpu_cpu_ratio, use_container_width=True)
    else:
        st.info("Both CPU and GPU data needed for this visualization")

with row2_col2:
    # Overall CPU vs GPU usage pattern across batches and model sizes
    device_summary = filtered_df.groupby(['device', 'model_size', 'batch_size'])[metric_col].agg(agg_function).reset_index()
    
    fig_device_model_batch = px.line(
        device_summary,
        x='batch_size',
        y=metric_col,
        color='device',
        line_dash='model_size',
        markers=True,
        title=f'{agg_method} {metric_name} by Device, Model Size and Batch Size',
        labels={
            'batch_size': 'Batch Size',
            metric_col: f'{metric_name} ({unit})',
            'device': 'Device',
            'model_size': 'Model Size'
        },
        template='plotly_white'
    )
    fig_device_model_batch.update_layout(height=600)
    st.plotly_chart(fig_device_model_batch, use_container_width=True)

# ----------------- MODEL & BATCH SIZE IMPACT -----------------
st.markdown('<div class="section-header">📏 Model Size & Batch Size Impact</div>', unsafe_allow_html=True)

# Analysis by Model Size and Batch Size
row3_col1, row3_col2 = st.columns(2)

with row3_col1:
    # Heatmap of memory usage by model size and batch size
    model_batch_summary = filtered_df.groupby(['model_size', 'batch_size'])[metric_col].agg(agg_function).reset_index()
    pivot_heatmap = model_batch_summary.pivot(index='model_size', columns='batch_size', values=metric_col)
    
    # Sort the indices for proper display order
    model_size_order = ['small', 'medium', 'large']
    available_sizes = [size for size in model_size_order if size in pivot_heatmap.index]
    pivot_heatmap = pivot_heatmap.reindex(available_sizes)
    
    fig_heatmap = px.imshow(
        pivot_heatmap,
        labels=dict(x="Batch Size", y="Model Size", color=f"{metric_name} ({unit})"),
        x=pivot_heatmap.columns,
        y=pivot_heatmap.index,
        title=f'{agg_method} {metric_name} by Model Size and Batch Size',
        color_continuous_scale='viridis',
        text_auto=True
    )
    fig_heatmap.update_layout(height=400)
    st.plotly_chart(fig_heatmap, use_container_width=True)

with row3_col2:
    # Impact of batch size on memory usage across frameworks
    batch_framework_summary = filtered_df.groupby(['batch_size', 'framework', 'model_size'])[metric_col].agg(agg_function).reset_index()
    
    fig_batch_impact = px.line(
        batch_framework_summary,
        x='batch_size',
        y=metric_col,
        color='framework',
        line_dash='model_size',
        markers=True,
        title=f'Batch Size Impact on {agg_method} {metric_name}',
        labels={
            'batch_size': 'Batch Size',
            metric_col: f'{metric_name} ({unit})',
            'framework': 'Framework',
            'model_size': 'Model Size'
        },
        template='plotly_white'
    )
    fig_batch_impact.update_layout(height=400)
    st.plotly_chart(fig_batch_impact, use_container_width=True)

# ----------------- TRAINING vs INFERENCE -----------------
st.markdown('<div class="section-header">🔄 Training vs Inference Mode Analysis</div>', unsafe_allow_html=True)

row4_col1, row4_col2 = st.columns(2)

with row4_col1:
    # Memory usage comparison between training and inference
    mode_summary = filtered_df.groupby(['mode', 'framework', 'model_size'])[metric_col].agg(agg_function).reset_index()
    
    fig_mode_comp = px.bar(
        mode_summary,
        x='framework',
        y=metric_col,
        color='mode',
        facet_col='model_size',
        barmode='group',
        title=f'{agg_method} {metric_name}: Training vs Inference',
        labels={
            'framework': 'Framework',
            metric_col: f'{metric_name} ({unit})',
            'mode': 'Mode',
            'model_size': 'Model Size'
        },
        template='plotly_white'
    )
    fig_mode_comp.update_layout(height=400)
    st.plotly_chart(fig_mode_comp, use_container_width=True)

with row4_col2:
    # Calculate the training to inference ratio
    mode_pivot = filtered_df.pivot_table(
        index=['framework', 'model_size', 'batch_size', 'device'],
        columns='mode',
        values=metric_col,
        aggfunc=agg_function
    ).reset_index()
    
    if 'train' in mode_pivot.columns and 'inference' in mode_pivot.columns:
        mode_pivot['train_to_inference_ratio'] = mode_pivot['train'] / mode_pivot['inference']
        
        fig_train_inf_ratio = px.box(
            mode_pivot,
            x='framework',
            y='train_to_inference_ratio',
            color='model_size',
            facet_col='device',
            points="all",
            title=f'Training to Inference {metric_name} Ratio',
            labels={
                'framework': 'Framework',
                'train_to_inference_ratio': 'Train/Inference Ratio',
                'model_size': 'Model Size',
                'device': 'Device'
            },
            template='plotly_white'
        )
        fig_train_inf_ratio.update_yaxes(range=[0, mode_pivot['train_to_inference_ratio'].quantile(0.95) * 1.1])
        fig_train_inf_ratio.update_layout(height=400)
        st.plotly_chart(fig_train_inf_ratio, use_container_width=True)
    else:
        st.info("Both training and inference data needed for this visualization")

# ----------------- ADDITIONAL METRICS ANALYSIS -----------------
if 'cpu_percent' in df.columns:
    st.markdown('<div class="section-header">🖥️ CPU Usage Analysis</div>', unsafe_allow_html=True)
    
    cpu_summary = filtered_df.groupby(['framework', 'model_size', 'batch_size', 'mode', 'device'])['cpu_percent'].agg(agg_function).reset_index()
    
    fig_cpu = px.bar(
        cpu_summary,
        x='framework',
        y='cpu_percent',
        color='model_size',
        facet_col='device',
        facet_row='mode',
        barmode='group',
        title=f'{agg_method} CPU Usage (%)',
        labels={
            'framework': 'Framework',
            'cpu_percent': 'CPU Usage (%)',
            'model_size': 'Model Size',
            'device': 'Device',
            'mode': 'Mode'
        },
        template='plotly_white'
    )
    fig_cpu.update_layout(height=600)
    st.plotly_chart(fig_cpu, use_container_width=True)

if 'page_faults' in df.columns:
    st.markdown('<div class="section-header">⚠️ Fault Analysis</div>', unsafe_allow_html=True)
    
    row5_col1, row5_col2 = st.columns(2)
    
    with row5_col1:
        faults_summary = filtered_df.groupby(['framework', 'model_size', 'mode', 'device'])['page_faults'].agg(agg_function).reset_index()
        
        fig_faults = px.bar(
            faults_summary,
            x='framework',
            y='page_faults',
            color='model_size',
            facet_col='device',
            barmode='group',
            title=f'{agg_method} Page Faults',
            labels={
                'framework': 'Framework',
                'page_faults': 'Page Faults',
                'model_size': 'Model Size',
                'device': 'Device'
            },
            template='plotly_white'
        )
        fig_faults.update_layout(height=400)
        st.plotly_chart(fig_faults, use_container_width=True)
    
    with row5_col2:
        if 'major_faults' in df.columns and 'minor_faults' in df.columns:
            faults_data = []
            for framework in selected_frameworks:
                for device in selected_devices:
                    major = filtered_df[(filtered_df['framework'] == framework) & (filtered_df['device'] == device)]['major_faults'].agg(agg_function)
                    minor = filtered_df[(filtered_df['framework'] == framework) & (filtered_df['device'] == device)]['minor_faults'].agg(agg_function)
                    faults_data.append({
                        'framework': framework,
                        'device': device,
                        'fault_type': 'Major Faults',
                        'value': major
                    })
                    faults_data.append({
                        'framework': framework,
                        'device': device,
                        'fault_type': 'Minor Faults',
                        'value': minor
                    })
            
            faults_df = pd.DataFrame(faults_data)
            
            fig_fault_types = px.bar(
                faults_df,
                x='framework',
                y='value',
                color='fault_type',
                facet_col='device',
                barmode='group',
                title=f'{agg_method} Fault Types',
                labels={
                    'framework': 'Framework',
                    'value': 'Number of Faults',
                    'fault_type': 'Fault Type',
                    'device': 'Device'
                },
                template='plotly_white'
            )
            fig_fault_types.update_layout(height=400)
            st.plotly_chart(fig_fault_types, use_container_width=True)
        else:
            st.info("Major and minor faults data needed for this visualization")

# ----------------- DETAILED CONFIGURATION COMPARISON -----------------
st.markdown('<div class="section-header">🔍 Detailed Configuration Comparison</div>', unsafe_allow_html=True)

# Create a radar chart to compare different configurations
st.markdown('<div class="subsection-header">Radar Chart: Configuration Performance</div>', unsafe_allow_html=True)

# Select configurations for radar chart
unique_configs = filtered_df['source_directory'].unique()
default_configs = unique_configs[:min(5, len(unique_configs))]
selected_configs = st.multiselect(
    "Select configurations to compare:",
    options=unique_configs,
    default=default_configs,
    max_selections=7  # Limit to avoid overcrowding
)

if selected_configs:
    # Prepare radar chart data
    radar_df = filtered_df[filtered_df['source_directory'].isin(selected_configs)]
    
    # Determine metrics to include in radar chart
    radar_metrics = ['memory_used_mb']
    radar_metric_names = ['Memory (MB)']
    
    if 'cpu_percent' in filtered_df.columns:
        radar_metrics.append('cpu_percent')
        radar_metric_names.append('CPU (%)')
    
    if 'page_faults' in filtered_df.columns:
        radar_metrics.append('page_faults')
        radar_metric_names.append('Page Faults')
    
    # Compute metrics for each configuration
    radar_data = []
    for config in selected_configs:
        config_df = radar_df[radar_df['source_directory'] == config]
        metrics_values = []
        
        for metric in radar_metrics:
            # Get the metric value and normalize it
            value = config_df[metric].agg(agg_function) 
            metrics_values.append(value)
        
        # Calculate percentages for the radar chart (normalize to max value)
        max_values = []
        for i, metric in enumerate(radar_metrics):
            max_val = radar_df[metric].max()
            if max_val > 0:
                metrics_values[i] = metrics_values[i] / max_val * 100
            else:
                metrics_values[i] = 0
            max_values.append(max_val)
            
        radar_data.append({
            'config': config,
            'metrics': metrics_values,
            'metric_names': radar_metric_names,
            'raw_values': [config_df[metric].agg(agg_function) for metric in radar_metrics],
            'max_values': max_values
        })
    
    # Create radar chart
    fig_radar = go.Figure()
    
    for data in radar_data:
        fig_radar.add_trace(go.Scatterpolar(
            r=data['metrics'],
            theta=data['metric_names'],
            fill='toself',
            name=data['config']
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title="Normalized Performance Metrics Comparison",
        height=600
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Display the actual values in a table
    st.markdown('<div class="subsection-header">Actual Values</div>', unsafe_allow_html=True)
    
    comparison_data = []
    for data in radar_data:
        row = {'Configuration': data['config']}
        for i, name in enumerate(data['metric_names']):
            row[name] = f"{data['raw_values'][i]:.2f}"
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    # Display the actual values in a table
    st.markdown('<div class="subsection-header">Actual Values</div>', unsafe_allow_html=True)
    
    comparison_data = []
    for data in radar_data:
        row = {'Configuration': data['config']}
        for i, name in enumerate(data['metric_names']):
            row[name] = f"{data['raw_values'][i]:.2f}"
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

# ----------------- RAW DATA DISPLAY -----------------
st.markdown('<div class="section-header">📋 Raw Data & Experiment Details</div>', unsafe_allow_html=True)

# Display experiment details
show_experiment_details = st.checkbox("Show Experiment Details", value=False)

if show_experiment_details:
    # Show detailed experiment settings
    experiments_summary = filtered_df.groupby(['source_directory']).agg({
        'memory_used_mb': ['mean', 'max', 'min', 'std', 'count'],
        'framework': 'first',
        'model_size': 'first',
        'batch_size': 'first',
        'mode': 'first',
        'device': 'first'
    }).reset_index()
    
    experiments_summary.columns = [
        'Experiment', 'Mean Memory (MB)', 'Max Memory (MB)', 'Min Memory (MB)', 
        'Std Dev (MB)', 'Sample Count', 'Framework', 'Model Size', 'Batch Size', 
        'Mode', 'Device'
    ]
    
    st.dataframe(experiments_summary, use_container_width=True)

# Show raw data table
show_raw_data = st.checkbox("Show Raw Data", value=False)

if show_raw_data:
    # Add pagination or limit rows to prevent browser slowdown
    page_size = st.slider("Rows per page", min_value=10, max_value=100, value=50, step=10)
    
    # Calculate the number of pages
    num_pages = (len(filtered_df) + page_size - 1) // page_size
    page_num = st.number_input("Page", min_value=1, max_value=max(1, num_pages), value=1, step=1)
    
    # Display the current page
    start_idx = (page_num - 1) * page_size
    end_idx = min(start_idx + page_size, len(filtered_df))
    
    st.dataframe(filtered_df.iloc[start_idx:end_idx], use_container_width=True)
    st.write(f"Showing rows {start_idx+1} to {end_idx} of {len(filtered_df)}")

# ----------------- DOWNLOAD SECTION -----------------
st.markdown('<div class="section-header">💾 Export Data</div>', unsafe_allow_html=True)

# Option to download the filtered data
if not filtered_df.empty:
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="ml_memory_profile_filtered_data.csv",
        mime="text/csv",
    )

    # Option to download aggregated summary
    agg_columns = ['framework', 'model_size', 'batch_size', 'mode', 'device']
    summary_df = filtered_df.groupby(agg_columns)['memory_used_mb'].agg([
        ('mean', 'mean'),
        ('max', 'max'),
        ('min', 'min'),
        ('std', 'std'),
        ('median', 'median'),
        ('count', 'count')
    ]).reset_index()
    
    summary_csv = summary_df.to_csv(index=False)
    st.download_button(
        label="Download Summary Data as CSV",
        data=summary_csv,
        file_name="ml_memory_profile_summary.csv",
        mime="text/csv",
    )

# ----------------- FOOTER -----------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    ML Memory Profiler Dashboard • Created with Streamlit and Plotly
</div>
""", unsafe_allow_html=True)