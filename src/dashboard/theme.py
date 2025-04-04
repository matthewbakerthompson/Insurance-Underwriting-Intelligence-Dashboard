"""
Theme module for the Insurance Underwriting Intelligence Dashboard.
Contains color definitions, plot layouts, and CSS styling.
"""
import streamlit as st

# Global color definitions for consistent theming
COLORS = {
    'primary': '#1E40AF',      # Darker Blue for better contrast
    'secondary': '#047857',    # Darker Green for better contrast
    'accent': '#B45309',       # Darker Amber for better contrast
    'text': '#0F172A',         # Dark slate
    'background': '#F8FAFC',   # Light slate background
    'light_gray': '#CBD5E1',   # Slightly darker light gray for better contrast
    'medium_gray': '#64748B',  # Darker medium gray for better contrast
    'dark_gray': '#334155',    # Dark gray for secondary text
    
    # Property type colors
    'residential': '#1E40AF',  # Darker Blue
    'commercial': '#047857',   # Darker Green
    'industrial': '#B45309',   # Darker Amber
    
    # Industry colors
    'retail': '#7C3AED',       # Darker Purple
    'manufacturing': '#DB2777', # Darker Pink
    'healthcare': '#0891B2',   # Darker Cyan
    'technology': '#0F766E',   # Darker Teal
    'finance': '#EA580C',      # Darker Orange
    'construction': '#C2410C', # Darker Orange variant
    'food_service': '#0D9488', # Darker Teal variant
    'transportation': '#4F46E5', # Darker Indigo
}

# Define consistent plot layout for all visualizations
PLOT_LAYOUT = {
    'template': 'plotly_white',
    'plot_bgcolor': '#FFFFFF',       # White background for better contrast
    'paper_bgcolor': '#FFFFFF',      # White paper background
    'font': {'family': 'Segoe UI, Arial, sans-serif', 'size': 14, 'color': COLORS['text']},  # Larger font size
    'margin': {'t': 50, 'b': 50, 'l': 50, 'r': 50},
    'xaxis': {
        'gridcolor': COLORS['light_gray'], 
        'zerolinecolor': COLORS['dark_gray'],
        'title': {'font': {'color': COLORS['text'], 'size': 16, 'weight': 'bold'}},  # Bolder, larger titles
        'tickfont': {'color': COLORS['text'], 'size': 14}  # Ensure tick labels are visible
    },
    'yaxis': {
        'gridcolor': COLORS['light_gray'], 
        'zerolinecolor': COLORS['dark_gray'],
        'title': {'font': {'color': COLORS['text'], 'size': 16, 'weight': 'bold'}},  # Bolder, larger titles
        'tickfont': {'color': COLORS['text'], 'size': 14}  # Ensure tick labels are visible
    },
    'legend': {
        'bgcolor': 'rgba(255,255,255,1.0)', 
        'bordercolor': COLORS['dark_gray'], 
        'font': {'color': COLORS['text'], 'size': 14}
    },
    'title': {'font': {'color': COLORS['text'], 'size': 18, 'weight': 'bold'}}
}

def apply_theme():
    """Apply the custom CSS styling to the Streamlit app."""
    st.markdown("""
    <style>
        /* Global styling */
        .stApp {
            background-color: #F8FAFC;
            font-family: 'Segoe UI', Arial, sans-serif;
            color: #0F172A;
        }
        
        /* Fix top bar styling */
        header[data-testid="stHeader"] {
            background-color: #1E40AF !important;
        }
        
        /* Fix hamburger menu icon */
        button[kind="header"] {
            color: white !important;
        }
        
        /* Improve text contrast */
        p, span, div, label, .stMarkdown {
            color: #0F172A !important;
            font-size: 16px !important;
        }
        
        .main .block-container {
            padding: 2rem;
            max-width: 90%;
        }
        
        /* Header styling */
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E3A8A;
            border-bottom: 2px solid #1E40AF;
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
        }
        
        .sub-header {
            font-size: 1.1rem;
            color: #475569;
            margin-top: 0;
            margin-bottom: 1.5rem;
        }
        
        /* Section headers */
        .section-header {
            font-size: 1.6rem;
            color: #0F172A;
            font-weight: 700;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            padding: 0.5rem 0;
            border-bottom: 2px solid #1E40AF;
            background-color: transparent !important;
            display: block;
        }
        
        /* Fix section header containers */
        h3.section-header {
            background-color: transparent !important;
            color: #0F172A !important;
        }
        
        /* Custom header container to avoid Streamlit's default styling */
        .header-container {
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin-bottom: 1rem !important;
            display: block !important;
        }
        
        /* Remove Streamlit's default container styling around headers */
        [data-testid="stVerticalBlock"] > div:has(.section-header),
        .stMarkdown:has(.section-header),
        .css-1fcdlhc:has(.section-header),
        .css-1aehpvj:has(.section-header),
        .css-16idsys:has(.section-header),
        .css-10oheav:has(.section-header),
        .css-ue6h4q:has(.section-header),
        .css-1544g2n:has(.section-header) {
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        
        /* Remove any container styling that might create boxes */
        .element-container:has(.section-header),
        div:has(> .section-header),
        .block-container div:has(.section-header) {
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* Fix for the yellow highlighted boxes */
        .stExpander, .streamlit-expanderHeader, .streamlit-expanderContent {
            border: none !important;
            box-shadow: none !important;
            background-color: transparent !important;
        }
        
        /* Card styling for metrics and content sections */
        .metric-card {
            background-color: #FFFFFF;
            border-radius: 0.5rem;
            padding: 1.2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
            border: 1px solid #94A3B8;
        }
        
        .metric-label {
            font-size: 1rem;
            font-weight: 600;
            color: #1E293B;
            margin-bottom: 0.3rem;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: #0F172A;
        }
        
        /* Improve sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #1E40AF;
            border-right: 1px solid #1E3A8A;
        }
        
        /* Style the sidebar text */
        [data-testid="stSidebar"] [data-testid="stMarkdown"] p {
            color: white !important;
            font-size: 16px !important;
            font-weight: 500 !important;
        }
        
        /* Fix navigation styling - ensure ALL text in sidebar is white */
        [data-testid="stSidebar"] * {
            color: white !important;
        }
        
        /* Navigation styling - specific selectors to override default styles */
        .css-16idsys p, .css-16idsys span, [data-testid="stSidebar"] .css-pkbazv,
        [data-testid="stSidebar"] span, [data-testid="stSidebar"] div,
        [data-testid="stSidebar"] a, [data-testid="stSidebar"] button {
            color: white !important;
            font-weight: 600 !important;
            font-size: 16px !important;
        }
        
        /* Make sidebar navigation items more visible */
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            padding: 0.5rem 0;
        }
        
        /* Add hover effect to sidebar items */
        [data-testid="stSidebar"] .element-container:hover {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
        }
        
        /* Add branding */
        .branding {
            position: fixed;
            bottom: 20px;
            left: 20px;
            font-size: 1rem;
            color: white;
            opacity: 1.0;
            font-weight: 500;
        }
        
        /* Content container styling */
        .content-container {
            background-color: #FFFFFF;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 1.5rem;
            border: 1px solid #E2E8F0;
            display: block !important;
            width: 100% !important;
        }
        
        /* Chart container styling - target Streamlit elements directly */
        [data-testid="stPlotlyChart"], .stPlotlyChart, .element-container:has(.stPlotlyChart) {
            background-color: #FFFFFF !important;
            padding: 1rem !important;
            border-radius: 0.5rem !important;
            border: 1px solid #E2E8F0 !important;
            margin-bottom: 1rem !important;
            width: 100% !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
        }
        
        /* Fix chart container headers */
        .stHeading, .css-10oheav, .css-ue6h4q, .css-1544g2n, .css-1fcdlhc, .css-1aehpvj, .css-16idsys {
            color: #0F172A !important;
            background-color: transparent !important;
            padding: 0.5rem 0 !important;
            font-weight: 600 !important;
        }
        
        /* Ensure chart titles and legend text are visible */
        .gtitle .plotly, .gtitle, .js-plotly-plot .plotly .gtitle,
        .legendtext, .legendtitletext, .g-legend-title, .legend text {
            fill: #0F172A !important;
            color: #0F172A !important;
            font-family: 'Segoe UI', Arial, sans-serif !important;
            font-weight: 600 !important;
        }
        
        /* Make sure all chart text is visible */
        .xtick text, .ytick text {
            fill: #334155 !important;
        }
        
        /* BALANCED CONTAINER STYLING */
        /* Style containers for charts and visualizations */
        .stPlotlyChart, .element-container:has(.stPlotlyChart) {
            background-color: white !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 0.5rem !important;
            padding: 1rem !important;
            margin-bottom: 1rem !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
        }
        
        /* Style expanders consistently */
        div[data-testid="stExpander"] {
            border: 1px solid #E2E8F0 !important;
            border-radius: 0.5rem !important;
            margin-bottom: 1rem !important;
            background-color: white !important;
        }
        
        /* Fix for section headers to ensure they're not in containers */
        div:has(> h2), div:has(> h3), div:has(> h4),
        div:has(> .section-header), div:has(> .header-container) {
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
        }
        
        /* Style specific elements consistently */
        .stDataFrame {
            background-color: white !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 0.5rem !important;
            padding: 1rem !important;
            margin-bottom: 1rem !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
        }
        
        /* Style buttons consistently */
        .stButton > button {
            background-color: #1E40AF !important;
            color: white !important;
            border: none !important;
            border-radius: 0.25rem !important;
            padding: 0.5rem 1rem !important;
            font-weight: 600 !important;
        }
        
        /* Improve button contrast on hover */
        .stButton > button:hover {
            background-color: #2563EB !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        }
        
        /* Style the slider */
        .stSlider > div > div > div > div {
            background-color: #1E40AF !important;
        }
        
        /* Ensure chart titles and legend text are visible */
        .gtitle .plotly, .gtitle, .js-plotly-plot .plotly .gtitle,
        .legendtext, .legendtitletext, .g-legend-title, .legend text {
            fill: #0F172A !important;
            font-size: 16px !important;
            font-weight: 600 !important;
            color: #0F172A !important;
            font-family: 'Segoe UI', Arial, sans-serif !important;
            opacity: 1 !important;
        }
        
        /* Ensure all text in the content container has good contrast */
        .content-container p, .content-container span, .content-container div {
            color: #0F172A !important;
        }
        
        /* Improve contrast for Streamlit elements */
        .stSelectbox label, .stSlider label, .stNumberInput label {
            color: #0F172A !important;
            font-weight: 500 !important;
            font-size: 16px !important;
        }
        
        /* Fix dropdown text color - more specific selectors */
        .stSelectbox div[data-baseweb="select"] div, 
        .stSelectbox [data-testid="stWidgetLabel"],
        select, option, .stSelectbox span,
        div[role="combobox"] span,
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div,
        div[data-baseweb="base-input"] div,
        div[data-baseweb="popover"] div,
        div[data-baseweb="select"] input,
        div[data-baseweb="select"] svg {
            color: #0F172A !important;
        }
        
        /* Force background color for select boxes */
        div[data-baseweb="select"],
        div[data-baseweb="select"] > div,
        div[data-baseweb="select"] > div > div,
        div[data-baseweb="select"] > div > div > div,
        div[role="combobox"],
        div[role="combobox"] > div {
            background-color: white !important;
        }
        
        /* Ensure dropdown options are visible */
        [role="listbox"] li, 
        [role="option"],
        div[role="option"],
        ul[role="listbox"] li {
            color: #0F172A !important;
            background-color: white !important;
        }
        
        /* Style dropdown on hover */
        [role="listbox"] li:hover, 
        [role="option"]:hover,
        div[role="option"]:hover {
            background-color: #EFF6FF !important;
        }
        
        /* Ensure plot labels have good contrast */
        .js-plotly-plot .plotly .gtitle, .js-plotly-plot .plotly .xtitle, .js-plotly-plot .plotly .ytitle {
            fill: #0F172A !important;
            font-weight: 600 !important;
        }
    </style>
    
    <div class="branding">
        <p>Matthew Thompson<br>Insurance Underwriting Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

def merge_layout(custom_layout=None):
    """Merge the default plot layout with custom layout settings.
    This prevents duplicate parameters by prioritizing custom settings."""
    if custom_layout is None:
        return PLOT_LAYOUT
    
    # Start with a copy of the default layout
    merged = PLOT_LAYOUT.copy()
    
    # Handle nested dictionaries like xaxis, yaxis, legend, etc.
    for key, value in custom_layout.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # For nested dictionaries, merge them recursively
            merged_nested = merged[key].copy()
            merged_nested.update(value)
            merged[key] = merged_nested
        else:
            # For simple values or when the key doesn't exist in the default layout,
            # just use the custom value
            merged[key] = value
    
    return merged

def styled_container(content_function):
    """Create a styled container and execute the provided function inside it."""
    # This is a direct wrapper that doesn't add any HTML
    # The chart styling is handled by CSS targeting the Streamlit elements directly
    return content_function()

def get_property_type_colors():
    """Return a dictionary mapping property types to their colors."""
    return {
        'Residential': COLORS['residential'],
        'Commercial': COLORS['secondary'],
        'Industrial': COLORS['accent']
    }

def get_industry_colors():
    """Return a dictionary mapping industries to their colors."""
    return {
        'Retail': COLORS['retail'],
        'Manufacturing': COLORS['manufacturing'],
        'Healthcare': COLORS['healthcare'],
        'Technology': COLORS['technology'],
        'Finance': COLORS['finance'],
        'Construction': COLORS['construction'],
        'Food Service': COLORS['food_service'],
        'Transportation': COLORS['transportation']
    }
