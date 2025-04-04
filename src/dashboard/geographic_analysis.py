"""
Geographic Risk Visualization module for the Insurance Underwriting Intelligence Dashboard.
Contains functions for generating geographic risk visualizations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from theme import COLORS, PLOT_LAYOUT, styled_container, merge_layout

def generate_geographic_data(property_data):
    """
    Generate geographic data for risk visualization.
    
    Args:
        property_data (pd.DataFrame): Property data
        
    Returns:
        pd.DataFrame: Geographic data with locations and risk metrics
    """
    # US states
    states = [
        'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 
        'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 
        'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 
        'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 
        'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 
        'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 
        'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 
        'Wisconsin', 'Wyoming'
    ]
    
    # State abbreviations for mapping
    state_abbr = {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
        'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
        'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
        'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
        'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
        'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
        'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
        'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
        'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
        'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
    }
    
    # Create base geographic data
    np.random.seed(42)  # For reproducibility
    geo_data = pd.DataFrame({
        'state': states,
        'state_code': [state_abbr[state] for state in states]
    })
    
    # Generate property counts with realistic distribution
    # More properties in populous states
    population_factor = np.array([
        9, 1, 13, 6, 74, 11, 7, 2, 42, 21, 3, 3, 24, 13, 6,
        6, 9, 9, 3, 12, 14, 20, 11, 6, 12, 2, 4, 6, 3,
        17, 4, 38, 21, 2, 23, 8, 8, 25, 2, 10, 2,
        14, 57, 6, 1, 17, 15, 3, 12, 1
    ])
    
    # Normalize to get distribution
    population_factor = population_factor / population_factor.sum()
    
    # Generate property counts
    total_properties = len(property_data)
    # Convert to int explicitly to avoid dtype issues
    geo_data['property_count'] = np.random.multinomial(total_properties, population_factor).astype(int)
    
    # Generate risk scores with regional patterns
    # Base risk for each state (higher in coastal and disaster-prone areas)
    base_risk = {
        'FL': 7.2, 'LA': 7.0, 'TX': 6.8, 'CA': 6.5, 'MS': 6.3, 'AL': 6.2, 'NC': 6.0, 'SC': 5.9,
        'OK': 5.8, 'KS': 5.7, 'MO': 5.6, 'AR': 5.5, 'GA': 5.4, 'TN': 5.3, 'KY': 5.2,
        'VA': 5.1, 'WV': 5.0, 'MD': 4.9, 'DE': 4.8, 'NJ': 4.8, 'NY': 4.7, 'CT': 4.7,
        'RI': 4.7, 'MA': 4.6, 'NH': 4.5, 'VT': 4.4, 'ME': 4.4, 'PA': 4.5, 'OH': 4.6,
        'MI': 4.5, 'IN': 4.7, 'IL': 4.8, 'WI': 4.4, 'MN': 4.3, 'IA': 4.6, 'NE': 4.7,
        'SD': 4.5, 'ND': 4.4, 'MT': 4.6, 'WY': 4.7, 'CO': 4.9, 'NM': 5.0, 'AZ': 5.2,
        'UT': 4.8, 'ID': 4.7, 'NV': 5.0, 'OR': 4.8, 'WA': 4.9, 'AK': 5.3, 'HI': 5.5
    }
    
    # Add some random variation
    geo_data['avg_risk_score'] = [base_risk[code] + np.random.normal(0, 0.3) for code in geo_data['state_code']]
    geo_data['avg_risk_score'] = geo_data['avg_risk_score'].clip(0, 10)
    
    # Calculate flood risk (higher in coastal states)
    coastal_states = ['FL', 'LA', 'TX', 'MS', 'AL', 'GA', 'SC', 'NC', 'VA', 'MD', 'DE', 'NJ', 
                      'NY', 'CT', 'RI', 'MA', 'NH', 'ME', 'CA', 'OR', 'WA', 'AK', 'HI']
    
    geo_data['flood_risk'] = geo_data.apply(
        lambda row: base_risk[row['state_code']] * 1.2 + np.random.normal(0, 0.3) 
        if row['state_code'] in coastal_states else base_risk[row['state_code']] * 0.8 + np.random.normal(0, 0.3),
        axis=1
    )
    geo_data['flood_risk'] = geo_data['flood_risk'].clip(0, 10)
    
    # Calculate fire risk (higher in western states)
    fire_prone_states = ['CA', 'OR', 'WA', 'ID', 'MT', 'WY', 'NV', 'UT', 'AZ', 'NM', 'CO', 'OK', 'TX']
    
    geo_data['fire_risk'] = geo_data.apply(
        lambda row: base_risk[row['state_code']] * 1.3 + np.random.normal(0, 0.3) 
        if row['state_code'] in fire_prone_states else base_risk[row['state_code']] * 0.7 + np.random.normal(0, 0.3),
        axis=1
    )
    geo_data['fire_risk'] = geo_data['fire_risk'].clip(0, 10)
    
    # Calculate claim rates based on risk scores
    geo_data['claim_rate'] = 0.05 + 0.01 * geo_data['avg_risk_score'] + np.random.normal(0, 0.01, len(geo_data))
    geo_data['claim_rate'] = geo_data['claim_rate'].clip(0, 1)
    
    return geo_data

def render_risk_map(geo_data, risk_type='avg_risk_score'):
    """
    Render a choropleth map of risk scores by state.
    
    Args:
        geo_data (pd.DataFrame): Geographic data
        risk_type (str): Type of risk to display ('avg_risk_score', 'flood_risk', or 'fire_risk')
    """
    # Set title based on risk type
    if risk_type == 'avg_risk_score':
        title = 'Overall Risk Score by State'
        color_scale = 'Reds'
    elif risk_type == 'flood_risk':
        title = 'Flood Risk Score by State'
        color_scale = 'Blues'
    elif risk_type == 'fire_risk':
        title = 'Fire Risk Score by State'
        color_scale = 'Oranges'
    else:
        title = 'Risk Score by State'
        color_scale = 'Reds'
    
    # Create choropleth map
    fig = px.choropleth(
        geo_data,
        locations='state_code',
        color=risk_type,
        hover_name='state',
        locationmode='USA-states',
        color_continuous_scale=color_scale,
        range_color=[3, 8],
        scope="usa",
        labels={
            'avg_risk_score': 'Risk Score',
            'flood_risk': 'Flood Risk',
            'fire_risk': 'Fire Risk'
        },
        hover_data={
            'state_code': False,
            'property_count': True,
            'claim_rate': ':.1%'
        }
    )
    
    # Update layout
    custom_layout = {
        'height': 500,
        'title': {'text': title, 'font': {'size': 20, 'color': '#0F172A'}},
        'margin': {"r": 0, "t": 40, "l": 0, "b": 0},
        'coloraxis_colorbar': {
            'title': 'Risk Score',
            'tickvals': [3, 4, 5, 6, 7, 8],
            'ticktext': ['3.0', '4.0', '5.0', '6.0', '7.0', '8.0']
        }
    }
    
    fig.update_layout(**merge_layout(custom_layout))
    st.plotly_chart(fig, use_container_width=True)

def render_claim_rate_map(geo_data):
    """
    Render a choropleth map of claim rates by state.
    
    Args:
        geo_data (pd.DataFrame): Geographic data
    """
    # Create choropleth map
    fig = px.choropleth(
        geo_data,
        locations='state_code',
        color='claim_rate',
        hover_name='state',
        locationmode='USA-states',
        color_continuous_scale='Reds',
        range_color=[0.05, 0.15],
        scope="usa",
        labels={'claim_rate': 'Claim Rate'},
        hover_data={
            'state_code': False,
            'property_count': True,
            'avg_risk_score': ':.1f'
        }
    )
    
    # Update layout
    custom_layout = {
        'height': 500,
        'title': {'text': 'Claim Rate by State', 'font': {'size': 20, 'color': '#0F172A'}},
        'margin': {"r": 0, "t": 40, "l": 0, "b": 0},
        'coloraxis_colorbar': {
            'title': 'Claim Rate',
            'tickvals': [0.05, 0.07, 0.09, 0.11, 0.13, 0.15],
            'ticktext': ['5%', '7%', '9%', '11%', '13%', '15%']
        }
    }
    
    fig.update_layout(**merge_layout(custom_layout))
    st.plotly_chart(fig, use_container_width=True)

def render_risk_comparison_chart(geo_data):
    """
    Render a bar chart comparing different risk types for top states.
    
    Args:
        geo_data (pd.DataFrame): Geographic data
    """
    # Get top 10 states by overall risk
    top_states = geo_data.sort_values('avg_risk_score', ascending=False).head(10)
    
    # Create a grouped bar chart
    fig = go.Figure()
    
    # Add bars for each risk type
    fig.add_trace(go.Bar(
        x=top_states['state_code'],
        y=top_states['avg_risk_score'],
        name='Overall Risk',
        marker_color=COLORS['primary']
    ))
    
    fig.add_trace(go.Bar(
        x=top_states['state_code'],
        y=top_states['flood_risk'],
        name='Flood Risk',
        marker_color=COLORS['healthcare']  # Using healthcare blue color instead of non-existent 'blue'
    ))
    
    fig.add_trace(go.Bar(
        x=top_states['state_code'],
        y=top_states['fire_risk'],
        name='Fire Risk',
        marker_color=COLORS['finance']  # Using finance orange color instead of non-existent 'orange'
    ))
    
    # Update layout
    custom_layout = {
        'height': 400,
        'title': {'text': 'Risk Comparison for Highest-Risk States', 'font': {'size': 20, 'color': '#0F172A'}},
        'xaxis_title': 'State',
        'yaxis_title': 'Risk Score (0-10)',
        'yaxis_range': [0, 10],
        'barmode': 'group',
        'legend': dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    }
    
    fig.update_layout(**merge_layout(custom_layout))
    st.plotly_chart(fig, use_container_width=True)

def render_geographic_analysis_page(property_data):
    """Render the Geographic Analysis page with interactive visualizations."""
    st.markdown('<h2 class="section-header">Geographic Risk Visualization</h2>', unsafe_allow_html=True)
    
    # Generate geographic data
    geo_data = generate_geographic_data(property_data)
    
    # Display page description
    st.markdown("""
    <p style='margin-bottom: 20px;'>This page provides geographic analysis of insurance risks across the United States, 
    helping underwriters identify high-risk regions and understand regional risk patterns.</p>
    """, unsafe_allow_html=True)
    
    # Create risk type selector
    risk_type = st.selectbox(
        "Select Risk Type to Visualize",
        options=[
            "Overall Risk Score", 
            "Flood Risk Score", 
            "Fire Risk Score",
            "Claim Rate"
        ],
        index=0
    )
    
    # Display the appropriate map based on selection
    if risk_type == "Overall Risk Score":
        render_risk_map(geo_data, 'avg_risk_score')
        
        # Add description
        st.markdown("""
        <p style='margin-top: 10px; margin-bottom: 20px;'>This map shows the overall risk score distribution across states.
        Coastal states and areas prone to natural disasters typically show higher risk scores.
        Florida, Louisiana, and Texas have the highest overall risk scores due to their exposure to multiple perils.</p>
        """, unsafe_allow_html=True)
        
    elif risk_type == "Flood Risk Score":
        render_risk_map(geo_data, 'flood_risk')
        
        # Add description
        st.markdown("""
        <p style='margin-top: 10px; margin-bottom: 20px;'>This map highlights flood risk distribution across states.
        Coastal states show significantly higher flood risk scores, with the Gulf Coast and Atlantic seaboard being particularly vulnerable.
        States like Florida, Louisiana, and Mississippi face the highest flood risks due to their low elevation and hurricane exposure.</p>
        """, unsafe_allow_html=True)
        
    elif risk_type == "Fire Risk Score":
        render_risk_map(geo_data, 'fire_risk')
        
        # Add description
        st.markdown("""
        <p style='margin-top: 10px; margin-bottom: 20px;'>This map displays fire risk distribution across states.
        Western states show significantly higher fire risk scores due to their drier climate and vegetation conditions.
        California, Oregon, and Arizona face the highest fire risks, exacerbated by drought conditions and climate change.</p>
        """, unsafe_allow_html=True)
        
    elif risk_type == "Claim Rate":
        render_claim_rate_map(geo_data)
        
        # Add description
        st.markdown("""
        <p style='margin-top: 10px; margin-bottom: 20px;'>This map shows claim rates across states, which closely correlate with risk scores.
        States with higher risk scores typically experience higher claim rates, though the relationship isn't perfectly linear.
        Factors like local building codes, infrastructure quality, and emergency response capabilities can influence claim rates beyond raw risk scores.</p>
        """, unsafe_allow_html=True)
    
    # Display risk comparison chart
    render_risk_comparison_chart(geo_data)
    
    # Add description for comparison chart
    st.markdown("""
    <p style='margin-top: 10px; margin-bottom: 20px;'>This chart compares different risk types for the highest-risk states.
    Some states face primarily flood risks (e.g., Florida, Louisiana), while others face primarily fire risks (e.g., California, Arizona).
    Understanding the specific risk profile of each region helps tailor underwriting guidelines and pricing strategies accordingly.</p>
    """, unsafe_allow_html=True)
