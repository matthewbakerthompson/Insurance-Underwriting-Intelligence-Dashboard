"""
Comparative Analysis Tool module for the Insurance Underwriting Intelligence Dashboard.
Contains functions for comparing multiple properties side-by-side.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from theme import COLORS, PLOT_LAYOUT, styled_container, merge_layout, get_property_type_colors

def render_property_selector(property_data):
    """
    Render a multi-select widget for choosing properties to compare.
    
    Args:
        property_data (pd.DataFrame): Property data
        
    Returns:
        list: Selected property IDs
    """
    st.subheader("Select Properties to Compare")
    
    # Create a dataframe with key property information for selection
    selection_data = property_data[['property_id', 'property_type', 'property_value', 'location_risk']].copy()
    selection_data['property_value_formatted'] = selection_data['property_value'].apply(lambda x: f"${x:,.0f}")
    selection_data['risk_score'] = selection_data['location_risk'].apply(lambda x: f"{x:.1f}")
    
    # Display properties as a dataframe (without selection)
    st.dataframe(
        selection_data[['property_id', 'property_type', 'property_value_formatted', 'risk_score']].rename(
            columns={
                'property_id': 'Property ID',
                'property_type': 'Type',
                'property_value_formatted': 'Value',
                'risk_score': 'Risk Score'
            }
        ),
        use_container_width=True,
        height=200,
        column_config={
            "Property ID": st.column_config.TextColumn(width="medium"),
            "Type": st.column_config.TextColumn(width="medium"),
            "Value": st.column_config.TextColumn(width="medium"),
            "Risk Score": st.column_config.TextColumn(width="medium")
        },
        hide_index=True
    )
    
    # Use multiselect for property selection
    all_property_ids = selection_data['property_id'].tolist()
    
    # Get a default selection of different property types
    default_selection = [
        property_data[property_data['property_type'] == 'Residential']['property_id'].iloc[0],
        property_data[property_data['property_type'] == 'Commercial']['property_id'].iloc[0],
        property_data[property_data['property_type'] == 'Industrial']['property_id'].iloc[0]
    ]
    
    selected_property_ids = st.multiselect(
        "Select properties to compare",
        options=all_property_ids,
        default=default_selection
    )
    
    # If no properties selected, use default selection
    if not selected_property_ids:
        selected_property_ids = default_selection
    
    # Show selected properties
    st.write(f"Selected {len(selected_property_ids)} properties for comparison")
    
    return selected_property_ids

def render_property_comparison_table(property_data, selected_ids):
    """
    Render a table comparing selected properties.
    
    Args:
        property_data (pd.DataFrame): Property data
        selected_ids (list): List of selected property IDs
    """
    st.subheader("Property Comparison")
    
    # Filter data for selected properties
    selected_data = property_data[property_data['property_id'].isin(selected_ids)].copy()
    
    # Format values for display
    selected_data['property_value_formatted'] = selected_data['property_value'].apply(lambda x: f"${x:,.0f}")
    selected_data['property_size_formatted'] = selected_data['property_size'].apply(lambda x: f"{x:,.0f} sq ft")
    selected_data['flood_risk_formatted'] = selected_data['flood_risk_score'].apply(lambda x: f"{x:.1f}/10")
    selected_data['fire_risk_formatted'] = selected_data['fire_risk_score'].apply(lambda x: f"{x:.1f}/10")
    selected_data['location_risk_formatted'] = selected_data['location_risk'].apply(lambda x: f"{x:.1f}/10")
    selected_data['claim_status'] = selected_data['claim_filed'].apply(lambda x: "Yes" if x == 1 else "No")
    
    # Create comparison table
    comparison_table = selected_data[[
        'property_id', 'property_type', 'property_value_formatted', 'property_age', 
        'property_size_formatted', 'flood_risk_formatted', 'fire_risk_formatted', 
        'location_risk_formatted', 'claim_status'
    ]].rename(columns={
        'property_id': 'Property ID',
        'property_type': 'Type',
        'property_value_formatted': 'Value',
        'property_age': 'Age (years)',
        'property_size_formatted': 'Size',
        'flood_risk_formatted': 'Flood Risk',
        'fire_risk_formatted': 'Fire Risk',
        'location_risk_formatted': 'Overall Risk',
        'claim_status': 'Claim Filed'
    })
    
    # Display the comparison table
    st.dataframe(
        comparison_table,
        use_container_width=True,
        hide_index=True
    )
    
    # Add description
    st.markdown("""
    <p style='margin-top: 10px; margin-bottom: 20px;'>This table provides a side-by-side comparison of key characteristics for the selected properties. 
    Compare property values, ages, sizes, and risk scores to identify similarities and differences. 
    This comparison helps underwriters make consistent decisions across similar properties.</p>
    """, unsafe_allow_html=True)

def render_radar_chart(property_data, selected_ids):
    """
    Render a radar chart comparing selected properties across multiple dimensions.
    
    Args:
        property_data (pd.DataFrame): Property data
        selected_ids (list): List of selected property IDs
    """
    st.subheader("Multi-dimensional Comparison")
    
    # Filter data for selected properties
    selected_data = property_data[property_data['property_id'].isin(selected_ids)].copy()
    
    # Normalize values for radar chart (0-1 scale)
    # For property value, size, and age, higher values mean higher risk
    selected_data['value_normalized'] = selected_data['property_value'] / property_data['property_value'].max()
    selected_data['size_normalized'] = selected_data['property_size'] / property_data['property_size'].max()
    selected_data['age_normalized'] = selected_data['property_age'] / property_data['property_age'].max()
    
    # For risk scores, they're already on a 0-10 scale, so divide by 10
    selected_data['flood_risk_normalized'] = selected_data['flood_risk_score'] / 10
    selected_data['fire_risk_normalized'] = selected_data['fire_risk_score'] / 10
    selected_data['location_risk_normalized'] = selected_data['location_risk'] / 10
    
    # Create radar chart
    fig = go.Figure()
    
    # Define categories for radar chart
    categories = ['Property Value', 'Property Size', 'Property Age', 
                 'Flood Risk', 'Fire Risk', 'Overall Risk']
    
    # Add a trace for each property
    for i, row in selected_data.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['value_normalized'], row['size_normalized'], row['age_normalized'],
               row['flood_risk_normalized'], row['fire_risk_normalized'], row['location_risk_normalized']],
            theta=categories,
            fill='toself',
            name=row['property_id']
        ))
    
    # Update layout
    custom_layout = {
        'height': 500,
        'title': 'Multi-dimensional Risk Profile Comparison',
        'polar': {
            'radialaxis': {
                'visible': True,
                'range': [0, 1]
            }
        },
        'showlegend': True,
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
    
    # Add description
    st.markdown("""
    <p style='margin-top: 10px; margin-bottom: 20px;'>This radar chart visualizes multiple risk dimensions simultaneously, allowing for quick identification of each property's risk profile. 
    Properties with larger radar areas generally represent higher overall risk. 
    The shape of each property's radar plot reveals its specific risk pattern - some may have high flood risk but low fire risk, while others may have balanced risk across all dimensions.</p>
    """, unsafe_allow_html=True)

def render_risk_comparison_chart(property_data, selected_ids):
    """
    Render a bar chart comparing risk scores for selected properties.
    
    Args:
        property_data (pd.DataFrame): Property data
        selected_ids (list): List of selected property IDs
    """
    st.subheader("Risk Score Comparison")
    
    # Filter data for selected properties
    selected_data = property_data[property_data['property_id'].isin(selected_ids)].copy()
    
    # Create a long-format dataframe for the risk scores
    risk_data = []
    
    for i, row in selected_data.iterrows():
        risk_data.append({
            'property_id': row['property_id'],
            'property_type': row['property_type'],
            'risk_type': 'Flood Risk',
            'risk_score': row['flood_risk_score']
        })
        risk_data.append({
            'property_id': row['property_id'],
            'property_type': row['property_type'],
            'risk_type': 'Fire Risk',
            'risk_score': row['fire_risk_score']
        })
        risk_data.append({
            'property_id': row['property_id'],
            'property_type': row['property_type'],
            'risk_type': 'Overall Risk',
            'risk_score': row['location_risk']
        })
    
    risk_df = pd.DataFrame(risk_data)
    
    # Create grouped bar chart
    fig = px.bar(
        risk_df,
        x='property_id',
        y='risk_score',
        color='risk_type',
        barmode='group',
        color_discrete_map={
            'Flood Risk': COLORS['healthcare'],  # Using healthcare instead of non-existent 'blue'
            'Fire Risk': COLORS['finance'],  # Using finance instead of non-existent 'orange'
            'Overall Risk': COLORS['primary']
        },
        hover_data=['property_type']
    )
    
    # Update layout
    custom_layout = {
        'height': 400,
        'xaxis_title': 'Property ID',
        'yaxis_title': 'Risk Score (0-10)',
        'yaxis_range': [0, 10],
        'title': 'Risk Comparison Across Properties',
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
    
    # Add description
    st.markdown("""
    <p style='margin-top: 10px; margin-bottom: 20px;'>This chart compares flood risk, fire risk, and overall risk scores across the selected properties. 
    The grouped bars make it easy to identify which properties have higher risk in specific categories. 
    This visualization helps underwriters identify outliers and ensure consistent risk assessment across similar properties.</p>
    """, unsafe_allow_html=True)

def render_comparative_analysis_page(property_data):
    """Render the Comparative Analysis page with interactive property comparison."""
    st.markdown('<h2 class="section-header">Comparative Analysis Tool</h2>', unsafe_allow_html=True)
    
    # Display page description
    st.markdown("""
    <p style='margin-bottom: 20px;'>This tool allows you to compare multiple properties side-by-side across various dimensions, 
    helping identify similarities, differences, and ensure consistent underwriting decisions.</p>
    """, unsafe_allow_html=True)
    
    # Render property selector
    selected_property_ids = render_property_selector(property_data)
    
    # Only proceed if properties are selected
    if selected_property_ids:
        # Render property comparison table
        render_property_comparison_table(property_data, selected_property_ids)
        
        # Create columns for the next two visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Render radar chart
            render_radar_chart(property_data, selected_property_ids)
        
        with col2:
            # Render risk comparison chart
            render_risk_comparison_chart(property_data, selected_property_ids)
    else:
        st.warning("Please select at least one property to compare.")
