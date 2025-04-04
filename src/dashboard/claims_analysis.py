"""
Claims Analysis module for the Insurance Underwriting Intelligence Dashboard.
Contains functions for generating claims analysis visualizations.
"""
import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
from theme import COLORS, PLOT_LAYOUT, styled_container, get_property_type_colors, merge_layout

def render_claims_by_value(property_data):
    """Render a bar chart showing claim rates by property value bins."""
    st.markdown('<div class="header-container"><h3 class="section-header">Claims by Property Value</h3></div>', unsafe_allow_html=True)
    
    def plot_claims_by_value():
        # Calculate claim rate by property value bin
        claims_by_value = property_data.groupby('value_bin')['claim_filed'].mean().reset_index()
        claims_by_value.columns = ['Property Value Range', 'Claim Rate']
        claims_by_value['Claim Rate'] = claims_by_value['Claim Rate'] * 100  # Convert to percentage
        
        # Create bar chart
        fig = px.bar(
            claims_by_value,
            x='Property Value Range',
            y='Claim Rate',
            color_discrete_sequence=[COLORS['secondary']],  # Use green instead of blue
            text_auto='.1f',
            labels={'Claim Rate': 'Claim Rate (%)'}
        )
        
        fig.update_traces(
            textposition='outside',
            texttemplate='%{y:.1f}%'
        )
        
        custom_layout = {
            'yaxis': dict(title="Claim Rate (%)", range=[0, max(claims_by_value['Claim Rate']) * 1.2]),
            'xaxis_title': "Property Value Range",
            'xaxis_tickangle': 45,
            'height': 400,
            'title': {"text": "Claim Rate by Property Value"}
        }
        fig.update_layout(**merge_layout(custom_layout))
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Use the styled container for consistent appearance
    styled_container(plot_claims_by_value)

def render_claims_by_flood_risk(property_data):
    """Render a bar chart showing claim rates by flood risk bins."""
    st.markdown('<div class="header-container"><h3 class="section-header">Claims by Flood Risk</h3></div>', unsafe_allow_html=True)
    
    def plot_claims_by_flood_risk():
        # Calculate claim rate by flood risk bin
        claims_by_flood = property_data.groupby('flood_risk_bin')['claim_filed'].mean().reset_index()
        claims_by_flood.columns = ['Flood Risk Level', 'Claim Rate']
        claims_by_flood['Claim Rate'] = claims_by_flood['Claim Rate'] * 100  # Convert to percentage
        
        # Create bar chart
        fig = px.bar(
            claims_by_flood,
            x='Flood Risk Level',
            y='Claim Rate',
            color_discrete_sequence=[COLORS['retail']],  # Use purple instead of blue
            text_auto='.1f',
            labels={'Claim Rate': 'Claim Rate (%)'}
        )
        
        fig.update_traces(
            textposition='outside',
            texttemplate='%{y:.1f}%'
        )
        
        custom_layout = {
            'yaxis': dict(title="Claim Rate (%)", range=[0, max(claims_by_flood['Claim Rate']) * 1.2]),
            'xaxis_title': "Flood Risk Level",
            'height': 400,
            'title': {"text": "Claim Rate by Flood Risk Level"}
        }
        fig.update_layout(**merge_layout(custom_layout))
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Use the styled container for consistent appearance
    styled_container(plot_claims_by_flood_risk)

def render_claims_by_fire_risk(property_data):
    """Render a bar chart showing claim rates by fire risk bins."""
    st.markdown('<div class="header-container"><h3 class="section-header">Claims by Fire Risk</h3></div>', unsafe_allow_html=True)
    
    def plot_claims_by_fire_risk():
        # Calculate claim rate by fire risk bin
        claims_by_fire = property_data.groupby('fire_risk_bin')['claim_filed'].mean().reset_index()
        claims_by_fire.columns = ['Fire Risk Level', 'Claim Rate']
        claims_by_fire['Claim Rate'] = claims_by_fire['Claim Rate'] * 100  # Convert to percentage
        
        # Create bar chart
        fig = px.bar(
            claims_by_fire,
            x='Fire Risk Level',
            y='Claim Rate',
            color_discrete_sequence=[COLORS['accent']],  # Use amber/orange instead of blue
            text_auto='.1f',
            labels={'Claim Rate': 'Claim Rate (%)'}
        )
        
        fig.update_traces(
            textposition='outside',
            texttemplate='%{y:.1f}%'
        )
        
        custom_layout = {
            'yaxis': dict(title="Claim Rate (%)", range=[0, max(claims_by_fire['Claim Rate']) * 1.2]),
            'xaxis_title': "Fire Risk Level",
            'height': 400,
            'title': {"text": "Claim Rate by Fire Risk Level"}
        }
        fig.update_layout(**merge_layout(custom_layout))
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Use the styled container for consistent appearance
    styled_container(plot_claims_by_fire_risk)

def render_claims_analysis_page(property_data):
    """Render the complete Claims Analysis page."""
    st.markdown('<h2 class="section-header">Claims Analysis</h2>', unsafe_allow_html=True)
    
    # Render claims by property value
    render_claims_by_value(property_data)
    
    # Render claims by flood risk
    render_claims_by_flood_risk(property_data)
    
    # Render claims by fire risk
    render_claims_by_fire_risk(property_data)
