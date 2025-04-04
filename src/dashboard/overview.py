"""
Overview module for the Insurance Underwriting Intelligence Dashboard.
Contains functions for generating overview visualizations.
"""
import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
from theme import COLORS, PLOT_LAYOUT, styled_container, get_property_type_colors, get_industry_colors, merge_layout

def render_metric_cards(property_data, business_data):
    """Render key metric cards at the top of the overview page."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <p class="metric-label">TOTAL PROPERTIES</p>
            <p class="metric-value">{len(property_data):,}</p>
        </div>
        ''', unsafe_allow_html=True)
        
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <p class="metric-label">TOTAL BUSINESSES</p>
            <p class="metric-value">{len(business_data):,}</p>
        </div>
        ''', unsafe_allow_html=True)
        
    with col3:
        st.markdown(f'''
        <div class="metric-card">
            <p class="metric-label">AVG PROPERTY VALUE</p>
            <p class="metric-value">${property_data['property_value'].mean():,.0f}</p>
        </div>
        ''', unsafe_allow_html=True)
        
    with col4:
        st.markdown(f'''
        <div class="metric-card">
            <p class="metric-label">CLAIM RATE</p>
            <p class="metric-value">{property_data['claim_filed'].mean():.1%}</p>
        </div>
        ''', unsafe_allow_html=True)

def render_property_distribution(property_data):
    """Render the property type distribution chart."""
    st.markdown('<div class="header-container"><h3 class="section-header">Property Type Distribution</h3></div>', unsafe_allow_html=True)
    
    def plot_property_distribution():
        # Prepare data
        property_type_counts = property_data['property_type'].value_counts().reset_index()
        property_type_counts.columns = ['Property Type', 'Count']
        
        # Get property type colors
        property_colors = list(get_property_type_colors().values())
        
        # Create pie chart
        fig = px.pie(
            property_type_counts, 
            names='Property Type', 
            values='Count',
            color='Property Type',
            color_discrete_sequence=property_colors,
            hole=0.4,
        )
        
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            textfont=dict(size=14, color='white')
        )
        
        custom_layout = {
            'showlegend': True,
            'legend': dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            'margin': dict(t=30, b=50, l=0, r=0),
            'height': 350,
            'title': {"text": "Property Type Distribution"}
        }
        fig.update_layout(**merge_layout(custom_layout))
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Use the styled container for consistent appearance
    styled_container(plot_property_distribution)

def render_industry_distribution(business_data):
    """Render the business industry distribution chart."""
    st.markdown('<div class="header-container"><h3 class="section-header">Business Industry Distribution</h3></div>', unsafe_allow_html=True)
    
    def plot_industry_distribution():
        # Prepare data
        industry_counts = business_data['industry'].value_counts().reset_index()
        industry_counts.columns = ['Industry', 'Count']
        
        # Sort by count
        industry_counts = industry_counts.sort_values('Count', ascending=False)
        
        # Get industry colors
        industry_colors = get_industry_colors()
        industry_color_list = [industry_colors.get(industry, COLORS['primary']) 
                              for industry in industry_counts['Industry']]
        
        # Create bar chart
        fig = px.bar(
            industry_counts,
            x='Industry',
            y='Count',
            color='Industry',
            color_discrete_sequence=industry_color_list,
            text='Count'
        )
        
        fig.update_traces(
            textposition='outside',
            textfont=dict(size=12)
        )
        
        custom_layout = {
            'showlegend': False,
            'xaxis': dict(title=None, tickangle=45),
            'yaxis': dict(title="Number of Businesses"),
            'margin': dict(t=30, b=80, l=0, r=0),
            'height': 350,
            'title': {"text": "Business Industry Distribution"}
        }
        fig.update_layout(**merge_layout(custom_layout))
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Use the styled container for consistent appearance
    styled_container(plot_industry_distribution)

def render_risk_score_distribution(property_data):
    """Render the risk score distribution chart."""
    st.markdown('<div class="header-container"><h3 class="section-header">Risk Score Distribution</h3></div>', unsafe_allow_html=True)
    
    def plot_risk_distribution():
        # Create histogram of risk scores
        fig = px.histogram(
            property_data,
            x='location_risk',
            color='property_type',
            color_discrete_map=get_property_type_colors(),
            nbins=20,
            opacity=0.8,
            barmode='overlay',
            histnorm='percent',
            labels={'location_risk': 'Location Risk Score', 'property_type': 'Property Type'}
        )
        
        # Explicitly set legend text color and font
        fig.update_traces(showlegend=True)
        
        custom_layout = {
            'xaxis_title': "Location Risk Score",
            'yaxis_title': "Percentage of Properties",
            'legend_title': "Property Type",
            'height': 350,
            'title': {"text": "Risk Score Distribution"},
            'legend': {
                'font': {'color': '#0F172A', 'size': 12, 'family': 'Arial, sans-serif'},
                'title': {'font': {'color': '#0F172A', 'size': 14, 'family': 'Arial, sans-serif'}},
                'bgcolor': 'rgba(255, 255, 255, 0.8)',
                'bordercolor': '#E2E8F0',
                'borderwidth': 1
            }
        }
        fig.update_layout(**merge_layout(custom_layout))
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Use the styled container for consistent appearance
    styled_container(plot_risk_distribution)

def render_claim_rate_by_property(property_data):
    """Render the claim rate by property type chart."""
    st.markdown('<div class="header-container"><h3 class="section-header">Claim Rate by Property Type</h3></div>', unsafe_allow_html=True)
    
    def plot_claim_rate():
        # Calculate claim rate by property type
        claim_by_type = property_data.groupby('property_type')['claim_filed'].mean().reset_index()
        claim_by_type.columns = ['Property Type', 'Claim Rate']
        claim_by_type['Claim Rate'] = claim_by_type['Claim Rate'] * 100  # Convert to percentage
        
        # Create bar chart
        fig = px.bar(
            claim_by_type,
            x='Property Type',
            y='Claim Rate',
            color='Property Type',
            color_discrete_map=get_property_type_colors(),
            text_auto='.1f',
            labels={'Claim Rate': 'Claim Rate (%)'}
        )
        
        fig.update_traces(
            textposition='outside',
            texttemplate='%{y:.1f}%'
        )
        
        custom_layout = {
            'yaxis': dict(title="Claim Rate (%)", range=[0, max(claim_by_type['Claim Rate']) * 1.2]),
            'xaxis_title': None,
            'height': 350,
            'showlegend': False,
            'title': {"text": "Claim Rate by Property Type"}
        }
        fig.update_layout(**merge_layout(custom_layout))
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Use the styled container for consistent appearance
    styled_container(plot_claim_rate)

def render_overview_page(property_data, business_data):
    """Render the complete Overview page."""
    st.markdown('<h2 class="section-header">Underwriting Dashboard Overview</h2>', unsafe_allow_html=True)
    
    # Render metric cards
    render_metric_cards(property_data, business_data)
    
    # Create a clean container for the entire page
    st.markdown('<div style="padding: 0; margin: 0; background-color: transparent; border: none;">', unsafe_allow_html=True)
    
    # Create two columns for the charts
    col1, col2 = st.columns(2)
    
    # Property type distribution
    with col1:
        render_property_distribution(property_data)
    
    # Industry distribution
    with col2:
        render_industry_distribution(business_data)
    
    # Risk score distribution
    with col1:
        render_risk_score_distribution(property_data)
    
    # Claim rate by property type
    with col2:
        render_claim_rate_by_property(property_data)
    
    st.markdown('</div>', unsafe_allow_html=True)
