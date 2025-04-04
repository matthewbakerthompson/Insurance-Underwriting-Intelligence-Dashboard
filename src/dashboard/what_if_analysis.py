"""
What-If Analysis module for the Insurance Underwriting Intelligence Dashboard.
Contains functions for generating what-if analysis visualizations.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from theme import COLORS, PLOT_LAYOUT, styled_container, get_property_type_colors, merge_layout

def render_what_if_analysis(property_data):
    """Render the What-If Analysis page with interactive controls."""
    st.markdown('<h2 class="section-header">What-If Analysis</h2>', unsafe_allow_html=True)
    
    # Display a regular text description instead of an info box
    st.markdown("<p style='margin-bottom: 20px;'>This interactive tool allows you to explore how changing various risk factors affects the predicted risk score and claim probability.</p>", unsafe_allow_html=True)
    
    # Create columns for the form and results
    col1, col2 = st.columns([1, 2])
    
    with col1:
        def render_controls():
            st.subheader("Adjust Risk Factors")
            
            # Property type selection
            property_type = st.selectbox(
                "Property Type",
                options=["Residential", "Commercial", "Industrial"],
                index=0
            )
            
            # Property value slider
            property_value = st.slider(
                "Property Value ($)",
                min_value=50000,
                max_value=1000000,
                value=250000,
                step=10000,
                format="$%d"
            )
            
            # Property age slider
            property_age = st.slider(
                "Property Age (years)",
                min_value=0,
                max_value=100,
                value=20,
                step=1
            )
            
            # Property size slider
            property_size = st.slider(
                "Property Size (sq ft)",
                min_value=500,
                max_value=10000,
                value=2000,
                step=100
            )
            
            # Flood risk score slider
            flood_risk = st.slider(
                "Flood Risk Score",
                min_value=0.0,
                max_value=10.0,
                value=3.0,
                step=0.1
            )
            
            # Fire risk score slider
            fire_risk = st.slider(
                "Fire Risk Score",
                min_value=0.0,
                max_value=10.0,
                value=2.5,
                step=0.1
            )
            
            # Calculate button with custom styling
            st.markdown("<style>.calculate-btn {background-color: #1E40AF !important; color: white !important; font-weight: bold !important; padding: 0.6rem 1.2rem !important; font-size: 1.1rem !important;}</style>", unsafe_allow_html=True)
            calculate = st.button("Calculate Risk", type="primary", key="calculate-risk-btn", use_container_width=True)
            
            return {
                "property_type": property_type,
                "property_value": property_value,
                "property_age": property_age,
                "property_size": property_size,
                "flood_risk": flood_risk,
                "fire_risk": fire_risk,
                "calculate": calculate
            }
        
        # Use the styled container for the controls
        inputs = styled_container(render_controls)
    
    with col2:
        def render_results():
            if inputs["calculate"]:
                st.subheader("Risk Assessment Results")
                
                # Calculate a simulated risk score based on the inputs
                # This is a simplified model for demonstration purposes
                base_score = 0
                
                # Property type factor
                type_factor = {
                    "Residential": 1.0,
                    "Commercial": 1.2,
                    "Industrial": 1.5
                }
                
                # Calculate normalized factors (0-1 scale)
                value_factor = min(1.0, inputs["property_value"] / 1000000)
                age_factor = min(1.0, inputs["property_age"] / 100)
                size_factor = min(1.0, inputs["property_size"] / 10000)
                flood_factor = inputs["flood_risk"] / 10
                fire_factor = inputs["fire_risk"] / 10
                
                # Calculate weighted risk score (0-100 scale)
                risk_score = (
                    type_factor[inputs["property_type"]] * 10 +
                    value_factor * 15 +
                    age_factor * 20 +
                    size_factor * 15 +
                    flood_factor * 20 +
                    fire_factor * 20
                )
                
                # Scale to 0-100
                risk_score = min(100, max(0, risk_score))
                
                # Calculate claim probability (simplified logistic function)
                claim_prob = 1 / (1 + np.exp(-0.05 * (risk_score - 50)))
                
                # Display the results
                col1, col2 = st.columns(2)
                
                with col1:
                    # Risk score gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=risk_score,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Risk Score"},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1},
                            'bar': {'color': COLORS['primary']},
                            'steps': [
                                {'range': [0, 33], 'color': "lightgreen"},
                                {'range': [33, 66], 'color': "gold"},
                                {'range': [66, 100], 'color': "salmon"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': risk_score
                            }
                        }
                    ))
                    
                    # Create a custom layout with our specific margin settings
                    custom_layout = merge_layout({
                        'height': 300,
                        'title': 'Claim Probability Prediction',
                        'margin': dict(t=50, b=0, l=25, r=25)
                    })
                    
                    fig.update_layout(**custom_layout)
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Claim probability gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=claim_prob * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Claim Probability (%)"},
                        number={'suffix': "%", 'valueformat': ".1f"},
                        delta={'reference': 15, 'valueformat': ".1f"},
                        gauge={
                            'axis': {'range': [0, 50], 'tickwidth': 1},
                            'bar': {'color': COLORS['secondary']},
                            'steps': [
                                {'range': [0, 10], 'color': "lightgreen"},
                                {'range': [10, 25], 'color': "gold"},
                                {'range': [25, 50], 'color': "salmon"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': claim_prob * 100
                            }
                        }
                    ))
                    
                    # Create a custom layout with our specific margin settings
                    custom_layout = merge_layout({
                        'height': 300,
                        'title': 'Claim Probability Prediction',
                        'margin': dict(t=50, b=0, l=25, r=25)
                    })
                    
                    fig.update_layout(**custom_layout)
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Risk factors breakdown
                st.subheader("Risk Factors Breakdown")
                
                # Calculate factor contributions
                factors = {
                    "Property Type": type_factor[inputs["property_type"]] * 10,
                    "Property Value": value_factor * 15,
                    "Property Age": age_factor * 20,
                    "Property Size": size_factor * 15,
                    "Flood Risk": flood_factor * 20,
                    "Fire Risk": fire_factor * 20
                }
                
                # Create DataFrame for the chart
                factors_df = pd.DataFrame({
                    'Factor': list(factors.keys()),
                    'Contribution': list(factors.values())
                })
                
                # Sort by contribution
                factors_df = factors_df.sort_values('Contribution', ascending=True)
                
                # Create horizontal bar chart
                fig = px.bar(
                    factors_df,
                    x='Contribution',
                    y='Factor',
                    orientation='h',
                    text='Contribution',
                    color='Contribution',
                    color_continuous_scale=['lightgreen', 'gold', 'salmon'],
                    labels={'Contribution': 'Risk Contribution'}
                )
                
                fig.update_traces(
                    texttemplate='%{text:.1f}',
                    textposition='outside'
                )
                
                # Create a custom layout with our specific settings
                custom_layout = merge_layout({
                    'height': 350,
                    'title': 'Risk Factors Contribution Analysis',
                    'yaxis_title': None,
                    'xaxis_title': "Contribution to Risk Score",
                    'coloraxis_showscale': False
                })
                
                fig.update_layout(**custom_layout)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations based on risk factors
                st.subheader("Risk Mitigation Recommendations")
                
                # Find the top risk factors
                top_factors = factors_df.tail(3)['Factor'].tolist()
                
                recommendations = {
                    "Property Type": "Consider different property types with lower risk profiles.",
                    "Property Value": "Higher value properties may have higher claim amounts. Consider additional coverage or security measures.",
                    "Property Age": "Older properties have higher risk. Consider renovations or updates to key systems (electrical, plumbing, roof).",
                    "Property Size": "Larger properties have more exposure. Consider dividing into separate policies or implementing zone-specific protections.",
                    "Flood Risk": "Implement flood mitigation measures such as improved drainage, flood barriers, or elevated equipment.",
                    "Fire Risk": "Install fire suppression systems, smoke detectors, and fire-resistant materials. Regular inspections recommended."
                }
                
                for factor in top_factors:
                    st.info(f"**{factor}**: {recommendations[factor]}")
            else:
                st.info("Adjust the risk factors on the left and click 'Calculate Risk' to see the results.")
        
        # Use the styled container for the results
        styled_container(render_results)

def render_what_if_analysis_page(property_data):
    """Render the complete What-If Analysis page."""
    render_what_if_analysis(property_data)
