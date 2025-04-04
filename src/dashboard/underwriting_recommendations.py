"""
Automated Underwriting Recommendations module for the Insurance Underwriting Intelligence Dashboard.
Contains functions for generating AI-powered underwriting recommendations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from theme import COLORS, PLOT_LAYOUT, styled_container, merge_layout

def generate_policy_recommendation(property_data, property_id=None, custom_inputs=None):
    """
    Generate automated underwriting recommendations based on property characteristics.
    
    Args:
        property_data (pd.DataFrame): Property data
        property_id (str, optional): Property ID to generate recommendations for
        custom_inputs (dict, optional): Custom property inputs for recommendation
        
    Returns:
        dict: Recommendation details
    """
    # If custom inputs are provided, use those instead of looking up a property
    if custom_inputs:
        # Extract values from custom inputs
        property_type = custom_inputs.get("property_type", "Residential")
        property_value = custom_inputs.get("property_value", 350000)
        property_age = custom_inputs.get("property_age", 20)
        property_size = custom_inputs.get("property_size", 2000)
        flood_risk = custom_inputs.get("flood_risk", 3.0)
        fire_risk = custom_inputs.get("fire_risk", 2.5)
        location_risk = (flood_risk * 0.4) + (fire_risk * 0.4) + (np.random.normal(5, 1) * 0.2)
        
        # Set a random seed based on inputs for reproducibility
        seed = int(property_value + property_age + property_size + flood_risk * 100 + fire_risk * 100)
        np.random.seed(seed)
        
    # If property ID is provided, look up the property
    elif property_id:
        property_row = property_data[property_data["property_id"] == property_id]
        
        if property_row.empty:
            return None
        
        # Extract values from property data
        property_type = property_row["property_type"].values[0]
        property_value = property_row["property_value"].values[0]
        property_age = property_row["property_age"].values[0]
        property_size = property_row["property_size"].values[0]
        flood_risk = property_row["flood_risk_score"].values[0]
        fire_risk = property_row["fire_risk_score"].values[0]
        location_risk = property_row["location_risk"].values[0]
        
        # Set a random seed based on property ID for reproducibility
        seed = int(property_id.replace("PROP-", ""))
        np.random.seed(seed)
        
    else:
        return None
    
    # Calculate base premium (simplified formula)
    # Base premium factors by property type
    base_premium_factors = {
        "Residential": 1.0,
        "Commercial": 1.8,
        "Industrial": 2.5
    }
    
    # Calculate base premium
    base_premium = (property_value * 0.002) * base_premium_factors[property_type]
    
    # Apply risk adjustments
    risk_multiplier = 1.0
    
    # Adjust for property age
    if property_age < 5:
        risk_multiplier *= 0.9  # Newer properties have lower risk
    elif property_age > 30:
        risk_multiplier *= 1.3  # Older properties have higher risk
    
    # Adjust for risk scores
    risk_multiplier *= (1.0 + (location_risk - 5) * 0.1)
    
    # Apply size adjustment
    size_factor = 1.0
    if property_type == "Residential":
        if property_size > 3000:
            size_factor = 1.2
        elif property_size < 1000:
            size_factor = 0.9
    elif property_type == "Commercial":
        if property_size > 5000:
            size_factor = 1.1
        elif property_size < 2000:
            size_factor = 0.95
    elif property_type == "Industrial":
        if property_size > 8000:
            size_factor = 1.15
        elif property_size < 3000:
            size_factor = 0.9
    
    # Calculate final premium
    final_premium = base_premium * risk_multiplier * size_factor
    
    # Determine coverage recommendations
    coverage_percentage = 1.0  # Full coverage by default
    
    # Adjust coverage based on risk
    if location_risk > 7:
        coverage_percentage = 0.9  # Higher risk properties get slightly lower coverage
    elif location_risk < 3:
        coverage_percentage = 1.1  # Lower risk properties can get enhanced coverage
    
    recommended_coverage = property_value * coverage_percentage
    
    # Determine deductible recommendations
    # Base deductible is a percentage of property value
    base_deductible_pct = 0.01  # 1% of property value
    
    # Adjust based on risk
    if location_risk > 7:
        base_deductible_pct = 0.015  # Higher risk properties get higher deductibles
    elif location_risk < 3:
        base_deductible_pct = 0.005  # Lower risk properties get lower deductibles
    
    recommended_deductible = property_value * base_deductible_pct
    
    # Round to nearest $500
    recommended_deductible = round(recommended_deductible / 500) * 500
    
    # Determine policy term recommendations
    if location_risk > 8:
        recommended_term = 1  # 1 year for very high risk
    elif location_risk > 6:
        recommended_term = 2  # 2 years for high risk
    else:
        recommended_term = 3  # 3 years for normal/low risk
    
    # Determine approval status
    if location_risk > 9:
        approval_status = "Refer to Senior Underwriter"
        approval_confidence = 0.3
    elif location_risk > 7:
        approval_status = "Approved with Conditions"
        approval_confidence = 0.7
    else:
        approval_status = "Approved"
        approval_confidence = 0.9
    
    # Generate specific conditions based on risk factors
    conditions = []
    
    if flood_risk > 7:
        conditions.append("Flood mitigation measures required")
    
    if fire_risk > 7:
        conditions.append("Fire safety inspection required")
    
    if property_age > 40:
        conditions.append("Building inspection required")
    
    if property_type == "Industrial" and location_risk > 6:
        conditions.append("Environmental risk assessment required")
    
    if not conditions and approval_status == "Approved with Conditions":
        conditions.append("Annual risk reassessment required")
    
    # Create recommendation dictionary
    recommendation = {
        "property_id": property_id if property_id else "Custom Property",
        "property_type": property_type,
        "property_value": property_value,
        "property_age": property_age,
        "property_size": property_size,
        "flood_risk": flood_risk,
        "fire_risk": fire_risk,
        "location_risk": location_risk,
        "base_premium": base_premium,
        "risk_multiplier": risk_multiplier,
        "size_factor": size_factor,
        "final_premium": final_premium,
        "recommended_coverage": recommended_coverage,
        "recommended_deductible": recommended_deductible,
        "recommended_term": recommended_term,
        "approval_status": approval_status,
        "approval_confidence": approval_confidence,
        "conditions": conditions
    }
    
    return recommendation

def render_property_search(property_data):
    """
    Render a search widget for finding properties.
    
    Args:
        property_data (pd.DataFrame): Property data
        
    Returns:
        str: Selected property ID or None
    """
    st.subheader("Property Search")
    
    # Create a dataframe with key property information for selection
    search_data = property_data[['property_id', 'property_type', 'property_value', 'location_risk']].copy()
    search_data['property_value_formatted'] = search_data['property_value'].apply(lambda x: f"${x:,.0f}")
    search_data['risk_score'] = search_data['location_risk'].apply(lambda x: f"{x:.1f}")
    
    # Create columns for search options
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Option to search by property ID
        property_ids = [""] + property_data["property_id"].tolist()
        selected_id = st.selectbox("Select Property ID", options=property_ids)
    
    with col2:
        # Option to filter by property type
        property_types = ["All Types"] + property_data["property_type"].unique().tolist()
        selected_type = st.selectbox("Filter by Property Type", options=property_types)
    
    # Filter data based on selections
    filtered_data = search_data
    
    if selected_type != "All Types":
        filtered_data = filtered_data[filtered_data["property_type"] == selected_type]
    
    # Display filtered properties
    st.dataframe(
        filtered_data[['property_id', 'property_type', 'property_value_formatted', 'risk_score']].rename(
            columns={
                'property_id': 'Property ID',
                'property_type': 'Type',
                'property_value_formatted': 'Value',
                'risk_score': 'Risk Score'
            }
        ),
        use_container_width=True,
        height=200,
        hide_index=True
    )
    
    return selected_id if selected_id else None

def render_custom_property_form():
    """
    Render a form for entering custom property details.
    
    Returns:
        dict: Custom property inputs or None if not submitted
    """
    st.subheader("Custom Property Analysis")
    
    # Use expander to save space
    with st.expander("Enter Custom Property Details"):
        # Use a form for custom property inputs
        with st.form("custom_property_form"):
            # Create columns for the form
            col1, col2 = st.columns(2)
            
            with col1:
                # Property characteristics
                property_type = st.selectbox(
                    "Property Type",
                    options=["Residential", "Commercial", "Industrial"],
                    index=0
                )
                
                property_value = st.number_input(
                    "Property Value ($)",
                    min_value=50000,
                    max_value=2000000,
                    value=350000,
                    step=10000
                )
                
                property_age = st.number_input(
                    "Property Age (years)",
                    min_value=1,
                    max_value=100,
                    value=20,
                    step=1
                )
            
            with col2:
                # Property characteristics continued
                property_size = st.number_input(
                    "Property Size (sq ft)",
                    min_value=500,
                    max_value=10000,
                    value=2000,
                    step=100
                )
                
                flood_risk = st.slider(
                    "Flood Risk Score",
                    min_value=0.0,
                    max_value=10.0,
                    value=3.0,
                    step=0.1
                )
                
                fire_risk = st.slider(
                    "Fire Risk Score",
                    min_value=0.0,
                    max_value=10.0,
                    value=2.5,
                    step=0.1
                )
            
            # Submit button
            submitted = st.form_submit_button("Generate Recommendation", type="primary", use_container_width=True)
            
            if submitted:
                # Create custom inputs dictionary
                custom_inputs = {
                    "property_type": property_type,
                    "property_value": property_value,
                    "property_age": property_age,
                    "property_size": property_size,
                    "flood_risk": flood_risk,
                    "fire_risk": fire_risk
                }
                
                return custom_inputs
    
    return None

def render_recommendation_details(recommendation):
    """
    Render detailed underwriting recommendations.
    
    Args:
        recommendation (dict): Recommendation details
    """
    st.subheader("Underwriting Recommendation")
    
    # Create columns for key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Recommended Premium", 
            value=f"${recommendation['final_premium']:,.2f}"
        )
    
    with col2:
        st.metric(
            label="Approval Status", 
            value=recommendation['approval_status']
        )
    
    with col3:
        st.metric(
            label="Confidence Score", 
            value=f"{recommendation['approval_confidence']:.0%}"
        )
    
    # Create columns for recommendation details
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Policy Recommendations")
        
        # Create a table of policy recommendations with better styling
        policy_details = pd.DataFrame([
            {"Metric": "Coverage Amount", "Value": f"${recommendation['recommended_coverage']:,.0f}"},
            {"Metric": "Deductible", "Value": f"${recommendation['recommended_deductible']:,.0f}"},
            {"Metric": "Policy Term", "Value": f"{recommendation['recommended_term']} years"},
            {"Metric": "Base Premium", "Value": f"${recommendation['base_premium']:,.2f}"},
            {"Metric": "Risk Multiplier", "Value": f"{recommendation['risk_multiplier']:.2f}"},
            {"Metric": "Size Factor", "Value": f"{recommendation['size_factor']:.2f}"}
        ])
        
        # Use a styled table with better formatting
        st.table(policy_details.set_index('Metric'))
    
    with col2:
        st.subheader("Risk Assessment")
        
        # Create a gauge chart for overall risk
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=recommendation['location_risk'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Risk Score", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': COLORS['primary']},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 3], 'color': 'green'},
                    {'range': [3, 7], 'color': 'yellow'},
                    {'range': [7, 10], 'color': 'red'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': recommendation['location_risk']
                }
            }
        ))
        
        # Update layout
        custom_layout = {
            'height': 300,
            'width': 400,
            'title': 'Property Risk Assessment',
            'margin': {"r": 25, "t": 50, "l": 25, "b": 25}
        }
        
        fig.update_layout(**merge_layout(custom_layout))
        st.plotly_chart(fig, use_container_width=True)
    
    # Display conditions if any
    if recommendation['conditions']:
        st.subheader("Conditions")
        
        for condition in recommendation['conditions']:
            st.warning(condition)
    
    # Create premium breakdown chart
    st.subheader("Premium Breakdown")
    
    # Calculate premium components
    base = recommendation['base_premium']
    risk_component = base * (recommendation['risk_multiplier'] - 1)
    size_component = base * recommendation['risk_multiplier'] * (recommendation['size_factor'] - 1)
    
    # Create data for the waterfall chart
    premium_breakdown = pd.DataFrame([
        {"Category": "Base Premium", "Amount": base},
        {"Category": "Risk Adjustment", "Amount": risk_component},
        {"Category": "Size Adjustment", "Amount": size_component},
        {"Category": "Final Premium", "Amount": recommendation['final_premium']}
    ])
    
    # Create waterfall chart
    fig = go.Figure(go.Waterfall(
        name="Premium Breakdown",
        orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=premium_breakdown["Category"],
        textposition="outside",
        text=premium_breakdown["Amount"].apply(lambda x: f"${x:,.2f}"),
        y=premium_breakdown["Amount"],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": COLORS["secondary"]}},  # Using secondary instead of non-existent 'green'
        increasing={"marker": {"color": COLORS["finance"]}},  # Using finance instead of non-existent 'orange'
        totals={"marker": {"color": COLORS["primary"]}}
    ))
    
    # Update layout
    custom_layout = {
        'height': 400,
        'xaxis_title': "Premium Component",
        'yaxis_title': "Amount ($)",
        'title': "Premium Calculation Breakdown",
        'yaxis': {
            'tickformat': "$,.2f"
        }
    }
    
    fig.update_layout(**merge_layout(custom_layout))
    st.plotly_chart(fig, use_container_width=True)
    
    # Add description
    st.markdown("""
    <p style='margin-top: 10px; margin-bottom: 20px;'>This automated recommendation is generated based on the property characteristics and risk factors. 
    The premium calculation considers the base rate for the property type, adjustments for specific risk factors, and property size. 
    The approval status and confidence score indicate the system's assessment of the risk, with lower confidence scores suggesting cases that may benefit from human review. 
    Any conditions listed must be met for policy issuance.</p>
    """, unsafe_allow_html=True)

def render_similar_properties(property_data, recommendation):
    """
    Render a comparison with similar properties.
    
    Args:
        property_data (pd.DataFrame): Property data
        recommendation (dict): Recommendation details
    """
    st.subheader("Similar Properties Comparison")
    
    # Filter to properties of the same type
    same_type = property_data[property_data["property_type"] == recommendation["property_type"]].copy()
    
    # Calculate similarity score based on property characteristics
    same_type["value_diff"] = abs(same_type["property_value"] - recommendation["property_value"]) / recommendation["property_value"]
    same_type["age_diff"] = abs(same_type["property_age"] - recommendation["property_age"]) / max(1, recommendation["property_age"])
    same_type["size_diff"] = abs(same_type["property_size"] - recommendation["property_size"]) / recommendation["property_size"]
    same_type["flood_diff"] = abs(same_type["flood_risk_score"] - recommendation["flood_risk"]) / max(1, recommendation["flood_risk"])
    same_type["fire_diff"] = abs(same_type["fire_risk_score"] - recommendation["fire_risk"]) / max(1, recommendation["fire_risk"])
    
    # Calculate overall similarity score (lower is more similar)
    same_type["similarity_score"] = (
        same_type["value_diff"] * 0.25 +
        same_type["age_diff"] * 0.2 +
        same_type["size_diff"] * 0.2 +
        same_type["flood_diff"] * 0.175 +
        same_type["fire_diff"] * 0.175
    )
    
    # Get the 5 most similar properties
    similar_properties = same_type.sort_values("similarity_score").head(5)
    
    # Calculate "expected" premium based on similar properties
    similar_premium = 0
    premium_weights = []
    
    for i, row in similar_properties.iterrows():
        # Calculate a simplified premium
        base = row["property_value"] * 0.002
        if row["property_type"] == "Commercial":
            base *= 1.8
        elif row["property_type"] == "Industrial":
            base *= 2.5
            
        risk_mult = 1.0 + (row["location_risk"] - 5) * 0.1
        premium = base * risk_mult
        
        # Weight is inverse of similarity score
        weight = 1 / max(0.01, row["similarity_score"])
        similar_premium += premium * weight
        premium_weights.append(weight)
    
    # Calculate weighted average
    similar_premium = similar_premium / sum(premium_weights)
    
    # Calculate premium difference
    premium_diff = recommendation["final_premium"] - similar_premium
    premium_diff_pct = premium_diff / similar_premium * 100
    
    # Display comparison metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Recommended Premium", 
            value=f"${recommendation['final_premium']:,.2f}"
        )
    
    with col2:
        st.metric(
            label="Similar Properties Avg Premium", 
            value=f"${similar_premium:,.2f}",
            delta=f"{premium_diff_pct:+.1f}%"
        )
    
    # Format similar properties for display
    display_properties = similar_properties[["property_id", "property_value", "property_age", "property_size", "location_risk", "claim_filed"]].copy()
    display_properties["property_value"] = display_properties["property_value"].apply(lambda x: f"${x:,.0f}")
    display_properties["property_size"] = display_properties["property_size"].apply(lambda x: f"{x:,} sq ft")
    display_properties["location_risk"] = display_properties["location_risk"].apply(lambda x: f"{x:.1f}")
    display_properties["claim_filed"] = display_properties["claim_filed"].apply(lambda x: "Yes" if x == 1 else "No")
    
    # Rename columns for display
    display_properties.columns = ["Property ID", "Value", "Age (years)", "Size", "Risk Score", "Claim Filed"]
    
    # Display similar properties
    st.dataframe(display_properties, use_container_width=True, hide_index=True)
    
    # Add description
    st.markdown("""
    <p style='margin-top: 10px; margin-bottom: 20px;'>This comparison shows similar properties and their calculated premiums to provide context for the recommendation. 
    A significant difference between the recommended premium and the average for similar properties may indicate unique risk factors or suggest the need for manual review. 
    Properties with claims history can provide insights into potential future claim patterns for similar risks.</p>
    """, unsafe_allow_html=True)

def render_underwriting_recommendations_page(property_data):
    """Render the Automated Underwriting Recommendations page with interactive property analysis."""
    st.markdown('<h2 class="section-header">Automated Underwriting Recommendations</h2>', unsafe_allow_html=True)
    
    # Display page description with clearer instructions
    st.markdown("""
    <p style='margin-bottom: 20px;'>This AI-powered tool provides automated underwriting recommendations, 
    including premium calculations, coverage suggestions, and approval status with confidence scores.</p>
    
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
        <h4 style='margin-top: 0;'>How to use this tool:</h4>
        <ul>
            <li><strong>Existing Property:</strong> Search for a property by ID or type to generate recommendations based on its characteristics.</li>
            <li><strong>Custom Property:</strong> Enter custom property details to see how different characteristics affect underwriting decisions.</li>
        </ul>
        <p>The tool will analyze risk factors and provide detailed premium calculations, coverage recommendations, and approval status.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different analysis options
    tab1, tab2 = st.tabs(["Existing Property", "Custom Property"])
    
    with tab1:
        # Render property search
        selected_property_id = render_property_search(property_data)
        
        # Generate recommendation if property is selected
        if selected_property_id:
            recommendation = generate_policy_recommendation(property_data, property_id=selected_property_id)
            
            if recommendation:
                # Render recommendation details
                render_recommendation_details(recommendation)
                
                # Render similar properties comparison
                render_similar_properties(property_data, recommendation)
    
    with tab2:
        # Render custom property form
        custom_inputs = render_custom_property_form()
        
        # Generate recommendation if form is submitted
        if custom_inputs:
            recommendation = generate_policy_recommendation(property_data, custom_inputs=custom_inputs)
            
            if recommendation:
                # Render recommendation details
                render_recommendation_details(recommendation)
                
                # Render similar properties comparison
                render_similar_properties(property_data, recommendation)
