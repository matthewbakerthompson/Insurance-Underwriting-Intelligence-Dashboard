"""
Scenario Planning module for the Insurance Underwriting Intelligence Dashboard.
Contains functions for creating and comparing multiple risk scenarios.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from theme import COLORS, PLOT_LAYOUT, styled_container, merge_layout, get_property_type_colors

def render_scenario_creator():
    """
    Render a form for creating and saving risk scenarios.
    
    Returns:
        dict: The created scenario if saved, None otherwise
    """
    st.subheader("Create Risk Scenario")
    
    # Use a form for scenario creation
    with st.form("scenario_form"):
        # Scenario name
        scenario_name = st.text_input("Scenario Name", "New Scenario")
        
        # Create columns for the form
        col1, col2 = st.columns(2)
        
        with col1:
            # Property characteristics
            st.write("Property Characteristics")
            property_type = st.selectbox(
                "Property Type",
                options=["Residential", "Commercial", "Industrial"],
                index=0
            )
            
            property_value = st.slider(
                "Property Value ($)",
                min_value=50000,
                max_value=2000000,
                value=350000,
                step=10000,
                format="$%d"
            )
            
            property_age = st.slider(
                "Property Age (years)",
                min_value=1,
                max_value=100,
                value=20,
                step=1
            )
            
            property_size = st.slider(
                "Property Size (sq ft)",
                min_value=500,
                max_value=10000,
                value=2000,
                step=100
            )
        
        with col2:
            # Risk factors
            st.write("Risk Factors")
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
            
            # Additional risk factors
            weather_risk = st.slider(
                "Weather Risk Score",
                min_value=0.0,
                max_value=10.0,
                value=4.0,
                step=0.1
            )
            
            location_quality = st.select_slider(
                "Location Quality",
                options=["Very Poor", "Poor", "Average", "Good", "Excellent"],
                value="Average"
            )
        
        # Submit button
        submitted = st.form_submit_button("Save Scenario", type="primary", use_container_width=True)
        
        if submitted:
            # Create scenario dictionary
            scenario = {
                "name": scenario_name,
                "property_type": property_type,
                "property_value": property_value,
                "property_age": property_age,
                "property_size": property_size,
                "flood_risk": flood_risk,
                "fire_risk": fire_risk,
                "weather_risk": weather_risk,
                "location_quality": location_quality,
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Calculate overall risk score (simplified formula)
            location_quality_map = {
                "Very Poor": 2.0,
                "Poor": 1.0,
                "Average": 0.0,
                "Good": -1.0,
                "Excellent": -2.0
            }
            
            base_risk = (flood_risk * 0.3) + (fire_risk * 0.3) + (weather_risk * 0.2)
            property_factor = (property_age / 100) * 2  # Age increases risk
            location_factor = location_quality_map[location_quality]
            
            scenario["risk_score"] = min(10, max(0, base_risk + property_factor + location_factor))
            
            # Calculate claim probability
            scenario["claim_probability"] = 1 / (1 + np.exp(-0.5 * (scenario["risk_score"] - 5)))
            
            return scenario
    
    return None

def load_saved_scenarios():
    """
    Load saved scenarios from session state.
    
    Returns:
        list: List of saved scenarios
    """
    if "scenarios" not in st.session_state:
        st.session_state.scenarios = []
    
    return st.session_state.scenarios

def save_scenario(scenario):
    """
    Save a scenario to session state.
    
    Args:
        scenario (dict): Scenario to save
    """
    if "scenarios" not in st.session_state:
        st.session_state.scenarios = []
    
    # Check if a scenario with the same name exists
    existing_names = [s["name"] for s in st.session_state.scenarios]
    if scenario["name"] in existing_names:
        # Append a number to make the name unique
        i = 1
        while f"{scenario['name']} ({i})" in existing_names:
            i += 1
        scenario["name"] = f"{scenario['name']} ({i})"
    
    st.session_state.scenarios.append(scenario)

def delete_scenario(scenario_name):
    """
    Delete a scenario from session state.
    
    Args:
        scenario_name (str): Name of the scenario to delete
    """
    if "scenarios" in st.session_state:
        st.session_state.scenarios = [s for s in st.session_state.scenarios if s["name"] != scenario_name]

def render_scenario_list():
    """
    Render a list of saved scenarios with options to view, compare, or delete.
    
    Returns:
        list: List of selected scenario names for comparison
    """
    st.subheader("Saved Scenarios")
    
    # Load saved scenarios
    scenarios = load_saved_scenarios()
    
    if not scenarios:
        st.info("No scenarios saved yet. Create a scenario using the form above.")
        return []
    
    # Create a dataframe for display
    scenario_df = pd.DataFrame([
        {
            "Scenario Name": s["name"],
            "Property Type": s["property_type"],
            "Risk Score": f"{s['risk_score']:.1f}",
            "Claim Probability": f"{s['claim_probability']:.1%}",
            "Created": s["timestamp"]
        }
        for s in scenarios
    ])
    
    # Display scenarios with selection
    selected_scenarios = st.multiselect(
        "Select scenarios to compare",
        options=scenario_df["Scenario Name"].tolist(),
        default=scenario_df["Scenario Name"].tolist()[:min(3, len(scenario_df))]
    )
    
    # Display the scenario table
    st.dataframe(
        scenario_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Add option to delete scenarios
    if scenarios:
        scenario_to_delete = st.selectbox(
            "Select a scenario to delete",
            options=[""] + scenario_df["Scenario Name"].tolist()
        )
        
        if scenario_to_delete:
            if st.button(f"Delete '{scenario_to_delete}'", type="secondary"):
                delete_scenario(scenario_to_delete)
                st.success(f"Deleted scenario: {scenario_to_delete}")
                st.rerun()
    
    return selected_scenarios

def render_scenario_comparison(selected_scenarios):
    """
    Render a comparison of selected scenarios.
    
    Args:
        selected_scenarios (list): List of scenario names to compare
    """
    st.subheader("Scenario Comparison")
    
    # Load saved scenarios
    all_scenarios = load_saved_scenarios()
    
    # Filter to selected scenarios
    scenarios = [s for s in all_scenarios if s["name"] in selected_scenarios]
    
    if not scenarios:
        st.info("Select scenarios to compare.")
        return
    
    # Create tabs for different comparisons
    tab1, tab2, tab3 = st.tabs(["Risk Comparison", "Property Comparison", "Detailed View"])
    
    with tab1:
        # Create risk score and claim probability comparison
        render_risk_comparison(scenarios)
    
    with tab2:
        # Create property characteristics comparison
        render_property_comparison(scenarios)
    
    with tab3:
        # Create detailed view of all factors
        render_detailed_comparison(scenarios)

def render_risk_comparison(scenarios):
    """
    Render a comparison of risk scores and claim probabilities.
    
    Args:
        scenarios (list): List of scenarios to compare
    """
    # Create a dataframe for the comparison
    risk_data = []
    
    for s in scenarios:
        risk_data.append({
            "Scenario": s["name"],
            "Metric": "Risk Score",
            "Value": s["risk_score"]
        })
        risk_data.append({
            "Scenario": s["name"],
            "Metric": "Claim Probability",
            "Value": s["claim_probability"] * 10  # Scale to make comparable
        })
    
    risk_df = pd.DataFrame(risk_data)
    
    # Create grouped bar chart
    fig = px.bar(
        risk_df,
        x="Scenario",
        y="Value",
        color="Metric",
        barmode="group",
        color_discrete_map={
            "Risk Score": COLORS["primary"],
            "Claim Probability": COLORS["secondary"]
        }
    )
    
    # Add a second y-axis for claim probability
    fig.update_layout(
        yaxis=dict(
            title="Risk Score (0-10)",
            range=[0, 10]
        ),
        yaxis2=dict(
            title="Claim Probability",
            titlefont=dict(color=COLORS["secondary"]),
            tickfont=dict(color=COLORS["secondary"]),
            anchor="x",
            overlaying="y",
            side="right",
            range=[0, 1],
            tickformat=".0%",
            showgrid=False
        )
    )
    
    # Update the claim probability bars to use the second y-axis
    for i, trace in enumerate(fig.data):
        if trace.name == "Claim Probability":
            trace.yaxis = "y2"
            trace.y = [y / 10 for y in trace.y]  # Scale back for display
    
    # Update layout
    custom_layout = {
        "height": 500,
        "title": "Property Characteristics Comparison Across Scenarios",
        "legend": dict(
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
    <p style='margin-top: 10px; margin-bottom: 20px;'>This chart compares the risk scores and claim probabilities across the selected scenarios. 
    Risk scores are shown on the left axis (0-10 scale), while claim probabilities are shown on the right axis (percentage). 
    This visualization helps identify which scenarios represent higher risk and how changes in risk factors affect the likelihood of claims.</p>
    """, unsafe_allow_html=True)

def render_property_comparison(scenarios):
    """
    Render a comparison of property characteristics.
    
    Args:
        scenarios (list): List of scenarios to compare
    """
    # Create a dataframe for the comparison
    property_data = []
    
    for s in scenarios:
        property_data.append({
            "Scenario": s["name"],
            "Characteristic": "Property Value",
            "Value": s["property_value"] / 1000  # Scale for display
        })
        property_data.append({
            "Scenario": s["name"],
            "Characteristic": "Property Age",
            "Value": s["property_age"]
        })
        property_data.append({
            "Scenario": s["name"],
            "Characteristic": "Property Size",
            "Value": s["property_size"] / 100  # Scale for display
        })
    
    property_df = pd.DataFrame(property_data)
    
    # Add a title for the chart
    st.subheader("Property Characteristics by Scenario")
    
    # Create grouped bar chart
    fig = px.bar(
        property_df,
        x="Scenario",
        y="Value",
        color="Characteristic",
        barmode="group",
        color_discrete_map={
            "Property Value": COLORS["primary"],  # Using primary instead of non-existent 'blue'
            "Property Age": COLORS["finance"],  # Using finance instead of non-existent 'orange'
            "Property Size": COLORS["secondary"]  # Using secondary instead of non-existent 'green'
        }
    )
    
    # Create multiple y-axes for different scales
    fig.update_layout(
        yaxis=dict(
            title="Property Value ($K)",
            range=[0, max(property_df[property_df["Characteristic"] == "Property Value"]["Value"]) * 1.1]
        ),
        yaxis2=dict(
            title="Property Age (years)",
            titlefont=dict(color=COLORS["finance"]),  # Using finance instead of non-existent 'orange'
            tickfont=dict(color=COLORS["finance"]),  # Using finance instead of non-existent 'orange'
            anchor="free",
            overlaying="y",
            side="right",
            position=0.85,
            range=[0, max(property_df[property_df["Characteristic"] == "Property Age"]["Value"]) * 1.1],
            showgrid=False
        ),
        yaxis3=dict(
            title="Property Size (100 sq ft)",
            titlefont=dict(color=COLORS["secondary"]),  # Using secondary instead of non-existent 'green'
            tickfont=dict(color=COLORS["secondary"]),  # Using secondary instead of non-existent 'green'
            anchor="free",
            overlaying="y",
            side="right",
            position=1.0,
            range=[0, max(property_df[property_df["Characteristic"] == "Property Size"]["Value"]) * 1.1],
            showgrid=False
        )
    )
    
    # Update the bars to use the appropriate y-axis
    for i, trace in enumerate(fig.data):
        if trace.name == "Property Age":
            trace.yaxis = "y2"
        elif trace.name == "Property Size":
            trace.yaxis = "y3"
    
    # Update layout
    custom_layout = {
        "height": 500,
        "title": "Property Characteristics Comparison Across Scenarios",
        "legend": dict(
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
    <p style='margin-top: 10px; margin-bottom: 20px;'>This chart compares the property characteristics across the selected scenarios. 
    Property value is shown in thousands of dollars (left axis), property age in years (first right axis), and property size in hundreds of square feet (second right axis). 
    This visualization helps identify how different property characteristics vary across scenarios and how they might contribute to risk differences.</p>
    """, unsafe_allow_html=True)

def render_detailed_comparison(scenarios):
    """
    Render a detailed comparison of all factors.
    
    Args:
        scenarios (list): List of scenarios to compare
    """
    # Create a dataframe for the comparison
    comparison_data = []
    
    for s in scenarios:
        comparison_data.append({
            "Scenario Name": s["name"],
            "Property Type": s["property_type"],
            "Property Value": f"${s['property_value']:,}",
            "Property Age": f"{s['property_age']} years",
            "Property Size": f"{s['property_size']:,} sq ft",
            "Flood Risk": f"{s['flood_risk']:.1f}/10",
            "Fire Risk": f"{s['fire_risk']:.1f}/10",
            "Weather Risk": f"{s['weather_risk']:.1f}/10",
            "Location Quality": s["location_quality"],
            "Risk Score": f"{s['risk_score']:.1f}/10",
            "Claim Probability": f"{s['claim_probability']:.1%}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display the comparison table
    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Create radar chart for multi-dimensional comparison
    fig = go.Figure()
    
    # Define categories for radar chart
    categories = [
        "Flood Risk", "Fire Risk", "Weather Risk", 
        "Property Age", "Risk Score", "Claim Probability"
    ]
    
    # Add a trace for each scenario
    for s in scenarios:
        # Normalize property age to 0-10 scale for comparison
        normalized_age = min(10, s["property_age"] / 10)
        
        fig.add_trace(go.Scatterpolar(
            r=[
                s["flood_risk"], 
                s["fire_risk"], 
                s["weather_risk"], 
                normalized_age, 
                s["risk_score"], 
                s["claim_probability"] * 10  # Scale to 0-10
            ],
            theta=categories,
            fill="toself",
            name=s["name"]
        ))
    
    # Update layout
    custom_layout = {
        "height": 500,
        "polar": {
            "radialaxis": {
                "visible": True,
                "range": [0, 10]
            }
        },
        "showlegend": True,
        "legend": dict(
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
    <p style='margin-top: 10px; margin-bottom: 20px;'>This radar chart provides a multi-dimensional comparison of the selected scenarios. 
    Each axis represents a different risk factor or outcome, with values normalized to a 0-10 scale for comparison. 
    The shape of each scenario's radar plot reveals its unique risk profile and helps identify which factors drive differences in overall risk.</p>
    """, unsafe_allow_html=True)

def render_scenario_planning_page():
    """Render the Scenario Planning page with interactive scenario creation and comparison."""
    st.markdown('<h2 class="section-header">Scenario Planning</h2>', unsafe_allow_html=True)
    
    # Display page description
    st.markdown("""
    <p style='margin-bottom: 20px;'>This tool allows you to create, save, and compare multiple risk scenarios, 
    helping underwriters evaluate different property characteristics and risk factors to make informed decisions.</p>
    """, unsafe_allow_html=True)
    
    # Create columns for scenario creation and list
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Render scenario creator
        new_scenario = render_scenario_creator()
        
        # Save new scenario if created
        if new_scenario:
            save_scenario(new_scenario)
            st.success(f"Saved scenario: {new_scenario['name']}")
            st.rerun()
    
    with col2:
        # Render scenario list
        selected_scenarios = render_scenario_list()
    
    # Render scenario comparison if scenarios are selected
    if selected_scenarios:
        render_scenario_comparison(selected_scenarios)
