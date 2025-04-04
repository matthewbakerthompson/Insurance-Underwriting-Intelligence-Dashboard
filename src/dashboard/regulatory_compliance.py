"""
Regulatory Compliance module for the Insurance Underwriting Intelligence Dashboard.
Contains functions for tracking and managing compliance requirements.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from theme import COLORS, PLOT_LAYOUT, styled_container, merge_layout

def generate_compliance_data():
    """
    Generate sample compliance data for demonstration.
    
    Returns:
        pd.DataFrame: Sample compliance data
    """
    # Define compliance categories
    categories = [
        "State Filing Requirements",
        "Rate Regulations",
        "Policy Form Requirements",
        "Consumer Protection",
        "Financial Solvency",
        "Market Conduct",
        "Privacy and Data Security",
        "Anti-Fraud Measures",
        "Catastrophe Response",
        "Environmental Regulations"
    ]
    
    # Define states
    states = [
        "California", "New York", "Texas", "Florida", "Illinois",
        "Pennsylvania", "Ohio", "Michigan", "Georgia", "North Carolina"
    ]
    
    # Define status options
    status_options = ["Compliant", "Pending Review", "Action Required", "Non-Compliant"]
    status_weights = [0.6, 0.2, 0.15, 0.05]  # Probability weights
    
    # Generate random data
    np.random.seed(42)  # For reproducibility
    
    data = []
    
    for category in categories:
        for state in states:
            # Not all categories apply to all states
            if np.random.random() < 0.8:  # 80% chance of a requirement for each state/category
                # Generate random dates
                last_review = datetime.now() - timedelta(days=np.random.randint(1, 365))
                next_review = last_review + timedelta(days=np.random.randint(30, 365))
                
                # Generate random status
                status = np.random.choice(status_options, p=status_weights)
                
                # Generate random compliance score
                if status == "Compliant":
                    score = np.random.uniform(0.9, 1.0)
                elif status == "Pending Review":
                    score = np.random.uniform(0.7, 0.9)
                elif status == "Action Required":
                    score = np.random.uniform(0.4, 0.7)
                else:  # Non-Compliant
                    score = np.random.uniform(0.0, 0.4)
                
                # Generate random risk level
                risk_level = "Low" if score > 0.8 else "Medium" if score > 0.5 else "High"
                
                # Generate random notes
                if status == "Compliant":
                    notes = "All requirements met"
                elif status == "Pending Review":
                    notes = "Documentation under review"
                elif status == "Action Required":
                    notes = "Updates needed by " + (datetime.now() + timedelta(days=np.random.randint(7, 30))).strftime("%Y-%m-%d")
                else:  # Non-Compliant
                    notes = "Immediate action required - potential penalties"
                
                data.append({
                    "Category": category,
                    "State": state,
                    "Status": status,
                    "Compliance Score": score,
                    "Risk Level": risk_level,
                    "Last Review": last_review.strftime("%Y-%m-%d"),
                    "Next Review": next_review.strftime("%Y-%m-%d"),
                    "Notes": notes
                })
    
    return pd.DataFrame(data)

def render_compliance_summary(compliance_data):
    """
    Render a summary of compliance status across categories and states.
    
    Args:
        compliance_data (pd.DataFrame): Compliance data
    """
    st.subheader("Compliance Summary")
    
    # Calculate overall compliance metrics
    total_requirements = len(compliance_data)
    compliant_count = len(compliance_data[compliance_data["Status"] == "Compliant"])
    pending_count = len(compliance_data[compliance_data["Status"] == "Pending Review"])
    action_required_count = len(compliance_data[compliance_data["Status"] == "Action Required"])
    non_compliant_count = len(compliance_data[compliance_data["Status"] == "Non-Compliant"])
    
    overall_compliance_rate = compliant_count / total_requirements
    
    # Create columns for key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Overall Compliance",
            value=f"{overall_compliance_rate:.1%}"
        )
    
    with col2:
        st.metric(
            label="Action Required",
            value=f"{action_required_count}",
            delta=f"{action_required_count / total_requirements:.1%}"
        )
    
    with col3:
        st.metric(
            label="Non-Compliant",
            value=f"{non_compliant_count}",
            delta=f"{non_compliant_count / total_requirements:.1%}",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            label="Pending Review",
            value=f"{pending_count}",
            delta=f"{pending_count / total_requirements:.1%}"
        )
    
    # Create compliance status by category chart
    status_by_category = compliance_data.groupby(["Category", "Status"]).size().reset_index(name="Count")
    
    st.subheader("Compliance Status by Category")
    fig = px.bar(
        status_by_category,
        x="Category",
        y="Count",
        color="Status",
        barmode="stack",
        color_discrete_map={
            "Compliant": COLORS["secondary"],  # Using secondary (green) instead of non-existent 'green'
            "Pending Review": COLORS["accent"],  # Using accent (amber) instead of non-existent 'yellow'
            "Action Required": COLORS["finance"],  # Using finance (orange) instead of non-existent 'orange'
            "Non-Compliant": COLORS["manufacturing"]  # Using manufacturing (pink) instead of non-existent 'red'
        }
    )
    
    # Update layout
    custom_layout = {
        "height": 400,
        "xaxis_title": "Compliance Category",
        "yaxis_title": "Number of Requirements",
        "legend_title": "Status",
        "title": "Compliance Status by Regulatory Category",
        "xaxis": {"tickangle": -45}
    }
    
    fig.update_layout(**merge_layout(custom_layout))
    st.plotly_chart(fig, use_container_width=True)
    
    # Add description
    st.markdown("""
    <p style='margin-top: 10px; margin-bottom: 20px;'>This chart shows the compliance status across different regulatory categories. 
    Each bar represents a category, with colors indicating the status of requirements within that category. 
    This visualization helps identify areas with compliance issues that may need immediate attention.</p>
    """, unsafe_allow_html=True)

def render_state_compliance_map(compliance_data):
    """
    Render a map of compliance status by state.
    
    Args:
        compliance_data (pd.DataFrame): Compliance data
    """
    st.subheader("State Compliance Map")
    
    # Calculate average compliance score by state
    state_compliance = compliance_data.groupby("State")["Compliance Score"].mean().reset_index()
    state_compliance["Compliance Score"] = state_compliance["Compliance Score"].round(2)
    
    # Create choropleth map
    st.subheader("Regulatory Compliance by State")
    fig = px.choropleth(
        state_compliance,
        locations="State",
        locationmode="USA-states",
        color="Compliance Score",
        scope="usa",
        color_continuous_scale=[[0, COLORS["manufacturing"]], [0.5, COLORS["accent"]], [1, COLORS["secondary"]]],
        range_color=[0, 1],
        labels={"Compliance Score": "Compliance Score"}
    )
    
    # Update layout
    custom_layout = {
        "height": 500,
        "geo": {
            "showlakes": True,
            "lakecolor": "rgb(255, 255, 255)"
        }
    }
    
    fig.update_layout(**merge_layout(custom_layout))
    st.plotly_chart(fig, use_container_width=True)
    
    # Add description
    st.markdown("""
    <p style='margin-top: 10px; margin-bottom: 20px;'>This map shows the average compliance score across all regulatory categories for each state. 
    Darker green indicates higher compliance, while yellow and red indicate areas of concern. 
    This visualization helps identify geographic regions where compliance efforts may need to be strengthened.</p>
    """, unsafe_allow_html=True)

def render_compliance_calendar(compliance_data):
    """
    Render a calendar of upcoming compliance reviews.
    
    Args:
        compliance_data (pd.DataFrame): Compliance data
    """
    st.subheader("Upcoming Compliance Reviews")
    
    # Convert date strings to datetime objects
    compliance_data["Next Review"] = pd.to_datetime(compliance_data["Next Review"])
    
    # Filter to upcoming reviews in the next 90 days
    today = pd.Timestamp.now().normalize()
    upcoming_reviews = compliance_data[compliance_data["Next Review"] <= (today + pd.Timedelta(days=90))].copy()
    upcoming_reviews = upcoming_reviews.sort_values("Next Review")
    
    # Format dates for display
    upcoming_reviews["Days Until Review"] = (upcoming_reviews["Next Review"] - today).dt.days
    upcoming_reviews["Next Review"] = upcoming_reviews["Next Review"].dt.strftime("%Y-%m-%d")
    
    # Add urgency indicator
    def get_urgency(days, status):
        if status == "Non-Compliant" or days <= 7:
            return "High"
        elif status == "Action Required" or days <= 30:
            return "Medium"
        else:
            return "Low"
    
    upcoming_reviews["Urgency"] = upcoming_reviews.apply(
        lambda x: get_urgency(x["Days Until Review"], x["Status"]), axis=1
    )
    
    # Display upcoming reviews
    if len(upcoming_reviews) > 0:
        # Create display dataframe
        display_reviews = upcoming_reviews[[
            "Category", "State", "Status", "Next Review", "Days Until Review", "Urgency"
        ]].copy()
        
        # Add color coding for urgency
        def highlight_urgency(val):
            if val == "High":
                return 'background-color: #FFCCCC'
            elif val == "Medium":
                return 'background-color: #FFFFCC'
            else:
                return 'background-color: #CCFFCC'
        
        # Display styled dataframe
        st.dataframe(
            display_reviews.style.applymap(highlight_urgency, subset=["Urgency"]),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No upcoming compliance reviews in the next 90 days.")

def render_compliance_details(compliance_data):
    """
    Render detailed compliance information with filtering options.
    
    Args:
        compliance_data (pd.DataFrame): Compliance data
    """
    st.subheader("Compliance Details")
    
    # Create filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Filter by category
        categories = ["All Categories"] + sorted(compliance_data["Category"].unique().tolist())
        selected_category = st.selectbox("Filter by Category", options=categories)
    
    with col2:
        # Filter by state
        states = ["All States"] + sorted(compliance_data["State"].unique().tolist())
        selected_state = st.selectbox("Filter by State", options=states)
    
    with col3:
        # Filter by status
        statuses = ["All Statuses"] + sorted(compliance_data["Status"].unique().tolist())
        selected_status = st.selectbox("Filter by Status", options=statuses)
    
    # Apply filters
    filtered_data = compliance_data.copy()
    
    if selected_category != "All Categories":
        filtered_data = filtered_data[filtered_data["Category"] == selected_category]
    
    if selected_state != "All States":
        filtered_data = filtered_data[filtered_data["State"] == selected_state]
    
    if selected_status != "All Statuses":
        filtered_data = filtered_data[filtered_data["Status"] == selected_status]
    
    # Display filtered data
    if len(filtered_data) > 0:
        # Create display dataframe
        display_data = filtered_data[[
            "Category", "State", "Status", "Risk Level", "Last Review", "Next Review", "Notes"
        ]].copy()
        
        # Add color coding for status
        def highlight_status(val):
            if val == "Non-Compliant":
                return 'background-color: #FFCCCC'
            elif val == "Action Required":
                return 'background-color: #FFDDAA'
            elif val == "Pending Review":
                return 'background-color: #FFFFCC'
            else:
                return 'background-color: #CCFFCC'
        
        # Display styled dataframe
        st.dataframe(
            display_data.style.applymap(highlight_status, subset=["Status"]),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No compliance requirements match the selected filters.")

def render_risk_heatmap(compliance_data):
    """
    Render a heatmap of compliance risk by category and state.
    
    Args:
        compliance_data (pd.DataFrame): Compliance data
    """
    st.subheader("Compliance Risk Heatmap")
    
    # Calculate average compliance score by category and state
    heatmap_data = compliance_data.pivot_table(
        index="Category",
        columns="State",
        values="Compliance Score",
        aggfunc="mean"
    ).round(2)
    
    # Create heatmap
    fig = px.imshow(
        heatmap_data,
        color_continuous_scale=[[0, COLORS["manufacturing"]], [0.5, COLORS["accent"]], [1, COLORS["secondary"]]],
        labels=dict(x="State", y="Category", color="Compliance Score"),
        zmin=0,
        zmax=1,
        aspect="auto"
    )
    
    # Add text annotations
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            value = heatmap_data.iloc[i, j]
            if not np.isnan(value):
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=f"{value:.2f}",
                    showarrow=False,
                    font=dict(
                        color="black" if 0.3 < value < 0.8 else "white"
                    )
                )
    
    # Update layout
    custom_layout = {
        "height": 600,
        "title": "Compliance Risk Heatmap by Category and State",
        "xaxis": {"tickangle": -45}
    }
    
    fig.update_layout(**merge_layout(custom_layout))
    st.plotly_chart(fig, use_container_width=True)
    
    # Add description
    st.markdown("""
    <p style='margin-top: 10px; margin-bottom: 20px;'>This heatmap shows compliance risk levels across categories and states. 
    Darker green indicates higher compliance (lower risk), while yellow and red indicate areas of concern. 
    This visualization helps identify specific combinations of regulatory categories and states that may require focused attention.</p>
    """, unsafe_allow_html=True)

def render_regulatory_compliance_page():
    """Render the Regulatory Compliance page with interactive compliance tracking."""
    st.markdown('<h2 class="section-header">Regulatory Compliance</h2>', unsafe_allow_html=True)
    
    # Display page description
    st.markdown("""
    <p style='margin-bottom: 20px;'>This tool helps track and manage compliance with insurance regulations across different states, 
    providing insights into compliance status, upcoming reviews, and areas of regulatory risk.</p>
    """, unsafe_allow_html=True)
    
    # Generate sample compliance data
    compliance_data = generate_compliance_data()
    
    # Create tabs for different compliance views
    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "State Map", "Calendar", "Details"])
    
    with tab1:
        # Render compliance summary
        render_compliance_summary(compliance_data)
        
        # Render risk heatmap
        render_risk_heatmap(compliance_data)
    
    with tab2:
        # Render state compliance map
        render_state_compliance_map(compliance_data)
    
    with tab3:
        # Render compliance calendar
        render_compliance_calendar(compliance_data)
    
    with tab4:
        # Render compliance details
        render_compliance_details(compliance_data)
