"""
Model Performance module for the Insurance Underwriting Intelligence Dashboard.
Contains functions for generating model performance visualizations.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from theme import COLORS, PLOT_LAYOUT, styled_container, merge_layout

def render_model_metrics(metrics):
    """Render model performance metrics in a styled card layout."""
    st.markdown('<h3 class="section-header">Model Performance Metrics</h3>', unsafe_allow_html=True)
    
    def display_metrics():
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.2f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.2f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.2f}")
        with col4:
            st.metric("F1 Score", f"{metrics['f1']:.2f}")
        with col5:
            st.metric("ROC AUC", f"{metrics['roc_auc']:.2f}")
    
    # Use the styled container for consistent appearance
    styled_container(display_metrics)

def render_feature_importance(feature_importance):
    """Render a bar chart showing feature importance."""
    st.markdown('<h3 class="section-header">Feature Importance</h3>', unsafe_allow_html=True)
    
    def plot_feature_importance():
        # Sort features by importance
        sorted_features = feature_importance.sort_values('Importance', ascending=True)
        
        fig = px.bar(
            sorted_features,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance',
            color_discrete_sequence=[COLORS['manufacturing']],  # Use pink instead of blue
            text='Importance'
        )
        
        # Update layout with consistent theming
        custom_layout = {
            'height': 400,
            'yaxis_title': None,
            'xaxis_title': "Importance Score",
            'margin': dict(l=150)  # Add more space for feature names
        }
        fig.update_layout(**merge_layout(custom_layout))
        
        # Format the text to show percentages with 1 decimal place
        fig.update_traces(
            texttemplate='%{text:.3f}',
            textposition='outside'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Use the styled container for consistent appearance
    styled_container(plot_feature_importance)

def render_roc_curve(metrics):
    """Render a ROC curve visualization."""
    st.markdown('<h3 class="section-header">ROC Curve</h3>', unsafe_allow_html=True)
    
    def plot_roc_curve():
        # Simulate ROC curve data
        fpr = np.linspace(0, 1, 100)
        tpr = 1 - np.exp(-5 * fpr)  # Simulated curve
        
        # Create a DataFrame for the ROC curve
        roc_df = pd.DataFrame({
            'False Positive Rate': fpr,
            'True Positive Rate': tpr
        })
        
        # Add the random classifier line data
        random_df = pd.DataFrame({
            'False Positive Rate': [0, 1],
            'True Positive Rate': [0, 1]
        })
        
        # Plot the ROC curve
        fig = px.line(
            roc_df,
            x='False Positive Rate',
            y='True Positive Rate',
            title=f'ROC Curve (AUC = {metrics["roc_auc"]:.2f})',
            line_shape='spline',  # Smooth curve
            color_discrete_sequence=[COLORS['healthcare']]  # Use cyan instead of blue
        )
        
        # Add the random classifier line (diagonal)
        fig.add_trace(
            go.Scatter(
                x=random_df['False Positive Rate'],
                y=random_df['True Positive Rate'],
                mode='lines',
                line=dict(color=COLORS['dark_gray'], dash='dash'),
                name='Random Classifier',
                showlegend=True
            )
        )
        
        # Update layout with consistent theming
        custom_layout = {
            'height': 400,
            'xaxis_title': "False Positive Rate",
            'yaxis_title': "True Positive Rate",
            'xaxis': dict(range=[0, 1], constrain='domain'),
            'yaxis': dict(range=[0, 1], scaleanchor="x", scaleratio=1),
            'legend': dict(
                yanchor="bottom",
                y=0.01,
                xanchor="right",
                x=0.99,
                font=dict(color='#0F172A')
            ),
            'title': {"text": "ROC Curve"}
        }
        fig.update_layout(**merge_layout(custom_layout))
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Use the styled container for consistent appearance
    styled_container(plot_roc_curve)

def render_confusion_matrix():
    """Render a confusion matrix visualization."""
    st.markdown('<h3 class="section-header">Confusion Matrix</h3>', unsafe_allow_html=True)
    
    def plot_confusion_matrix():
        # Simulate confusion matrix
        tn, fp = 70, 10
        fn, tp = 15, 55
        
        # Create the confusion matrix figure
        z = [[tp, fp], [fn, tn]]
        x = ['Predicted Positive', 'Predicted Negative']
        y = ['Actual Positive', 'Actual Negative']
        
        # Create annotation text
        annotations = []
        for i in range(2):
            for j in range(2):
                annotations.append(
                    dict(
                        showarrow=False,
                        text=f"{z[i][j]}",
                        font=dict(size=16, color='white'),
                        x=x[j],
                        y=y[i],
                        xref='x',
                        yref='y'
                    )
                )
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=[[0, COLORS['retail']], [1, COLORS['finance']]],  # Use purple and orange
            showscale=False
        ))
        
        # Add annotations
        custom_layout = {
            'annotations': annotations,
            'height': 400,
            'xaxis_title': None,
            'yaxis_title': None,
            'title': {'text': 'Confusion Matrix', 'font': {'size': 20, 'color': '#0F172A'}}
        }
        
        # Use merge_layout to properly combine layouts without duplicate keys
        fig.update_layout(**merge_layout(custom_layout))
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Use the styled container for consistent appearance
    styled_container(plot_confusion_matrix)

def render_model_performance_page(model_performance):
    """Render the complete Model Performance page."""
    st.markdown('<h2 class="section-header">Model Performance</h2>', unsafe_allow_html=True)
    
    # Add an overview of the model performance page
    st.markdown("""
    <p style='margin-bottom: 20px;'>This page presents key performance metrics and visualizations for our insurance risk prediction model. 
    The model was trained on historical property data to predict claim likelihood and risk scores.</p>
    """, unsafe_allow_html=True)
    
    metrics = model_performance['metrics']
    
    # Render model metrics
    render_model_metrics(metrics)
    
    # Render feature importance
    render_feature_importance(model_performance['feature_importance'])
    
    # Add description for feature importance after the chart
    st.markdown("""
    <p style='margin-top: 10px; margin-bottom: 20px;'>This chart shows the relative importance of each feature in the model's predictions. 
    Features with higher importance have a greater impact on the model's decision-making process. Understanding feature importance helps identify the key risk factors that drive claim predictions.</p>
    """, unsafe_allow_html=True)
    
    # Render ROC curve
    render_roc_curve(metrics)
    
    # Add description for ROC curve after the chart
    st.markdown("""
    <p style='margin-top: 10px; margin-bottom: 20px;'>The Receiver Operating Characteristic (ROC) curve illustrates the model's ability to discriminate between properties that will have claims and those that won't. 
    The Area Under the Curve (AUC) of 0.82 indicates strong predictive performance, where 1.0 would be perfect prediction and 0.5 would be random guessing.</p>
    """, unsafe_allow_html=True)
    
    # Render confusion matrix
    render_confusion_matrix()
    
    # Add description for confusion matrix after the chart
    st.markdown("""
    <p style='margin-top: 10px; margin-bottom: 20px;'>This visualization shows the model's prediction accuracy across different outcomes. 
    True Positives and True Negatives represent correct predictions, while False Positives (Type I errors) and False Negatives (Type II errors) represent incorrect predictions. 
    This helps underwriters understand the model's strengths and limitations in different scenarios.</p>
    """, unsafe_allow_html=True)
