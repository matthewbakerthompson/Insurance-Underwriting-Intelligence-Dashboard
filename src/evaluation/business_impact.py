"""
Business Impact Evaluation Module

This module provides functionality for assessing the business impact
of predictive models and data-driven insights in insurance underwriting.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.metrics import confusion_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BusinessImpactEvaluator:
    """Class for evaluating the business impact of predictive models in underwriting."""
    
    def __init__(self):
        """Initialize the business impact evaluator."""
        self.results = {}
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Loaded data as DataFrame
        """
        _, ext = os.path.splitext(file_path)
        
        if ext.lower() == '.csv':
            return pd.read_csv(file_path)
        elif ext.lower() == '.json':
            return pd.read_json(file_path)
        elif ext.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def calculate_cost_benefit(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        cost_matrix: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate cost-benefit analysis based on model predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            cost_matrix: Dictionary with costs/benefits for TP, TN, FP, FN
                Example: {'tp_benefit': 1000, 'tn_benefit': 100, 'fp_cost': 500, 'fn_cost': 2000}
            
        Returns:
            Dictionary with cost-benefit metrics
        """
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate costs and benefits
        tp_benefit = tp * cost_matrix.get('tp_benefit', 0)
        tn_benefit = tn * cost_matrix.get('tn_benefit', 0)
        fp_cost = fp * cost_matrix.get('fp_cost', 0)
        fn_cost = fn * cost_matrix.get('fn_cost', 0)
        
        # Calculate total impact
        total_benefit = tp_benefit + tn_benefit
        total_cost = fp_cost + fn_cost
        net_impact = total_benefit - total_cost
        roi = (net_impact / total_cost) if total_cost > 0 else float('inf')
        
        results = {
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'tp_benefit': tp_benefit,
            'tn_benefit': tn_benefit,
            'fp_cost': fp_cost,
            'fn_cost': fn_cost,
            'total_benefit': total_benefit,
            'total_cost': total_cost,
            'net_impact': net_impact,
            'roi': roi
        }
        
        self.results['cost_benefit'] = results
        
        logger.info(f"Cost-benefit analysis completed. Net impact: {net_impact:.2f}, ROI: {roi:.2f}")
        
        return results
    
    def calculate_efficiency_gains(
        self, 
        manual_time_per_case: float,
        automated_time_per_case: float,
        num_cases: int,
        hourly_cost: float
    ) -> Dict[str, float]:
        """
        Calculate efficiency gains from automation.
        
        Args:
            manual_time_per_case: Time (in hours) for manual processing per case
            automated_time_per_case: Time (in hours) for automated processing per case
            num_cases: Number of cases processed
            hourly_cost: Hourly cost of underwriter time
            
        Returns:
            Dictionary with efficiency metrics
        """
        # Calculate time savings
        manual_total_time = manual_time_per_case * num_cases
        automated_total_time = automated_time_per_case * num_cases
        time_saved = manual_total_time - automated_total_time
        
        # Calculate cost savings
        manual_total_cost = manual_total_time * hourly_cost
        automated_total_cost = automated_total_time * hourly_cost
        cost_saved = manual_total_cost - automated_total_cost
        
        # Calculate efficiency improvement
        efficiency_improvement = (time_saved / manual_total_time) * 100
        
        results = {
            'manual_total_time': manual_total_time,
            'automated_total_time': automated_total_time,
            'time_saved': time_saved,
            'manual_total_cost': manual_total_cost,
            'automated_total_cost': automated_total_cost,
            'cost_saved': cost_saved,
            'efficiency_improvement': efficiency_improvement
        }
        
        self.results['efficiency'] = results
        
        logger.info(f"Efficiency analysis completed. Time saved: {time_saved:.2f} hours, Cost saved: ${cost_saved:.2f}")
        
        return results
    
    def calculate_risk_reduction(
        self, 
        baseline_loss_ratio: float,
        predicted_loss_ratio: float,
        total_premium: float
    ) -> Dict[str, float]:
        """
        Calculate risk reduction impact.
        
        Args:
            baseline_loss_ratio: Historical loss ratio without model
            predicted_loss_ratio: Expected loss ratio with model
            total_premium: Total premium amount
            
        Returns:
            Dictionary with risk reduction metrics
        """
        # Calculate loss amounts
        baseline_loss = baseline_loss_ratio * total_premium
        predicted_loss = predicted_loss_ratio * total_premium
        
        # Calculate risk reduction
        loss_reduction = baseline_loss - predicted_loss
        loss_reduction_percent = (loss_reduction / baseline_loss) * 100
        
        results = {
            'baseline_loss_ratio': baseline_loss_ratio,
            'predicted_loss_ratio': predicted_loss_ratio,
            'total_premium': total_premium,
            'baseline_loss': baseline_loss,
            'predicted_loss': predicted_loss,
            'loss_reduction': loss_reduction,
            'loss_reduction_percent': loss_reduction_percent
        }
        
        self.results['risk_reduction'] = results
        
        logger.info(f"Risk reduction analysis completed. Loss reduction: ${loss_reduction:.2f} ({loss_reduction_percent:.2f}%)")
        
        return results
    
    def calculate_portfolio_optimization(
        self, 
        df: pd.DataFrame,
        risk_score_col: str,
        premium_col: str,
        loss_col: str,
        risk_threshold: float
    ) -> Dict[str, Any]:
        """
        Calculate portfolio optimization impact.
        
        Args:
            df: DataFrame with policy data
            risk_score_col: Column name for risk scores
            premium_col: Column name for premium amounts
            loss_col: Column name for loss amounts
            risk_threshold: Threshold for high-risk policies
            
        Returns:
            Dictionary with portfolio optimization metrics
        """
        # Identify high-risk policies
        high_risk = df[risk_score_col] >= risk_threshold
        
        # Calculate metrics for current portfolio
        current_policies = len(df)
        current_premium = df[premium_col].sum()
        current_loss = df[loss_col].sum()
        current_loss_ratio = current_loss / current_premium if current_premium > 0 else 0
        
        # Calculate metrics for optimized portfolio (excluding high-risk)
        optimized_policies = len(df[~high_risk])
        optimized_premium = df.loc[~high_risk, premium_col].sum()
        optimized_loss = df.loc[~high_risk, loss_col].sum()
        optimized_loss_ratio = optimized_loss / optimized_premium if optimized_premium > 0 else 0
        
        # Calculate impact
        policy_reduction = current_policies - optimized_policies
        premium_reduction = current_premium - optimized_premium
        loss_reduction = current_loss - optimized_loss
        loss_ratio_improvement = current_loss_ratio - optimized_loss_ratio
        
        results = {
            'current_policies': current_policies,
            'current_premium': current_premium,
            'current_loss': current_loss,
            'current_loss_ratio': current_loss_ratio,
            'optimized_policies': optimized_policies,
            'optimized_premium': optimized_premium,
            'optimized_loss': optimized_loss,
            'optimized_loss_ratio': optimized_loss_ratio,
            'policy_reduction': policy_reduction,
            'policy_reduction_percent': (policy_reduction / current_policies) * 100,
            'premium_reduction': premium_reduction,
            'premium_reduction_percent': (premium_reduction / current_premium) * 100,
            'loss_reduction': loss_reduction,
            'loss_reduction_percent': (loss_reduction / current_loss) * 100,
            'loss_ratio_improvement': loss_ratio_improvement
        }
        
        self.results['portfolio_optimization'] = results
        
        logger.info(f"Portfolio optimization analysis completed. Loss ratio improvement: {loss_ratio_improvement:.4f}")
        
        return results
    
    def plot_cost_benefit_analysis(self, output_path: str) -> None:
        """
        Plot cost-benefit analysis and save to file.
        
        Args:
            output_path: Path to save the plot
        """
        if 'cost_benefit' not in self.results:
            logger.warning("Cost-benefit analysis not performed")
            return
        
        results = self.results['cost_benefit']
        
        # Create data for plotting
        categories = ['Benefits', 'Costs']
        values = [results['total_benefit'], results['total_cost']]
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(categories, values)
        
        # Add net impact
        plt.axhline(y=results['net_impact'], color='r', linestyle='-', label='Net Impact')
        
        # Add labels and title
        plt.xlabel('Category')
        plt.ylabel('Amount')
        plt.title('Cost-Benefit Analysis')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}',
                    ha='center', va='bottom')
        
        plt.legend()
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the plot
        plt.savefig(output_path)
        logger.info(f"Cost-benefit analysis plot saved to {output_path}")
    
    def plot_efficiency_gains(self, output_path: str) -> None:
        """
        Plot efficiency gains and save to file.
        
        Args:
            output_path: Path to save the plot
        """
        if 'efficiency' not in self.results:
            logger.warning("Efficiency analysis not performed")
            return
        
        results = self.results['efficiency']
        
        # Create data for plotting
        categories = ['Manual', 'Automated']
        time_values = [results['manual_total_time'], results['automated_total_time']]
        cost_values = [results['manual_total_cost'], results['automated_total_cost']]
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Time plot
        bars1 = ax1.bar(categories, time_values)
        ax1.set_xlabel('Process')
        ax1.set_ylabel('Time (hours)')
        ax1.set_title('Time Comparison')
        
        # Cost plot
        bars2 = ax2.bar(categories, cost_values)
        ax2.set_xlabel('Process')
        ax2.set_ylabel('Cost ($)')
        ax2.set_title('Cost Comparison')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:,.0f} hrs',
                    ha='center', va='bottom')
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the plot
        plt.savefig(output_path)
        logger.info(f"Efficiency gains plot saved to {output_path}")
    
    def plot_risk_reduction(self, output_path: str) -> None:
        """
        Plot risk reduction impact and save to file.
        
        Args:
            output_path: Path to save the plot
        """
        if 'risk_reduction' not in self.results:
            logger.warning("Risk reduction analysis not performed")
            return
        
        results = self.results['risk_reduction']
        
        # Create data for plotting
        categories = ['Baseline', 'With Model']
        loss_values = [results['baseline_loss'], results['predicted_loss']]
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(categories, loss_values)
        
        # Add labels and title
        plt.xlabel('Scenario')
        plt.ylabel('Loss Amount ($)')
        plt.title('Loss Reduction Impact')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}',
                    ha='center', va='bottom')
        
        # Add reduction arrow
        plt.annotate(
            f'${results["loss_reduction"]:,.0f} reduction\n({results["loss_reduction_percent"]:.1f}%)',
            xy=(0.5, min(loss_values) + (results['loss_reduction'] / 2)),
            xytext=(0.5, min(loss_values) + (results['loss_reduction'] / 2)),
            arrowprops=dict(arrowstyle='<->', color='red'),
            ha='center'
        )
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the plot
        plt.savefig(output_path)
        logger.info(f"Risk reduction plot saved to {output_path}")
    
    def generate_impact_report(self, output_path: str) -> None:
        """
        Generate a comprehensive impact report and save to file.
        
        Args:
            output_path: Path to save the report
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate report content
        report = {
            'title': 'Business Impact Assessment Report',
            'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'results': self.results,
            'summary': {}
        }
        
        # Add summary metrics
        if 'cost_benefit' in self.results:
            report['summary']['net_impact'] = self.results['cost_benefit']['net_impact']
            report['summary']['roi'] = self.results['cost_benefit']['roi']
        
        if 'efficiency' in self.results:
            report['summary']['efficiency_improvement'] = self.results['efficiency']['efficiency_improvement']
            report['summary']['cost_saved'] = self.results['efficiency']['cost_saved']
        
        if 'risk_reduction' in self.results:
            report['summary']['loss_reduction'] = self.results['risk_reduction']['loss_reduction']
            report['summary']['loss_reduction_percent'] = self.results['risk_reduction']['loss_reduction_percent']
        
        # Save the report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Impact report saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    evaluator = BusinessImpactEvaluator()
    
    # Example cost-benefit analysis
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 0])
    
    cost_matrix = {
        'tp_benefit': 5000,  # Benefit from correctly identifying a claim
        'tn_benefit': 500,   # Benefit from correctly identifying a non-claim
        'fp_cost': 1000,     # Cost of falsely predicting a claim
        'fn_cost': 8000      # Cost of missing a claim
    }
    
    evaluator.calculate_cost_benefit(y_true, y_pred, cost_matrix)
    evaluator.plot_cost_benefit_analysis('../reports/figures/cost_benefit_analysis.png')
    
    # Example efficiency gains
    evaluator.calculate_efficiency_gains(
        manual_time_per_case=2.5,      # hours
        automated_time_per_case=0.5,   # hours
        num_cases=1000,
        hourly_cost=75                 # dollars
    )
    evaluator.plot_efficiency_gains('../reports/figures/efficiency_gains.png')
    
    # Example risk reduction
    evaluator.calculate_risk_reduction(
        baseline_loss_ratio=0.65,
        predicted_loss_ratio=0.58,
        total_premium=10000000
    )
    evaluator.plot_risk_reduction('../reports/figures/risk_reduction.png')
    
    # Generate impact report
    evaluator.generate_impact_report('../reports/business_impact_report.json')
