"""
Chart Generator - Generate matplotlib charts for Churn Analysis
===============================================================
Creates visualizations for:
- Churn distribution
- Feature importance
- Risk analysis
- Prediction results
"""

import logging
import io
import base64
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..analytics.data_handler import get_churn_stats, get_churn_by_category
from ..ml.predictor import churn_predictor

logger = logging.getLogger(__name__)

# Style settings
plt.style.use('default')
COLORS = {
    'primary': '#4CAF50',
    'secondary': '#2196F3', 
    'warning': '#FF9800',
    'danger': '#F44336',
    'info': '#00BCD4',
    'churn': '#E53935',
    'no_churn': '#43A047',
    'high_risk': '#D32F2F',
    'medium_risk': '#FFA000',
    'low_risk': '#388E3C'
}


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"


def _generate_empty_chart(message: str) -> str:
    """Generate a placeholder chart with message"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=14, 
            transform=ax.transAxes, color='gray')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return _fig_to_base64(fig)


def generate_churn_distribution_chart() -> Dict:
    """Generate pie chart showing churn vs no-churn distribution"""
    try:
        # Get stats from database
        stats_result = get_churn_stats()
        if not stats_result.get('success'):
            return {"success": False, "error": "Could not get churn stats"}
        
        stats = stats_result.get('stats', {})
        total = stats.get('total_customers', 0)
        
        if total == 0:
            return {"success": True, "chart": _generate_empty_chart("No Customer Data Available")}
        
        # Parse churn rate to get churned count
        churn_rate_str = stats.get('churn_rate', '0%')
        churn_rate = float(churn_rate_str.replace('%', '')) / 100
        churned = int(total * churn_rate)
        retained = total - churned
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart
        sizes = [retained, churned]
        labels = [f'Retained\n({retained:,})', f'Churned\n({churned:,})']
        colors = [COLORS['no_churn'], COLORS['churn']]
        explode = (0, 0.05)
        
        wedges, texts, autotexts = ax1.pie(
            sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90,
            textprops={'fontsize': 11}
        )
        ax1.set_title('Customer Churn Distribution', fontsize=14, fontweight='bold')
        
        # Summary stats bar
        metrics = ['Total\nCustomers', 'Churned', 'Retained']
        values = [total, churned, retained]
        
        bars = ax2.bar(metrics, values, color=[COLORS['info'], COLORS['churn'], COLORS['no_churn']])
        
        for bar, val in zip(bars, values):
            ax2.annotate(f'{val:,.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=12)
        
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title(f'Summary (Churn Rate: {churn_rate*100:.1f}%)', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return {"success": True, "chart": _fig_to_base64(fig)}
    
    except Exception as e:
        logger.error(f"Churn distribution chart error: {e}")
        return {"success": False, "error": str(e)}


def generate_churn_by_category_chart() -> Dict:
    """Generate bar chart showing churn rate by category"""
    try:
        # Get category data from database
        cat_result = get_churn_by_category()
        if not cat_result.get('success'):
            return {"success": False, "error": "Could not get category data"}
        
        # Find available category columns from results
        available_keys = [k for k in cat_result.keys() if k.startswith('by_')]
        
        if len(available_keys) == 0:
            return {"success": True, "chart": _generate_empty_chart("No category data available. Upload churn data first.")}
        
        # Create subplots based on available categories (max 3)
        num_charts = min(len(available_keys), 3)
        fig, axes = plt.subplots(1, num_charts, figsize=(6 * num_charts, 5))
        
        # Handle single chart case
        if num_charts == 1:
            axes = [axes]
        
        for i, key in enumerate(available_keys[:num_charts]):
            ax = axes[i]
            cat_data = cat_result.get(key, [])
            
            # Extract column name from key (e.g., "by_contract" -> "contract")
            col_name = key[3:]  # Remove "by_" prefix
            title = f"Churn by {col_name.replace('_', ' ').title()}"
            
            if not cat_data:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title, fontsize=11)
                continue
            
            # Get labels and values
            labels = [str(d.get('category', d.get(col_name, 'Unknown')))[:15] for d in cat_data]
            
            # Check if we have churn_rate data
            has_churn_rate = any(d.get('churn_rate') is not None for d in cat_data)
            
            if has_churn_rate:
                rates = [float(d.get('churn_rate', 0) or 0) for d in cat_data]
                totals = [d.get('total', d.get('count', 0)) for d in cat_data]
                
                # Color based on churn rate
                colors = [COLORS['high_risk'] if r >= 30 else (COLORS['medium_risk'] if r >= 20 else COLORS['low_risk']) 
                         for r in rates]
                
                bars = ax.bar(labels, rates, color=colors, alpha=0.8)
                
                for bar, rate, total in zip(bars, rates, totals):
                    ax.annotate(f'{rate:.1f}%\n(n={total})', 
                               xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                               xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
                
                ax.set_ylabel('Churn Rate (%)', fontsize=10)
            else:
                # No churn rate - just show counts
                counts = [d.get('total', d.get('count', 0)) for d in cat_data]
                bars = ax.bar(labels, counts, color=COLORS['secondary'], alpha=0.8)
                
                for bar, count in zip(bars, counts):
                    ax.annotate(f'{count:,}', 
                               xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                               xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
                
                ax.set_ylabel('Count', fontsize=10)
            
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return {"success": True, "chart": _fig_to_base64(fig)}
    
    except Exception as e:
        logger.error(f"Category chart error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def generate_feature_importance_chart() -> Dict:
    """Generate horizontal bar chart of feature importance"""
    try:
        # Get feature importance from trained model
        if not churn_predictor.is_trained:
            return {"success": False, "error": "Model not trained yet"}
        
        feature_importance = churn_predictor.feature_importance
        
        if not feature_importance:
            return {"success": True, "chart": _generate_empty_chart("No Feature Importance Data - Train Model First")}
        
        # Take top 15 features
        features = list(feature_importance.keys())[:15]
        importance = list(feature_importance.values())[:15]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Reverse for better display (most important at top)
        features = features[::-1]
        importance = importance[::-1]
        
        # Color gradient
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(features)))
        
        bars = ax.barh(features, importance, color=colors)
        
        # Add value labels
        for bar, val in zip(bars, importance):
            ax.annotate(f'{val:.3f}', xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                       xytext=(3, 0), textcoords="offset points", ha='left', va='center', fontsize=10)
        
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title('Top Features for Churn Prediction', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Clean up feature names for display
        clean_labels = [f.replace('_encoded', '').replace('_', ' ').title() for f in features]
        ax.set_yticklabels(clean_labels, fontsize=10)
        
        plt.tight_layout()
        return {"success": True, "chart": _fig_to_base64(fig)}
    
    except Exception as e:
        logger.error(f"Feature importance chart error: {e}")
        return {"success": False, "error": str(e)}


def generate_risk_distribution_chart() -> Dict:
    """Generate chart showing distribution of predicted risk levels"""
    try:
        # Get batch predictions from model
        if not churn_predictor.is_trained:
            return {"success": False, "error": "Model not trained yet"}
        
        batch_result = churn_predictor.predict_batch(limit=500)
        if not batch_result.get('success'):
            return {"success": False, "error": "Could not get predictions"}
        
        predictions = batch_result.get('predictions', [])
        
        if not predictions:
            return {"success": True, "chart": _generate_empty_chart("No Predictions Available")}
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Count risk levels
        high = sum(1 for p in predictions if p.get('risk_level') == 'HIGH')
        medium = sum(1 for p in predictions if p.get('risk_level') == 'MEDIUM')
        low = sum(1 for p in predictions if p.get('risk_level') == 'LOW')
        
        # Pie chart of risk distribution
        sizes = [high, medium, low]
        labels = [f'High Risk\n({high})', f'Medium Risk\n({medium})', f'Low Risk\n({low})']
        colors = [COLORS['high_risk'], COLORS['medium_risk'], COLORS['low_risk']]
        
        wedges, texts, autotexts = ax1.pie(
            sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 11}
        )
        ax1.set_title('Risk Level Distribution', fontsize=14, fontweight='bold')
        
        # Histogram of churn probabilities
        probs = [p.get('churn_probability', 0) for p in predictions]
        
        ax2.hist(probs, bins=20, color=COLORS['secondary'], edgecolor='white', alpha=0.7)
        ax2.axvline(x=0.4, color=COLORS['medium_risk'], linestyle='--', label='Medium threshold')
        ax2.axvline(x=0.7, color=COLORS['high_risk'], linestyle='--', label='High threshold')
        
        ax2.set_xlabel('Churn Probability', fontsize=12)
        ax2.set_ylabel('Number of Customers', fontsize=12)
        ax2.set_title('Distribution of Churn Probabilities', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return {"success": True, "chart": _fig_to_base64(fig)}
    
    except Exception as e:
        logger.error(f"Risk distribution chart error: {e}")
        return {"success": False, "error": str(e)}


def generate_prediction_comparison_chart() -> Dict:
    """Generate chart comparing predicted vs actual churn"""
    try:
        # Get batch predictions from model
        if not churn_predictor.is_trained:
            return {"success": False, "error": "Model not trained yet"}
        
        batch_result = churn_predictor.predict_batch(limit=500)
        if not batch_result.get('success'):
            return {"success": False, "error": "Could not get predictions"}
        
        predictions = batch_result.get('predictions', [])
        
        if not predictions:
            return {"success": True, "chart": _generate_empty_chart("No Predictions Available")}
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Confusion matrix style
        actual_churn = [p.get('actual_churn', 0) for p in predictions]
        predicted_churn = [1 if p.get('churn_probability', 0) >= 0.5 else 0 for p in predictions]
        
        # Calculate confusion matrix values
        tp = sum(1 for a, p in zip(actual_churn, predicted_churn) if a == 1 and p == 1)
        tn = sum(1 for a, p in zip(actual_churn, predicted_churn) if a == 0 and p == 0)
        fp = sum(1 for a, p in zip(actual_churn, predicted_churn) if a == 0 and p == 1)
        fn = sum(1 for a, p in zip(actual_churn, predicted_churn) if a == 1 and p == 0)
        
        # Confusion matrix heatmap
        matrix = np.array([[tn, fp], [fn, tp]])
        im = ax1.imshow(matrix, cmap='Blues')
        
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.set_xticklabels(['Predicted: No', 'Predicted: Yes'])
        ax1.set_yticklabels(['Actual: No', 'Actual: Yes'])
        
        for i in range(2):
            for j in range(2):
                ax1.text(j, i, f'{matrix[i, j]}', ha='center', va='center', 
                        fontsize=16, color='white' if matrix[i, j] > matrix.max()/2 else 'black')
        
        ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        # Metrics bar chart
        total = len(predictions)
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        values = [accuracy, precision, recall, f1]
        
        bars = ax2.bar(metrics, values, color=[COLORS['primary'], COLORS['secondary'], COLORS['info'], COLORS['warning']])
        
        for bar, val in zip(bars, values):
            ax2.annotate(f'{val:.1%}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=12)
        
        ax2.set_ylim(0, 1.1)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return {"success": True, "chart": _fig_to_base64(fig)}
    
    except Exception as e:
        logger.error(f"Prediction comparison chart error: {e}")
        return {"success": False, "error": str(e)}
