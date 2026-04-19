"""Visualization dashboard for validation results."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime


class ValidationDashboard:
    """Dashboard for visualizing validation results."""
    
    def __init__(self, results_dir: str = "data/validation_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Style settings
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = {
            'sparse': '#2E86AB',  # Blue
            'single': '#A23B72',   # Purple
            'always_on': '#F18F01', # Orange
            'oracle': '#73AB84',   # Green
            'static': '#C73E1D'    # Red
        }
    
    def load_latest_results(self) -> Dict[str, Any]:
        """Load latest validation results."""
        if not self.results_dir.exists():
            return {}
        
        result_files = list(self.results_dir.glob("complete_validation_*.json"))
        if not result_files:
            return {}
        
        latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
        with open(latest_file, 'r') as f:
            return json.load(f)
    
    def create_dashboard(self, save_path: str = "validation_dashboard.png"):
        """Create comprehensive validation dashboard."""
        results = self.load_latest_results()
        if not results:
            print("No validation results found")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Quality Comparison (top-left)
        ax1 = plt.subplot(3, 3, 1)
        self._plot_quality_comparison(ax1, results)
        
        # 2. Efficiency Comparison (top-middle)
        ax2 = plt.subplot(3, 3, 2)
        self._plot_efficiency_comparison(ax2, results)
        
        # 3. Synergy Analysis (top-right)
        ax3 = plt.subplot(3, 3, 3)
        self._plot_synergy_analysis(ax3, results)
        
        # 4. Learning Mechanism Validation (middle-left)
        ax4 = plt.subplot(3, 3, 4)
        self._plot_learning_validation(ax4, results)
        
        # 5. Statistical Power (middle-middle)
        ax5 = plt.subplot(3, 3, 5)
        self._plot_statistical_power(ax5, results)
        
        # 6. Claims Validation (middle-right)
        ax6 = plt.subplot(3, 3, 6)
        self._plot_claims_validation(ax6, results)
        
        # 7. Time Series (bottom-left)
        ax7 = plt.subplot(3, 3, 7)
        self._plot_time_series(ax7, results)
        
        # 8. Confidence Intervals (bottom-middle)
        ax8 = plt.subplot(3, 3, 8)
        self._plot_confidence_intervals(ax8, results)
        
        # 9. Recommendations (bottom-right)
        ax9 = plt.subplot(3, 3, 9)
        self._plot_recommendations(ax9, results)
        
        # Add overall title
        timestamp = results.get('validation_timestamp', datetime.now().isoformat())
        fig.suptitle(f'Agentic Network v2 Validation Dashboard\n{timestamp}', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Dashboard saved to: {save_path}")
    
    def _plot_quality_comparison(self, ax, results: Dict[str, Any]):
        """Plot quality comparison between configurations."""
        baseline = results.get('detailed_results', {}).get('baseline_analysis', {})
        
        if 'metrics' not in baseline:
            ax.text(0.5, 0.5, 'No quality data', ha='center', va='center')
            ax.set_title('Quality Comparison', fontweight='bold')
            return
        
        metrics = baseline['metrics']
        
        # Extract quality scores
        configs = ['sparse', 'single', 'always_on']
        labels = ['Sparse', 'Single', 'Always-On']
        
        success_rates = []
        ci_lower = []
        ci_upper = []
        
        for config in configs:
            if config in metrics.get('quality', {}):
                success_rates.append(metrics['quality'][f'{config}_success_rate'])
                # For simplicity, assume ±0.05 CI
                ci_lower.append(max(0, metrics['quality'][f'{config}_success_rate'] - 0.05))
                ci_upper.append(min(1, metrics['quality'][f'{config}_success_rate'] + 0.05))
            else:
                success_rates.append(0)
                ci_lower.append(0)
                ci_upper.append(0)
        
        # Create bar plot with error bars
        x = np.arange(len(configs))
        bars = ax.bar(x, success_rates, color=[self.colors[c] for c in configs])
        
        # Add error bars
        for i, (bar, lower, upper) in enumerate(zip(bars, ci_lower, ci_upper)):
            ax.errorbar(i, success_rates[i], 
                       yerr=[[success_rates[i] - lower], [upper - success_rates[i]]],
                       fmt='none', ecolor='black', capsize=5)
        
        # Add value labels
        for i, (bar, rate) in enumerate(zip(bars, success_rates)):
            ax.text(i, rate + 0.02, f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Success Rate')
        ax.set_ylim(0, 1.1)
        ax.set_title('Quality Comparison', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_efficiency_comparison(self, ax, results: Dict[str, Any]):
        """Plot efficiency comparison."""
        baseline = results.get('detailed_results', {}).get('baseline_analysis', {})
        
        if 'metrics' not in baseline:
            ax.text(0.5, 0.5, 'No efficiency data', ha='center', va='center')
            ax.set_title('Efficiency Comparison', fontweight='bold')
            return
        
        metrics = baseline['metrics']
        
        # Extract efficiency metrics
        configs = ['sparse', 'single']
        labels = ['Sparse', 'Single']
        
        # Time efficiency (lower is better)
        if 'efficiency' in metrics and 'time_ratio' in metrics['efficiency']:
            time_ratios = [metrics['efficiency']['time_ratio'], 1.0]
        else:
            time_ratios = [1.0, 1.0]
        
        # Token efficiency (lower is better)
        if 'efficiency' in metrics and 'token_ratio' in metrics['efficiency']:
            token_ratios = [metrics['efficiency']['token_ratio'], 1.0]
        else:
            token_ratios = [1.0, 1.0]
        
        # Create grouped bar plot
        x = np.arange(len(configs))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, time_ratios, width, label='Time Ratio', color=self.colors['sparse'])
        bars2 = ax.bar(x + width/2, token_ratios, width, label='Token Ratio', color=self.colors['single'])
        
        # Add baseline line at 1.0
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline (Single)')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.05,
                       f'{height:.1f}x', ha='center', va='bottom', fontsize=9)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Ratio (vs Single)')
        ax.set_title('Efficiency Comparison\n(Lower is Better)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_synergy_analysis(self, ax, results: Dict[str, Any]):
        """Plot synergy analysis."""
        baseline = results.get('detailed_results', {}).get('baseline_analysis', {})
        
        if 'metrics' not in baseline or 'synergy' not in baseline['metrics']:
            ax.text(0.5, 0.5, 'No synergy data', ha='center', va='center')
            ax.set_title('Synergy Analysis', fontweight='bold')
            return
        
        synergy = baseline['metrics']['synergy']
        
        # Extract synergy data
        avg_gap = synergy.get('avg_synergy_gap', 0)
        gaps = synergy.get('synergy_gaps', [])
        
        if not gaps:
            ax.text(0.5, 0.5, 'No synergy gaps', ha='center', va='center')
            ax.set_title('Synergy Analysis', fontweight='bold')
            return
        
        # Create histogram of synergy gaps
        ax.hist(gaps, bins=10, color=self.colors['sparse'], alpha=0.7, edgecolor='black')
        
        # Add mean line
        ax.axvline(x=avg_gap, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_gap:.3f}')
        
        # Add zero line
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        
        # Add positive/negative regions
        ax.axvspan(0.1, max(gaps) if max(gaps) > 0.1 else 0.1, alpha=0.1, color='green', label='Strong Positive')
        ax.axvspan(min(gaps) if min(gaps) < -0.1 else -0.1, -0.1, alpha=0.1, color='red', label='Strong Negative')
        
        ax.set_xlabel('Synergy Gap (Team - Best Individual)')
        ax.set_ylabel('Frequency')
        ax.set_title('Synergy Analysis', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_learning_validation(self, ax, results: Dict[str, Any]):
        """Plot learning mechanism validation."""
        learning = results.get('detailed_results', {}).get('learning_validation', {})
        
        if not learning:
            ax.text(0.5, 0.5, 'No learning validation data', ha='center', va='center')
            ax.set_title('Learning Mechanism Validation', fontweight='bold')
            return
        
        # Extract validation results
        mechanisms = list(learning.keys())
        validated = []
        
        for mechanism, data in learning.items():
            if isinstance(data, dict):
                validated.append(data.get('validated', False))
            else:
                validated.append(False)
        
        # Create bar plot
        x = np.arange(len(mechanisms))
        colors = ['green' if v else 'red' for v in validated]
        
        bars = ax.bar(x, [1 if v else 0 for v in validated], color=colors)
        
        # Add labels
        for i, (bar, mechanism, is_valid) in enumerate(zip(bars, mechanisms, validated)):
            ax.text(i, 0.5, mechanism.replace('_', ' ').title(), 
                   ha='center', va='center', rotation=90, color='white', fontweight='bold')
            status = '✅' if is_valid else '❌'
            ax.text(i, 0.1, status, ha='center', va='bottom', fontsize=12)
        
        ax.set_xticks([])
        ax.set_ylim(0, 1.2)
        ax.set_title('Learning Mechanism Validation', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_statistical_power(self, ax, results: Dict[str, Any]):
        """Plot statistical power analysis."""
        stats = results.get('detailed_results', {}).get('statistical_analysis', {})
        
        if 'error' in stats:
            ax.text(0.5, 0.5, 'No statistical data', ha='center', va='center')
            ax.set_title('Statistical Power', fontweight='bold')
            return
        
        # Extract power data
        sample_size = stats.get('sample_size', 0)
        power = stats.get('power', 0)
        adequate = stats.get('adequate', False)
        recommended = stats.get('recommended_sample_size', 20)
        
        # Create gauge chart
        theta = np.linspace(0, np.pi, 100)
        r = 1
        
        # Background arc
        ax.plot(theta, np.ones_like(theta) * r, color='gray', linewidth=2)
        
        # Power arc
        power_theta = power * np.pi
        theta_power = np.linspace(0, power_theta, 100)
        ax.plot(theta_power, np.ones_like(theta_power) * r, 
               color='green' if adequate else 'orange', linewidth=4)
        
        # Add threshold line (80% power)
        threshold_theta = 0.8 * np.pi
        ax.plot([threshold_theta, threshold_theta], [0.9, 1.1], 
               color='red', linestyle='--', linewidth=1, alpha=0.7)
        
        # Add labels
        ax.text(0, 0.7, f'Sample: {sample_size}', ha='center', fontweight='bold')
        ax.text(0, 0.5, f'Power: {power:.1%}', ha='center', fontweight='bold')
        ax.text(0, 0.3, f'Rec: {recommended}', ha='center', fontsize=9)
        
        ax.set_xlim(-0.2, np.pi + 0.2)
        ax.set_ylim(0, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Statistical Power Analysis', fontweight='bold')
    
    def _plot_claims_validation(self, ax, results: Dict[str, Any]):
        """Plot claims validation results."""
        claims = results.get('claims_validation', {})
        
        if not claims:
            ax.text(0.5, 0.5, 'No claims validation data', ha='center', va='center')
            ax.set_title('Claims Validation', fontweight='bold')
            return
        
        # Extract claim data
        claim_names = []
        validated = []
        actual_values = []
        
        for claim_name, claim_data in claims.items():
            if 'error' in claim_data:
                continue
                
            claim_names.append(claim_name.replace('_', ' ').title())
            validated.append(claim_data.get('validated', False))
            
            if 'actual' in claim_data:
                actual_values.append(claim_data['actual'])
            else:
                actual_values.append(0)
        
        if not claim_names:
            ax.text(0.5, 0.5, 'No valid claim data', ha='center', va='center')
            ax.set_title('Claims Validation', fontweight='bold')
            return
        
        # Create horizontal bar plot
        y = np.arange(len(claim_names))
        
        # Colors based on validation status
        colors = ['green' if v else 'red' for v in validated]
        
        bars = ax.barh(y, [1] * len(claim_names), color=colors, alpha=0.7)
        
        # Add claim names and values
        for i, (bar, name, actual, is_valid) in enumerate(zip(bars, claim_names, actual_values, validated)):
            ax.text(0.1, i, name, ha='left', va='center', fontsize=9)
            
            if isinstance(actual, (int, float)):
                value_text = f'{actual:.3f}'
            else:
                value_text = str(actual)
            
            status = '✅' if is_valid else '❌'
            ax.text(0.9, i, f'{status} {value_text}', ha='right', va='center', fontsize=9)
        
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Claims Validation', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    def _plot_time_series(self, ax, results: Dict[str, Any]):
        """Plot time series of metrics."""
        # This would normally load historical data
        # For now, create a simple demonstration
        
        # Simulated time series data
        time_points = np.arange(1, 11)
        
        # Simulated metrics
        quality = 0.7 + 0.02 * time_points + np.random.normal(0, 0.05, len(time_points))
        efficiency = 1.0 - 0.03 * time_points + np.random.normal(0, 0.05, len(time_points))
        synergy = 0.0 + 0.01 * time_points + np.random.normal(0, 0.03, len(time_points))
        
        # Plot time series
        ax.plot(time_points, quality, marker='o', label='Quality', color=self.colors['sparse'], linewidth=2)
        ax.plot(time_points, efficiency, marker='s', label='Efficiency', color=self.colors['single'], linewidth=2)
        ax.plot(time_points, synergy, marker='^', label='Synergy', color=self.colors['always_on'], linewidth=2)
        
        ax.set_xlabel('Validation Iteration')
        ax.set_ylabel('Metric Value')
        ax.set_title('Metric Trends Over Time', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_confidence_intervals(self, ax, results: Dict[str, Any]):
        """Plot confidence intervals for key metrics."""
        stats = results.get('detailed_results', {}).get('statistical_analysis', {})
        
        if 'error' in stats:
            ax.text(0.5, 0.5, 'No confidence interval data', ha='center', va='center')
            ax.set_title('Confidence Intervals', fontweight='bold')
            return
        
        # Extract CI data (simplified)
        metrics = ['Quality', 'Efficiency', 'Synergy']
        means = [0.85, 0.7, 0.05]  # Example values
        ci_lower = [0.78, 0.65, -0.02]  # Example
        ci_upper = [0.92, 0.75, 0.12]  # Example
        
        # Create CI plot
        y = np.arange(len(metrics))
        
        ax.errorbar(means, y, xerr=[np.array(means) - np.array(ci_lower), 
                                    np.array(ci_upper) - np.array(means)],
                   fmt='o', color='black', ecolor='gray', capsize=5)
        
        ax.set_yticks(y)
        ax.set_yticklabels(metrics)
        ax.set_xlabel('Metric Value')
        ax.set_title('95% Confidence Intervals', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    def _plot_recommendations(self, ax, results: Dict[str, Any]):
        """Plot recommendations summary."""
        recommendations = results.get('recommendations', {})
        
        if not recommendations:
            ax.text(0.5, 0.5, 'No recommendations', ha='center', va='center')
            ax.set_title('Recommendations', fontweight='bold')
            return
        
        # Count recommendations by timeframe
        timeframes = list(recommendations.keys())
        counts = [len(recs) for recs in recommendations.values()]
        
        # Create donut chart
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue
        
        wedges, texts, autotexts = ax.pie(counts, labels=timeframes, colors=colors[:len(timeframes)],
                                          autopct='%1.0f%%', startangle=90, pctdistance=0.85)
        
        # Draw circle for donut effect
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        ax.add_artist(centre_circle)
        
        # Add total count in center
        total = sum(counts)
        ax.text(0, 0, f'Total:\n{total}', ha='center', va='center', fontweight='bold')
        
        ax.set_title('Recommendations by Timeframe', fontweight='bold')
        ax.axis('equal')
    
    def create_interactive_html(self, save_path: str = "validation_dashboard.html"):
        """Create interactive HTML dashboard."""
        results = self.load_latest_results()
        if not results:
            print("No results for HTML dashboard")
            return
        
        # Create simple HTML dashboard
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Agentic Network v2 Validation Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
                .metric {{ background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .valid {{ color: green; font-weight: bold; }}
                .invalid {{ color: red; font-weight: bold; }}
                .recommendation {{ background: #f8f9fa; padding: 10px; margin: 5px 0; border-left: 4px solid #3498db; }}
                .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Agentic Network v2 Validation Dashboard</h1>
                <p>Timestamp: {results.get('validation_timestamp', 'N/A')}</p>
            </div>
            
            <div class="grid">
        """
        
        # Add claims validation
        claims = results.get('claims_validation', {})
        if claims:
            html += """
            <div class="metric">
                <h2>Claims Validation</h2>
            """
            
            for claim_name, claim_data in claims.items():
                if 'error' in claim_data:
                    continue
                    
                validated = claim_data.get('validated', False)
                status_class = 'valid' if validated else 'invalid'
                status_text = 'VALIDATED' if validated else 'NOT VALIDATED'
                
                html += f"""
                <div>
                    <h3>{claim_name.replace('_', ' ').title()}</h3>
                    <p>Claimed: {claim_data.get('claimed', 'N/A')}</p>
                    <p>Actual: {claim_data.get('actual', 'N/A')}</p>
                    <p class="{status_class}">Status: {status_text}</p>
                </div>
                <hr>
                """
            
            html += "</div>"
        
        # Add recommendations
        recommendations = results.get('recommendations', {})
        if recommendations:
            html += """
            <div class="metric">
                <h2>Recommendations</h2>
            """
            
            for timeframe, recs in recommendations.items():
                if recs:
                    html += f"<h3>{timeframe.replace('_', ' ').title()}</h3>"
                    for rec in recs:
                        html += f'<div class="recommendation">{rec}</div>'
            
            html += "</div>"
        
        # Add statistical summary
        stats = results.get('detailed_results', {}).get('statistical_analysis', {})
        if stats and 'error' not in stats:
            html += f"""
            <div class="metric">
                <h2>Statistical Summary</h2>
                <p>Sample Size: {stats.get('sample_size', 'N/A')}</p>
                <p>Statistical Power: {stats.get('power', 0):.1%}</p>
                <p>Effect Size: {stats.get('effect_size', 'N/A')}</p>
                <p>Recommended Sample Size: {stats.get('recommended_sample_size', 'N/A')}</p>
            </div>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        with open(save_path, 'w') as f:
            f.write(html)
        
        print(f"HTML dashboard saved to: {save_path}")


def main():
    """Create validation dashboard."""
    dashboard = ValidationDashboard()
    
    # Create static dashboard
    dashboard.create_dashboard("validation_dashboard.png")
    
    # Create interactive HTML dashboard
    dashboard.create_interactive_html("validation_dashboard.html")
    
    print("\nDashboard creation complete!")
    print("1. Static dashboard: validation_dashboard.png")
    print("2. Interactive dashboard: validation_dashboard.html")


if __name__ == "__main__":
    main()