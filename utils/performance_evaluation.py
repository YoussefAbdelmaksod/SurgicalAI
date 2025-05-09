"""
Performance evaluation module for SurgicalAI.

This module implements comprehensive performance evaluation for laparoscopic
surgeries, including mistake analysis, time efficiency, and personalized
recommendations for improvement.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from collections import defaultdict


class SurgicalPerformanceEvaluator:
    """
    Comprehensive evaluator for surgical performance, generating detailed
    post-surgery reports with metrics, visualizations, and personalized feedback.
    """
    
    def __init__(self, phase_reference_times=None, tool_reference_usage=None):
        """
        Initialize the surgical performance evaluator.
        
        Args:
            phase_reference_times (dict): Reference times for each phase (expert benchmarks)
            tool_reference_usage (dict): Reference tool usage patterns
        """
        self.phase_reference_times = phase_reference_times or {}
        self.tool_reference_usage = tool_reference_usage or {}
        
        # Standard phase names for cholecystectomy
        self.standard_phases = [
            'preparation',
            'calot_triangle_dissection',
            'clipping_and_cutting',
            'gallbladder_dissection',
            'gallbladder_packaging',
            'cleaning_and_coagulation',
            'gallbladder_extraction'
        ]
        
        # Define mistake severity levels
        self.severity_thresholds = {
            'critical': 0.8,  # Mistakes with risk level >= 0.8
            'major': 0.5,     # Mistakes with risk level 0.5-0.79
            'minor': 0.0      # Mistakes with risk level < 0.5
        }
        
        # Expert-defined tool-phase associations (which tools should be used in which phase)
        self.phase_tool_mapping = {
            'preparation': ['grasper', 'hook'],
            'calot_triangle_dissection': ['grasper', 'hook', 'scissors'],
            'clipping_and_cutting': ['grasper', 'clipper', 'scissors'],
            'gallbladder_dissection': ['grasper', 'hook', 'scissors', 'bipolar'],
            'gallbladder_packaging': ['grasper', 'specimen_bag'],
            'cleaning_and_coagulation': ['grasper', 'irrigator', 'hook', 'bipolar'],
            'gallbladder_extraction': ['grasper']
        }
    
    def evaluate_session(self, session_data):
        """
        Generate a comprehensive performance evaluation report.
        
        Args:
            session_data (dict): Dictionary with session data including:
                - mistakes: List of detected mistakes
                - phase_durations: Time spent in each phase
                - tool_usage: Tool usage statistics
                - total_duration: Total procedure duration
                - surgical_actions: Timeline of actions performed
                - phase_transitions: Timestamps of phase transitions
        
        Returns:
            dict: Comprehensive performance report
        """
        # Extract session data
        mistakes = session_data.get('mistakes', [])
        phase_durations = session_data.get('phase_durations', {})
        tool_usage = session_data.get('tool_usage', {})
        total_duration = session_data.get('total_duration', 0)
        surgical_actions = session_data.get('surgical_actions', [])
        phase_transitions = session_data.get('phase_transitions', [])
        
        # Calculate core metrics
        metrics = self._calculate_metrics(
            mistakes, phase_durations, tool_usage, total_duration
        )
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(metrics)
        
        # Identify strengths and areas for improvement
        strengths = self._identify_strengths(metrics, phase_durations)
        improvements = self._identify_improvements(
            metrics, mistakes, phase_durations, tool_usage
        )
        
        # Generate visualizations
        visualizations = self._generate_visualizations(
            mistakes, phase_durations, tool_usage, phase_transitions
        )
        
        # Compile comprehensive report
        report = {
            'overall_score': performance_score,
            'performance_level': self._get_performance_level(performance_score),
            'metrics': metrics,
            'strengths': strengths,
            'areas_for_improvement': improvements,
            'visualizations': visualizations,
            'phase_analysis': self._analyze_phases(phase_durations, mistakes),
            'tool_usage_analysis': self._analyze_tool_usage(tool_usage, phase_durations),
            'mistake_analysis': self._analyze_mistakes(mistakes, phase_durations),
            'personalized_recommendations': self._generate_recommendations(
                metrics, mistakes, phase_durations, tool_usage
            ),
            'summary': self._generate_summary(performance_score, strengths, improvements)
        }
        
        return report
    
    def _calculate_metrics(self, mistakes, phase_durations, tool_usage, total_duration):
        """Calculate core performance metrics."""
        # Count mistakes by severity
        mistake_counts = {
            'total': len(mistakes),
            'critical': sum(1 for m in mistakes if m['risk_level'] >= self.severity_thresholds['critical']),
            'major': sum(1 for m in mistakes if self.severity_thresholds['major'] <= m['risk_level'] < self.severity_thresholds['critical']),
            'minor': sum(1 for m in mistakes if m['risk_level'] < self.severity_thresholds['major'])
        }
        
        # Calculate mistake rate (mistakes per minute)
        metrics = {
            'mistake_counts': mistake_counts,
            'mistake_rate': mistake_counts['total'] / (total_duration / 60) if total_duration > 0 else 0,
            'critical_mistake_rate': mistake_counts['critical'] / (total_duration / 60) if total_duration > 0 else 0,
            'total_duration_minutes': total_duration / 60,
        }
        
        # Calculate phase efficiency (compared to reference times if available)
        phase_efficiency = {}
        for phase, duration in phase_durations.items():
            reference_time = self.phase_reference_times.get(phase)
            if reference_time:
                efficiency = reference_time / duration if duration > 0 else 0
                phase_efficiency[phase] = efficiency
        
        metrics['phase_efficiency'] = phase_efficiency
        
        # Calculate overall efficiency
        if phase_efficiency:
            metrics['overall_efficiency'] = sum(phase_efficiency.values()) / len(phase_efficiency)
        else:
            metrics['overall_efficiency'] = 1.0  # Default when no reference data
        
        # Calculate tool usage efficiency
        tool_efficiency = {}
        for phase, tools in tool_usage.items():
            recommended_tools = self.phase_tool_mapping.get(phase, [])
            if recommended_tools:
                # Check if appropriate tools were used
                appropriate_tools = sum(1 for tool in tools if tool in recommended_tools)
                inappropriate_tools = sum(1 for tool in tools if tool not in recommended_tools)
                
                tool_efficiency[phase] = appropriate_tools / (appropriate_tools + inappropriate_tools) if (appropriate_tools + inappropriate_tools) > 0 else 1.0
        
        metrics['tool_efficiency'] = tool_efficiency
        if tool_efficiency:
            metrics['overall_tool_efficiency'] = sum(tool_efficiency.values()) / len(tool_efficiency)
        else:
            metrics['overall_tool_efficiency'] = 1.0
        
        return metrics
    
    def _calculate_performance_score(self, metrics):
        """Calculate overall performance score (0-100)."""
        # Base score starts at 100
        score = 100
        
        # Deduct for mistakes based on severity
        mistake_penalty = (
            (metrics['mistake_counts']['critical'] * 10) +
            (metrics['mistake_counts']['major'] * 5) +
            (metrics['mistake_counts']['minor'] * 2)
        )
        mistake_penalty = min(mistake_penalty, 50)  # Cap penalty at 50 points
        
        # Adjust for efficiency (can add up to 10 bonus points)
        efficiency_bonus = (metrics['overall_efficiency'] - 1.0) * 10 if metrics['overall_efficiency'] > 1.0 else 0
        efficiency_bonus = min(efficiency_bonus, 10)
        
        # Adjust for tool usage efficiency
        tool_factor = (metrics['overall_tool_efficiency'] * 20) - 10  # -10 to +10 range
        
        # Calculate final score
        final_score = max(0, min(100, score - mistake_penalty + efficiency_bonus + tool_factor))
        
        return final_score
    
    def _get_performance_level(self, score):
        """Map performance score to a descriptive level."""
        if score >= 90:
            return "Expert"
        elif score >= 80:
            return "Advanced"
        elif score >= 70:
            return "Proficient"
        elif score >= 60:
            return "Competent"
        elif score >= 50:
            return "Developing"
        else:
            return "Novice"
    
    def _identify_strengths(self, metrics, phase_durations):
        """Identify areas of strength based on performance metrics."""
        strengths = []
        
        # No critical mistakes
        if metrics['mistake_counts']['critical'] == 0:
            strengths.append({
                'category': 'safety',
                'description': 'Maintained safety throughout procedure with no critical errors'
            })
        
        # Low overall mistake rate
        if metrics['mistake_rate'] < 0.5:  # Less than 1 mistake per 2 minutes
            strengths.append({
                'category': 'precision',
                'description': 'Demonstrated good precision with minimal errors'
            })
        
        # Good efficiency in specific phases
        for phase, efficiency in metrics['phase_efficiency'].items():
            if efficiency >= 1.1:  # At least 10% faster than reference
                strengths.append({
                    'category': 'efficiency',
                    'description': f'Excellent efficiency in {phase.replace("_", " ")} phase'
                })
        
        # Good tool usage
        for phase, efficiency in metrics['tool_efficiency'].items():
            if efficiency >= 0.9:  # At least 90% appropriate tool usage
                strengths.append({
                    'category': 'tool_handling',
                    'description': f'Appropriate tool selection in {phase.replace("_", " ")} phase'
                })
        
        # Phases with no mistakes
        mistake_phases = set(m['phase'] for m in metrics.get('mistakes', []))
        phases_without_mistakes = set(phase_durations.keys()) - mistake_phases
        
        for phase in phases_without_mistakes:
            strengths.append({
                'category': 'technique',
                'description': f'Flawless execution during {phase.replace("_", " ")} phase'
            })
        
        return strengths[:5]  # Return top 5 strengths
    
    def _identify_improvements(self, metrics, mistakes, phase_durations, tool_usage):
        """Identify areas for improvement."""
        improvements = []
        
        # Phases with high mistake rates
        mistake_by_phase = defaultdict(list)
        for mistake in mistakes:
            mistake_by_phase[mistake['phase']].append(mistake)
        
        for phase, phase_mistakes in mistake_by_phase.items():
            phase_duration = phase_durations.get(phase, 0)
            if phase_duration > 0:
                mistake_rate = len(phase_mistakes) / (phase_duration / 60)
                if mistake_rate > 1.0:  # More than 1 mistake per minute
                    improvements.append({
                        'category': 'technique',
                        'description': f'Improve precision during {phase.replace("_", " ")} phase',
                        'priority': 'high' if any(m['risk_level'] >= 0.8 for m in phase_mistakes) else 'medium'
                    })
        
        # Phases with poor efficiency
        for phase, efficiency in metrics['phase_efficiency'].items():
            if efficiency < 0.8:  # More than 20% slower than reference
                improvements.append({
                    'category': 'efficiency',
                    'description': f'Work on improving speed during {phase.replace("_", " ")} phase',
                    'priority': 'medium'
                })
        
        # Poor tool selection
        for phase, efficiency in metrics['tool_efficiency'].items():
            if efficiency < 0.7:  # Less than 70% appropriate tool usage
                improvements.append({
                    'category': 'tool_handling',
                    'description': f'Improve tool selection during {phase.replace("_", " ")} phase',
                    'priority': 'medium'
                })
        
        # Common mistake types
        mistake_types = defaultdict(int)
        for mistake in mistakes:
            mistake_types[mistake['type']] += 1
        
        common_mistakes = sorted(mistake_types.items(), key=lambda x: x[1], reverse=True)
        for mistake_type, count in common_mistakes[:3]:  # Top 3 common mistakes
            if count >= 3:  # Only if it occurred at least 3 times
                improvements.append({
                    'category': 'technique',
                    'description': f'Practice reducing "{mistake_type}" errors',
                    'priority': 'high' if mistake_type in [m['type'] for m in mistakes if m['risk_level'] >= 0.8] else 'medium'
                })
        
        return improvements[:5]  # Return top 5 areas for improvement
    
    def _generate_visualizations(self, mistakes, phase_durations, tool_usage, phase_transitions):
        """Generate visualizations for the report."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import io
        import base64
        from collections import defaultdict
        import numpy as np
        
        visualizations = {}
        
        # Generate mistake timeline visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        if mistakes:
            mistake_times = [m.get('timestamp', 0) for m in mistakes]
            mistake_risks = [m.get('risk_level', 0) for m in mistakes]
            mistake_types = [m.get('type', 'Unknown') for m in mistakes]
            
            scatter = ax.scatter(mistake_times, mistake_risks, c=np.arange(len(mistakes)), 
                      cmap='viridis', s=100, alpha=0.7)
            
            for i, txt in enumerate(mistake_types):
                ax.annotate(txt, (mistake_times[i], mistake_risks[i]), 
                           xytext=(5, 5), textcoords='offset points')
            
            # Add phase transition lines if available
            if phase_transitions:
                for transition in phase_transitions:
                    ax.axvline(x=transition.get('timestamp', 0), color='r', linestyle='--', alpha=0.5)
                    ax.text(transition.get('timestamp', 0), 0.1, 
                           transition.get('phase', ''), rotation=90, alpha=0.7)
            
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Risk Level')
            ax.set_title('Timeline of Surgical Mistakes')
            ax.grid(True, alpha=0.3)
            
            # Save to base64
            buf = io.BytesIO()
            fig.tight_layout()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            visualizations['mistake_timeline'] = {
                'description': 'Timeline of surgical mistakes throughout the procedure',
                'format': 'png',
                'data': img_str
            }
        
        # Generate phase duration comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        if phase_durations:
            phases = list(phase_durations.keys())
            durations = list(phase_durations.values())
            reference_durations = [self.phase_reference_times.get(phase, 0) for phase in phases]
            
            x = np.arange(len(phases))
            width = 0.35
            
            ax.bar(x - width/2, durations, width, label='Actual Duration')
            ax.bar(x + width/2, reference_durations, width, label='Reference Duration')
            
            ax.set_xlabel('Surgical Phase')
            ax.set_ylabel('Duration (seconds)')
            ax.set_title('Phase Duration Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels([p.replace('_', ' ').title() for p in phases], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save to base64
            buf = io.BytesIO()
            fig.tight_layout()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            visualizations['phase_duration'] = {
                'description': 'Comparison of phase durations with reference times',
                'format': 'png',
                'data': img_str
            }
        
        # Generate tool usage patterns
        fig, ax = plt.subplots(figsize=(12, 8))
        if tool_usage:
            # Flatten tool usage by phase into counts
            tool_counts = defaultdict(lambda: defaultdict(int))
            for phase, tools in tool_usage.items():
                for tool in tools:
                    tool_counts[phase][tool] += 1
            
            phases = list(tool_counts.keys())
            all_tools = set()
            for phase_tools in tool_counts.values():
                all_tools.update(phase_tools.keys())
            all_tools = sorted(list(all_tools))
            
            # Create a matrix for the heatmap
            data = np.zeros((len(phases), len(all_tools)))
            for i, phase in enumerate(phases):
                for j, tool in enumerate(all_tools):
                    data[i, j] = tool_counts[phase][tool]
            
            im = ax.imshow(data, cmap='YlOrRd')
            
            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Usage Count', rotation=-90, va="bottom")
            
            # Show all ticks and label them
            ax.set_xticks(np.arange(len(all_tools)))
            ax.set_yticks(np.arange(len(phases)))
            ax.set_xticklabels([t.title() for t in all_tools])
            ax.set_yticklabels([p.replace('_', ' ').title() for p in phases])
            
            # Rotate the tick labels and set their alignment
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            ax.set_title("Tool Usage Across Surgical Phases")
            
            # Loop over data dimensions and create text annotations
            for i in range(len(phases)):
                for j in range(len(all_tools)):
                    text = ax.text(j, i, int(data[i, j]),
                                  ha="center", va="center", color="black" if data[i, j] < np.max(data)/2 else "white")
            
            # Save to base64
            buf = io.BytesIO()
            fig.tight_layout()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            visualizations['tool_usage'] = {
                'description': 'Tool usage patterns across surgical phases',
                'format': 'png',
                'data': img_str
            }
        
        # Generate mistake heatmap by phase
        fig, ax = plt.subplots(figsize=(10, 6))
        if mistakes and phase_durations:
            # Count mistakes by phase
            mistakes_by_phase = defaultdict(int)
            for mistake in mistakes:
                mistakes_by_phase[mistake.get('phase', 'unknown')] += 1
            
            phases = list(phase_durations.keys())
            mistake_counts = [mistakes_by_phase.get(phase, 0) for phase in phases]
            
            # Calculate mistake density (mistakes per minute)
            mistake_density = []
            for phase in phases:
                duration = phase_durations.get(phase, 0)
                count = mistakes_by_phase.get(phase, 0)
                density = (count / (duration / 60)) if duration > 0 else 0
                mistake_density.append(density)
            
            # Create bar chart
            ax.bar(phases, mistake_density, color='crimson')
            ax.set_xlabel('Surgical Phase')
            ax.set_ylabel('Mistakes per Minute')
            ax.set_title('Mistake Frequency by Surgical Phase')
            ax.set_xticklabels([p.replace('_', ' ').title() for p in phases], rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # Add count labels
            for i, count in enumerate(mistake_counts):
                ax.text(i, mistake_density[i] + 0.05, f'{count} mistakes', 
                       ha='center', va='bottom')
            
            # Save to base64
            buf = io.BytesIO()
            fig.tight_layout()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            visualizations['mistake_heatmap'] = {
                'description': 'Heatmap of mistake frequency by surgical phase',
                'format': 'png',
                'data': img_str
            }
        
        return visualizations
    
    def _analyze_phases(self, phase_durations, mistakes):
        """Perform detailed analysis of each surgical phase."""
        phase_analysis = {}
        
        # Group mistakes by phase
        mistakes_by_phase = defaultdict(list)
        for mistake in mistakes:
            mistakes_by_phase[mistake['phase']].append(mistake)
        
        for phase, duration in phase_durations.items():
            phase_mistakes = mistakes_by_phase.get(phase, [])
            
            # Calculate phase-specific metrics
            reference_time = self.phase_reference_times.get(phase)
            time_efficiency = reference_time / duration if reference_time and duration > 0 else None
            
            mistake_rate = len(phase_mistakes) / (duration / 60) if duration > 0 else 0
            critical_mistakes = sum(1 for m in phase_mistakes if m['risk_level'] >= 0.8)
            
            phase_analysis[phase] = {
                'duration_seconds': duration,
                'reference_duration_seconds': reference_time,
                'time_efficiency': time_efficiency,
                'mistake_count': len(phase_mistakes),
                'mistake_rate_per_minute': mistake_rate,
                'critical_mistakes': critical_mistakes,
                'common_mistakes': self._get_common_mistakes(phase_mistakes)
            }
        
        return phase_analysis
    
    def _get_common_mistakes(self, mistakes):
        """Get most common mistake types from a list of mistakes."""
        if not mistakes:
            return []
            
        mistake_types = defaultdict(int)
        for mistake in mistakes:
            mistake_types[mistake['type']] += 1
            
        common_mistakes = sorted(mistake_types.items(), key=lambda x: x[1], reverse=True)
        return [{'type': t, 'count': c} for t, c in common_mistakes[:3]]
    
    def _analyze_tool_usage(self, tool_usage, phase_durations):
        """Analyze tool usage patterns across phases."""
        tool_analysis = {}
        
        for phase, tools in tool_usage.items():
            recommended_tools = self.phase_tool_mapping.get(phase, [])
            
            # Analyze appropriate vs. inappropriate tool usage
            tool_counts = defaultdict(int)
            for tool in tools:
                tool_counts[tool] += 1
                
            appropriate_tools = {t: c for t, c in tool_counts.items() if t in recommended_tools}
            inappropriate_tools = {t: c for t, c in tool_counts.items() if t not in recommended_tools}
            
            # Calculate efficiency
            total_tools = sum(tool_counts.values())
            appropriate_count = sum(appropriate_tools.values())
            
            tool_analysis[phase] = {
                'recommended_tools': recommended_tools,
                'tool_counts': dict(tool_counts),
                'appropriate_tools': appropriate_tools,
                'inappropriate_tools': inappropriate_tools,
                'appropriate_percentage': (appropriate_count / total_tools * 100) if total_tools > 0 else 100
            }
        
        return tool_analysis
    
    def _analyze_mistakes(self, mistakes, phase_durations):
        """Perform detailed analysis of surgical mistakes."""
        if not mistakes:
            return {
                'total_count': 0,
                'severity_distribution': {
                    'critical': 0,
                    'major': 0,
                    'minor': 0
                },
                'common_types': []
            }
            
        # Count by severity
        severity_counts = {
            'critical': sum(1 for m in mistakes if m['risk_level'] >= self.severity_thresholds['critical']),
            'major': sum(1 for m in mistakes if self.severity_thresholds['major'] <= m['risk_level'] < self.severity_thresholds['critical']),
            'minor': sum(1 for m in mistakes if m['risk_level'] < self.severity_thresholds['major'])
        }
        
        # Identify common mistake types
        mistake_types = defaultdict(int)
        for mistake in mistakes:
            mistake_types[mistake['type']] += 1
            
        common_types = sorted(mistake_types.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate mistake rate by phase
        mistake_rates = {}
        mistakes_by_phase = defaultdict(list)
        for mistake in mistakes:
            mistakes_by_phase[mistake['phase']].append(mistake)
            
        for phase, phase_mistakes in mistakes_by_phase.items():
            duration = phase_durations.get(phase, 0)
            if duration > 0:
                mistake_rates[phase] = len(phase_mistakes) / (duration / 60)
            else:
                mistake_rates[phase] = 0
        
        return {
            'total_count': len(mistakes),
            'severity_distribution': severity_counts,
            'common_types': [{'type': t, 'count': c} for t, c in common_types],
            'mistake_rates_by_phase': mistake_rates
        }
    
    def _generate_recommendations(self, metrics, mistakes, phase_durations, tool_usage):
        """Generate personalized recommendations for improvement."""
        recommendations = []
        
        # Add specific recommendations based on performance
        if metrics['mistake_counts']['critical'] > 0:
            recommendations.append({
                'focus_area': 'Safety',
                'description': 'Review critical safety protocols, particularly for high-risk steps',
                'suggested_training': 'Simulation training focused on critical error prevention'
            })
        
        # Check phases with highest mistake rates
        mistake_by_phase = defaultdict(list)
        for mistake in mistakes:
            mistake_by_phase[mistake['phase']].append(mistake)
        
        phase_mistake_rates = {}
        for phase, phase_mistakes in mistake_by_phase.items():
            duration = phase_durations.get(phase, 0)
            if duration > 0:
                phase_mistake_rates[phase] = len(phase_mistakes) / (duration / 60)
            
        if phase_mistake_rates:
            worst_phase = max(phase_mistake_rates.items(), key=lambda x: x[1])
            if worst_phase[1] > 1.0:  # More than 1 mistake per minute
                recommendations.append({
                    'focus_area': f'{worst_phase[0].replace("_", " ")} technique',
                    'description': f'Focus on improving technique during {worst_phase[0].replace("_", " ")} phase',
                    'suggested_training': f'Targeted practice of {worst_phase[0].replace("_", " ")} on surgical simulator'
                })
        
        # Tool selection recommendations
        poor_tool_phases = []
        for phase, tools in tool_usage.items():
            recommended_tools = self.phase_tool_mapping.get(phase, [])
            if recommended_tools:
                inappropriate_count = sum(1 for tool in tools if tool not in recommended_tools)
                if inappropriate_count > 2:
                    poor_tool_phases.append(phase)
        
        if poor_tool_phases:
            phase_names = ', '.join([p.replace('_', ' ') for p in poor_tool_phases[:2]])
            recommendations.append({
                'focus_area': 'Tool selection',
                'description': f'Review appropriate tool selection for {phase_names} phases',
                'suggested_training': 'Tool selection exercises with expert feedback'
            })
        
        # Common mistake types
        mistake_types = defaultdict(int)
        for mistake in mistakes:
            mistake_types[mistake['type']] += 1
            
        common_mistakes = sorted(mistake_types.items(), key=lambda x: x[1], reverse=True)
        if common_mistakes:
            top_mistake = common_mistakes[0][0]
            recommendations.append({
                'focus_area': 'Error reduction',
                'description': f'Work on reducing "{top_mistake}" errors',
                'suggested_training': f'Focused practice with expert feedback on {top_mistake} prevention'
            })
        
        # Efficiency recommendations
        if metrics.get('overall_efficiency', 1.0) < 0.7:
            recommendations.append({
                'focus_area': 'Procedural efficiency',
                'description': 'Improve overall procedural speed while maintaining safety',
                'suggested_training': 'Timed practice sessions with expert guidance'
            })
        
        return recommendations
    
    def _generate_summary(self, performance_score, strengths, improvements):
        """Generate an executive summary of the performance."""
        # Create a concise summary based on the performance level
        performance_level = self._get_performance_level(performance_score)
        
        summary_text = f"Overall performance level: {performance_level} ({performance_score:.1f}/100). "
        
        # Add top strength
        if strengths:
            summary_text += f"Key strength: {strengths[0]['description']}. "
        
        # Add top area for improvement
        if improvements:
            summary_text += f"Focus area: {improvements[0]['description']}."
        
        return summary_text 