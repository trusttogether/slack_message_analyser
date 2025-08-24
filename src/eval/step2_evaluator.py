#!/usr/bin/env python3
"""
Step 2 Evaluation Script
Analyzes merge/split results and compares with benchmark topics
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Set
import matplotlib.pyplot as plt
import seaborn as sns

class Step2Evaluator:
    def __init__(self, results_file: str, benchmark_file: str):
        self.results_file = results_file
        self.benchmark_file = benchmark_file
        self.results = None
        self.benchmark_topics = None
        
    def load_data(self):
        """Load results and benchmark data"""
        with open(self.results_file, 'r') as f:
            self.results = json.load(f)
        
        with open(self.benchmark_file, 'r') as f:
            benchmark_data = json.load(f)
            self.benchmark_topics = benchmark_data['topics']
    
    def analyze_cluster_merges(self) -> Dict:
        """Analyze which clusters were merged and why"""
        initial_count = len(self.results.get('initial_clusters', []))
        final_count = len(self.results['final_clusters'])
        merges_performed = initial_count - final_count
        
        merge_analysis = {
            'initial_clusters': initial_count,
            'final_clusters': final_count,
            'merges_performed': merges_performed,
            'merge_rate': merges_performed / initial_count if initial_count > 0 else 0
        }
        
        return merge_analysis
    
    def calculate_detailed_metrics(self) -> Dict:
        """Calculate detailed precision, recall, and F1 metrics"""
        benchmark_sets = [set(topic['messages']) for topic in self.benchmark_topics]
        final_sets = [set(cluster['message_ids']) for cluster in self.results['final_clusters']]
        
        detailed_metrics = []
        
        for i, benchmark_set in enumerate(benchmark_sets):
            topic_metrics = {
                'topic_id': self.benchmark_topics[i]['id'],
                'topic_title': self.benchmark_topics[i]['title'],
                'benchmark_messages': len(benchmark_set),
                'best_match_cluster': None,
                'best_precision': 0,
                'best_recall': 0,
                'best_f1': 0,
                'matching_clusters': []
            }
            
            for j, final_set in enumerate(final_sets):
                if len(final_set) > 0:
                    intersection = benchmark_set.intersection(final_set)
                    precision = len(intersection) / len(final_set)
                    recall = len(intersection) / len(benchmark_set)
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    if f1 > topic_metrics['best_f1']:
                        topic_metrics['best_f1'] = f1
                        topic_metrics['best_precision'] = precision
                        topic_metrics['best_recall'] = recall
                        topic_metrics['best_match_cluster'] = self.results['final_clusters'][j]['cluster_id']
                    
                    if precision > 0.5 or recall > 0.5:
                        topic_metrics['matching_clusters'].append({
                            'cluster_id': self.results['final_clusters'][j]['cluster_id'],
                            'precision': precision,
                            'recall': recall,
                            'f1': f1
                        })
            
            detailed_metrics.append(topic_metrics)
        
        return detailed_metrics
    
    def identify_issues(self) -> List[Dict]:
        """Identify potential issues with the merge/split operations"""
        issues = []
        
        # Check for over-clustering (too many final clusters)
        expected_clusters = len(self.benchmark_topics)
        actual_clusters = len(self.results['final_clusters'])
        
        if actual_clusters > expected_clusters * 1.5:
            issues.append({
                'type': 'over_clustering',
                'description': f'Too many final clusters: {actual_clusters} vs expected ~{expected_clusters}',
                'severity': 'high'
            })
        
        # Check for under-clustering (too few final clusters)
        if actual_clusters < expected_clusters * 0.5:
            issues.append({
                'type': 'under_clustering',
                'description': f'Too few final clusters: {actual_clusters} vs expected ~{expected_clusters}',
                'severity': 'high'
            })
        
        # Check for clusters with too many messages
        for cluster in self.results['final_clusters']:
            if len(cluster['message_ids']) > 100:
                issues.append({
                    'type': 'large_cluster',
                    'description': f'Cluster {cluster["cluster_id"]} has {len(cluster["message_ids"])} messages',
                    'severity': 'medium',
                    'cluster_id': cluster['cluster_id']
                })
        
        # Check for clusters with too few messages
        for cluster in self.results['final_clusters']:
            if len(cluster['message_ids']) < 3:
                issues.append({
                    'type': 'small_cluster',
                    'description': f'Cluster {cluster["cluster_id"]} has only {len(cluster["message_ids"])} messages',
                    'severity': 'medium',
                    'cluster_id': cluster['cluster_id']
                })
        
        return issues
    
    def generate_report(self) -> Dict:
        """Generate comprehensive evaluation report"""
        self.load_data()
        
        # Basic metrics
        evaluation = self.results['evaluation']
        
        # Detailed analysis
        merge_analysis = self.analyze_cluster_merges()
        detailed_metrics = self.calculate_detailed_metrics()
        issues = self.identify_issues()
        
        # Calculate additional metrics
        avg_messages_per_cluster = np.mean([len(cluster['message_ids']) for cluster in self.results['final_clusters']])
        cluster_size_std = np.std([len(cluster['message_ids']) for cluster in self.results['final_clusters']])
        
        report = {
            'summary': {
                'similarity_threshold': self.results.get('similarity_threshold', 'unknown'),
                'overall_f1': evaluation['avg_f1'],
                'overall_precision': evaluation['avg_precision'],
                'overall_recall': evaluation['avg_recall'],
                'expected_clusters': evaluation['expected_clusters'],
                'actual_clusters': evaluation['actual_clusters'],
                'cluster_count_difference': abs(evaluation['expected_clusters'] - evaluation['actual_clusters'])
            },
            'merge_analysis': merge_analysis,
            'detailed_metrics': detailed_metrics,
            'issues': issues,
            'cluster_statistics': {
                'avg_messages_per_cluster': avg_messages_per_cluster,
                'cluster_size_std': cluster_size_std,
                'min_cluster_size': min([len(cluster['message_ids']) for cluster in self.results['final_clusters']]),
                'max_cluster_size': max([len(cluster['message_ids']) for cluster in self.results['final_clusters']])
            }
        }
        
        return report
    
    def save_report(self, report: Dict, output_file: str):
        """Save evaluation report to file"""
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Evaluation report saved to {output_file}")
    
    def print_summary(self, report: Dict):
        """Print a summary of the evaluation results"""
        print("\n" + "="*60)
        print("STEP 2 EVALUATION SUMMARY")
        print("="*60)
        
        summary = report['summary']
        print(f"Similarity Threshold: {summary['similarity_threshold']}")
        print(f"Overall F1 Score: {summary['overall_f1']:.3f}")
        print(f"Overall Precision: {summary['overall_precision']:.3f}")
        print(f"Overall Recall: {summary['overall_recall']:.3f}")
        print(f"Expected Clusters: {summary['expected_clusters']}")
        print(f"Actual Clusters: {summary['actual_clusters']}")
        print(f"Cluster Count Difference: {summary['cluster_count_difference']}")
        
        merge_analysis = report['merge_analysis']
        print(f"\nMerge Analysis:")
        print(f"  Initial Clusters: {merge_analysis['initial_clusters']}")
        print(f"  Final Clusters: {merge_analysis['final_clusters']}")
        print(f"  Merges Performed: {merge_analysis['merges_performed']}")
        print(f"  Merge Rate: {merge_analysis['merge_rate']:.2%}")
        
        if report['issues']:
            print(f"\nIssues Found ({len(report['issues'])}):")
            for issue in report['issues']:
                print(f"  [{issue['severity'].upper()}] {issue['description']}")
        else:
            print("\nNo issues found!")
        
        print("\n" + "="*60)

def main():
    """Main execution function"""
    evaluator = Step2Evaluator(
        "deemerge_test/data/step2_results.json",
        "deemerge_test/data/benchmark_topics_corrected_fixed.json"
    )
    
    # Generate and save report
    report = evaluator.generate_report()
    evaluator.save_report(report, "deemerge_test/data/step2_evaluation_report.json")
    
    # Print summary
    evaluator.print_summary(report)

if __name__ == "__main__":
    main()


