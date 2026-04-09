import torch
import pyhessian
import time
import numpy as np
from typing import List
def search_top_k_dominant_bulk_space(eigenvalues: List[float], method: str='gap', threshold: float=0.0003):
    pass
    sorted_eigenvalues = sorted(eigenvalues, reverse=True)
    if method == 'gap':
        gaps = [sorted_eigenvalues[i] - sorted_eigenvalues[i + 1] for i in range(len(sorted_eigenvalues) - 1)]
        max_gap_value = max(gaps)
        max_gap_index = gaps.index(max_gap_value)
        gaps_copy = gaps.copy()
        gaps_copy.pop(max_gap_index)
        second_max_gap_value = max(gaps_copy)
        second_max_gap_index = gaps.index(second_max_gap_value)
        if max_gap_index < second_max_gap_index:
            dominant_start = 0
            dominant_end = max_gap_index + 1
            bulk_start = dominant_end
            bulk_end = second_max_gap_index + 1
        else:
            dominant_start = 0
            dominant_end = second_max_gap_index + 1
            bulk_start = dominant_end
            bulk_end = max_gap_index + 1
        zero_start = bulk_end
        zero_end = len(sorted_eigenvalues)
        return {
            'dominant_start': dominant_start,
            'dominant_end': dominant_end,
            'bulk_start': bulk_start,
            'bulk_end': bulk_end,
            'zero_start': zero_start,
            'zero_end': zero_end,
            'max_gap_index': max_gap_index,
            'second_max_gap_index': second_max_gap_index,
            'max_gap_value': max_gap_value,
            'second_max_gap_value': second_max_gap_value
        }
    elif method == 'log_gap':
        log_gaps = []
        for i in range(len(sorted_eigenvalues) - 1):
            ratio = sorted_eigenvalues[i] / sorted_eigenvalues[i + 1]
            if ratio <= 0:
                log_gaps.append(-10)
            else:
                log_gaps.append(np.log(ratio))
        max_log_gap_index = log_gaps.index(max(log_gaps))
        top_k = max_log_gap_index + 1
        return top_k, log_gaps
    elif method == 'threshold':
        filtered_eigenvalues = [ev for ev in sorted_eigenvalues if ev > threshold]
        if len(filtered_eigenvalues) == 0:
            return 0, []
        if len(filtered_eigenvalues) == 1:
            return 1, [0.0]
        gaps = []
        for i in range(len(filtered_eigenvalues) - 1):
            gap = filtered_eigenvalues[i] - filtered_eigenvalues[i + 1]
            gaps.append(gap)
        max_gap_index = gaps.index(max(gaps))
        top_k = max_gap_index + 1
        for i, gap in enumerate(gaps):
            mark = " <- MAX GAP" if i == max_gap_index else ""
            print(f"   Gap {i+1}: {gap:.6f} (between λ{i+1} and λ{i+2}){mark}")
        return top_k, gaps
    elif method == 'adaptive_threshold':
        if len(sorted_eigenvalues) == 0:
            return {'dominant_k': 0, 'bulk_start': 0, 'bulk_end': 0, 'gaps': []}
        percentage = 0.01
        max_eigenval = sorted_eigenvalues[0]
        adaptive_threshold = max_eigenval * percentage
        filtered_eigenvalues = [ev for ev in sorted_eigenvalues if ev > adaptive_threshold]
        if len(filtered_eigenvalues) <= 1:
            return {'dominant_k': len(filtered_eigenvalues), 'bulk_start': len(filtered_eigenvalues), 'bulk_end': len(filtered_eigenvalues), 'gaps': [0.0]}
        gaps = []
        for i in range(len(filtered_eigenvalues) - 1):
            gap = filtered_eigenvalues[i] - filtered_eigenvalues[i + 1]
            gaps.append(gap)
        max_gap_value = max(gaps)
        max_gap_index = gaps.index(max_gap_value)
        dominant_k = max_gap_index + 1
        if max_gap_index + 1 < len(gaps):
            remaining_gaps = gaps[max_gap_index + 1:]
            second_max_gap_value = max(remaining_gaps)
            second_max_gap_index = gaps.index(second_max_gap_value, max_gap_index + 1)
            bulk_end = second_max_gap_index + 1
        else:
            second_max_gap_value = 0.0
            second_max_gap_index = len(gaps) - 1
            bulk_end = len(filtered_eigenvalues)
        bulk_start = dominant_k
        return {
            'dominant_k': dominant_k,
            'bulk_start': bulk_start,
            'bulk_end': bulk_end,
            'gaps': gaps,
            'max_gap_index': max_gap_index,
            'second_max_gap_index': second_max_gap_index,
            'max_gap_value': max_gap_value,
            'second_max_gap_value': second_max_gap_value,
            'filtered_eigenvalues': filtered_eigenvalues
        }
def analyze_eigenvalue_distribution(eigenvalues: List[float], thresholds: List[float] = None):
    pass
    if thresholds is None:
        thresholds = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1]
    sorted_eigenvalues = sorted(eigenvalues, reverse=True)
    for threshold in thresholds:
        filtered_count = sum(1 for ev in sorted_eigenvalues if ev > threshold)
    return {
        'total_count': len(sorted_eigenvalues),
        'max_eigenvalue': sorted_eigenvalues[0],
        'min_eigenvalue': sorted_eigenvalues[-1],
        'positive_count': sum(1 for ev in sorted_eigenvalues if ev > 0),
        'negative_count': sum(1 for ev in sorted_eigenvalues if ev < 0),
        'near_zero_count': sum(1 for ev in sorted_eigenvalues if abs(ev) < 1e-8),
        'threshold_analysis': {str(t): sum(1 for ev in sorted_eigenvalues if ev > t) for t in thresholds}
    }
def analyze_eigenvalue_spaces(eigenvalues: List[float], method: str='adaptive_threshold', threshold: float=0.0003):
    pass
    result = search_top_k_dominant_bulk_space(eigenvalues, method, threshold)
    sorted_eigenvalues = sorted(eigenvalues, reverse=True)
    dominant_eigenvalues = sorted_eigenvalues[:result['dominant_k']]
    for i, ev in enumerate(dominant_eigenvalues):
        print(f"   λ{i+1}: {ev:.6f}")
    if result['bulk_start'] < result['bulk_end']:
        bulk_eigenvalues = sorted_eigenvalues[result['bulk_start']:result['bulk_end']]
        for i, ev in enumerate(bulk_eigenvalues):
            print(f"   λ{result['bulk_start']+i+1}: {ev:.6f}")
    near_zero_start = result['bulk_end']
    if near_zero_start < len(sorted_eigenvalues):
        near_zero_eigenvalues = sorted_eigenvalues[near_zero_start:]
    return result
