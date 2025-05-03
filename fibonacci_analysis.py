"""
Advanced Fibonacci Analysis Module
---------------------------------
This module provides sophisticated Fibonacci analysis tools for Elliott Wave trading,
including multi-layered Fibonacci relationships, clusters, and time projections.
"""

import numpy as np
import pandas as pd
import math
from datetime import timedelta

# --- Constants ---
# Standard Fibonacci levels
FIB_RETRACEMENT_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786, 0.886]
FIB_EXTENSION_LEVELS = [1.0, 1.272, 1.414, 1.618, 2.0, 2.618, 3.618, 4.236]
FIB_PROJECTION_LEVELS = [0.618, 1.0, 1.618, 2.0, 2.618]

# Tolerance for Fibonacci level hits
PRICE_TOLERANCE = 0.03  # 3% tolerance for price levels
TIME_TOLERANCE = 0.1    # 10% tolerance for time projections

# --- Fibonacci Price Analysis ---

def calculate_retracement_levels(start_price, end_price):
    """
    Calculate Fibonacci retracement levels from a price move.
    
    Args:
        start_price: Starting price of the move
        end_price: Ending price of the move
        
    Returns:
        Dictionary of Fibonacci retracement levels
    """
    price_range = end_price - start_price
    levels = {}
    
    for fib in FIB_RETRACEMENT_LEVELS:
        levels[fib] = end_price - (price_range * fib)
    
    return levels

def calculate_extension_levels(start_price, end_price, retracement_price=None):
    """
    Calculate Fibonacci extension levels from a price move.
    
    Args:
        start_price: Starting price of the move
        end_price: Ending price of the move
        retracement_price: Price after retracement (for projections)
        
    Returns:
        Dictionary of Fibonacci extension levels
    """
    price_range = end_price - start_price
    levels = {}
    
    base_price = retracement_price if retracement_price is not None else end_price
    
    for fib in FIB_EXTENSION_LEVELS:
        levels[fib] = base_price + (price_range * fib)
    
    return levels

def find_fibonacci_clusters(levels_list, tolerance=PRICE_TOLERANCE):
    """
    Find price zones where multiple Fibonacci levels cluster together.
    
    Args:
        levels_list: List of dictionaries containing Fibonacci levels
        tolerance: Percentage tolerance for clustering
        
    Returns:
        List of clusters with price ranges and strength
    """
    # Flatten all levels into a single list
    all_levels = []
    for levels in levels_list:
        for fib, price in levels.items():
            all_levels.append((price, fib, levels.get('type', 'unknown')))
    
    # Sort by price
    all_levels.sort(key=lambda x: x[0])
    
    # Find clusters
    clusters = []
    i = 0
    while i < len(all_levels):
        base_price = all_levels[i][0]
        cluster_levels = [all_levels[i]]
        
        # Find all levels within tolerance
        j = i + 1
        while j < len(all_levels) and abs(all_levels[j][0] - base_price) / base_price <= tolerance:
            cluster_levels.append(all_levels[j])
            j += 1
        
        # If we found a cluster (more than one level)
        if len(cluster_levels) > 1:
            # Calculate average price and range
            prices = [level[0] for level in cluster_levels]
            avg_price = sum(prices) / len(prices)
            min_price = min(prices)
            max_price = max(prices)
            
            # Create cluster info
            cluster = {
                'center': avg_price,
                'min': min_price,
                'max': max_price,
                'strength': len(cluster_levels),
                'levels': [(level[1], level[2]) for level in cluster_levels]
            }
            clusters.append(cluster)
            
            i = j  # Skip to after this cluster
        else:
            i += 1
    
    # Sort clusters by strength (descending)
    clusters.sort(key=lambda x: x['strength'], reverse=True)
    
    return clusters

def analyze_wave_relationships(wave_points, is_up=True):
    """
    Analyze Fibonacci relationships between waves in an Elliott Wave pattern.
    
    Args:
        wave_points: DataFrame of wave points
        is_up: Boolean indicating if this is an uptrend (True) or downtrend (False)
        
    Returns:
        Dictionary of Fibonacci relationships between waves
    """
    if len(wave_points) < 3:
        return {"error": "Insufficient points for analysis"}
    
    # Extract price levels based on trend direction
    price_levels = {}
    for i, point in enumerate(wave_points.itertuples()):
        if i % 2 == 0:  # Waves 0, 2, 4 (even indices)
            price_levels[i] = point.Low if is_up else point.High
        else:  # Waves 1, 3, 5 (odd indices)
            price_levels[i] = point.High if is_up else point.Low
    
    relationships = {}
    
    # Wave 2 retracement of Wave 1
    if len(price_levels) >= 3:
        wave1_range = abs(price_levels[1] - price_levels[0])
        wave2_retrace = abs(price_levels[1] - price_levels[2]) / wave1_range
        relationships["wave2_retrace"] = wave2_retrace
        relationships["wave2_fib_level"] = find_closest_fib(wave2_retrace, FIB_RETRACEMENT_LEVELS)
    
    # Wave 3 extension of Wave 1
    if len(price_levels) >= 4:
        wave1_range = abs(price_levels[1] - price_levels[0])
        wave3_range = abs(price_levels[3] - price_levels[2])
        wave3_extension = wave3_range / wave1_range
        relationships["wave3_extension"] = wave3_extension
        relationships["wave3_fib_level"] = find_closest_fib(wave3_extension, FIB_EXTENSION_LEVELS)
    
    # Wave 4 retracement of Wave 3
    if len(price_levels) >= 5:
        wave3_range = abs(price_levels[3] - price_levels[2])
        wave4_retrace = abs(price_levels[3] - price_levels[4]) / wave3_range
        relationships["wave4_retrace"] = wave4_retrace
        relationships["wave4_fib_level"] = find_closest_fib(wave4_retrace, FIB_RETRACEMENT_LEVELS)
    
    # Wave 5 relationships
    if len(price_levels) >= 6:
        # Wave 5 to Wave 1 relationship
        wave1_range = abs(price_levels[1] - price_levels[0])
        wave5_range = abs(price_levels[5] - price_levels[4])
        wave5_to_wave1 = wave5_range / wave1_range
        relationships["wave5_to_wave1"] = wave5_to_wave1
        relationships["wave5_to_wave1_fib"] = find_closest_fib(wave5_to_wave1, FIB_EXTENSION_LEVELS)
        
        # Wave 5 extension from Wave 1 to 3
        wave1to3_range = abs(price_levels[3] - price_levels[0])
        wave5_extension = wave5_range / wave1to3_range
        relationships["wave5_extension"] = wave5_extension
        relationships["wave5_extension_fib"] = find_closest_fib(wave5_extension, FIB_EXTENSION_LEVELS)
    
    return relationships

def find_closest_fib(ratio, fib_levels, tolerance=PRICE_TOLERANCE):
    """
    Find the closest Fibonacci level to a given ratio.
    
    Args:
        ratio: The ratio to check
        fib_levels: List of Fibonacci levels to check against
        tolerance: Maximum acceptable deviation
        
    Returns:
        Closest Fibonacci level or None if none are within tolerance
    """
    if math.isnan(ratio):
        return None
    
    closest_level = None
    min_diff = float('inf')
    
    for level in fib_levels:
        diff = abs(ratio - level)
        if diff < min_diff:
            min_diff = diff
            closest_level = level
    
    # Check if within tolerance
    if min_diff <= tolerance:
        return closest_level
    else:
        return None

# --- Fibonacci Time Analysis ---

def calculate_time_projections(start_date, end_date, base="forward"):
    """
    Calculate Fibonacci time projections from a time range.
    
    Args:
        start_date: Starting date of the time range
        end_date: Ending date of the time range
        base: Direction for projections ("forward" or "backward")
        
    Returns:
        Dictionary of Fibonacci time projections
    """
    if not isinstance(start_date, pd.Timestamp):
        start_date = pd.to_datetime(start_date)
    if not isinstance(end_date, pd.Timestamp):
        end_date = pd.to_datetime(end_date)
    
    time_range = (end_date - start_date).total_seconds()
    projections = {}
    
    for fib in FIB_EXTENSION_LEVELS:
        time_delta = timedelta(seconds=time_range * fib)
        if base == "forward":
            projections[fib] = end_date + time_delta
        else:  # backward
            projections[fib] = start_date - time_delta
    
    return projections

def find_time_clusters(projections_list, tolerance_days=5):
    """
    Find time zones where multiple Fibonacci time projections cluster together.
    
    Args:
        projections_list: List of dictionaries containing time projections
        tolerance_days: Number of days tolerance for clustering
        
    Returns:
        List of time clusters with date ranges and strength
    """
    # Flatten all projections into a single list
    all_projections = []
    for projections in projections_list:
        for fib, date in projections.items():
            all_projections.append((date, fib, projections.get('type', 'unknown')))
    
    # Sort by date
    all_projections.sort(key=lambda x: x[0])
    
    # Find clusters
    clusters = []
    i = 0
    while i < len(all_projections):
        base_date = all_projections[i][0]
        cluster_dates = [all_projections[i]]
        
        # Find all dates within tolerance
        j = i + 1
        tolerance = timedelta(days=tolerance_days)
        while j < len(all_projections) and abs(all_projections[j][0] - base_date) <= tolerance:
            cluster_dates.append(all_projections[j])
            j += 1
        
        # If we found a cluster (more than one date)
        if len(cluster_dates) > 1:
            # Calculate average date and range
            dates = [date[0] for date in cluster_dates]
            min_date = min(dates)
            max_date = max(dates)
            
            # Create cluster info
            cluster = {
                'min_date': min_date,
                'max_date': max_date,
                'strength': len(cluster_dates),
                'levels': [(level[1], level[2]) for level in cluster_dates]
            }
            clusters.append(cluster)
            
            i = j  # Skip to after this cluster
        else:
            i += 1
    
    # Sort clusters by strength (descending)
    clusters.sort(key=lambda x: x['strength'], reverse=True)
    
    return clusters

# --- Target and Support/Resistance Analysis ---

def calculate_target_zones(current_price, wave_points, pattern_type, is_up=True):
    """
    Calculate target price zones based on the Elliott Wave pattern.
    
    Args:
        current_price: Current price
        wave_points: DataFrame of wave points
        pattern_type: Type of pattern (impulse, zigzag, etc.)
        is_up: Boolean indicating if this is an uptrend (True) or downtrend (False)
        
    Returns:
        Dictionary of target zones with confidence levels
    """
    targets = {
        "conservative": None,
        "moderate": None,
        "aggressive": None,
        "stop_loss": None,
        "invalidation": None
    }
    
    # Extract price levels based on trend direction
    price_levels = {}
    for i, point in enumerate(wave_points.itertuples()):
        if i % 2 == 0:  # Waves 0, 2, 4 (even indices)
            price_levels[i] = point.Low if is_up else point.High
        else:  # Waves 1, 3, 5 (odd indices)
            price_levels[i] = point.High if is_up else point.Low
    
    # Calculate targets based on pattern type
    if pattern_type == "impulse":
        # Determine which wave we're in
        wave_count = len(price_levels)
        
        if wave_count == 3:  # After Wave 2, projecting Wave 3
            wave1_range = abs(price_levels[1] - price_levels[0])
            
            if is_up:
                base = price_levels[2]  # Wave 2 low
                targets["conservative"] = base + wave1_range * 1.618
                targets["moderate"] = base + wave1_range * 2.0
                targets["aggressive"] = base + wave1_range * 2.618
                targets["stop_loss"] = min(price_levels[0], price_levels[2]) * 0.99
                targets["invalidation"] = price_levels[0]
            else:
                base = price_levels[2]  # Wave 2 high
                targets["conservative"] = base - wave1_range * 1.618
                targets["moderate"] = base - wave1_range * 2.0
                targets["aggressive"] = base - wave1_range * 2.618
                targets["stop_loss"] = max(price_levels[0], price_levels[2]) * 1.01
                targets["invalidation"] = price_levels[0]
        
        elif wave_count == 5:  # After Wave 4, projecting Wave 5
            wave1_range = abs(price_levels[1] - price_levels[0])
            
            if is_up:
                base = price_levels[4]  # Wave 4 low
                targets["conservative"] = base + wave1_range * 0.618
                targets["moderate"] = base + wave1_range * 1.0
                targets["aggressive"] = base + wave1_range * 1.618
                targets["stop_loss"] = price_levels[4] * 0.99
                targets["invalidation"] = price_levels[2]
            else:
                base = price_levels[4]  # Wave 4 high
                targets["conservative"] = base - wave1_range * 0.618
                targets["moderate"] = base - wave1_range * 1.0
                targets["aggressive"] = base - wave1_range * 1.618
                targets["stop_loss"] = price_levels[4] * 1.01
                targets["invalidation"] = price_levels[2]
    
    elif pattern_type == "zigzag":
        # For a zigzag correction, project Wave C
        if len(price_levels) >= 3:
            wave_a_range = abs(price_levels[1] - price_levels[0])
            
            if is_up:
                base = price_levels[2]  # Wave B low
                targets["conservative"] = base + wave_a_range * 0.618
                targets["moderate"] = base + wave_a_range * 1.0
                targets["aggressive"] = base + wave_a_range * 1.618
                targets["stop_loss"] = price_levels[2] * 0.99
                targets["invalidation"] = price_levels[0]
            else:
                base = price_levels[2]  # Wave B high
                targets["conservative"] = base - wave_a_range * 0.618
                targets["moderate"] = base - wave_a_range * 1.0
                targets["aggressive"] = base - wave_a_range * 1.618
                targets["stop_loss"] = price_levels[2] * 1.01
                targets["invalidation"] = price_levels[0]
    
    return targets

def identify_support_resistance(data, lookback=100):
    """
    Identify key support and resistance levels from historical data.
    
    Args:
        data: DataFrame with OHLCV data
        lookback: Number of periods to look back
        
    Returns:
        Dictionary of support and resistance levels with strength
    """
    if len(data) < lookback:
        lookback = len(data)
    
    recent_data = data.iloc[-lookback:].copy()
    
    # Find swing highs and lows
    highs = []
    lows = []
    
    for i in range(2, len(recent_data) - 2):
        # Swing high
        if (recent_data.iloc[i].High > recent_data.iloc[i-1].High and 
            recent_data.iloc[i].High > recent_data.iloc[i-2].High and
            recent_data.iloc[i].High > recent_data.iloc[i+1].High and
            recent_data.iloc[i].High > recent_data.iloc[i+2].High):
            highs.append(recent_data.iloc[i].High)
        
        # Swing low
        if (recent_data.iloc[i].Low < recent_data.iloc[i-1].Low and 
            recent_data.iloc[i].Low < recent_data.iloc[i-2].Low and
            recent_data.iloc[i].Low < recent_data.iloc[i+1].Low and
            recent_data.iloc[i].Low < recent_data.iloc[i+2].Low):
            lows.append(recent_data.iloc[i].Low)
    
    # Group nearby levels
    resistance_zones = group_price_levels(highs, tolerance=0.02)
    support_zones = group_price_levels(lows, tolerance=0.02)
    
    return {
        "resistance": resistance_zones,
        "support": support_zones
    }

def group_price_levels(prices, tolerance=0.02):
    """
    Group nearby price levels into zones.
    
    Args:
        prices: List of price levels
        tolerance: Percentage tolerance for grouping
        
    Returns:
        List of price zones with center price and strength
    """
    if not prices:
        return []
    
    # Sort prices
    sorted_prices = sorted(prices)
    
    # Group nearby prices
    zones = []
    current_zone = [sorted_prices[0]]
    
    for i in range(1, len(sorted_prices)):
        current_price = sorted_prices[i]
        prev_price = sorted_prices[i-1]
        
        # If price is within tolerance of previous price, add to current zone
        if (current_price - prev_price) / prev_price <= tolerance:
            current_zone.append(current_price)
        else:
            # Create a new zone
            avg_price = sum(current_zone) / len(current_zone)
            zones.append({
                "price": avg_price,
                "strength": len(current_zone)
            })
            current_zone = [current_price]
    
    # Add the last zone
    if current_zone:
        avg_price = sum(current_zone) / len(current_zone)
        zones.append({
            "price": avg_price,
            "strength": len(current_zone)
        })
    
    # Sort zones by strength
    zones.sort(key=lambda x: x["strength"], reverse=True)
    
    return zones
