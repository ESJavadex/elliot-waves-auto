"""
Trade Signal Generation Module
------------------------------
This module provides enhanced trade signal generation based on Elliott Wave analysis,
with specific entry and exit points, risk management, and profit targets.
"""

import numpy as np
import pandas as pd
import math

# --- Constants ---
# Risk management
MAX_RISK_PERCENT = 0.02  # Maximum risk per trade (2% of account)
DEFAULT_STOP_ATR_MULT = 1.5  # Default ATR multiplier for stop loss
PROFIT_TARGET_RATIO = 2.0  # Risk:Reward ratio (minimum)

# Fibonacci levels for targets
FIB_TARGETS = {
    "conservative": [0.618, 1.0],
    "moderate": [1.0, 1.618],
    "aggressive": [1.618, 2.618]
}

def generate_trade_signals(wave_points, analysis_results, data_full):
    """
    Generate trade signals based on Elliott Wave analysis.
    
    Args:
        wave_points: DataFrame of labeled wave points
        analysis_results: Results from Elliott Wave analysis
        data_full: DataFrame with price and indicator data
        
    Returns:
        Dictionary with trade signals and risk management parameters
    """
    signals = {
        "entry": None,
        "stop_loss": None,
        "targets": [],
        "risk_reward": None,
        "confidence": "low",
        "direction": None,
        "pattern_type": None,
        "entry_criteria": [],
        "exit_criteria": []
    }
    
    # Check if we have valid analysis results
    if not analysis_results.get("found_impulse") and not analysis_results.get("details", {}).get("corrective_wave"):
        return signals
    
    # Get the last labeled point
    last_label = analysis_results.get("last_label")
    if not last_label:
        return signals
    
    # Determine if we have an impulse or corrective pattern
    has_impulse = analysis_results.get("found_impulse", False)
    has_corrective = "corrective_wave" in analysis_results.get("details", {})
    
    # Calculate ATR for stop loss placement
    atr = None
    if "ATR" in data_full.columns:
        atr = data_full["ATR"].iloc[-1]
    
    # Get current price (last close)
    current_price = data_full["Close"].iloc[-1]
    
    # --- Impulse Wave Signals ---
    if has_impulse:
        details = analysis_results.get("details", {})
        is_upward = details.get("is_upward", True)
        signals["direction"] = "long" if is_upward else "short"
        signals["pattern_type"] = "impulse"
        
        # Determine confidence based on score and guidelines
        score = details.get("score", 0)
        guidelines = details.get("guidelines", {})
        
        if score > 500:
            signals["confidence"] = "high"
        elif score > 300:
            signals["confidence"] = "medium"
        
        # Generate signals based on wave count
        if last_label == "5":
            # End of impulse - prepare for reversal
            signals["entry_criteria"].append("Wait for confirmation of reversal (A-B-C pattern)")
            signals["pattern_type"] = "impulse_complete"
            
            # Potential reversal entry after wave 5
            if is_upward:
                # Bearish setup after bullish impulse
                signals["direction"] = "short"
                signals["entry_criteria"].append("Enter short on break below wave 4")
                
                # Find wave 4 point for stop reference
                wave4_points = wave_points[wave_points["EW_Label"] == "4"]
                if not wave4_points.empty:
                    wave4_low = wave4_points.iloc[0]["Low"]
                    signals["entry"] = wave4_low * 0.99  # Entry slightly below wave 4
                    
                    # Stop loss above wave 5
                    wave5_points = wave_points[wave_points["EW_Label"] == "5"]
                    if not wave5_points.empty:
                        wave5_high = wave5_points.iloc[0]["High"]
                        signals["stop_loss"] = wave5_high * 1.01
                        
                        # Calculate risk and targets
                        risk = signals["stop_loss"] - signals["entry"]
                        signals["targets"] = [
                            signals["entry"] - risk * 1.0,  # 1:1
                            signals["entry"] - risk * 1.618,  # 1.618:1
                            signals["entry"] - risk * 2.618   # 2.618:1
                        ]
                        signals["risk_reward"] = 1.618  # Minimum target
            else:
                # Bullish setup after bearish impulse
                signals["direction"] = "long"
                signals["entry_criteria"].append("Enter long on break above wave 4")
                
                # Find wave 4 point for stop reference
                wave4_points = wave_points[wave_points["EW_Label"] == "4"]
                if not wave4_points.empty:
                    wave4_high = wave4_points.iloc[0]["High"]
                    signals["entry"] = wave4_high * 1.01  # Entry slightly above wave 4
                    
                    # Stop loss below wave 5
                    wave5_points = wave_points[wave_points["EW_Label"] == "5"]
                    if not wave5_points.empty:
                        wave5_low = wave5_points.iloc[0]["Low"]
                        signals["stop_loss"] = wave5_low * 0.99
                        
                        # Calculate risk and targets
                        risk = signals["entry"] - signals["stop_loss"]
                        signals["targets"] = [
                            signals["entry"] + risk * 1.0,  # 1:1
                            signals["entry"] + risk * 1.618,  # 1.618:1
                            signals["entry"] + risk * 2.618   # 2.618:1
                        ]
                        signals["risk_reward"] = 1.618  # Minimum target
        
        elif last_label == "3":
            # End of wave 3 - potential wave 4 pullback then wave 5 continuation
            signals["pattern_type"] = "impulse_wave3"
            
            if is_upward:
                signals["direction"] = "long"
                signals["entry_criteria"].append("Wait for wave 4 pullback (38.2-50% of wave 3)")
                signals["entry_criteria"].append("Enter long on reversal from wave 4 with confirmation")
                
                # Find wave 3 point
                wave3_points = wave_points[wave_points["EW_Label"] == "3"]
                wave2_points = wave_points[wave_points["EW_Label"] == "2"]
                
                if not wave3_points.empty and not wave2_points.empty:
                    wave3_high = wave3_points.iloc[0]["High"]
                    wave2_low = wave2_points.iloc[0]["Low"]
                    wave3_size = wave3_high - wave2_low
                    
                    # Estimate wave 4 pullback zone (38.2-50% retracement of wave 3)
                    wave4_38_level = wave3_high - (wave3_size * 0.382)
                    wave4_50_level = wave3_high - (wave3_size * 0.5)
                    
                    signals["entry"] = (wave4_38_level + wave4_50_level) / 2  # Middle of zone
                    signals["stop_loss"] = wave2_low * 0.99  # Below wave 2
                    
                    # Calculate risk and targets
                    risk = signals["entry"] - signals["stop_loss"]
                    
                    # Wave 5 targets (equal to wave 1, or 0.618 of waves 1-3)
                    wave1_points = wave_points[wave_points["EW_Label"] == "1"]
                    if not wave1_points.empty:
                        wave1_size = wave1_points.iloc[0]["High"] - wave1_points.iloc[0]["Low"]
                        wave5_target = signals["entry"] + wave1_size
                        
                        signals["targets"] = [
                            signals["entry"] + risk * 1.0,  # 1:1
                            wave5_target,  # Wave 5 = Wave 1
                            wave3_high + (wave3_size * 0.382)  # Extension beyond wave 3
                        ]
                        signals["risk_reward"] = (wave5_target - signals["entry"]) / risk
            else:
                # Bearish impulse
                signals["direction"] = "short"
                signals["entry_criteria"].append("Wait for wave 4 pullback (38.2-50% of wave 3)")
                signals["entry_criteria"].append("Enter short on reversal from wave 4 with confirmation")
                
                # Similar logic as above but for bearish setup
                wave3_points = wave_points[wave_points["EW_Label"] == "3"]
                wave2_points = wave_points[wave_points["EW_Label"] == "2"]
                
                if not wave3_points.empty and not wave2_points.empty:
                    wave3_low = wave3_points.iloc[0]["Low"]
                    wave2_high = wave2_points.iloc[0]["High"]
                    wave3_size = wave2_high - wave3_low
                    
                    # Estimate wave 4 pullback zone (38.2-50% retracement of wave 3)
                    wave4_38_level = wave3_low + (wave3_size * 0.382)
                    wave4_50_level = wave3_low + (wave3_size * 0.5)
                    
                    signals["entry"] = (wave4_38_level + wave4_50_level) / 2  # Middle of zone
                    signals["stop_loss"] = wave2_high * 1.01  # Above wave 2
                    
                    # Calculate risk and targets
                    risk = signals["stop_loss"] - signals["entry"]
                    
                    # Wave 5 targets
                    wave1_points = wave_points[wave_points["EW_Label"] == "1"]
                    if not wave1_points.empty:
                        wave1_size = wave1_points.iloc[0]["High"] - wave1_points.iloc[0]["Low"]
                        wave5_target = signals["entry"] - wave1_size
                        
                        signals["targets"] = [
                            signals["entry"] - risk * 1.0,  # 1:1
                            wave5_target,  # Wave 5 = Wave 1
                            wave3_low - (wave3_size * 0.382)  # Extension beyond wave 3
                        ]
                        signals["risk_reward"] = (signals["entry"] - wave5_target) / risk
    
    # --- Corrective Wave Signals ---
    elif has_corrective or last_label in ["A", "B", "C", "(A?)", "(B?)", "(C?)"]:
        corrective_details = analysis_results.get("details", {}).get("corrective_wave", {})
        if not corrective_details and "correction_guess" in analysis_results.get("details", {}):
            # Using the basic correction guess
            signals["pattern_type"] = "correction_guess"
            signals["confidence"] = "low"
            
            # Determine direction based on the correction
            # After a bearish correction (A down, B up, C down), expect bullish move
            # After a bullish correction (A up, B down, C up), expect bearish move
            
            correction_points = analysis_results["details"]["correction_guess"]
            if correction_points["A"]["Close"] > correction_points["C"]["Close"]:
                # Bearish correction, expect bullish move
                signals["direction"] = "long"
                signals["entry_criteria"].append("Enter long on break above point B")
                signals["stop_loss"] = correction_points["C"]["Low"] * 0.99
                signals["entry"] = correction_points["B"]["High"] * 1.01
            else:
                # Bullish correction, expect bearish move
                signals["direction"] = "short"
                signals["entry_criteria"].append("Enter short on break below point B")
                signals["stop_loss"] = correction_points["C"]["High"] * 1.01
                signals["entry"] = correction_points["B"]["Low"] * 0.99
            
            # Calculate risk and targets
            if signals["entry"] and signals["stop_loss"]:
                risk = abs(signals["entry"] - signals["stop_loss"])
                if signals["direction"] == "long":
                    signals["targets"] = [
                        signals["entry"] + risk * 1.0,
                        signals["entry"] + risk * 1.618,
                        signals["entry"] + risk * 2.618
                    ]
                else:
                    signals["targets"] = [
                        signals["entry"] - risk * 1.0,
                        signals["entry"] - risk * 1.618,
                        signals["entry"] - risk * 2.618
                    ]
                signals["risk_reward"] = 1.618
        else:
            # Using the improved corrective wave detection
            signals["pattern_type"] = "corrective_wave"
            pattern_type = corrective_details.get("pattern_type", "zigzag")
            is_up = corrective_details.get("is_up", True)
            
            # For a zigzag (A-B-C) correction in a larger uptrend, expect continuation up after C
            # For a zigzag correction in a larger downtrend, expect continuation down after C
            
            if pattern_type == "zigzag":
                signals["confidence"] = "medium"
                
                # Find the labeled points
                a_points = wave_points[wave_points["EW_Label"] == "A"]
                b_points = wave_points[wave_points["EW_Label"] == "B"]
                c_points = wave_points[wave_points["EW_Label"] == "C"]
                
                if not a_points.empty and not b_points.empty and not c_points.empty:
                    a_point = a_points.iloc[0]
                    b_point = b_points.iloc[0]
                    c_point = c_points.iloc[0]
                    
                    if is_up:
                        # Bullish setup after a bearish correction
                        signals["direction"] = "long"
                        signals["entry_criteria"].append("Enter long on break above point B")
                        signals["entry"] = b_point["High"] * 1.01
                        signals["stop_loss"] = c_point["Low"] * 0.99
                        
                        # Calculate risk and targets
                        risk = signals["entry"] - signals["stop_loss"]
                        
                        # Targets based on the size of the correction
                        correction_size = a_point["High"] - c_point["Low"]
                        signals["targets"] = [
                            signals["entry"] + risk * 1.0,  # 1:1
                            c_point["Low"] + correction_size,  # 100% of correction
                            c_point["Low"] + correction_size * 1.618  # 161.8% of correction
                        ]
                    else:
                        # Bearish setup after a bullish correction
                        signals["direction"] = "short"
                        signals["entry_criteria"].append("Enter short on break below point B")
                        signals["entry"] = b_point["Low"] * 0.99
                        signals["stop_loss"] = c_point["High"] * 1.01
                        
                        # Calculate risk and targets
                        risk = signals["stop_loss"] - signals["entry"]
                        
                        # Targets based on the size of the correction
                        correction_size = a_point["Low"] - c_point["High"]
                        signals["targets"] = [
                            signals["entry"] - risk * 1.0,  # 1:1
                            c_point["High"] + correction_size,  # 100% of correction
                            c_point["High"] + correction_size * 1.618  # 161.8% of correction
                        ]
                    
                    signals["risk_reward"] = abs(signals["targets"][1] - signals["entry"]) / risk
            
            elif pattern_type == "flat":
                signals["confidence"] = "medium"
                # Similar logic for flat corrections
                # ...
    
    # Add exit criteria
    signals["exit_criteria"] = [
        "Exit at target levels",
        "Exit if stop loss is hit",
        "Exit if pattern invalidation occurs"
    ]
    
    # Calculate position size based on risk
    if signals["entry"] is not None and signals["stop_loss"] is not None:
        risk_per_share = abs(signals["entry"] - signals["stop_loss"])
        if risk_per_share > 0:
            signals["max_position_size"] = f"{MAX_RISK_PERCENT * 100}% of account / ${risk_per_share:.2f} per share"
    
    return signals

def calculate_position_size(account_size, risk_percent, entry_price, stop_loss_price):
    """
    Calculate the appropriate position size based on risk management parameters.
    
    Args:
        account_size: Total account size in currency units
        risk_percent: Maximum risk percentage (e.g., 0.02 for 2%)
        entry_price: Entry price per share
        stop_loss_price: Stop loss price per share
        
    Returns:
        Dictionary with position sizing information
    """
    risk_amount = account_size * risk_percent
    risk_per_share = abs(entry_price - stop_loss_price)
    
    if risk_per_share <= 0:
        return {
            "shares": 0,
            "risk_amount": 0,
            "risk_percent": 0,
            "error": "Invalid risk per share"
        }
    
    shares = int(risk_amount / risk_per_share)
    actual_risk = shares * risk_per_share
    actual_risk_percent = actual_risk / account_size
    
    return {
        "shares": shares,
        "risk_amount": actual_risk,
        "risk_percent": actual_risk_percent,
        "max_loss": actual_risk
    }

def analyze_trade_opportunity(wave_points, analysis_results, data_full, account_size=10000):
    print("  [Trade Analysis] Analyzing trade opportunities with enhanced criteria:")
    print("    - Elliott Wave pattern completion status")
    print("    - Fibonacci entry and exit points")
    print("    - Risk-to-reward calculation")
    print("    - Position sizing based on risk management")
    """
    Analyze a potential trade opportunity based on Elliott Wave analysis.
    
    Args:
        wave_points: DataFrame of labeled wave points
        analysis_results: Results from Elliott Wave analysis
        data_full: DataFrame with price and indicator data
        account_size: Account size for position sizing
        
    Returns:
        Dictionary with trade analysis and recommendations
    """
    # Generate trade signals
    signals = generate_trade_signals(wave_points, analysis_results, data_full)
    
    # Calculate position size if we have valid entry and stop loss
    position_sizing = None
    if signals["entry"] is not None and signals["stop_loss"] is not None:
        position_sizing = calculate_position_size(
            account_size=account_size,
            risk_percent=MAX_RISK_PERCENT,
            entry_price=signals["entry"],
            stop_loss_price=signals["stop_loss"]
        )
    
    # Combine results
    trade_analysis = {
        "signals": signals,
        "position_sizing": position_sizing,
        "current_price": data_full["Close"].iloc[-1],
        "timestamp": data_full.index[-1],
        "recommendation": "No trade" if signals["confidence"] == "low" else f"{signals['direction'].upper()} - {signals['confidence'].upper()} confidence"
    }
    
    return trade_analysis
