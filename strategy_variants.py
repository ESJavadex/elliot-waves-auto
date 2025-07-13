"""
Elliott Wave Strategy Variants
============================
Multiple Elliott Wave + Fibonacci strategies with different risk profiles and approaches.
Each strategy focuses on different aspects of Elliott Wave theory and Fibonacci analysis.
"""

import numpy as np
import pandas as pd
from fibonacci_analysis import analyze_wave_relationships, calculate_target_zones, find_fibonacci_clusters

class ElliotWaveStrategies:
    """Collection of Elliott Wave trading strategies with different approaches."""
    
    def __init__(self):
        self.strategies = {
            'conservative': self.conservative_elliott_wave,
            'aggressive': self.aggressive_elliott_wave,
            'fibonacci_confluence': self.fibonacci_confluence_strategy,
            'wave_validation': self.wave_count_validation,
            'hybrid_momentum': self.hybrid_momentum_strategy
        }
    
    def get_strategy(self, strategy_name):
        """Get a specific strategy function."""
        return self.strategies.get(strategy_name, self.conservative_elliott_wave)
    
    def list_strategies(self):
        """Return list of available strategies with descriptions."""
        return {
            'conservative': {
                'name': 'Conservative Elliott Wave',
                'description': 'High confidence trades with strict Elliott Wave rules',
                'risk_level': 'Low',
                'win_rate': 'High',
                'trade_frequency': 'Low'
            },
            'aggressive': {
                'name': 'Aggressive Elliott Wave', 
                'description': 'More frequent trades with relaxed thresholds',
                'risk_level': 'High',
                'win_rate': 'Medium',
                'trade_frequency': 'High'
            },
            'fibonacci_confluence': {
                'name': 'Fibonacci Confluence',
                'description': 'Focus on Fibonacci level convergence and clusters',
                'risk_level': 'Medium',
                'win_rate': 'High',
                'trade_frequency': 'Medium'
            },
            'wave_validation': {
                'name': 'Wave Count Validation',
                'description': 'Multiple timeframe wave count confirmation',
                'risk_level': 'Low',
                'win_rate': 'Very High',
                'trade_frequency': 'Very Low'
            },
            'hybrid_momentum': {
                'name': 'Hybrid Momentum',
                'description': 'Elliott Wave + RSI/Volume momentum confirmation',
                'risk_level': 'Medium',
                'win_rate': 'Medium-High',
                'trade_frequency': 'Medium'
            }
        }

    def conservative_elliott_wave(self, data, analysis, sr_levels, current_price):
        """
        Conservative Elliott Wave Strategy
        - High confidence threshold (>350 points)
        - Strict Elliott Wave rule compliance
        - Conservative Fibonacci targets
        - Higher risk-reward ratios (3:1 minimum)
        """
        signal = self._init_signal()
        
        if not analysis["valid"] or analysis["confidence_score"] < 350:
            signal["reasoning"].append("Confidence score below conservative threshold (350)")
            return signal
            
        pattern_type = analysis["pattern_type"]
        is_up = analysis["is_up"]
        current_wave = analysis["current_wave"]["wave"]
        
        # Only trade after Wave 2 and Wave 4 completions
        if pattern_type == "impulse" and current_wave in ["2", "4"]:
            signal["signal"] = "buy" if is_up else "sell"
            signal["direction"] = "long" if is_up else "short"
            signal["confidence"] = "high"
            
            wave_points = analysis["details"].get("wave_points", [])
            targets = calculate_target_zones(current_price, wave_points, pattern_type, is_up)
            
            if targets["conservative"]:
                signal["entry"] = current_price
                signal["stop_loss"] = targets["stop_loss"]
                
                # Conservative targets - only use conservative and moderate
                signal["targets"] = [
                    targets["conservative"],
                    targets["moderate"]
                ]
                
                # Calculate risk-reward (must be >3:1 for conservative)
                risk = abs(signal["entry"] - signal["stop_loss"])
                reward = abs(signal["targets"][0] - signal["entry"])
                signal["risk_reward"] = reward / risk if risk > 0 else 0
                
                if signal["risk_reward"] < 3.0:
                    signal["signal"] = "no_trade"
                    signal["reasoning"].append(f"Risk-reward ratio ({signal['risk_reward']:.2f}) below conservative minimum (3.0)")
                else:
                    signal["reasoning"].append(f"Conservative Wave {current_wave} completion with RR: {signal['risk_reward']:.2f}")
        
        return signal

    def aggressive_elliott_wave(self, data, analysis, sr_levels, current_price):
        """
        Aggressive Elliott Wave Strategy
        - Lower confidence threshold (>150 points)
        - More lenient Elliott Wave interpretation
        - Aggressive Fibonacci targets
        - Lower risk-reward ratios (1.5:1 minimum)
        """
        signal = self._init_signal()
        
        if not analysis["valid"] or analysis["confidence_score"] < 150:
            signal["reasoning"].append("Confidence score below aggressive threshold (150)")
            return signal
            
        pattern_type = analysis["pattern_type"]
        is_up = analysis["is_up"]
        current_wave = analysis["current_wave"]["wave"]
        
        # Trade more wave completions including Wave 1 and 3
        if pattern_type == "impulse" and current_wave in ["1", "2", "3", "4"]:
            signal["signal"] = "buy" if is_up else "sell"
            signal["direction"] = "long" if is_up else "short"
            signal["confidence"] = "medium" if current_wave in ["1", "3"] else "high"
            
            wave_points = analysis["details"].get("wave_points", [])
            targets = calculate_target_zones(current_price, wave_points, pattern_type, is_up)
            
            if targets["conservative"]:
                signal["entry"] = current_price
                signal["stop_loss"] = targets["stop_loss"]
                
                # Aggressive targets - use all levels
                signal["targets"] = [
                    targets["conservative"],
                    targets["moderate"],
                    targets["aggressive"]
                ]
                
                # Lower risk-reward requirement
                risk = abs(signal["entry"] - signal["stop_loss"])
                reward = abs(signal["targets"][1] - signal["entry"])  # Use moderate target
                signal["risk_reward"] = reward / risk if risk > 0 else 0
                
                if signal["risk_reward"] < 1.5:
                    signal["signal"] = "no_trade"
                    signal["reasoning"].append(f"Risk-reward ratio ({signal['risk_reward']:.2f}) below aggressive minimum (1.5)")
                else:
                    signal["reasoning"].append(f"Aggressive Wave {current_wave} signal with RR: {signal['risk_reward']:.2f}")
        
        # Also trade corrective pattern completions
        elif pattern_type in ["zigzag", "flat"] and current_wave == "C":
            signal["signal"] = "buy" if is_up else "sell"
            signal["direction"] = "long" if is_up else "short"
            signal["confidence"] = "medium"
            signal["reasoning"].append(f"Aggressive corrective {pattern_type} completion")
        
        return signal

    def fibonacci_confluence_strategy(self, data, analysis, sr_levels, current_price):
        """
        Fibonacci Confluence Strategy
        - Focus on Fibonacci level convergence
        - Trade when multiple Fibonacci levels align
        - Use Fibonacci clusters for entries and exits
        """
        signal = self._init_signal()
        
        if not analysis["valid"] or analysis["confidence_score"] < 200:
            return signal
            
        pattern_type = analysis["pattern_type"]
        is_up = analysis["is_up"]
        current_wave = analysis["current_wave"]["wave"]
        wave_points = analysis["details"].get("wave_points", [])
        
        # Find Fibonacci clusters around current price
        clusters = find_fibonacci_clusters(wave_points, current_price, tolerance=0.02)
        
        if clusters and len(clusters) >= 2:  # At least 2 Fibonacci levels converging
            signal["signal"] = "buy" if is_up else "sell"
            signal["direction"] = "long" if is_up else "short"
            signal["confidence"] = "high"
            
            # Entry at Fibonacci confluence
            signal["entry"] = current_price
            
            # Stop loss beyond the next major Fibonacci level
            if is_up:
                signal["stop_loss"] = min([cluster["level"] for cluster in clusters]) * 0.98
            else:
                signal["stop_loss"] = max([cluster["level"] for cluster in clusters]) * 1.02
            
            # Targets at next Fibonacci extensions
            targets = calculate_target_zones(current_price, wave_points, pattern_type, is_up)
            if targets["conservative"]:
                signal["targets"] = [
                    targets["conservative"],
                    targets["moderate"],
                    targets["aggressive"]
                ]
                
                risk = abs(signal["entry"] - signal["stop_loss"])
                reward = abs(signal["targets"][1] - signal["entry"])
                signal["risk_reward"] = reward / risk if risk > 0 else 0
                
                cluster_levels = [c["type"] for c in clusters]
                signal["reasoning"].append(f"Fibonacci confluence at {cluster_levels} with RR: {signal['risk_reward']:.2f}")
        
        return signal

    def wave_count_validation(self, data, analysis, sr_levels, current_price):
        """
        Wave Count Validation Strategy
        - Multiple timeframe confirmation
        - Very strict wave counting rules
        - Only trade when wave count is unambiguous
        """
        signal = self._init_signal()
        
        if not analysis["valid"] or analysis["confidence_score"] < 400:
            signal["reasoning"].append("Confidence score below validation threshold (400)")
            return signal
            
        pattern_type = analysis["pattern_type"]
        is_up = analysis["is_up"]
        current_wave = analysis["current_wave"]["wave"]
        
        # Additional validation checks
        guidelines = analysis["details"].get("guidelines", {})
        
        # Check Elliott Wave guidelines compliance
        guideline_score = 0
        if guidelines.get("alternation", False):
            guideline_score += 1
        if guidelines.get("fibonacci_relationships", False):
            guideline_score += 1
        if guidelines.get("volume_confirmation", False):
            guideline_score += 1
        
        # Only trade if at least 2 guidelines are met
        if guideline_score < 2:
            signal["reasoning"].append(f"Insufficient guideline compliance ({guideline_score}/3)")
            return signal
            
        # Only trade Wave 2 and Wave 4 completions with high validation
        if pattern_type == "impulse" and current_wave in ["2", "4"]:
            signal["signal"] = "buy" if is_up else "sell"
            signal["direction"] = "long" if is_up else "short"
            signal["confidence"] = "very_high"
            
            wave_points = analysis["details"].get("wave_points", [])
            targets = calculate_target_zones(current_price, wave_points, pattern_type, is_up)
            
            if targets["conservative"]:
                signal["entry"] = current_price
                signal["stop_loss"] = targets["stop_loss"]
                signal["targets"] = [targets["conservative"], targets["moderate"]]
                
                risk = abs(signal["entry"] - signal["stop_loss"])
                reward = abs(signal["targets"][0] - signal["entry"])
                signal["risk_reward"] = reward / risk if risk > 0 else 0
                
                signal["reasoning"].append(f"Validated Wave {current_wave} with {guideline_score}/3 guidelines met")
        
        return signal

    def hybrid_momentum_strategy(self, data, analysis, sr_levels, current_price):
        """
        Hybrid Momentum Strategy
        - Elliott Wave + RSI divergence
        - Volume confirmation
        - Momentum indicators alignment
        """
        signal = self._init_signal()
        
        if not analysis["valid"] or analysis["confidence_score"] < 250:
            return signal
            
        pattern_type = analysis["pattern_type"]
        is_up = analysis["is_up"]
        current_wave = analysis["current_wave"]["wave"]
        
        # Check for momentum indicators
        has_rsi = 'RSI' in data.columns
        has_volume = 'Volume' in data.columns
        
        momentum_score = 0
        momentum_reasons = []
        
        # RSI divergence check
        if has_rsi and len(data) >= 20:
            rsi = data['RSI'].iloc[-1]
            if pattern_type == "impulse":
                if is_up and current_wave == "2" and rsi < 40:  # Oversold on Wave 2
                    momentum_score += 1
                    momentum_reasons.append("RSI oversold on Wave 2")
                elif not is_up and current_wave == "2" and rsi > 60:  # Overbought on Wave 2
                    momentum_score += 1
                    momentum_reasons.append("RSI overbought on Wave 2")
        
        # Volume confirmation
        if has_volume and len(data) >= 20:
            avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
            current_volume = data['Volume'].iloc[-1]
            if current_volume > avg_volume * 1.2:  # 20% above average
                momentum_score += 1
                momentum_reasons.append("Above average volume")
        
        # Price momentum
        if len(data) >= 10:
            price_change = (data['Close'].iloc[-1] - data['Close'].iloc[-10]) / data['Close'].iloc[-10]
            if abs(price_change) > 0.05:  # 5% move in 10 periods
                momentum_score += 1
                momentum_reasons.append(f"Strong price momentum ({price_change*100:.1f}%)")
        
        # Require at least 2 momentum confirmations
        if momentum_score >= 2 and pattern_type == "impulse" and current_wave in ["2", "4"]:
            signal["signal"] = "buy" if is_up else "sell"
            signal["direction"] = "long" if is_up else "short"
            signal["confidence"] = "high" if momentum_score >= 3 else "medium"
            
            wave_points = analysis["details"].get("wave_points", [])
            targets = calculate_target_zones(current_price, wave_points, pattern_type, is_up)
            
            if targets["conservative"]:
                signal["entry"] = current_price
                signal["stop_loss"] = targets["stop_loss"]
                signal["targets"] = [
                    targets["conservative"],
                    targets["moderate"],
                    targets["aggressive"]
                ]
                
                risk = abs(signal["entry"] - signal["stop_loss"])
                reward = abs(signal["targets"][1] - signal["entry"])
                signal["risk_reward"] = reward / risk if risk > 0 else 0
                
                signal["reasoning"].append(f"Hybrid Wave {current_wave} with momentum: {', '.join(momentum_reasons)}")
        
        return signal

    def _init_signal(self):
        """Initialize empty signal structure with all required fields."""
        return {
            "signal": "no_trade",
            "direction": None,
            "confidence": "low",
            "entry": None,
            "stop_loss": None,
            "targets": [],
            "risk_reward": None,
            "position_size": None,
            "reasoning": ["No valid trade signal generated by strategy"],
            # Additional fields for compatibility
            "status": "No Trade",
            "currency": "USD",
            "risk_percent": 1.0,
            "account_size": 10000,
            "confidence_score": 0,
            "notes": "Strategy did not identify a valid trade setup"
        }

# Strategy factory function
def get_strategy_variants():
    """Return instance of strategy variants."""
    return ElliotWaveStrategies()

# Strategy execution function
def execute_strategy(strategy_name, data, analysis, sr_levels, current_price):
    """Execute a specific strategy and return signals."""
    strategies = get_strategy_variants()
    strategy_func = strategies.get_strategy(strategy_name)
    return strategy_func(data, analysis, sr_levels, current_price)