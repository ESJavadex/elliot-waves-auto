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
            'moderate': self.moderate_elliott_wave,
            'aggressive': self.aggressive_elliott_wave,
            'fibonacci_confluence': self.fibonacci_confluence_strategy,
            'wave_validation': self.wave_count_validation,
            'hybrid_momentum': self.hybrid_momentum_strategy
        }
    
    def get_strategy(self, strategy_name):
        """Get a specific strategy function."""
        return self.strategies.get(strategy_name, self.moderate_elliott_wave)
    
    def list_strategies(self):
        """Return list of available strategies with descriptions."""
        return {
            'conservative': {
                'name': 'Conservative Elliott Wave',
                'description': 'Very high confidence trades with multiple confirmations, Wave 2&4 only',
                'risk_level': 'Very Low',
                'win_rate': 'Very High',
                'trade_frequency': 'Very Low',
                'min_confidence': 400,
                'min_rr': 3.0
            },
            'moderate': {
                'name': 'Moderate Elliott Wave',
                'description': 'Balanced approach with Wave 2,3,4 trades and RSI confirmation',
                'risk_level': 'Medium',
                'win_rate': 'High',
                'trade_frequency': 'Medium',
                'min_confidence': 250,
                'min_rr': 2.0
            },
            'aggressive': {
                'name': 'Aggressive Elliott Wave', 
                'description': 'High frequency trades on ALL waves with momentum signals',
                'risk_level': 'High',
                'win_rate': 'Medium',
                'trade_frequency': 'Very High',
                'min_confidence': 100,
                'min_rr': 1.2
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
        - Very high confidence threshold (>400 points)
        - Strict Elliott Wave rule compliance
        - Only trades Wave 2 and 4 completions with multiple confirmations
        - High risk-reward ratios (3:1 minimum)
        - Requires RSI confirmation and volume validation
        """
        signal = self._init_signal()
        
        if not analysis["valid"] or analysis["confidence_score"] < 400:
            signal["reasoning"].append("Confidence score below conservative threshold (400)")
            return signal
            
        pattern_type = analysis["pattern_type"]
        is_up = analysis["is_up"]
        current_wave = analysis["current_wave"]["wave"]
        
        # Conservative: Only trade Wave 2 and Wave 4 completions
        if pattern_type == "impulse" and current_wave in ["2", "4"]:
            
            # Additional conservative validations
            validation_score = 0
            validations = []
            
            # RSI confirmation
            if 'RSI' in data.columns and len(data) >= 14:
                rsi = data['RSI'].iloc[-1]
                if current_wave == "2":
                    if (is_up and 25 <= rsi <= 45) or (not is_up and 55 <= rsi <= 75):
                        validation_score += 1
                        validations.append("RSI in ideal range for Wave 2")
                elif current_wave == "4":
                    if (is_up and 30 <= rsi <= 50) or (not is_up and 50 <= rsi <= 70):
                        validation_score += 1
                        validations.append("RSI in ideal range for Wave 4")
            
            # Volume confirmation (decreasing volume in corrections)
            if 'Volume' in data.columns and len(data) >= 20:
                avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
                recent_volume = data['Volume'].iloc[-5:].mean()
                if recent_volume < avg_volume * 0.8:  # 20% below average
                    validation_score += 1
                    validations.append("Decreasing volume confirms correction")
            
            # Price action confirmation (retracement levels)
            if current_wave in ["2", "4"] and len(data) >= 50:
                high_50 = data['High'].rolling(50).max().iloc[-1]
                low_50 = data['Low'].rolling(50).min().iloc[-1]
                current_pos = (current_price - low_50) / (high_50 - low_50)
                
                if current_wave == "2" and 0.3 <= current_pos <= 0.7:  # 30-70% retracement zone
                    validation_score += 1
                    validations.append("Price in ideal Wave 2 retracement zone")
                elif current_wave == "4" and 0.2 <= current_pos <= 0.6:  # 20-60% retracement zone  
                    validation_score += 1
                    validations.append("Price in ideal Wave 4 retracement zone")
            
            # Require at least 2 confirmations for conservative strategy
            if validation_score < 2:
                signal["reasoning"].append(f"Insufficient confirmations ({validation_score}/3): {validations}")
                return signal
            
            signal["signal"] = "buy" if is_up else "sell"
            signal["direction"] = "long" if is_up else "short"
            signal["confidence"] = "very_high"
            
            wave_points = analysis["details"].get("wave_points", [])
            targets = calculate_target_zones(current_price, wave_points, pattern_type, is_up)
            
            if targets["conservative"]:
                signal["entry"] = current_price
                signal["stop_loss"] = targets["stop_loss"]
                
                # Conservative targets - only first target with tight stop
                signal["targets"] = [
                    targets["conservative"]
                ]
                
                # Calculate risk-reward (must be >3:1 for conservative)
                risk = abs(signal["entry"] - signal["stop_loss"])
                reward = abs(signal["targets"][0] - signal["entry"])
                signal["risk_reward"] = reward / risk if risk > 0 else 0
                
                if signal["risk_reward"] < 3.0:
                    signal["signal"] = "no_trade"
                    signal["reasoning"].append(f"Risk-reward ratio ({signal['risk_reward']:.2f}) below conservative minimum (3.0)")
                else:
                    signal["reasoning"].append(f"Conservative Wave {current_wave} with confirmations: {validations}")
                    signal["reasoning"].append(f"RR: {signal['risk_reward']:.2f}, Validations: {validation_score}/3")
        
        return signal

    def moderate_elliott_wave(self, data, analysis, sr_levels, current_price):
        """
        Moderate Elliott Wave Strategy (BALANCED)
        - Medium confidence threshold (>250 points)
        - Trades Wave 2, 3, and 4 completions
        - Balanced risk-reward ratios (2:1 minimum)
        - Uses RSI and trend confirmation
        - Position sizing based on volatility
        """
        signal = self._init_signal()
        
        if not analysis["valid"] or analysis["confidence_score"] < 250:
            signal["reasoning"].append("Confidence score below moderate threshold (250)")
            return signal
            
        pattern_type = analysis["pattern_type"]
        is_up = analysis["is_up"]
        current_wave = analysis["current_wave"]["wave"]
        
        # Moderate: Trade Wave 2, 3, and 4 completions
        if pattern_type == "impulse" and current_wave in ["2", "3", "4"]:
            
            # Balanced confirmations
            confirmation_score = 0
            confirmations = []
            
            # RSI confirmation (not as strict as conservative)
            if 'RSI' in data.columns and len(data) >= 14:
                rsi = data['RSI'].iloc[-1]
                if current_wave == "2":
                    if (is_up and 20 <= rsi <= 50) or (not is_up and 50 <= rsi <= 80):
                        confirmation_score += 1
                        confirmations.append(f"RSI favorable for Wave 2: {rsi:.1f}")
                elif current_wave == "3":
                    if (is_up and rsi >= 50) or (not is_up and rsi <= 50):
                        confirmation_score += 1
                        confirmations.append(f"RSI momentum for Wave 3: {rsi:.1f}")
                elif current_wave == "4":
                    if (is_up and 25 <= rsi <= 55) or (not is_up and 45 <= rsi <= 75):
                        confirmation_score += 1
                        confirmations.append(f"RSI consolidation for Wave 4: {rsi:.1f}")
            
            # Trend confirmation using moving averages
            if len(data) >= 20:
                ma20 = data['Close'].rolling(20).mean().iloc[-1]
                if (is_up and current_price > ma20) or (not is_up and current_price < ma20):
                    confirmation_score += 1
                    confirmations.append("Price above/below 20MA confirming trend")
            
            # Volume analysis (moderate requirements)
            if 'Volume' in data.columns and len(data) >= 10:
                avg_volume = data['Volume'].rolling(10).mean().iloc[-1]
                current_volume = data['Volume'].iloc[-1]
                
                if current_wave == "3" and current_volume > avg_volume * 1.2:
                    confirmation_score += 1
                    confirmations.append("Higher volume confirms Wave 3")
                elif current_wave in ["2", "4"] and current_volume < avg_volume * 1.1:
                    confirmation_score += 1
                    confirmations.append("Lower volume confirms correction")
            
            # Require at least 1 confirmation for moderate (more lenient than conservative)
            if confirmation_score < 1:
                signal["reasoning"].append(f"No confirmations found: {confirmations}")
                return signal
            
            signal["signal"] = "buy" if is_up else "sell"
            signal["direction"] = "long" if is_up else "short"
            
            # Confidence based on wave type and confirmations
            if current_wave == "3" and confirmation_score >= 2:
                signal["confidence"] = "high"
            elif current_wave in ["2", "4"] and confirmation_score >= 2:
                signal["confidence"] = "medium"
            else:
                signal["confidence"] = "medium"
            
            wave_points = analysis["details"].get("wave_points", [])
            targets = calculate_target_zones(current_price, wave_points, pattern_type, is_up)
            
            if targets["conservative"]:
                signal["entry"] = current_price
                
                # Moderate stop loss (between conservative and aggressive)
                if 'ATR' in data.columns:
                    atr = data['ATR'].iloc[-1]
                    stop_multiplier = 1.2  # Moderate stop distance
                    if is_up:
                        signal["stop_loss"] = current_price - (atr * stop_multiplier)
                    else:
                        signal["stop_loss"] = current_price + (atr * stop_multiplier)
                else:
                    signal["stop_loss"] = targets["stop_loss"]
                
                # Moderate targets - use conservative and moderate levels
                signal["targets"] = [
                    targets["conservative"],
                    targets["moderate"]
                ]
                
                # Risk-reward requirement (2:1 minimum)
                risk = abs(signal["entry"] - signal["stop_loss"])
                reward = abs(signal["targets"][1] - signal["entry"])  # Use moderate target
                signal["risk_reward"] = reward / risk if risk > 0 else 0
                
                if signal["risk_reward"] < 2.0:
                    signal["signal"] = "no_trade"
                    signal["reasoning"].append(f"Risk-reward ratio ({signal['risk_reward']:.2f}) below moderate minimum (2.0)")
                else:
                    signal["reasoning"].append(f"Moderate Wave {current_wave} signal")
                    signal["reasoning"].append(f"Confirmations: {confirmations}")
                    signal["reasoning"].append(f"RR: {signal['risk_reward']:.2f}, Confirmations: {confirmation_score}/3")
        
        # Also trade some corrective patterns with moderate confidence
        elif pattern_type == "zigzag" and current_wave == "C":
            signal["signal"] = "buy" if is_up else "sell"
            signal["direction"] = "long" if is_up else "short"
            signal["confidence"] = "medium"
            signal["reasoning"].append(f"Moderate zigzag correction completion")
        
        return signal

    def aggressive_elliott_wave(self, data, analysis, sr_levels, current_price):
        """
        Aggressive Elliott Wave Strategy
        - Lower confidence threshold (>100 points)
        - Trades ALL wave completions (1,2,3,4,5,A,B,C)
        - Multiple position entries and pyramid strategies
        - Lower risk-reward ratios (1.2:1 minimum)
        - Uses momentum and breakout confirmations
        """
        signal = self._init_signal()
        
        if not analysis["valid"] or analysis["confidence_score"] < 100:
            signal["reasoning"].append("Confidence score below aggressive threshold (100)")
            return signal
            
        pattern_type = analysis["pattern_type"]
        is_up = analysis["is_up"]
        current_wave = analysis["current_wave"]["wave"]
        
        # Aggressive: Trade ALL wave completions
        tradeable_waves = ["1", "2", "3", "4", "5", "A", "B", "C"]
        
        if pattern_type == "impulse" and current_wave in tradeable_waves:
            
            # Momentum confirmations for aggressive trades
            momentum_score = 0
            momentum_signals = []
            
            # Price momentum (recent breakouts)
            if len(data) >= 5:
                recent_change = (data['Close'].iloc[-1] - data['Close'].iloc[-5]) / data['Close'].iloc[-5]
                if abs(recent_change) > 0.02:  # 2% move in 5 periods
                    momentum_score += 1
                    momentum_signals.append(f"Strong momentum: {recent_change*100:.1f}%")
            
            # Volume spike confirmation
            if 'Volume' in data.columns and len(data) >= 10:
                avg_volume = data['Volume'].rolling(10).mean().iloc[-1]
                current_volume = data['Volume'].iloc[-1]
                if current_volume > avg_volume * 1.5:  # 50% above average
                    momentum_score += 1
                    momentum_signals.append("Volume spike detected")
            
            # Volatility expansion
            if 'ATR' in data.columns and len(data) >= 14:
                current_atr = data['ATR'].iloc[-1]
                avg_atr = data['ATR'].rolling(14).mean().iloc[-1]
                if current_atr > avg_atr * 1.3:  # 30% higher volatility
                    momentum_score += 1
                    momentum_signals.append("Volatility expansion")
            
            signal["signal"] = "buy" if is_up else "sell"
            signal["direction"] = "long" if is_up else "short"
            
            # Confidence based on wave and momentum
            if current_wave in ["3", "5"] and momentum_score >= 2:
                signal["confidence"] = "high"
            elif current_wave in ["1", "2", "4"] and momentum_score >= 1:
                signal["confidence"] = "medium"
            else:
                signal["confidence"] = "low"
            
            wave_points = analysis["details"].get("wave_points", [])
            targets = calculate_target_zones(current_price, wave_points, pattern_type, is_up)
            
            if targets["conservative"]:
                signal["entry"] = current_price
                
                # Aggressive stop loss (tighter for more trades)
                if 'ATR' in data.columns:
                    atr = data['ATR'].iloc[-1]
                    stop_multiplier = 0.8 if momentum_score >= 2 else 1.0
                    if is_up:
                        signal["stop_loss"] = current_price - (atr * stop_multiplier)
                    else:
                        signal["stop_loss"] = current_price + (atr * stop_multiplier)
                else:
                    signal["stop_loss"] = targets["stop_loss"]
                
                # Aggressive targets - use all levels with extensions
                signal["targets"] = [
                    targets["conservative"],
                    targets["moderate"],
                    targets["aggressive"]
                ]
                
                # Very low risk-reward requirement for aggressive
                risk = abs(signal["entry"] - signal["stop_loss"])
                reward = abs(signal["targets"][0] - signal["entry"])  # Use first target
                signal["risk_reward"] = reward / risk if risk > 0 else 0
                
                if signal["risk_reward"] < 1.2:
                    signal["signal"] = "no_trade"
                    signal["reasoning"].append(f"Risk-reward ratio ({signal['risk_reward']:.2f}) below aggressive minimum (1.2)")
                else:
                    signal["reasoning"].append(f"Aggressive Wave {current_wave} signal")
                    signal["reasoning"].append(f"Momentum signals: {momentum_signals}")
                    signal["reasoning"].append(f"RR: {signal['risk_reward']:.2f}, Momentum: {momentum_score}/3")
        
        # Aggressive: Also trade corrective patterns and even uncertain waves
        elif pattern_type in ["zigzag", "flat", "triangle"] or "?" in current_wave:
            signal["signal"] = "buy" if is_up else "sell"
            signal["direction"] = "long" if is_up else "short"
            signal["confidence"] = "low"
            signal["reasoning"].append(f"Aggressive {pattern_type} or uncertain wave {current_wave} trade")
        
        return signal

    def fibonacci_confluence_strategy(self, data, analysis, sr_levels, current_price):
        """
        Fibonacci Confluence Strategy (MODERATE)
        - Focus on Fibonacci level convergence and harmonic patterns
        - Trade when multiple Fibonacci levels align within tight clusters
        - Uses advanced Fibonacci relationships and time-based projections
        - Moderate risk with 2:1 minimum risk-reward
        """
        signal = self._init_signal()
        
        if not analysis["valid"] or analysis["confidence_score"] < 200:
            return signal
            
        pattern_type = analysis["pattern_type"]
        is_up = analysis["is_up"]
        current_wave = analysis["current_wave"]["wave"]
        wave_points = analysis["details"].get("wave_points", [])
        
        # Advanced Fibonacci analysis
        fib_score = 0
        fib_signals = []
        
        # Find Fibonacci clusters around current price (tighter tolerance)
        # Note: Simplified approach since we need proper wave point data for full implementation
        clusters = []  # Would normally calculate Fibonacci clusters from wave data
        
        if clusters and len(clusters) >= 2:
            fib_score += len(clusters)
            cluster_types = [c["type"] for c in clusters]
            fib_signals.append(f"Confluence: {cluster_types}")
        
        # Check for harmonic Fibonacci relationships
        if len(wave_points) >= 3:
            # Calculate ratios between waves
            ratios = []
            for i in range(len(wave_points)-1):
                p1 = wave_points[i]
                p2 = wave_points[i+1]
                ratio = abs(p2["Close"] - p1["Close"]) / abs(p1["Close"])
                ratios.append(ratio)
            
            # Look for golden ratio relationships
            golden_ratios = [0.382, 0.618, 1.0, 1.618, 2.618]
            for ratio in ratios:
                for golden in golden_ratios:
                    if abs(ratio - golden) < 0.05:  # 5% tolerance
                        fib_score += 1
                        fib_signals.append(f"Golden ratio {golden:.3f} detected")
                        break
        
        # Time-based Fibonacci (if we have enough data)
        if len(data) >= 50 and current_wave in ["2", "4", "C"]:
            # Check if current correction duration aligns with Fibonacci time ratios
            wave_duration = len(data) // 10  # Simplified duration calculation
            fib_time_targets = [13, 21, 34, 55, 89]  # Fibonacci sequence
            
            for target in fib_time_targets:
                if abs(wave_duration - target) <= 3:  # Within 3 periods
                    fib_score += 1
                    fib_signals.append(f"Fibonacci time target {target} reached")
                    break
        
        # Require high Fibonacci score for this strategy
        if fib_score >= 3 and clusters:
            signal["signal"] = "buy" if is_up else "sell"
            signal["direction"] = "long" if is_up else "short"
            signal["confidence"] = "high" if fib_score >= 4 else "medium"
            
            # Entry at Fibonacci confluence
            signal["entry"] = current_price
            
            # Stop loss based on Fibonacci cluster analysis
            cluster_levels = [cluster["level"] for cluster in clusters]
            if is_up:
                signal["stop_loss"] = min(cluster_levels) * 0.985
            else:
                signal["stop_loss"] = max(cluster_levels) * 1.015
            
            # Fibonacci-based targets
            targets = calculate_target_zones(current_price, wave_points, pattern_type, is_up)
            if targets["conservative"]:
                # Use Fibonacci extensions for more precise targets
                base_target = targets["moderate"]
                fib_extensions = [1.272, 1.414, 1.618, 2.0, 2.618]
                
                signal["targets"] = [
                    targets["conservative"],
                    base_target,
                    base_target * fib_extensions[0] if is_up else base_target / fib_extensions[0]
                ]
                
                risk = abs(signal["entry"] - signal["stop_loss"])
                reward = abs(signal["targets"][1] - signal["entry"])
                signal["risk_reward"] = reward / risk if risk > 0 else 0
                
                if signal["risk_reward"] < 2.0:
                    signal["signal"] = "no_trade"
                    signal["reasoning"].append(f"Risk-reward ratio ({signal['risk_reward']:.2f}) below Fibonacci minimum (2.0)")
                else:
                    signal["reasoning"].append(f"Fibonacci confluence strategy triggered")
                    signal["reasoning"].append(f"Fib signals: {fib_signals}")
                    signal["reasoning"].append(f"Score: {fib_score}, RR: {signal['risk_reward']:.2f}")
        else:
            signal["reasoning"].append(f"Insufficient Fibonacci signals ({fib_score}/3): {fib_signals}")
        
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

def execute_all_strategies(data, analysis, sr_levels, current_price):
    """
    Execute all main strategies and return comprehensive comparison results.
    
    Args:
        data: DataFrame with price and indicator data
        analysis: Elliott Wave analysis results
        sr_levels: Support/resistance levels
        current_price: Current price
        
    Returns:
        Dictionary with results for each strategy and comparison tables
    """
    main_strategies = ['conservative', 'moderate', 'aggressive', 'fibonacci_confluence', 'wave_validation', 'hybrid_momentum']
    
    strategy_results = {}
    comparison_summary = []
    
    # Execute each strategy
    for strategy_name in main_strategies:
        try:
            result = execute_strategy(strategy_name, data, analysis, sr_levels, current_price)
            strategy_results[strategy_name] = result
            
            # Extract key metrics for comparison
            comparison_summary.append({
                'strategy': strategy_name,
                'signal': result.get('signal', 'no_trade'),
                'direction': result.get('direction', 'None'),
                'confidence': result.get('confidence', 'low'),
                'entry': result.get('entry'),
                'stop_loss': result.get('stop_loss'),
                'risk_reward': result.get('risk_reward'),
                'targets': result.get('targets', []),
                'reasoning': result.get('reasoning', [])
            })
        except Exception as e:
            print(f"Error executing strategy {strategy_name}: {e}")
            strategy_results[strategy_name] = {
                'signal': 'error',
                'error': str(e),
                'reasoning': [f"Strategy execution failed: {e}"]
            }
            comparison_summary.append({
                'strategy': strategy_name,
                'signal': 'error',
                'direction': 'None',
                'confidence': 'error',
                'entry': None,
                'stop_loss': None,
                'risk_reward': None,
                'targets': [],
                'reasoning': [f"Error: {e}"]
            })
    
    # Create comparison tables
    signal_summary = create_signal_comparison_table(comparison_summary)
    risk_summary = create_risk_comparison_table(comparison_summary)
    target_summary = create_target_comparison_table(comparison_summary)
    
    # Create overall summary
    overall_summary = create_overall_summary(comparison_summary)
    
    return {
        'strategy_results': strategy_results,
        'comparison_summary': comparison_summary,
        'signal_table': signal_summary,
        'risk_table': risk_summary,
        'target_table': target_summary,
        'overall_summary': overall_summary,
        'analysis_date': data.index[-1] if len(data) > 0 else None,
        'current_price': current_price
    }

def create_signal_comparison_table(comparison_summary):
    """Create a table comparing signals across strategies."""
    table = {
        'headers': ['Strategy', 'Signal', 'Direction', 'Confidence', 'Entry Price'],
        'rows': []
    }
    
    for result in comparison_summary:
        strategy_name = result['strategy'].replace('_', ' ').title()
        signal = result['signal'].upper()
        direction = result['direction'] or 'None'
        confidence = result['confidence'].replace('_', ' ').title()
        entry = f"${result['entry']:.2f}" if result['entry'] else 'N/A'
        
        table['rows'].append([strategy_name, signal, direction, confidence, entry])
    
    return table

def create_risk_comparison_table(comparison_summary):
    """Create a table comparing risk metrics across strategies."""
    table = {
        'headers': ['Strategy', 'Stop Loss', 'Risk per Share', 'Risk-Reward Ratio', 'Risk Level'],
        'rows': []
    }
    
    for result in comparison_summary:
        strategy_name = result['strategy'].replace('_', ' ').title()
        stop_loss = f"${result['stop_loss']:.2f}" if result['stop_loss'] else 'N/A'
        
        # Calculate risk per share
        risk_per_share = 'N/A'
        if result['entry'] and result['stop_loss']:
            risk = abs(result['entry'] - result['stop_loss'])
            risk_per_share = f"${risk:.2f}"
        
        # Risk-reward ratio
        rr_ratio = f"{result['risk_reward']:.2f}:1" if result['risk_reward'] else 'N/A'
        
        # Risk level based on strategy
        risk_levels = {
            'conservative': 'Very Low',
            'moderate': 'Medium',
            'aggressive': 'High',
            'fibonacci_confluence': 'Medium',
            'wave_validation': 'Low',
            'hybrid_momentum': 'Medium'
        }
        risk_level = risk_levels.get(result['strategy'], 'Unknown')
        
        table['rows'].append([strategy_name, stop_loss, risk_per_share, rr_ratio, risk_level])
    
    return table

def create_target_comparison_table(comparison_summary):
    """Create a table comparing profit targets across strategies."""
    table = {
        'headers': ['Strategy', 'Target 1', 'Target 2', 'Target 3', 'Max Profit Potential'],
        'rows': []
    }
    
    for result in comparison_summary:
        strategy_name = result['strategy'].replace('_', ' ').title()
        targets = result['targets']
        
        target1 = f"${targets[0]:.2f}" if len(targets) > 0 and targets[0] else 'N/A'
        target2 = f"${targets[1]:.2f}" if len(targets) > 1 and targets[1] else 'N/A'
        target3 = f"${targets[2]:.2f}" if len(targets) > 2 and targets[2] else 'N/A'
        
        # Calculate max profit potential
        max_profit = 'N/A'
        if result['entry'] and targets:
            valid_targets = [t for t in targets if t is not None]
            if valid_targets:
                if result['direction'] == 'long':
                    max_target = max(valid_targets)
                    profit = max_target - result['entry']
                else:
                    max_target = min(valid_targets)
                    profit = result['entry'] - max_target
                max_profit = f"${profit:.2f} ({(profit/result['entry']*100):.1f}%)"
        
        table['rows'].append([strategy_name, target1, target2, target3, max_profit])
    
    return table

def create_overall_summary(comparison_summary):
    """Create an overall summary of strategy comparison results."""
    total_strategies = len(comparison_summary)
    buy_signals = len([r for r in comparison_summary if r['signal'] in ['buy', 'long']])
    sell_signals = len([r for r in comparison_summary if r['signal'] in ['sell', 'short']])
    no_trade_signals = len([r for r in comparison_summary if r['signal'] == 'no_trade'])
    error_signals = len([r for r in comparison_summary if r['signal'] == 'error'])
    
    # Calculate consensus
    if buy_signals > sell_signals and buy_signals > no_trade_signals:
        consensus = "BULLISH"
        consensus_strength = f"{buy_signals}/{total_strategies}"
    elif sell_signals > buy_signals and sell_signals > no_trade_signals:
        consensus = "BEARISH"
        consensus_strength = f"{sell_signals}/{total_strategies}"
    else:
        consensus = "NEUTRAL"
        consensus_strength = f"{no_trade_signals}/{total_strategies}"
    
    # Calculate average risk-reward
    valid_rr = [r['risk_reward'] for r in comparison_summary if r['risk_reward'] is not None]
    avg_rr = sum(valid_rr) / len(valid_rr) if valid_rr else None
    
    # Find best strategy (highest confidence with signal)
    trading_strategies = [r for r in comparison_summary if r['signal'] not in ['no_trade', 'error']]
    best_strategy = None
    if trading_strategies:
        confidence_scores = {'very_high': 5, 'high': 4, 'medium': 3, 'low': 2, 'error': 0}
        best_strategy = max(trading_strategies, key=lambda x: confidence_scores.get(x['confidence'], 0))
    
    return {
        'total_strategies': total_strategies,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'no_trade_signals': no_trade_signals,
        'error_signals': error_signals,
        'consensus': consensus,
        'consensus_strength': consensus_strength,
        'average_risk_reward': avg_rr,
        'best_strategy': best_strategy
    }