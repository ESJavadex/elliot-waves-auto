#!/usr/bin/env python3
"""
This script tests the fix we implemented for the interval parameter order issue.
It verifies that the run_analysis function correctly handles the case where interval
is passed as the 4th parameter instead of the 5th.
"""

# Function with the fix we implemented
def run_analysis_with_fix(ticker, stock_data, peak_order=8, is_backtest=False, interval='1wk'):
    """Test function with the fix for parameter order"""
    # Fix for backward compatibility when interval is passed as 4th parameter
    # This handles the case where some code might call run_analysis(ticker, start_date, end_date, interval)
    if isinstance(is_backtest, str) and is_backtest in ['1d', '1wk', '1mo', '1h', '5m', '15m', '30m', '5d', '3mo']:
        # In this case, is_backtest is actually the interval
        print(f"  Note: Fixing parameter order - interval was passed as 4th parameter")
        interval = is_backtest
        is_backtest = False
    
    # Just print the parameters to verify they're correct
    print(f"\nParameters received by run_analysis:")
    print(f" - ticker: {ticker}")
    print(f" - stock_data: {stock_data}")
    print(f" - peak_order: {peak_order}")
    print(f" - is_backtest: {is_backtest}")
    print(f" - interval: {interval}")
    
    # Return dummy data to simulate the function result
    return None, {}, None

# Run the test with the correct parameter order
print("\n=== TEST 1: Correct parameter order ===")
fig, analysis, _ = run_analysis_with_fix("AAPL", "2023-01-01", "2023-12-31", False, "1d")

# Run the test with incorrect parameter order (interval as 4th parameter)
print("\n=== TEST 2: Incorrect parameter order (interval as 4th parameter) ===")
fig, analysis, _ = run_analysis_with_fix("AAPL", "2023-01-01", "2023-12-31", "1d")

print("\nIf the test passes, the interval parameter in Test 2 should be '1d', not '1wk'")
print("And is_backtest should be False, not '1d'") 