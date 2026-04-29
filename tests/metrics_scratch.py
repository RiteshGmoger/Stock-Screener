import numpy as np;

def sharpe(returns, risk_free = 0.04):
	risk_free_per_year = risk_free / 252;
	excess = [ret - risk_free_per_year for ret in returns]; # remove risk_free from the returns
	
	if np.std(excess, ddof = 1) == 0 : # divides by 1 sample std
		return 0;
	
	return (np.mean(excess) * 252) / (np.std(excess, ddof = 1) * np.sqrt(252));
	
def max_drawdown(equity_curves): # daily close value
	peak = equity_curves[0];
	max_dd = 0;
	
	for val in equity_curves:
		if(val > peak):
			peak = val;
		dd = (val - peak) / peak;
		
		if dd < max_dd: # drawdown will be nagative so most negative one
			max_dd =  dd; 
	
	return max_dd;
	
print(max_drawdown([100, 110, 115, 90, 95, 120]))
print(sharpe([0.01, -0.02, 0.015, 0.005]))
