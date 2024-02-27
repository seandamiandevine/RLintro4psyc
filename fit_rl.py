"""
Companion code to:
    Introduction to Reinforcement Learning. Tutorial. Nijmegen, 2024

    contact: seandamiandevine@gmail.com

"""

import numpy as np 
import pandas as pd
from scipy import optimize
from scipy.stats import norm, beta, gamma
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm 

def softmax(beta:float,V:np.ndarray) -> np.ndarray:
	"""Softmax choice rule.
	
	Args:
	    beta (float): Inverse temperature parameter value
	    V (np.ndarray): Array 1x2 [value of left option, value of right option]
	
	Returns:
	    np.ndarray: Array 1x2 of choice probabilities
	"""
	if not isinstance(V, np.ndarray):
		try:
			V = np.array(V)
		except:
			raise Exception("V must be a numpy array")
	return np.exp(beta*V)/np.sum(np.exp(beta*V))

def sim_simple_rl(alpha:float, beta:float, ntrials:int=30, 
	detailed_output:bool=False) -> pd.core.frame.DataFrame :
	"""
	Simulate data on a basic two-arm bandit RL task for a single subject. 
	Simualted subjects respond according to R-W rules with softmax choice rule. 
	
	Args:
	    alpha (float): Learning rate
	    beta (float): Inverse temperature
	    ntrials (int, optional): Number of trials
	    detailed_output(bool, optional): Whether the output should contained intermediate values, like PEs or predicted choice probailities (default False)
	
	Returns:
	    pd.core.frame.DataFrame: Pandas df for a single simulated subject

	"""
	## Basic experimental setup
	MU    = [-1, 1] ## mu_triangle, mu_square
	SIGMA = 1       ## shared std. dev. of reward distributions

	## store values and deltas
	Vs      = np.zeros((ntrials+1,2))
	deltas  = np.zeros(ntrials)
	choices = np.zeros(ntrials)
	rewards = np.zeros(ntrials)
	pA      = np.zeros(ntrials)
	pB      = np.zeros(ntrials)

	for t in range(1,ntrials):

		## make choice
		p = softmax(beta, Vs[t,:])
		pA[t], pB[t] = p
		choices[t] = np.random.choice([0,1], size=1, p=p)[0]
		rewards[t] = np.random.normal(loc=MU[int(choices[t])], scale=SIGMA, size=1)[0]

		## update value
		idx = int(choices[t])
		deltas[t] = rewards[t] - Vs[t, idx]
		Vs[t+1:,idx] = Vs[t,idx] + alpha*deltas[t]

	pid = np.random.randint(0,100000,size=1)[0]
	df = pd.DataFrame({"PID":pid, 'trial':np.arange(ntrials)+1,"choice": choices, "rewards":rewards, 
		"V_triangle":Vs[1:,0], "V_square":Vs[1:,1], "delta":deltas, 'p_square':pB, 'p_triangle':pA,
		'alpha':alpha, 'beta':beta})

	if not detailed_output:
		return df[["PID","trial","choice","rewards"]]

	return df

# ****************************************************************************
# *                     ## Simulate data for 30 subjects                     *
# ****************************************************************************

## Note: this was not covered in the tutorial. This is just here for completeness. 

np.random.seed(2024)
N      = 20
alphas = np.random.uniform(.1, .4, size=N)
betas  = np.random.uniform(1, 4, size=N)
df     = pd.concat([sim_simple_rl(alphas[i], betas[i],detailed_output=True, ntrials=500) for i in range(N)])

## Visualize choice trajectory per subject
df["trialbin"] = pd.cut(df.trial,20).apply(lambda x: x.left).astype(int)
df['PID'] = df.PID.astype(str)
fig, ax = plt.subplots(1)
sns.lineplot(data=df, x='trialbin', y='choice', hue='PID', ax=ax, errorbar=None)
ax.set(xlabel="Trial", ylabel="P(Choose Square)", ylim=(.5,1.1))
ax.legend(ncol=4, title="PID", frameon=False)
# plt.show()
fig.savefig("figs/rl_fit_demo/choice_trajectory.png", dpi=300)

# ****************************************************************************
# *                       ## Fit model to each subject                       *
# ****************************************************************************


def rl_lik(pars:list, choices:np.ndarray, rewards:np.ndarray, to_return:str="nll") -> float:
	"""
	Args:
	    pars (list): List of two parameters, alpha [float] and beta [float]
	    choices (np.ndarray): An array of binary choices (0,1)
	    rewards (np.ndarray): An array of received rewards
	    to_return (str): What to return? Negative log-likelihoods ("nll"; default) or choice probabilities ("prob")
	
	Returns:
	    float: Negative log-likelihood or choice probabilities
	
	"""

	## add catch to avoid impossible values
	alpha, beta = pars
	if alpha < 0 or alpha > 1:
		return 1e10
	if beta < 0 or beta > 20:
		return 1e10

	## set up initial values
	V = [0,0] ## initial value for triangle and square
	probs = np.zeros(len(choices))

	# compute log likelihood
	for t,(c,r) in enumerate(zip(choices,rewards)):
		
		## make choice
		probs[t] = softmax(beta, V)[c]

		## update value
		delta = r - V[c]
		V[c]  = V[c] + alpha*delta

	if to_return=="prob":
		return probs

	nll = -np.sum(np.log(probs))
	return nll


# example use of `rl_lik`
choices = df.choice[df.PID == '52638'].values.astype(int)
rewards = df.rewards[df.PID == '52638'].values
rl_lik([0.3, 1.1], choices, rewards)
rl_lik([0.2, 1.21], choices, rewards)


## Prepare to store result
N = df.PID.unique().shape[0]

rl_output = pd.DataFrame({
	"PID":np.zeros(N), 
	"alpha_hat":np.zeros(N),
	"beta_hat":np.zeros(N),
	"true_alpha":np.zeros(N),
	"true_beta":np.zeros(N),
	"ll": np.zeros(N),
	"conv":np.zeros(N)
	})

for i, (pid, subj_df) in enumerate(df.groupby("PID")):

	print(f"Fitting RL model for subject {i+1}/{N}")

	## create objective function
	obj = lambda pars: rl_lik(pars, 
		choices=subj_df.choice.values.astype(int), 
		rewards=subj_df.rewards.values)

	best_ll  = 1e10
	for _ in tqdm(range(5)):
		x0 = [np.random.uniform(), np.random.uniform(1,5)]
		opt = optimize.minimize(fun=obj, x0=x0)
		if opt.fun < best_ll:
			best_ll  = opt.fun
			best_opt = opt

	## Save results
	rl_output.loc[i,"PID"]        = pid
	rl_output.loc[i,"alpha_hat"]  = opt.x[0]
	rl_output.loc[i,"beta_hat"]   = opt.x[1]
	rl_output.loc[i,"true_alpha"] = subj_df.alpha.iloc[0]
	rl_output.loc[i,"true_beta"]  = subj_df.beta.iloc[0]
	rl_output.loc[i,"ll"]         = -opt.fun ## add negative to convert this to simple log likelihood
	rl_output.loc[i,"conv"]       = opt.status

rl_output

# ****************************************************************************
# *                    ## Visualize parameter distribution                   *
# ****************************************************************************

fig, ax = plt.subplots(1,2, figsize=(8,4), constrained_layout=True)

sns.histplot(data=rl_output, x="alpha_hat", kde=True, ax=ax[0])
ax[0].set(title=r"$\alpha$", xlabel="")
sns.histplot(data=rl_output, x="beta_hat",  kde=True, ax=ax[1])
ax[1].set(title=r"$\beta$", xlabel="")

# fig.savefig("figs/rl_fit_demo/param_distribution.png", dpi=300)

# plt.show()

# ****************************************************************************
# *                           ## Parameter recovery                          *
# ****************************************************************************

fig, ax = plt.subplots(1,2, figsize=(8,4), constrained_layout=True)

sns.regplot(data=rl_output,x="true_alpha",y="alpha_hat", ax=ax[0],
 scatter_kws={"color":'grey'}, line_kws={"color":'red'})
ax[0].set(xlabel=r"True $\alpha$", ylabel=r"Estimated $\alpha$")
sns.regplot(data=rl_output,x="true_beta",y="beta_hat", ax=ax[1],
	 scatter_kws={"color":'grey'}, line_kws={"color":'red'})
ax[1].set(xlabel=r"True $\beta$", ylabel=r"Estimated $\beta$")

fig.savefig("figs/rl_fit_demo/param_recovery.png", dpi=300)

# plt.show()

# ****************************************************************************
# *                          ## Predictive checking                          *
# ****************************************************************************


subj_dfs = []

for i, (pid, subj_df) in enumerate(df.groupby("PID")):

	print(f"Fitting RL model for subject {i+1}/{N}")

	## get subj parameters
	a,b = rl_output[rl_output.PID==pid][["alpha_hat","beta_hat"]].values[0]

	## compute choice probabilities based on these parameters
	subj_df['cp'] = rl_lik(pars=[a,b], 
						choices=subj_df.choice.values.astype(int), 
						rewards=subj_df.rewards.values, 
						to_return="prob")

	## Convert choice probabilities to P(choice==1)
	subj_df.cp[subj_df.choice==0] = 1-subj_df.cp

	## Store results
	subj_dfs.append(subj_df)


df = pd.concat(subj_dfs)

## Visualize predictive check

df["trialbin"] = pd.cut(df.trial,20).apply(lambda x: x.left).astype(int)

fig, ax = plt.subplots(1)
sns.lineplot(data=df, x='trialbin', y='choice', ax=ax, errorbar=None, label="Subject")
sns.lineplot(data=df, x='trialbin', y='cp', ax=ax, label="Model")
ax.set(xlabel="Trial", ylabel="P(Choose Square)", ylim=(.5,1))
ax.legend(loc='lower right', prop={"size":20})
# plt.show()
fig.savefig("figs/rl_fit_demo/predictive_check.png", dpi=300)

