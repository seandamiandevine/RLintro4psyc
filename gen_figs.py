"""
Generate figures that appear in tutorial slides. 

contact: seandamiandevine@gmail.com
"""

import numpy as np 
import pandas as pd
from scipy.stats import norm, beta, gamma
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

## Slide 15
x = np.arange(10)
y = np.exp(-x)

fig, ax = plt.subplots(1)
ax.plot(x,y,'-o')
ax.set(yticks=[0,1], yticklabels=["Low","High"], xlabel="Trial", ylabel="Surprise")
fig.savefig('figs/slide15.png', dpi=300)
# plt.show()


## Slide 26
n = 20
trials = np.arange(n)
Vs = np.zeros(n+1)
deltas = np.zeros(n)
rewards = np.ones(n)
alpha = 0.5

for t in range(n):
	deltas[t] = rewards[t] - Vs[t]
	Vs[t+1] = Vs[t] + alpha*deltas[t]

fig, ax = plt.subplots(1)
ax.plot(trials, Vs[:-1], label=r"$V_t$")
ax.plot(trials, deltas, label=r"$\delta_t$")
ax.legend(loc="right", prop={"size":20})
ax.set(xticks=np.arange(n, step=5), xlabel="Trial")
fig.savefig('figs/slide26.png', dpi=300)

# plt.show()

## Slide 27
n = 20
trials = np.arange(n)
deltas = np.zeros(n)
rewards = np.ones(n)
alphas = np.linspace(0,1,num=5)
Vs = np.zeros( ((n+1),alphas.shape[0]))

for i,a in enumerate(alphas):
	for t in range(n):
		deltas[t] = rewards[t] - Vs[t,i]
		Vs[t+1,i] = Vs[t,i] + a*deltas[t]

fig, ax = plt.subplots(1)
for i,a in enumerate(alphas):
	ax.plot(trials, Vs[:-1,i], label=r"$\alpha = $"+str(a))
ax.set(xticks=np.arange(n, step=5), xlabel="Trial", ylabel=r"$V_t$")
ax.legend(loc='right')
# plt.show()
fig.savefig('figs/slide27.png', dpi=300)


## Slide 31
x  = np.linspace(-10, 10, num=1000)
y1 = norm.pdf(x, loc=1, scale=1)
y2 = norm.pdf(x, loc=-1, scale=1)

fig, ax = plt.subplots(1)
ax.plot(x,y1,label="Square")
ax.plot(x,y2,label="Triangle")
ax.set(xlabel="Reward", ylabel="Density")
ax.legend()
# plt.show()
fig.savefig('figs/slide31.png', dpi=300)

## Slide 32+
## Simulate data

MU = [-1, 1] ## mu_triangle, mu_square
SIGMA = 1 ## share std. dev.

def softmax(beta,V):
	return np.exp(beta*V)/np.sum(np.exp(beta*V))

def sim_simple_rl(alpha:float, beta:float, ntrials:int=30):

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
	return df

## simulate for multiple subjects
np.random.seed(2023)
N = 20
alphas = np.random.uniform(.1,.3, size=N)
betas  = np.random.uniform(1, 3, size=N)
df = pd.concat([sim_simple_rl(alphas[i], betas[i]) for i in range(N)])

## normalize variables for plotting
df['V_square_01'] = df.groupby("PID")["V_square"].transform(lambda x: (x-x.min()) / (x.max()-x.min()))
df['V_triangle_01'] = df.groupby("PID")["V_triangle"].transform(lambda x: (x-x.min()) / (x.max()-x.min()))

## visualize choice trajectory
fig, ax = plt.subplots(1)
sns.lineplot(data=df[df.trial>1], x='trial', y='choice', color="black", ax=ax, errorbar=None)
sns.lineplot(data=df[df.trial>1], x='trial', y='V_square_01', color="darkblue", ls='--', ax=ax, errorbar=None)
sns.lineplot(data=df[df.trial>1], x='trial', y='V_triangle_01', color="lightblue", ls='--', ax=ax, errorbar=None)
ax.set(xlabel="Trial", ylabel="P(Choose Square)", ylim=(0,1.1))
# plt.show()
fig.savefig('figs/slide32.png', dpi=300)

## Slide 35
V1 = np.linspace(-10, 10, num=100)
V2 = np.linspace(-10, 10, num=100)[::-1]
dV = V1-V2
betas = np.linspace(0.5, 2.5, num=5)
p = np.zeros((dV.shape[0], betas.shape[0]))

fig, ax = plt.subplots(1)
for i in range(betas.shape[0]):
	p[:,i] = [softmax(betas[i], np.array([V1[v], V2[v]]))[0] for v in range(100)]
	ax.plot(dV, p[:,i], label=betas[i])
ax.set(xlabel=r"$V_1 - V_2$", ylabel=r"P(Choose $O_1$)")
ax.legend(loc="right", title=r"$\beta$")
# plt.show()
fig.savefig('figs/slide35.png', dpi=300)


## Slide 44
fig, ax = plt.subplots(1)
sns.lineplot(data=df[df.trial>1], x='trial', y='p_square', color="black", ax=ax, errorbar=None, label="Predicted P(Square)")
sns.lineplot(data=df[df.trial>1], x='trial', y='V_square_01', color="darkblue", ls='--', ax=ax, errorbar=None, label=r"$V(Square)$")
sns.lineplot(data=df[df.trial>1], x='trial', y='V_triangle_01', color="lightblue", ls='--', ax=ax, errorbar=None, label=r"$V(Triangle)$")
ax.set(xlabel="Trial", ylabel="P(Choose Square)", ylim=(0,1.3), 
	yticks=[0,.25,.5,.75,1])
ax.legend()
ax.axhline(0.5, ls='--', c='grey')
# plt.show()
fig.savefig('figs/slide44a.png', dpi=300)


fig, ax = plt.subplots(1)
sns.lineplot(data=df[df.trial>1], x='trial', y='choice', color="black", ax=ax, errorbar=None, label="Real Choice")
sns.lineplot(data=df[df.trial>1], x='trial', y='p_square', color="blue", ls='--', ax=ax, label="Model Predicted")
ax.set(xlabel="Trial", ylabel="P(Choose Square)", ylim=(0,1.3), 
	yticks=[0,.25,.5,.75,1])
ax.legend()
ax.axhline(0.5, ls='--', c='grey')
# plt.show()
fig.savefig('figs/slide44b.png', dpi=300)

## Slide 49
flips = np.array([0]*3 + [1]*7)
x = np.linspace(0,1,num=1000)
y = beta.pdf(x,4,4)

fig,ax=plt.subplots(1)
ax.plot(x,y)
ax.set(xlabel="Probability of Coin Landing Heads", 
	ylabel="Density",
	title="Beta Distribution for 10 Coin Tosses")
fig.savefig('figs/slide49.png', dpi=300)

## Slide 50
min_y = y[np.argmin(np.abs(x-.7))]
fig,ax=plt.subplots(1)
ax.plot(x,y)
ax.set(xlabel="Probability of Coin Landing Heads", 
	ylabel="Density",
	title="Beta Distribution for 10 Coin Tosses")
ax.plot(0.7,min_y , marker='x', c='red', ms=30)
ax.axvline(0.7, ymax=min_y/max(y), ls='--', color='red')
ax.axhline(min_y, xmax=0.7, ls='--', color='red')
# plt.show()
fig.savefig('figs/slide50.png', dpi=300)

## Slide 51
y2 = beta.pdf(x,6,2)
fig,ax=plt.subplots(1)
ax.plot(x,y)
ax.plot(x,y2)
ax.set(xlabel="Probability of Coin Landing Heads", 
	ylabel="Density",
	title="Beta Distribution for 10 Coin Tosses")
# plt.show()
fig.savefig('figs/slide51.png', dpi=300)

## Slide 52
min_y = y2[np.argmin(np.abs(x-.7))]
fig,ax=plt.subplots(1)
ax.plot(x,y)
ax.plot(x,y2)
ax.set(xlabel="Probability of Coin Landing Heads", 
	ylabel="Density",
	title="Beta Distribution for 10 Coin Tosses")
ax.plot(0.7,min_y , marker='x', c='red', ms=30)
ax.axvline(0.7, ymax=min_y/max(y2), ls='--', color='red')
ax.axhline(min_y, xmax=0.7, ls='--', color='red')
# plt.show()
fig.savefig('figs/slide52.png', dpi=300)

## Slide 53
fig,ax=plt.subplots(1)
ax.plot(x,y, label="Beta(4,4)")
ax.plot(x,y2, label="Beta(6,2)")
ax.set(xlabel="Probability of Coin Landing Heads", 
	ylabel="Density",
	title="Beta Distribution for 10 Coin Tosses")
ax.legend(loc='upper left', prop={"size":18})
# plt.show()
fig.savefig('figs/slide53.png', dpi=300)

## Slide 54
x_p = flips.mean()
resolution = 100
par1 = np.linspace(0.01,20,num=resolution)
par2 = np.linspace(0.01,20,num=resolution)

liks = np.zeros((resolution,resolution))

for i,a in enumerate(par1):
	for j,b in enumerate(par2):
		liks[i,j] = beta.pdf(x_p, a, b)

fig, ax = plt.subplots(1, figsize=(8,6))
sns.heatmap(liks, cbar=True)
ax.set(xlabel="a", ylabel="b", title="Likelihoods for Beta distribution for 7/10 heads", 
	xticks=np.linspace(0,resolution,5), xticklabels=np.linspace(0,par1.max(),5), 
	yticks=np.linspace(0,resolution,5), yticklabels=np.linspace(0,par2.max(),5))
ax.invert_yaxis()
max_idx = np.unravel_index(liks.argmax(), liks.shape)
ahat, bhat = par1[max_idx[0]], par2[max_idx[1]] 
ax.plot(max_idx[1], max_idx[0]-1, marker='x', c='red', ms=20)
# plt.show()
fig.savefig('figs/slide54a.png', dpi=300)

#----

fig, ax = plt.subplots(1)
x = np.linspace(0,1,num=100)
y = beta.pdf(x,ahat,bhat)
min_y = y[np.argmin(np.abs(x-x_p))]
ax.plot(x,y)
ax.set(xlabel="Probability of Coin Landing Heads", 
	ylabel="Density",
	title="Beta(20,9.10)")
ax.plot(x_p,min_y , marker='x', c='red', ms=30)
ax.axvline(x_p, ymax=min_y/max(y), ls='--', color='red')
ax.axhline(min_y, xmax=x_p, ls='--', color='red')
fig.savefig('figs/slide54b.png', dpi=300)

## slide 80

p = np.linspace(0,1,num=1000)
logp = np.log(p)

fig, ax = plt.subplots(1)
ax.plot(p, logp)
ax.set(xlabel="Probability", ylabel="Log(Probability)")
fig.savefig('figs/slide80.png', dpi=300)


## Slide 83
## use normal likelihood function for illustration
x = np.random.normal(size=1000)
mus = np.linspace(-10,10,num=100)
nll = np.array([-np.log(norm.pdf(x, loc=mu)).sum() for mu in mus])
nll *= -np.sin(mus) / 100
nll[:10] += 100

fig, ax = plt.subplots(1, figsize=(6,4))
ax.plot(mus, nll)
ax.set(xlabel=r"$\theta$", ylabel="Negative Log-Likelihood", xticks=[])
# plt.show()
fig.savefig('figs/slide83.png', dpi=300)

