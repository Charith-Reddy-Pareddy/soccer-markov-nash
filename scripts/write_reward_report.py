"""Build the reward appendix from measured audit results, without retraining."""
import json
from pathlib import Path


def table(matrix):
    head='<tr><th>P0 / P1</th><th>U</th><th>D</th><th>L</th><th>R</th></tr>'
    return '<table>'+head+''.join('<tr><th>'+a+'</th>'+''.join(f'<td>{x:.6f}</td>' for x in row)+'</tr>' for a,row in zip('UDLR',matrix))+'</table>'


def main():
    data=json.loads(Path('experiments/reward_q_audit.json').read_text())
    cert=json.loads(Path('experiments/reward_reference_audit.json').read_text())
    sparse=data['models']['sparse'];dense=data['models']['possession_progress']
    html=f'''<section id="reward-audit"><h2>Reward audit: why Jae's Q differs</h2>
<p>Verified against Jae Woo Song's public viewer and deterministic data,
September 28, 2026. Same state: <b>(3,4,4,4,0)</b>; gamma = 0.9.
Rows and columns are U, D, L, R; every entry is player 0's return.</p>
<p><b>Your default:</b> +1/-1 at a terminal goal; zero otherwise.
<b>Jae:</b> the same terminal rewards, but every nonterminal next state pays
0.005 + 0.005*x0 when player 0 carries, or
-(0.005 + 0.005*(6-x1)) when player 1 carries. No bonus on terminal goals.
The bonus rewards position held, not change in distance.</p>
<h3>Your sparse-reward Q: V = 0</h3>{table(sparse['cases'][2]['Q'])}
<h3>Jae's reward, independently solved here: V = 0.2</h3>{table(dense['cases'][2]['Q'])}
<p>For (U,U), possession remains with player 0 at x=3: reward = 0.020,
so Q = 0.020 + 0.9*0.2 = 0.2. For (U,L), possession changes:
reward = -0.015 and next value = -0.15, giving Q = -0.15.
These are reward units, <b>not winning probabilities</b>.</p>
<p>The reproduced matrix matches all 16 screenshot entries to six decimal
places (maximum difference {cert['screenshot_max_error']:.2e}).
Across all 38,080 state/action pairs, transition and reward mismatches:
<b>{cert['mismatches']}</b>. Maximum difference from Jae's published values:
{cert['value_max_error']:.2e}; reconstructed Q: {cert['Q_max_error']:.2e}.
His stored V values use eight decimal places.</p></section>
<section><h2>Zero matrices and neural cross-checks</h2>
<p><b>The professor's concern needs a full check.</b> One zero-reward self-loop
is insufficient proof: deviations may lead to a win. We check every action
pair and each policy's guaranteed payoff at every state. Both bounds must
agree with V. The sparse board's maximum residual is zero at machine
precision. An additional certificate checks every Bellman equality with exact
fractions at gamma=9/10: {sparse['all_zero_matrices']} of 2,380 matrices are exactly zero
before formatting. The screenshot state has 14 zeros, not 16.
A zero minimax value does not imply that every future policy draws.</p>
<p>Jae's reward produces {cert['mixed_states']} mixed states; its maximum
equilibrium gap is {cert['max_equilibrium_gap']:.2e} and Bellman residual
is {cert['max_bellman_residual']:.2e}. It changes the objective and can change
the equilibrium, not merely the printed scale.</p>
<p><b>Learned experiments:</b> two random seeds, hidden width 64, 400 fitted
minimax-DQN epochs and 1,000 PG iterations with 100-step rollouts per reward
model. No exact targets were provided during training. MAE and maximum
absolute error below cover all 38,080 Q entries.</p>
<table><tr><th>Reward / seed</th><th>DQN MAE</th><th>DQN max</th><th>PG Q-policy MAE</th><th>PG max</th></tr>'''
    for name,model in data['models'].items():
        for run in model['training']:
            html+='<tr><th>'+('Sparse' if name=='sparse' else 'Jae')+' / '+str(run['seed'])+'</th>'+''.join(f'<td>{run[k]:.4f}</td>' for k in ['dqn_Q_mae','dqn_Q_max_error','pg_policy_Q_mae','pg_policy_Q_max_error'])+'</tr>'
    html+='''</table><p>DQN approximates equilibrium Q. PG returns probabilities,
not a Q matrix: its Q-policy here is computed by exact evaluation of the
learned pair, followed by r + gamma*V-policy(next). It equals equilibrium Q
only if the learned policies reach an equilibrium. These finite runs do not
establish neural convergence. Full matrices for all eight cases are in
experiments/reward_q_audit.json.</p>
<p><b>Defect fixed:</b> PG restarted after goals but its return calculation
included the next game's rewards. Both independent and shared trainers now
stop returns at terminal boundaries. The regression example [0,+1,-1]
with the last two steps terminal now returns [.9,1,-1], not [.09,.1,-1].
The sparse exact solver's zeros were not caused by this PG defect.</p>
<p><b>Sources:</b> <a href="https://jwsong118.github.io/soccer-mpe/soccer-viewer.js">Jae's reward/transition source</a>;
<a href="https://cs.uwaterloo.ca/~klarson/teaching/W06-886/papers/Littman94.pdf">Littman (1994), minimax Markov games</a>;
<a href="https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf">Ng et al. (1999), potential-based shaping</a>;
<a href="https://arxiv.org/abs/2102.04540">Wei et al. (2021), specialized optimistic actor-critic convergence</a>.
The latter is not a guarantee for vanilla REINFORCE. Arbitrary recurring
bonuses should not be assumed policy-invariant. Reproduction and source
hashes: <a href="reward-q-audit.md">reward-q-audit.md</a>.</p></section>'''
    Path('docs/reward_q_appendix.html').write_text(html)
    p=Path('docs/a10_cases.html');original=p.read_text()
    start=original.find('<section id="reward-audit">')
    if start>=0: original=original[:start]+'</html>'
    p.write_text(original.replace('</html>',html+'</html>'))

if __name__=='__main__': main()
