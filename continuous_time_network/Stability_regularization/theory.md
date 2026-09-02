# Why the continuous-time network fails past two periods, and what the literature offers

*Theory section for the stability-regularisation study. Draft for review, 2026-09-02. Every claim about a paper is taken from the paper itself (arXiv 2203.13648, 2509.11768, 2604.23528); every claim about our own runs points at the folder that produced it.*

---

In this section I set out, from first principles, why a network trained on the residual of the van der Pol equation stops reproducing the oscillation after about two periods, why that failure is a property of the training objective rather than of the network or the optimiser, and what three recent papers say about fixing it. I start with the objective itself and show that a fixed point of the flow is a perfect minimiser of its residual part (Section 2). I then show that the fixed point is only the simplest member of a whole family of perfect minimisers, and that the size of that family is what grows with the horizon (Sections 3 and 4). Sections 5 to 7 summarise the three papers in enough detail that a reader can judge our implementation of each against the source. Section 8 states, before the results are read, what each proposed fix should and should not change on our problem. The results section that follows tests those statements.

## 1. The objective, and the vocabulary I will use

The system is the van der Pol oscillator with unit damping,

$$
\dot y_1 = y_2, \qquad \dot y_2 = (1 - y_1^2)\, y_2 - y_1 ,
$$

which I write as $\dot y = f(y)$ with $y = (y_1, y_2)$. Every solution except the one that starts exactly at the origin ends up circulating on the same closed loop, the *limit cycle*, whose period is $6.663$ time units (measured in the reference study, `RK_Truth/`). Our starting state throughout is $y(0) = (2, 0)$, which lies on the loop.

The *continuous-time physics-informed network* is a small fully connected network $y_\theta(t)$ with one input, the time, and two outputs, the two state components; $\theta$ denotes its weights. It is trained to make two things small at once. The first is the *initial-condition term*, the squared distance between the network's value at $t = 0$ and the prescribed start,

$$
L_{\mathrm{IC}}(\theta) = \| y_\theta(0) - y(0) \|^2 .
$$

The second is the *residual term*. The *residual* of a candidate function at a time $t$ is how badly it violates the equation there,

$$
r_\theta(t) = \dot y_\theta(t) - f\big(y_\theta(t)\big),
$$

where $\dot y_\theta$ is the exact time derivative of the network, obtained by automatic differentiation rather than by finite differences. The residual is measured at a finite set of times $\{t_i\}_{i=1}^{N}$ inside $[0, T]$, the *collocation points*, and averaged,

$$
L_{\mathrm{res}}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \| r_\theta(t_i) \|^2 .
$$

The training objective is the plain sum $L = L_{\mathrm{IC}} + L_{\mathrm{res}}$. No trajectory data enters anywhere: the equation supplies the supervision, and a reference solution, computed separately by a verified sixth-order integrator, is used only to score the result afterwards. Success is declared when the relative $L_2$ error against that reference over a grid of spacing $0.01$ is below $0.15$, the threshold used by the two Graz papers summarised below, so that our success rates are comparable with theirs.

A worked micro-example fixes the ideas. Take the constant candidate $y_\theta(t) \equiv (0, 0)$. Its derivative is zero everywhere and $f(0, 0) = (0, 0)$, so its residual is exactly zero at every time and $L_{\mathrm{res}} = 0$. Its initial-condition term is $\|(0,0) - (2,0)\|^2 = 4$. Now take the candidate that equals the reference solution for the first period and then sits at the origin for the rest of the window. Its initial-condition term is zero, its residual is zero on the first period and zero while it sits at the origin, and it is nonzero only during the hand-over between the two, however that hand-over is made. This second candidate is the shape of every failed run in our earlier studies, and the rest of this section explains why the objective likes it.

## 2. Fixed points are perfect minimisers of the residual term

A *fixed point* of the flow is a state $y^*$ with $f(y^*) = 0$: a system started there stays there. The van der Pol system has exactly one, the origin. Its *stability* is read from the Jacobian of $f$, the matrix of partial derivatives, evaluated at the fixed point:

$$
J(y) = \begin{pmatrix} 0 & 1 \\ -1 - 2 y_1 y_2 & 1 - y_1^2 \end{pmatrix}, \qquad
J(0, 0) = \begin{pmatrix} 0 & 1 \\ -1 & 1 \end{pmatrix}.
$$

The eigenvalues of $J(0,0)$ solve $\lambda^2 - \lambda + 1 = 0$, so $\lambda = \tfrac12 \pm \tfrac{\sqrt 3}{2}\, i$. Both have positive real part $\tfrac12$, so a small displacement from the origin grows like $e^{t/2}$ while rotating: the origin is an *unstable spiral*. It is not a *saddle*, which would have one positive and one negative real eigenvalue and a direction along which displacements shrink. This distinction matters in Section 6, because the one documented limitation of the regulariser we test concerns saddles.

The observation at the heart of the first paper (Rohrhofer et al., Section 5 below) is elementary once stated. If the network output is the constant $y^*$, its derivative is zero and $f(y^*)$ is zero, so the residual is zero at every collocation point regardless of where the points sit. The residual term therefore takes its global minimum, zero, at every fixed point of the flow. Moreover, the residual depends continuously on the weights, so weights close to those producing the constant $y^*$ produce a small residual: each fixed point sits at the bottom of a basin in the landscape of the residual term. Nothing in the residual term prefers the true solution to a constant at the origin. Only the initial-condition term does, and it is a single number, while the residual term is an average over $N$ points.

For the true solution the residual term is also zero, so the objective has at least two exact minimisers of its residual part, and the initial-condition term is the sole arbiter between them. The candidate of the micro-example above, true for one period then parked, pays nothing on the initial condition and nothing on the residual except in the hand-over. Whether the hand-over is cheap is the whole question, and Section 3 answers it.

## 3. The fixed point is one member of a family: the splice construction

The second paper (Wang et al., Section 7 below) proves a statement that I paraphrase for our setting and then extend.

**The splice.** Fix the collocation set $\{t_i\}$ and a time $t_0$. Let $\alpha(t)$ be a smooth function equal to $1$ in a small neighbourhood of every collocation point before $t_0$, equal to $0$ in a small neighbourhood of every collocation point after $t_0$, and equal to $0$ for all $t \ge t_0$. Define the *spliced candidate*

$$
y^\dagger(t) = \alpha(t)\, y^*(t),
$$

where $y^*$ is the true solution. Then $y^\dagger$ satisfies the initial condition, is identically zero after $t_0$, and has residual *exactly zero at every collocation point*: near each collocation point before $t_0$ it coincides with the true solution, which solves the equation, and near each collocation point after $t_0$ it coincides with the zero function, which also solves the equation because the origin is a fixed point. Its residual is nonzero only inside the *transition layer*, the stretch where $\alpha$ moves from $1$ to $0$, and that stretch has been placed between collocation points, where nothing is measured. The empirical objective is therefore exactly zero on $y^\dagger$, the same value it takes on the true solution.

The proof needs nothing about the origin except that it solves the equation. That is the extension I want to record, because it matches what we see when the origin is made expensive. For an autonomous system, *every trajectory of the flow* solves the equation, and so does every *time-shift* of a trajectory. So in place of the zero function one may splice in any other trajectory $v(t)$: the candidate that follows $y^*$ near the collocation points before $t_0$ and follows $v$ near the collocation points after $t_0$, with the hand-over hidden between points, also has exactly zero empirical residual. The spurious set is a family, and the fixed point is only its simplest, cheapest member. Two other members are worth naming now:

1. **The phase-shifted loop.** The limit cycle traversed with a different phase is a trajectory of the flow. A network that rides the loop correctly for a lap and then continues on a phase-shifted copy has zero residual away from one hand-over band. The residual cannot see phase, because the equation has no explicit time dependence. Only the anchor at $t = 0$ fixes the phase, and the splice lets the network drop it after a lap.
2. **Off-loop trajectories.** Any trajectory spiralling in from outside the loop, or out from near the origin, is a trajectory of the flow. A hand-over onto one of these produces the *wandering* solutions, which leave the loop in phase space and rejoin it, or do not, as the flow dictates.

**What resampling changes.** The exact zero depends on the collocation set being fixed, so that the transition layer can be tucked between points. If fresh points are drawn every training step, the layer is eventually sampled. Its cost is then finite and computable. Inside a layer of width $h$ the residual is of order $|y^*(t_0)| / h$, since the candidate must move by a distance of order $|y^*(t_0)|$ within a time $h$, and the mean-squared residual over the whole window collects that squared and integrated over the layer, giving a contribution of order

$$
\frac{1}{T} \int_{\mathrm{layer}} \Big| \frac{y^*(t_0)}{h} \Big|^2 dt \;\sim\; \frac{|y^*(t_0)|^2}{h\,T} .
$$

This is Wang et al.'s remark 2.2: under resampling the spliced candidate is no longer free, but its price falls like $1/(hT)$ and can be paid. A micro-example with our numbers: a hand-over from the loop to the origin that takes about a period, $h \approx 2$, at amplitude $|y^*| \approx 2$, in a window $T = 27$, costs of order $4 / (2 \times 27) \approx 0.07$. The resampled runs in this study park with a loss of $0.03$ and a gradient of zero, which is that estimate to within its own crudeness. A sharper hand-over costs more under resampling and less under a fixed set, which is why the two protocols find different parked solutions of the same family.

## 4. Why the horizon grows the basin

Three effects push in the same direction as $T$ grows.

1. **The price of a splice falls like $1/T$.** The hand-over costs a fixed amount, of order $|y^*|^2 / h$ integrated over the layer, and the residual term divides that by the number of points, which is proportional to $T$ at fixed density. The anchor is one number and does not grow. So the objective's preference for the true solution over a spliced one weakens in direct proportion to the horizon.
2. **The number of opportunities grows like $T$.** A splice can be placed anywhere; every additional period adds a period's worth of hand-over positions, each priced at $1/T$.
3. **The price of the true solution rises.** To represent the true solution the network must produce $T / 6.663$ oscillations at the correct phase throughout. Fully connected networks with smooth activations learn low-frequency content first and high-frequency content slowly, the effect the literature calls *spectral bias*, and every extra lap is more high-frequency content that must be held in phase against a single anchor at $t = 0$.

Together these explain the pattern already measured in our earlier work: the unweighted objective fails past one period, the causally weighted objective past two, and no resource, neither network size nor collocation density nor placement nor a fourfold budget, moves those walls. Resources change the price of representing the truth; none of them changes the price of a splice, which is set by the structure of the objective.

## 5. Rohrhofer, Posch, Gößnitzer and Geiger (2023): fixed points shape the loss landscape

**Reference.** *On the Role of Fixed Points of Dynamical Systems in Training Physics-Informed Neural Networks*, Transactions on Machine Learning Research 2023, arXiv 2203.13648. Code at `github.com/frohrhofer/PINNs_fixed_points`.

**Claim.** Fixed points of a dynamical system are global minima of the physics loss with non-trivial basins of attraction, and these basins create local optima and saddle points in the landscape seen by the optimiser. The argument is the continuity argument of Section 2: at a fixed point the residual vanishes identically, and it stays small under small changes of the weights. The paper's phrasing is that the network "can only learn correct behaviour from the given initial and boundary data", which is our statement that the anchor is the sole arbiter.

**Systems and protocol.** Two ordinary differential equations are studied in detail, a one-dimensional toy problem with an unstable fixed point at zero and the undamped pendulum with a stable fixed point at $0^\circ$ and an unstable one at $180^\circ$, plus two partial differential equations, Allen–Cahn and vortex shedding behind a cylinder. The baseline network is four hidden layers of fifty units with the hyperbolic tangent, trained with Adam at learning rate $10^{-3}$ for 50,000 epochs, one hundred random initialisations per setting. Success is a relative $L_2$ error below 15 percent.

**Findings that transfer to us.**

1. Success falls as the horizon $T$ grows and as the starting state approaches a fixed point. Both trends are monotone across their tables.
2. Larger networks improve the success rate slightly but no architecture resolves the failures at long horizons or near fixed points. The authors conclude that the difficulty is "bound to the optimisation complexity, rather than insufficient expressive power". An ablation over learning rate, number of collocation points, loss weighting and initialisation changes nothing notable. This is exactly the null result of our network-size study over 600 runs and our density study over 72 runs, obtained independently on a different system.
3. When the starting state is very close to a fixed point, the non-physical minimum can be *better* than the true one, so that convergence to the truth becomes not merely hard but unfavourable. They call these "economical" solutions. Our baseline runs at $T = 27$ that park with a loss of $10^{-7}$, the same level as the successes at $T = 14$, are economical in this sense.
4. Reducing $T$ makes the spurious minimum disappear from the landscape; the loss-landscape plots show the local optimum fading as the domain shrinks. This is the paper's explanation of why time-marching, curriculum and domain-decomposition methods work: they train on windows short enough that no splice pays.
5. Longer training can escape. Their Allen–Cahn run was trapped at a fixed-point-like solution after 50,000 epochs and escaped to the correct solution after about 90,000, with the loss landscape showing the trap as a saddle. This is the direct precedent for our decision to train every arm to demonstrated convergence before reading its failure mode.

**What the paper does not do.** It proposes no remedy. It also does not distinguish the fixed point from other zero-residual candidates; the family of Section 3 is not in it.

## 6. Babic, Rohrhofer and Geiger (2025): penalise sitting still at an unstable fixed point

**Reference.** *Stabilizing PINNs: A Regularization Scheme for PINN Training to Avoid Unstable Fixed Points of Dynamical Systems*, arXiv 2509.11768. This is the paper whose fix we implement.

**The term.** The regulariser is a product of three factors, averaged over the collocation points and added to the objective. For each collocation point $t_i$:

1. The *local-stability factor* is the sum of the positive real parts of the eigenvalues of the Jacobian of $f$ at the network's output,
   $$R_{\mathrm{LS}}(t_i) = \sum_{\lambda \in \sigma(J(y_\theta(t_i)))} \max(\mathrm{Re}\,\lambda, 0).$$
   It is positive only where the flow is locally unstable. For van der Pol the trace of $J$ is $1 - y_1^2$ and the determinant is $1 + 2 y_1 y_2$, so the factor is positive only inside the band $|y_1| < 1$; at the origin it equals $2 \times \tfrac12 = 1$, and on the outer parts of the loop it is zero.
2. The *stillness factor* is a Gaussian in the network's own speed,
   $$R_{\mathrm{SE}}(t_i) = \exp\!\big(-\|\dot y_\theta(t_i)\|^2 / \varepsilon\big),$$
   with $\varepsilon = 0.01$. It equals one only where the network's trajectory has stopped, and is negligible wherever the speed exceeds a few times $\sqrt{\varepsilon} = 0.1$. The authors' reason for including it is that "stability is a statement about fixed points", so the penalty is meaningless unless the candidate is at one. On the true limit cycle the slowest speed squared is $0.59$, so this factor is at most $e^{-59}$ there: the term is inert on the correct solution by construction.
3. The *coefficient* decays linearly from $C_0 \gamma$ to zero at a fraction $\gamma$ of the training epochs and stays at zero afterwards, $C = \max\big(C_0(\gamma - \mathrm{epoch}/N_{\mathrm{epochs}}), 0\big)$, with $C_0 = 1$ and $\gamma = 0.5$. The authors' justification is that "it may suffice to steer away the candidate solution from unstable fixed points early during training", after which the unmodified objective is handed back.

The full objective is $L = L_{\mathrm{IC}} + L_{\mathrm{res}} + C \cdot \frac{1}{N}\sum_i R_{\mathrm{SE}}(t_i)\, R_{\mathrm{LS}}(t_i)$.

**A worked value.** At initialisation a tanh network with small weights outputs an almost flat curve, so its speed is nearly zero everywhere and the stillness factor is close to one at every point. On our network at $T = 27$ the term evaluates to $0.96$ at initialisation. The gradient of the term then pushes the whole flat curve out of the band $|y_1| < 1$, away from the origin, which is the intended early effect. Once the network is moving, the stillness factor switches the term off point by point.

**Their protocol.** Four hidden layers of fifty units with the Swish activation; Adam at learning rate $10^{-3}$ for 25,000 epochs; *1024 collocation points drawn afresh every epoch*; for second-order systems such as van der Pol the initial condition is a soft loss term with unit weight, exactly as in ours. Ten networks per configuration. Success is the relative $L_2$ error below $0.15$ on a grid of spacing $0.01$.

**Their van der Pol results.** Starts at $(x_0, 0)$ for $x_0 \in \{0.1, \dots, 0.5\}$, all near the origin, and horizons $T \in \{11, \dots, 15\}$. Without the term, success collapses from near-complete at $T = 11$ to zero at $T \ge 13$ for every start. With the term, success is complete at $T \le 12$ for most starts and decays to between $0.1$ and $0.6$ at $T = 15$. In every failed unmodified run, training converged to the unstable fixed point. A second experiment samples twenty starts from a Gaussian centred at the origin with $T = 12.5$ and reports success rising from 0 percent to 100 percent. A sensitivity table on the Duffing oscillator shows success flat between $0.3$ and $0.6$ as $C_0$ ranges over five orders of magnitude, $\varepsilon$ over six, and $\gamma$ from $0.1$ to $0.9$. An ablation shows that on van der Pol the stillness factor alone, without the stability factor, does equally well, which the authors attribute to the system having no stable fixed point to protect.

**Their stated limitation.** Failures that survive the term are traced, by manual inspection, to *saddle points*: a candidate on the wrong side of a saddle's stable manifold is pushed further into the repelling region and, once the term decays, is drawn back. Van der Pol's origin is an unstable spiral, not a saddle, so this limitation does not apply to our problem. The authors propose combining the term with causality-respecting or sequence-to-sequence methods.

**Two facts to hold onto when reading our results.** Their horizons stop at $T = 15$, about two periods, so they never observed a case in which training is still descending when the term switches off; ours at $T \ge 27$ is that case. And their collocation points are resampled every epoch, so the published "regularised PINN" is the regulariser *plus resampling*; the regulariser with a fixed collocation set is our own variant.

**Implementation check.** Our term reproduces equations 3 to 7 of the paper. The eigenvalue sum is computed in closed form from the trace and determinant and agrees with a numerical eigen-decomposition at test points covering both the complex-pair and the real-pair case; its gradients are finite everywhere. The stillness factor uses the network's automatic derivative, as the equation states. The decay schedule is the paper's, with the switch-off at 30,000 epochs in our budget against 12,500 in theirs. One normalisation difference: we average the two state components in the initial-condition and residual terms where the paper sums them, so our two loss terms are half theirs and our $C_0 = 1$ corresponds to their $C_0 = 2$, well inside the flat region of their sensitivity table.

## 7. Wang, Koohy, Lu and Perdikaris (2026): the empirical residual admits spurious solutions; pseudo-time stepping exposes them

**Reference.** *When PINNs Go Wrong: Pseudo-Time Stepping Against Spurious Solutions*, arXiv 2604.23528. Code at `github.com/sifanexisted/jaxpi2`.

**The diagnosis.** Theorem 2.1 is the splice of Section 3, stated for a homogeneous parabolic problem with zero boundary data: for any finite collocation set and any $t_0 > 0$ there is a smooth function that satisfies the initial and boundary conditions, is identically zero after $t_0$, and has exactly zero empirical residual. The construction is $u^\dagger = \alpha(t)\, u^*$ with a cutoff $\alpha$ that is constant near every collocation point. The authors' conclusion is that "the optimisation landscape may contain many poor global minima associated with trivial or spurious solutions" and that "minimising this objective alone is insufficient to prevent convergence to such undesirable states". Remark 2.2 adds the resampling estimate: a transition layer of width $h$ contributes of order $h^{-1}$ to the mean-squared residual, so "unless the optimisation is able to reduce the PDE residual loss below this transition-layer scale, such spurious solutions may not be effectively excluded, even when collocation points are randomly resampled".

**The fix.** *Pseudo-time stepping* replaces the residual at training step $k$ by a relaxed version that also penalises change from the previous iterate,

$$
L_k(\theta) = \frac{1}{N}\sum_{i=1}^{N} \Big\| \frac{y_\theta(t_i) - y_{\theta_{k-1}}(t_i)}{\tau_k} + r_\theta(t_i) \Big\|^2 + L_{\mathrm{IC}}(\theta),
$$

where $\theta_{k-1}$ are the weights before the current step, held fixed, $\tau_k$ is the *pseudo-time step*, and the collocation points are drawn afresh at every step. This is the implicit Euler discretisation of the artificial dynamics $\partial u / \partial s = -r[u]$ in a pseudo-time $s$. Theorem 2.5 gives the mechanism. Apply one explicit pseudo-time update $u^{\dagger,+} = u^\dagger - \tau\, r[u^\dagger]$ to a spliced candidate whose layer has width $h$. On freshly sampled points the expected residual loss of $u^\dagger$ is of order $h^{-1}$, but that of $u^{\dagger,+}$ is of order $h^{-1} + \tau^2 h^{-3}$, because the update differentiates the layer once more. The hidden defect is amplified by a factor $\tau^2 / h^2$, which for a sharp layer is enormous. The relaxed loss therefore penalises spliced candidates far more strongly than the plain residual does. The authors are explicit that resampling is essential to the mechanism and that on a fixed collocation set pseudo-time stepping can fail while attaining a *smaller* training loss.

**Choosing the step.** Larger $\tau$ amplifies more but makes the relaxed objective harder to optimise stably, and the right value cannot be read from the training loss. The adaptive rule estimates the largest locally stable step from a finite-difference surrogate of the residual Jacobian along the current update direction: the ratio of the change in prediction to the change in residual between consecutive iterates on the current points, smoothed with momentum and clipped. A *shrink factor* $\gamma \in [0.1, 1]$, a cosine decay driven by how many decades the plain residual loss has fallen since the start (from two decades to six in their defaults), is folded into the rule so that the damping toward the previous iterate fades as the residual converges and the final solution is a minimiser of the plain residual. On this last point the paper's text and its reference code disagree: the text says the effective step is reduced late in training, the code and its documentation apply the factor to the weight $1/\tau$, so that the step grows and the damping vanishes. We follow the code, which produced the published numbers.

**Their evidence.** Ten partial-differential-equation benchmarks, from Allen–Cahn to Kolmogorov flow at Reynolds number $10^4$, on top of a strong baseline that already includes a residual-adaptive architecture, causal weighting, adaptive loss balancing and a quasi-second-order optimiser. Adaptive pseudo-time stepping gives the lowest error in every case, in several cases with a training loss comparable to or higher than the baseline's, which the authors present as further evidence that the benefit is residual amplification, not conditioning. The baseline's failures on Kolmogorov flow and Rayleigh–Taylor are of the parked kind: "after some time, its prediction becomes nearly stationary and no longer evolves with time". They note the method helps most where spurious solutions are likely, namely sharp transitions and long-time evolution, and little where the baseline already works.

**What transfers, and what we add.** The theorem is stated for a linear spatial operator and a homogeneous problem; the proof of the exact zero on a fixed set uses only that the two spliced pieces solve the equation near the collocation points, which holds for any autonomous ordinary differential equation and any pair of its trajectories. The paper does not consider ordinary differential equations, fixed points, or the phase of an oscillator. Our implementation follows Algorithm 1 with the reference code's constants: initial $\tau = 1$, first update at step 100 and every 1000 thereafter, momentum $0.9$, clipping of $1/\tau$ to $[10^{-2}, 10^{2}]$, shrink thresholds two and six decades with floor $0.1$. The polish stage of our protocol runs on the plain objective, as it does for every arm.

## 8. What each fix prices, and what to expect on our problem

The family of Section 3 gives a single frame for every method under test. A spliced candidate has three properties that a method can act on: it *sits still* only if the spliced-in trajectory is the fixed point; its *transition layer* is invisible on a fixed collocation set and costs $h^{-1}$ under resampling; and it is *cheaper than the truth by a margin that shrinks like $1/T$*.

1. **The stability regulariser** prices sitting still inside the unstable band. It closes exactly one member of the family, the origin. It is inert on every moving candidate, including the phase-shifted loop and the off-loop trajectories, because their speed switches the stillness factor off. So it should convert parking into some other member of the family at long horizons, and cure the failures at horizons where the origin was the only member in play. Keeping the term switched on throughout training should change nothing on a wandering candidate, because the term does not see it.
2. **Resampling alone** turns the exact zero of the fixed set into a finite price of order $|y^*|^2 / (hT)$. Whether that price is paid depends on whether the optimiser can push the residual below the layer scale. At our density and horizons remark 2.2 predicts a parked stationary point with a loss at the layer scale, of order $10^{-2}$.
3. **Pseudo-time stepping with resampling** raises the price of any transition layer to order $\tau^2 h^{-3}$, whichever trajectory is spliced in. It is the only method under test whose mechanism prices the whole family rather than one member. It should be judged at $T \ge 27$, where every other arm fails.
4. **Causal weighting** changes the order in which the objective is exposed to the window rather than the price of a splice: the residual at later times is down-weighted until earlier times are satisfied, so a splice can only be placed at the front as it advances. Its measured effect in our earlier work, moving the wall from one period to two, is consistent with a method that delays the splice without removing its incentive.
5. **Longer training** cannot by itself remove a minimiser that is exact. It can only reveal which minimiser a run has reached. That is why every arm in this study is trained to a demonstrated plateau and polished to a stall before its failure mode is read, and why the fixed-budget numbers of the earlier pass are reported only as a "before" picture.

The results section tests each of these statements against ten seeds per arm and horizon, with every term of the objective logged separately through training, and classifies every failed run by which member of the family it reached.
