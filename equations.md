$$X_{adv} = X + \epsilon \nabla_X L(X,Y)$$

$$\delta = \max_{\lVert \delta\rVert_p\leq \epsilon} \biggl(L(X+\delta,Y)\biggr)$$

$$    \min_\theta \max_{\lVert\delta\rVert_p\leq \epsilon} \biggl(L(X+\delta,Y;\theta)\biggr) $$
