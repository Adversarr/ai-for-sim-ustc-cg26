# Equations for the double pendulum system

$$
\begin{aligned}
\dot{\theta}_1 &= \omega_1 \qquad
\dot{\theta}_2 = \omega_2   \\
\dot{\omega}_1 &= \frac{m_2 g \sin(\theta_2)\cos(\theta_1-\theta_2) - m_2\sin(\theta_1-\theta_2)(l_1\omega_1^2\cos(\theta_1-\theta_2) + l_2\omega_2^2) - (m_1+m_2)g\sin(\theta_1)}{l_1 D} \\
\dot{\omega}_2 &= \frac{(m_1+m_2)(l_1\omega_1^2\sin(\theta_1-\theta_2) - g\sin(\theta_2) + g\sin(\theta_1)\cos(\theta_1-\theta_2)) + m_2 l_2 \omega_2^2\sin(\theta_1-\theta_2)\cos(\theta_1-\theta_2)}{l_2 D}
\end{aligned}
$$

## Numerical Method: RK3 (3rd-order Runge-Kutta)

The system of ODEs is solved using the **RK3 method** (Kutta's third-order method), an explicit Runge-Kutta scheme with local truncation error O(h⁴) and global error O(h³).

For the initial value problem ($\mathbf{y}=[\theta_1, \theta_2, \omega_1, \omega_2]$):

$$
\frac{d\mathbf{y}}{dt} = \mathbf{f}(t, \mathbf{y}), \quad \mathbf{y}(t_0) = \mathbf{y}_0
$$

The RK3 update formula is:
$$
\begin{aligned}
k_1 &= \mathbf{f}(t_n, \mathbf{y}_n) \\
k_2 &= \mathbf{f}(t_n + \frac{h}{2}, \mathbf{y}_n + \frac{h}{2}k_1) \\
k_3 &= \mathbf{f}(t_n + h, \mathbf{y}_n - h k_1 + 2h k_2) \\
\mathbf{y}_{n+1} &= \mathbf{y}_n + \frac{h}{6}(k_1 + 4k_2 + k_3)
\end{aligned}
$$

where:
- $h$ is the time step
- $\mathbf{y}_n$ is the state vector at time $t_n$
- $k_1, k_2, k_3$ are the stage derivatives