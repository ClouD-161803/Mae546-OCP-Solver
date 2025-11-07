from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Tuple, Dict, Any, Optional, List, Type
from scipy.optimize import minimize, Bounds, NonlinearConstraint

Array = np.ndarray

@dataclass
class LanderParams:
    g: float = 1.5
    umax: float = 3.0
    N: int = 50

@dataclass
class LanderIC:
    x0: float = 10.0
    v0: float = -2.0

@dataclass
class LanderSolution:
    success: bool
    message: str
    t: Array
    x: Array
    v: Array
    u: Array
    tf: float
    J: float
    nit: int
    info: Dict[str, Any]

class BaseSolver:
    def __init__(self, p: LanderParams, ic: LanderIC):
        self.p, self.ic = p, ic

    def solve(self, **kwargs) -> LanderSolution:
        raise NotImplementedError

    @staticmethod
    def step_const_acc(x: float, v: float, a: float, dt: float) -> Tuple[float, float]:
        return x + v*dt + 0.5*a*dt*dt, v + a*dt

    def sweep(self, g_list: List[float], umax_list: List[float], Solver: Type['BaseSolver'],
              N: int = 40, maxiter: int = 600) -> Tuple[Dict[float, List[float]], Dict[float, List[float]]]:
        tf_map, J_map = {}, {}
        for g in g_list:
            tfs, Js = [], []
            for umax in umax_list:
                p = LanderParams(g=g, umax=umax, N=N)
                sol = Solver(p, self.ic).solve(maxiter=maxiter)
                tfs.append(sol.tf if sol.success else np.nan)
                Js.append(sol.J if sol.success else np.nan)
            tf_map[g] = tfs; J_map[g] = Js
        return tf_map, J_map

class DirectShootingSolver(BaseSolver):
    def simulate(self, u: Array, tf: float) -> Tuple[Array, Array, Array]:
        N, dt = u.size, tf/u.size
        t = np.linspace(0.0, tf, N+1)
        x = np.zeros(N+1); v = np.zeros(N+1)
        x[0], v[0] = self.ic.x0, self.ic.v0
        for i in range(N):
            a = -self.p.g + float(u[i])
            x[i+1], v[i+1] = self.step_const_acc(x[i], v[i], a, dt)
        return t, x, v

    def objective(self, q: Array) -> float:
        u, tf = q[:-1], q[-1]
        return (tf/self.p.N)*float(np.sum(u))

    def terminal(self, q: Array) -> Array:
        u, tf = q[:-1], q[-1]
        _, x, v = self.simulate(u, tf)
        return np.array([x[-1], v[-1]])

    def solve(self, u_init: Optional[Array]=None, tf_init: Optional[float]=None, maxiter: int=600, **kwargs) -> LanderSolution:
        N = self.p.N
        if u_init is None:
            u_init = np.full(N, min(max(self.p.g, 0.1), self.p.umax))
        if tf_init is None:
            tf_init = max(1.0, 2.0*self.ic.x0/max(1.0, (self.p.umax - self.p.g + 1e-6)))
        q0 = np.hstack([u_init, [tf_init]])
        lb = np.hstack([np.zeros(N), 1e-3]); ub = np.hstack([np.full(N, self.p.umax), 1e3])
        bounds: Bounds = Bounds(lb, ub)  # type: ignore[arg-type]
        nlc = NonlinearConstraint(lambda q: self.terminal(q), lb=np.zeros(2), ub=np.zeros(2))
        res = minimize(lambda q: self.objective(q), q0, method="SLSQP",
                       bounds=bounds, constraints=[nlc],
                       options=dict(maxiter=maxiter, ftol=1e-9, disp=False))
        u_star, tf_star = res.x[:-1], res.x[-1]
        t, x, v = self.simulate(u_star, tf_star)
        J = self.objective(res.x)
        info = dict(status=res.status, constr_violation=getattr(res, "constr_violation", None))
        return LanderSolution(bool(res.success), res.message, t, x, v, u_star, tf_star, J, res.nit, info)

class HSSolver(BaseSolver):
    def _sizes(self) -> Tuple[int,int,int,int,int,int]:
        N = self.p.N
        return N+1, N+1, N, N, N+1, N

    def unpack(self, z: Array):
        n_xn, n_vn, n_xm, n_vm, n_un, n_um = self._sizes()
        ofs = 0
        x_n = z[ofs:ofs+n_xn]; ofs += n_xn
        v_n = z[ofs:ofs+n_vn]; ofs += n_vn
        x_m = z[ofs:ofs+n_xm]; ofs += n_xm
        v_m = z[ofs:ofs+n_vm]; ofs += n_vm
        u_n = z[ofs:ofs+n_un]; ofs += n_un
        u_m = z[ofs:ofs+n_um]; ofs += n_um
        tf  = float(z[ofs])
        return x_n, v_n, x_m, v_m, u_n, u_m, tf

    @staticmethod
    def f_vec(x: float, v: float, u: float, g: float) -> Array:
        return np.array([v, -g + u], dtype=float)

    def simpson_cost(self, u_nodes: Array, u_mids: Array, tf: float) -> float:
        N, dt = u_mids.size, tf/u_mids.size
        s = 0.0
        for i in range(N):
            s += (u_nodes[i] + 4.0*u_mids[i] + u_nodes[i+1])
        return (dt/6.0)*s

    def hs_constraints(self, z: Array) -> Array:
        N = self.p.N
        x_n, v_n, x_m, v_m, u_n, u_m, tf = self.unpack(z)
        dt = tf / N
        g = []
        for i in range(N):
            xi, vi   = x_n[i],   v_n[i]
            xi1, vi1 = x_n[i+1], v_n[i+1]
            xm, vm   = x_m[i],   v_m[i]
            ui, ui1, um = u_n[i], u_n[i+1], u_m[i]
            Fi  = self.f_vec(xi,  vi,  ui,  self.p.g)
            Fi1 = self.f_vec(xi1, vi1, ui1, self.p.g)
            Fm  = self.f_vec(xm,  vm,  um,  self.p.g)
            g.append(xm - 0.5*(xi + xi1) - (dt/8.0)*(Fi[0] - Fi1[0]))
            g.append(vm - 0.5*(vi + vi1) - (dt/8.0)*(Fi[1] - Fi1[1]))
            g.append((xi1 - xi) - (dt/6.0)*(Fi[0] + 4.0*Fm[0] + Fi1[0]))
            g.append((vi1 - vi) - (dt/6.0)*(Fi[1] + 4.0*Fm[1] + Fi1[1]))
        return np.array(g, dtype=float)

    def solve(self, x_init: Optional[Array]=None, v_init: Optional[Array]=None,
              u_init: Optional[Array]=None, um_init: Optional[Array]=None,
              tf_init: Optional[float]=None, maxiter: int=1000, **kwargs) -> LanderSolution:
        N = self.p.N
        if tf_init is None:
            tf_init = max(1.0, 2.0*self.ic.x0/max(1.0, (self.p.umax - self.p.g + 1e-6)))
        tgrid = np.linspace(0.0, 1.0, N+1)
        if x_init is None: x_init = self.ic.x0*(1.0 - tgrid)
        if v_init is None: v_init = self.ic.v0*(1.0 - tgrid)
        if u_init is None: u_init = np.full(N+1, min(max(self.p.g, 0.1), self.p.umax))
        if um_init is None: um_init = np.full(N,   min(max(self.p.g, 0.1), self.p.umax))
        x_mid0 = 0.5*(x_init[:-1] + x_init[1:])
        v_mid0 = 0.5*(v_init[:-1] + v_init[1:])
        z0 = np.hstack([x_init, v_init, x_mid0, v_mid0, u_init, um_init, [tf_init]])

        big = 1e6
        lb_xn = np.full(N+1, -big); ub_xn = np.full(N+1, big)
        lb_vn = np.full(N+1, -big); ub_vn = np.full(N+1, big)
        lb_xm = np.full(N,   -big); ub_xm = np.full(N,   big)
        lb_vm = np.full(N,   -big); ub_vm = np.full(N,   big)
        lb_xn[0] = ub_xn[0] = self.ic.x0
        lb_vn[0] = ub_vn[0] = self.ic.v0
        lb_xn[-1] = ub_xn[-1] = 0.0
        lb_vn[-1] = ub_vn[-1] = 0.0
        lb_un = np.zeros(N+1); ub_un = np.full(N+1, self.p.umax)
        lb_um = np.zeros(N);   ub_um = np.full(N,   self.p.umax)
        lb_tf = 1e-3;          ub_tf = 1e3
        lb = np.hstack([lb_xn, lb_vn, lb_xm, lb_vm, lb_un, lb_um, [lb_tf]])
        ub = np.hstack([ub_xn, ub_vn, ub_xm, ub_vm, ub_un, ub_um, [ub_tf]])
        bounds: Bounds = Bounds(lb, ub)  # type: ignore[arg-type]

        nlc = NonlinearConstraint(lambda z: self.hs_constraints(z), lb=np.zeros(4*N), ub=np.zeros(4*N))
        def objective(z: Array) -> float:
            x_n, v_n, x_m, v_m, u_n, u_m, tf = self.unpack(z)
            return self.simpson_cost(u_n, u_m, tf)

        res = minimize(objective, z0, method="SLSQP",
                       bounds=bounds, constraints=[nlc],
                       options=dict(maxiter=maxiter, ftol=1e-9, disp=False))

        x_n, v_n, x_m, v_m, u_n, u_m, tf = self.unpack(res.x)
        t_nodes = np.linspace(0.0, tf, N+1)
        J = self.simpson_cost(u_n, u_m, tf)
        info = dict(status=res.status, constr_violation=getattr(res, "constr_violation", None))
        return LanderSolution(bool(res.success), res.message, t_nodes, x_n, v_n, u_n, tf, J, res.nit, info)