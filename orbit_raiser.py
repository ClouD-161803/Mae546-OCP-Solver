from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from scipy.integrate import solve_ivp
from scipy.optimize import root

Array = np.ndarray


@dataclass
class OrbitParams:
    """Normalised problem parameters."""
    mu: float = 1.0
    T_norm: float = 0.0
    beta_norm: float = 0.0

@dataclass
class OrbitIC:
    """Normalised initial conditions."""
    r0: float = 1.0
    u0: float = 0.0
    v0: float = 1.0
    m0: float = 1.0

@dataclass
class OrbitSolution:
    """Class for storing the BVP solution."""
    success: bool
    message: str
    t: Array
    r: Array
    u: Array
    v: Array
    m: Array
    pr: Array
    pu: Array
    pv: Array
    pm: Array
    phi: Array
    tf: float
    p0: Optional[Array] = None
    info: Optional[Dict[str, Any]] = None


@dataclass
class UnitConverter:
    """
    Helper class to manage unit conversions.
    DU = r0, MU = m0, and mu_norm = 1.0.
    """
    du: float  # Distance Unit (km)
    mu: float  # Gravitational Parameter (km^3/s^2)
    m0: float  # Mass Unit (kg)
    mu_val: float = 1.0

    def __post_init__(self):
        self.tu = np.sqrt(self.du**3 / (self.mu / self.mu_val))
        self.vu = self.du / self.tu
        self.mu_norm = self.mu_val

    def __str__(self):
        return (
            f"--- Unit Converter ---\n"
            f"DU = {self.du:.3e} km\n"
            f"TU = {self.tu:.3e} s  ({self.tu / (24*3600):.2f} days)\n"
            f"MU = {self.m0:.3e} kg\n"
            f"VU = {self.vu:.3e} km/s\n"
            "------------------------"
        )

    def normalize_dist(self, d_phys: float) -> float:
        return d_phys / self.du

    def normalize_vel(self, v_phys: float) -> float:
        return v_phys / self.vu

    def normalize_time(self, t_phys: float) -> float:
        return t_phys / self.tu

    def normalize_mass(self, m_phys: float) -> float:
        return m_phys / self.m0

    def normalize_mdot(self, mdot_phys: float) -> float:
        return mdot_phys * self.tu / self.m0

    def normalize_thrust(self, T_phys: float) -> float:
        return T_phys * (self.tu**2) / (self.du * self.m0)

    def denormalize_dist(self, d_norm: float) -> float:
        return d_norm * self.du

    def denormalize_mass(self, m_norm: float) -> float:
        return m_norm * self.m0


class IndirectShootingSolver:
    """
    Solves the 10.1 BVP using an indirect shooting method.
    Finds the 4 unknown initial costates p0 that satisfy the
    4 final transversality/boundary conditions.
    """

    def __init__(self, p: OrbitParams, ic: OrbitIC, tf_norm: float):
        self.p = p
        self.ic = ic
        self.tf = tf_norm
        self.y0_known = np.array([ic.r0, ic.u0, ic.v0, ic.m0])
        self.sol_cache: Optional[Any] = None
        self.tol: float = 1e-10

    def _ode_system(self, t: float, y: Array) -> Array:
        """The 8-state ODE system for states and costates."""
        r, u, v, m, pr, pu, pv, pm = y
        
        lambda_p = np.sqrt(pu**2 + pv**2)
        
        if lambda_p < 1e-12:
            sin_phi, cos_phi = 0.0, 1.0
        else:
            sin_phi = pu / lambda_p
            cos_phi = pv / lambda_p

        T_m = self.p.T_norm / m
        mu_r2 = self.p.mu / (r**2)
        v2_r = v**2 / r
        uv_r = u * v / r

        dr = u
        du = v2_r - mu_r2 + T_m * sin_phi
        dv = -uv_r + T_m * cos_phi
        dm = -self.p.beta_norm

        dpr = pu * (v2_r / r - 2 * mu_r2 / r) + pv * (uv_r / r)
        dpu = -pr + pv * (v / r)
        dpv = -pu * (2 * v / r) + pv * (u / r)
        dpm = (self.p.T_norm * lambda_p) / (m**2)

        return np.array([dr, du, dv, dm, dpr, dpu, dpv, dpm])

    def _calculate_residuals(self, p0_guess: Array) -> Array:
        """
        Integrates the ODEs and returns the 4 final boundary condition
        residuals. This is the function for the root-finder.
        """
        y0 = np.hstack([self.y0_known, p0_guess])
        
        sol = solve_ivp(
            self._ode_system,
            [0, self.tf],
            y0,
            method="RK45",
            rtol=self.tol,
            atol=self.tol
        )
        
        self.sol_cache = sol 

        if not sol.success:
            return np.full(4, 1e10)

        Yf = sol.y[:, -1]
        rf, uf, vf, mf, prf, puf, pvf, pmf = Yf
        
        res_u = uf
        res_v = vf - np.sqrt(self.p.mu / rf)
        res_pr = prf - (1.0 + (pvf / 2.0) * np.sqrt(self.p.mu / (rf**3)))
        res_pm = pmf

        return np.array([res_u, res_v, res_pr, res_pm])

    def solve(self, 
              p0_guess: Array, 
              tol: float = 1e-10, 
              ftol: float = 1e-10, 
              verbose: bool = False,
              **kwargs) -> OrbitSolution:
        """
        Solves the BVP using scipy.optimize.root.
        """
        self.tol = tol
        self.sol_cache = None
        
        res = root(
            self._calculate_residuals,
            p0_guess,
            method="lm",
            options={'ftol': ftol, 'xtol': ftol, **kwargs}
        )
        
        if verbose:
            print(res)

        if res.success:
            sol_ivp = self.sol_cache
            assert sol_ivp is not None, "sol_cache should not be None when res.success is True"
            y_full = sol_ivp.y
            
            return OrbitSolution(
                success=True,
                message=res.message,
                t=sol_ivp.t,
                r=y_full[0], u=y_full[1], v=y_full[2], m=y_full[3],
                pr=y_full[4], pu=y_full[5], pv=y_full[6], pm=y_full[7],
                phi=np.empty_like(sol_ivp.t),
                tf=self.tf,
                p0=res.x,
                info=dict(res)
            )
        else:
            return OrbitSolution(
                success=False,
                message=res.message,
                t=np.array([0]), r=np.array([0]), u=np.array([0]), 
                v=np.array([0]), m=np.array([0]), pr=np.array([0]),
                pu=np.array([0]), pv=np.array([0]), pm=np.array([0]),
                phi=np.array([0]), tf=self.tf, p0=p0_guess,
                info=dict(res)
            )

    def post_process(self, sol: OrbitSolution) -> OrbitSolution:
        """
        Calculates the optimal control phi along the trajectory.
        """
        if not sol.success:
            return sol
            
        lambda_p = np.sqrt(sol.pu**2 + sol.pv**2)
        sol.phi = np.arctan2(sol.pu, sol.pv) # atan2(y, x) -> atan2(pu, pv)
        
        return sol