import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.linalg import solve_banded
from multiprocessing import Pool
from itertools import product


class qclSolver:

    fundamentals = {

        "e-charge": -1.6021766208 * (10 ** (-19)),
        "planck": 1.054571800 * (10 ** (-34)),
        "planck_e": 6.582119569 * (10 ** (-16)),
        "m0": 9.10938356 * (10 ** (-31)),
        "k-bol": 1.38064852 * (10 ** (-23)),
        "c": 3 * 10 ** 8,
        "eps0": 8.85418781762 * (10 ** (-12)),

    }
    def __init__(self, struct, interval=2, step=0.05, side=5., TE=400.):

        self.step = step
        self.struct = struct
        self.side = side

        self.TL = struct.TL
        self.TE = TE

        self.index, self.last = qclSolver.layerExtrap(struct, side)
        self.grid = np.arange(0, self.last, step)

        self.meff = np.array([(struct.layers[int(self.index(z))].material.params['meff']) for z in self.grid])
        self.Valloy = np.array([(struct.layers[int(self.index(z))].material.params['Valloy']) for z in self.grid])
        self.Perm = np.array(
            [(self.struct.layers[int(self.index(z))].material.params['eps0']) for z in self.grid]) * qclSolver.fundamentals["eps0"]
        self.lattconst = np.array(
            [(self.struct.layers[int(self.index(z))].material.params['lattconst']) for z in self.grid])
        self.comp = np.array([(self.struct.layers[int(self.index(z))].material.x) for z in self.grid])
        self.Ep = np.array([(self.struct.layers[int(self.index(z))].material.params['Ep']) for z in self.grid])

        self.dop = 0.
        for i in range(0, len(struct.dopings)):
            self.dop += np.piecewise(
                self.grid, [self.grid < struct.dopings[i][0],
                            self.grid >= struct.dopings[i][0],
                            self.grid >= struct.dopings[i][1]],
                [0, struct.dopings[i][2], 0]
            )

        self.N_carr = step / 10 ** 9 * self.dop.sum() * (10 ** 6)

        self.tau_pure = 0.1 * 10 ** (-12)  # pure dephasing time


        self.periods = 30
        self.dim_l = 0.3  # cm
        self.dim_h = 4 / 10 ** 4  # cm
        self.dim_w = 0.15  # cm


        if interval == 2:
            self.refr = 3.4
            self.alpha_m = 7.5
            self.alpha_w = 7.5
            self.Gamma_overlap = 1
        else:
            self.refr = 3.4
            self.alpha_m = 3.
            self.alpha_w = 3.
            self.Gamma_overlap = 1

        self.evaluate_W = True
        self.TPop = True
        self.P = -1

    # ============================================
    # ================== EIGS ====================
    # ============================================

    def eigTM(self, resolution=10 ** (-3)):
        m = self.meff * qclSolver.fundamentals["m0"]
        step = self.step / 10 ** 9
        Ep = self.Ep

        Energy = np.arange(np.amin(self.potential), np.amax(self.potential), resolution)
        boundary = lambda E: np.dot(qclSolver.buildTM(E, self.potential, Ep, m, step)[:, :, -1], [1, -1]).sum()

        val = []
        eig = []
        psi = []

        old_settings = np.seterr(all='ignore')

        for E in Energy:
            val.append(boundary(E).real)

        for i in range(0, np.size(Energy) - 1):
            if (val[i] * val[i + 1] < 0):
                eig.append(brentq(lambda E: boundary(E).real, Energy[i], Energy[i + 1], xtol=1e-20))

        for E in eig:
            matArray = qclSolver.buildTM(E, self.potential, Ep, m, step)
            psi_tmp = np.sum(np.matmul(np.transpose(matArray, (2, 0, 1)), [1, -1]), axis=1).real
            nrm = ((psi_tmp ** 2).sum() * step)
            psi_tmp = psi_tmp / np.sqrt(nrm)

            psi.append(np.append(0., psi_tmp))

        np.seterr(**old_settings)

        self.eigs = np.array(eig)[::-1]
        self.psi = (np.array(psi)[::-1][:]).transpose()
        self.selectPsi()

    def buildTM(E, Ev, Ep, m, step):
        dE = (E - Ev) * 1.60218e-19

        planck = qclSolver.fundamentals["planck"]
        m0 = qclSolver.fundamentals["m0"]

        m_np = m * (1 + (E - Ev) / (m / m0 * Ep))
        k = np.sqrt(2 * m_np * dE + 0j) / planck
        kt = k / m_np
        kp = (k[:-1] + k[1:]) / 2
        km = (k[:-1] - k[1:]) / 2

        a = (kt[1:] + kt[:-1]) / 2 / kt[1:] * np.exp(1j * kp * step)
        b = (kt[1:] - kt[:-1]) / 2 / kt[1:] * np.exp(-1j * km * step)
        c = (kt[1:] - kt[:-1]) / 2 / kt[1:] * np.exp(1j * km * step)
        d = (kt[1:] + kt[:-1]) / 2 / kt[1:] * np.exp(-1j * kp * step)

        matArray = np.array([[a, b], [c, d]])

        for i in range(1, len(Ev) - 1):
            matArray[:, :, i] = np.matmul(matArray[:, :, i], matArray[:, :, i - 1])

        return matArray

    def eigDiag(self, ):  # not implemented

        self.eigs = 1
        self.psi = 1
        return 0

    # ============================================
    # ================ TECHNICAL =================
    # ============================================

    def layerExtrap(struct, side=5.):
        shift = side - struct.layers[0].width
        length = struct.length + shift
        z = np.array([struct.layerPos(i) + shift for i in range(0, struct.Nl)] + [length] + [length + side])
        n = np.arange(0, struct.Nl + 2, 1)
        z[0] = 0.
        n[-2:] = 0
        return interp1d(z, n, kind='previous'), z[-1]

    def setPotential(self, U, hart=0.):
        self.U = U
        self.potential = np.array(
            [(self.struct.layers[int(self.index(z))].material.params['Ec']) for z in
             self.grid]) - U * self.grid / 10 ** 7 + hart

    def setBGDoping(self, BGDoping):

        self.dop[self.dop > 0] += -BGDoping
        self.dop += BGDoping

        self.alpha_w *= self.step / 10 ** 9 * self.dop.sum() * (10 ** 6) / self.N_carr
        self.N_carr = self.step / 10 ** 9 * self.dop.sum() * (10 ** 6)

    def selectPsi(self, deloc_ext=1):
        step = self.step / 10 ** 9
        eigs = self.eigs
        psi = self.psi
        potential = self.potential
        U = self.U / 10 ** 7
        ind = np.zeros(0, dtype=int)
        Ep = self.Ep

        for i in range(0, len(eigs)):
            if eigs[i] > potential[-1]:
                left = (psi[np.nonzero(potential[0] - U * self.grid - eigs[i] > 0), i] ** 2).sum()
                right = (psi[np.nonzero(eigs[i] - potential[0] + U * self.grid > 0), i] ** 2).sum()
                if left < deloc_ext * right:
                    ind = np.append(ind, i)

        eigs = np.delete(eigs, ind)
        psi = np.delete(psi, ind, 1)

        mass_sub = np.zeros_like(eigs)

        for i in range(0, len(eigs)):
            mass_sub[i] = step * (
                        self.meff * (1 + (eigs[i] - self.potential) / self.meff / Ep) * (psi[:, i] ** 2)).sum()

        self.eigs, self.psi, self.mass_sub = eigs, psi, mass_sub * qclSolver.fundamentals["m0"]

    # ============================================

    def solvePoisson(self):
        h = self.step / 10 ** 9
        side = self.side / 10 ** 9
        z = self.grid / 10 ** 9
        el = qclSolver.fundamentals["e-charge"]
        N = len(z)
        mass_sub = self.mass_sub

        Perm = self.Perm

        N_car_tot = self.N_carr

        front = np.argwhere(z <= side)[-1][-1]
        back = np.argwhere(z <= side + self.struct.length / 10 ** 9)[-1][-1]

        if self.TPop:
            mu = brentq(lambda mu_t: (N_car_tot - qclSolver.TDistr(mass_sub, self.eigs, mu_t, self.TE, self.TL).sum()),
                        -1,
                        1,
                        xtol=1e-30)
            Population = qclSolver.TDistr(mass_sub, self.eigs, mu, self.TE, self.TL)
            self.Population = Population
        else:
            Population = self.Population

        f_carr = self.psi ** 2 * Population[np.newaxis, :]
        f_carr = np.sum(f_carr, axis=1)

        Ro_n = -el * (self.dop[front:back] * (10 ** 6) - f_carr[front:back])

        s_n = 1 / 2 / el / h / h * (Perm[front - 1:back - 1] + Perm[front:back])
        d_n = -1 / 2 / el / h / h * (Perm[front - 1:back - 1] + 2 * Perm[front:back] + Perm[front + 1:back + 1])

        M_ro = np.vstack((s_n, d_n, np.append(s_n[1:], 0)))
        V_ro = -solve_banded((1, 1), M_ro, Ro_n) / 1.60218e-19

        return np.append(np.append(np.zeros(front), V_ro), np.zeros(N - back))

    def TDistr(m, E, mu, TE, TL):
        k = qclSolver.fundamentals["k-bol"]
        planck = qclSolver.fundamentals["planck"]

        return m / np.pi / planck / planck * k * TL * np.logaddexp(0, (mu - E) * 1.60218e-19 / k / TE)

    def SPSolve(self, Resolution=10 ** (-3), iteration=3):
        self.eigTM(Resolution)
        for i in range(0, iteration):
            self.setPotential(self.U, hart=self.solvePoisson())
            self.eigTM(Resolution)

    def RESolve(self, r_iter=3, ncpu):

        el = qclSolver.fundamentals["e-charge"]
        if self.evaluate_W:
            self.Build_W(ncpu=ncpu)
            self.evaluate_W = False

        W = self.W

        for i in range(0, r_iter):
            R_1, R_2 = self.Build_R()

            population = np.linalg.eig((W + R_1 + R_2).transpose())

            ind = np.argmin(np.abs(population[0].real))
            population = population[1][:, ind]
            population = population.real / (population.real.sum()) * self.N_carr / 10 ** 4

        for i in range(0, len(self.eigs)):
            R_1[i, i] = 0
            R_2[i, i] = 0

        self.J_d = -el * (np.sum(R_1 - R_2, axis=1) * population).sum()
        self.Population = population
        self.TPop = False

    # ============================================
    # =============== SCATTERING =================
    # ============================================

    def w_lo_ph(self, k_i, init, fin, pi_pts=150, istep=0.2):

        if istep > self.step:
            skip = np.around(istep / self.step).astype(int)
        else:
            skip = 1

        theta = np.linspace(0., np.pi, pi_pts)
        th_step = (theta[1] - theta[0])
        h = skip * self.step / 10 ** 9
        el = qclSolver.fundamentals["e-charge"]
        planck = qclSolver.fundamentals["planck"]
        eps0 = qclSolver.fundamentals["eps0"]
        E_lo = self.struct.layers[0].material.params['ELO'] * 1.60218e-19
        perm0 = self.struct.layers[0].material.params['eps0']
        perm_inf = self.struct.layers[0].material.params['epsinf']
        k_bol = qclSolver.fundamentals["k-bol"]

        m_i = self.mass_sub[init]
        m_f = self.mass_sub[fin]
        E_f = self.eigs[fin] * 1.60218e-19
        E_i = self.eigs[init] * 1.60218e-19

        qs_s = el ** 2 * (h / skip * self.dop.sum() * 10 ** 6 / (
                    self.struct.length / 10 ** 9)) / eps0 / perm0 / k_bol / self.TE  # perm0 should be avareged

        k_f_a = m_f / m_i * k_i + 2 * m_f * (E_i - E_f + E_lo) / planck ** 2
        k_f_e = m_f / m_i * k_i + 2 * m_f * (E_i - E_f - E_lo) / planck ** 2

        Psi_i = self.psi[::skip, init]
        Psi_f = self.psi[::skip, fin]
        z = self.grid[::skip] / 10 ** 9

        # ------------------------
        I_e = 0
        if k_f_e > 0:
            q_e = np.sqrt(k_f_e + k_i - 2 * np.sqrt(k_f_e) * np.sqrt(k_i) * np.cos(theta) + qs_s)  # vector:q^2 < 0?
            for i in range(0, pi_pts):
                I_e += h ** 2 / q_e[i] * th_step * (Psi_i * Psi_f * (Psi_i * Psi_f)[:, np.newaxis] * np.exp(
                    -q_e[i] * np.abs(z - z[:, np.newaxis]))).sum()
        I_a = 0
        if k_f_a > 0:
            q_a = np.sqrt(k_f_a + k_i - 2 * np.sqrt(k_f_a) * np.sqrt(k_i) * np.cos(theta) + qs_s)
            for i in range(0, pi_pts):
                I_a += h ** 2 / q_a[i] * th_step * (Psi_i * Psi_f * (Psi_i * Psi_f)[:, np.newaxis] * np.exp(
                    -q_a[i] * np.abs(z - z[:, np.newaxis]))).sum()

        w_e = m_f * (el ** 2) * E_lo / 4 / np.pi / (planck ** 3) / eps0 * (1 / perm_inf - 1 / perm0) * (
                    1 / (np.exp(E_lo / k_bol / self.TL) - 1) + 1) * I_e
        w_a = m_f * (el ** 2) * E_lo / 4 / np.pi / (planck ** 3) / eps0 * (1 / perm_inf - 1 / perm0) * (
                    1 / (np.exp(E_lo / k_bol / self.TL) - 1)) * I_a

        return w_a + w_e

    def w_ad(self, k_i, init, fin, istep=0.2):

        if istep > self.step:
            skip = np.around(istep / self.step).astype(int)
        else:
            skip = 1

        planck = qclSolver.fundamentals["planck"]
        h = skip * self.step / 10 ** 9
        m_i = self.mass_sub[init]
        m_f = self.mass_sub[fin]
        E_f = self.eigs[fin] * 1.60218e-19
        E_i = self.eigs[init] * 1.60218e-19

        if E_i + k_i * planck ** 2 / 2 / m_i - E_f < 0:
            return 0

        Psi_i = self.psi[::skip, init]
        Psi_f = self.psi[::skip, fin]
        Valloy = self.Valloy[::skip] * 1.60218e-19
        a = self.lattconst[::skip] / 10 ** 10
        x = self.comp[::skip]

        w = h * m_f / 8 / planck ** 3 * (a ** 3 * Valloy ** 2 * x * (1 - x) * Psi_i ** 2 * Psi_f ** 2).sum()

        return w

    def w_dop(self, k_i, init, fin, pi_pts=150, istep=0.2):

        if istep > self.step:
            skip = np.around(istep / self.step).astype(int)
        else:
            skip = 1

        h = skip * self.step / 10 ** 9
        m_i = self.mass_sub[init]
        m_f = self.mass_sub[fin]
        E_f = self.eigs[fin] * 1.60218e-19
        E_i = self.eigs[init] * 1.60218e-19

        el = qclSolver.fundamentals["e-charge"]
        planck = qclSolver.fundamentals["planck"]
        eps0 = qclSolver.fundamentals["eps0"]

        k_f = m_f / m_i * k_i + 2 * m_f * (E_i - E_f) / planck ** 2

        if k_f < 0:
            return 0

        Psi_i = self.psi[::skip, init]
        Psi_f = self.psi[::skip, fin]
        z = self.grid[::skip] / 10 ** 9
        dop = self.dop * 10 ** 6  # skip?

        perm0 = self.struct.layers[1].material.params['eps0']
        Perm = self.Perm

        theta = np.linspace(0., np.pi, pi_pts)
        th_step = (theta[1] - theta[0])

        k_bol = qclSolver.fundamentals["k-bol"]

        qs_s = el ** 2 * (h / skip * dop.sum() / (
                    self.struct.length / 10 ** 9)) / eps0 / perm0 / k_bol / self.TE  # perm0 should be averaged

        q = np.sqrt(k_f + k_i - 2 * np.sqrt(k_f) * np.sqrt(k_i) * np.cos(theta) + qs_s)
        I = 0

        for i in range(0, pi_pts):
            I += h ** 2 / skip / q[i] ** 2 * th_step * (dop[dop > 0] / (Perm[dop > 0]) ** 2 * (
                np.sum((Psi_i * Psi_f) * np.exp(-q[i] * np.abs(z - self.grid[dop > 0, np.newaxis] / 10 ** 9)),
                       axis=1)) ** 2).sum()

        w = m_f * el ** 4 / 4 / np.pi / planck ** 3 * eps0 * I
        return w

    def w_ifr(self, k_i, init, fin, pi_pts=300, kappa=1.5):

        m_i = self.mass_sub[init]
        m_f = self.mass_sub[fin]
        E_f = self.eigs[fin] * 1.60218e-19
        E_i = self.eigs[init] * 1.60218e-19
        el = qclSolver.fundamentals["e-charge"]
        planck = qclSolver.fundamentals["planck"]

        k_f = m_f / m_i * k_i + 2 * m_f * (E_i - E_f) / planck / planck

        if k_f < 0:
            return 0

        h = self.step / 10 ** 9
        Delta = self.struct.eta / 10 ** 9  # нм -> м
        Lambda = self.struct.lam / 10 ** 9  # нм -> м
        potential = self.potential * 1.60218e-19
        kappa /= 10 ** 9

        theta = np.linspace(0., np.pi, pi_pts)
        th_step = (theta[1] - theta[0])

        q_sq = k_f + k_i - 2 * np.sqrt(k_f) * np.sqrt(k_i) * np.cos(theta)

        Psi_i = self.psi[:, init]
        Psi_f = self.psi[:, fin]

        z_int = []
        n_int = []
        for i in range(1, len(self.struct.layers) + 1):
            z_int = np.append(z_int, self.struct.layerPos(i))
        z_int = z_int - z_int[0] + self.side

        for i in range(0, len(z_int)):
            n_int = np.append(n_int, np.argwhere(self.grid < z_int[i])[-1])
        n_int = n_int.astype(int)

        c_mat = np.exp(-h * np.abs(n_int - n_int[:, np.newaxis]) / kappa)
        F_line = -(potential[n_int + 1] - potential[n_int]) * Psi_i[n_int] * Psi_f[n_int]
        H_step_f = np.piecewise(q_sq, [q_sq < 0, q_sq >= 0], [0, 1])

        I = (c_mat * (F_line * F_line[:, np.newaxis])).sum()
        I *= th_step * (np.exp(-Lambda ** 2 / 4 * q_sq) * H_step_f).sum()
        w = m_f / planck ** 3 * Delta ** 2 * Lambda ** 2 * I

        return w

    def w_m(self, i, f, E_pts=50):

        m = self.mass_sub[i]
        Emax = self.potential[0] * 1.60218e-19
        E_i = self.eigs[i] * 1.60218e-19
        planck = qclSolver.fundamentals["planck"]
        k_bol = qclSolver.fundamentals["k-bol"]
        TE = self.TE

        kGrid = np.linspace(0, np.sqrt(2 * m * (Emax - E_i) / planck ** 2), E_pts)
        step = kGrid[1] - kGrid[0]

        sigma = planck ** 2 / m / k_bol / TE * step

        w_1 = lambda ks: self.w_lo_ph(ks, i, f)
        w_2 = lambda ks: self.w_ad(ks, i, f)
        w_3 = lambda ks: self.w_dop(ks, i, f)
        w_4 = lambda ks: self.w_ifr(ks, i, f)

        I1 = (np.array([w_1(k ** 2) for k in kGrid]) * kGrid * np.exp(
            -planck ** 2 * kGrid ** 2 / 2 / m / k_bol / TE)).sum()
        I2 = (np.array([w_2(k ** 2) for k in kGrid]) * kGrid * np.exp(
            -planck ** 2 * kGrid ** 2 / 2 / m / k_bol / TE)).sum()
        I3 = (np.array([w_3(k ** 2) for k in kGrid]) * kGrid * np.exp(
            -planck ** 2 * kGrid ** 2 / 2 / m / k_bol / TE)).sum()
        I4 = (np.array([w_4(k ** 2) for k in kGrid]) * kGrid * np.exp(
            -planck ** 2 * kGrid ** 2 / 2 / m / k_bol / TE)).sum()

        return sigma * (I1 + I2 + I3 + I4)

    def Build_W(self, ncpu=4):

        if ncpu > 1:

            with Pool(processes=ncpu) as pool:
                W = pool.starmap(self.w_m, product(range(0, len(self.eigs)), repeat=2))
            W = np.array(W).reshape(len(self.eigs), len(self.eigs))

            for i in range(0, len(self.eigs)):
                W[i][i] = 0.
                W[i][i] = -W[i][:].sum()
            self.W = W  # ???

        else:

            W = np.zeros((len(self.eigs), len(self.eigs)))
            for i in range(0, len(self.eigs)):
                for f in range(0, len(self.eigs)):
                    if i != f:
                        W[i][f] = self.w_m(i, f)

            for i in range(0, len(self.eigs)):
                W[i][i] = -W[i][:].sum()
            self.W = W

    # ============================================
    # =============== TUNNELLING =================
    # ============================================

    def findOmega(self, i, f):

        planck = qclSolver.fundamentals["planck"]
        m0 = qclSolver.fundamentals["m0"]
        U = self.U / 10 ** 7
        side = self.side
        grid = self.grid
        length = self.struct.length
        E_i = self.eigs[i]
        E_f = self.eigs[f]
        Ep = self.Ep

        front = np.argwhere(grid <= side)[-1][-1]
        back = np.argwhere(grid <= side + length)[-1][-1]

        Ep_coupl = np.append(Ep[:back], Ep[front:])
        m_coupl = np.append(self.meff[:back], self.meff[front:]) * m0
        v_coupl = np.append(self.potential[:back], self.potential[front:] - U * length) * 1.60218e-19
        h = self.step / 10 ** 9
        n_ap = len(v_coupl) - len(self.psi[:, f])

        psi_l = np.append(self.psi[:, i], np.zeros(n_ap))
        psi_r = np.append(np.zeros(n_ap), self.psi[:, f])

        der_l = (np.append(psi_l, 0.) - np.append(0., psi_l))[1:] / 2 / h
        der_r = (np.append(psi_r, 0.) - np.append(0., psi_r))[1:] / 2 / h

        m_l = m_coupl * (1 + (E_i - v_coupl / 1.60218e-19) / (m_coupl / m0 * Ep_coupl))
        m_r = m_coupl * (1 + (E_f - U * grid[back] - v_coupl / 1.60218e-19) / (m_coupl / m0 * Ep_coupl))

        K_lr = planck ** 2 * h / 2 * (m_l * der_l / m_l * der_r / m_r).sum()
        K_rl = planck ** 2 * h / 2 * (m_r * der_l / m_l * der_r / m_r).sum()
        K_l = planck ** 2 * h / 2 * (m_l * der_l / m_l * der_l / m_l).sum()
        K_r = planck ** 2 * h / 2 * (m_r * der_r / m_r * der_r / m_r).sum()

        T_lr = h * (v_coupl * psi_r * psi_l).sum()
        T_rl = h * (v_coupl * psi_r * psi_l).sum()
        T_l = h * (v_coupl * psi_l * psi_l).sum()
        T_r = h * (v_coupl * psi_r * psi_r).sum()

        r = h * (psi_r * psi_l).sum()
        delta = 1 - r ** 2

        h_lr = ((K_lr + T_lr) - r * (K_r + T_r)) / delta
        h_rl = ((K_rl + T_rl) - r * (K_l + T_l)) / delta

        return np.sqrt(np.abs(h_lr * h_rl)) / planck

    def Build_R(self):
        eigs = self.eigs
        N = len(eigs)
        planck = qclSolver.fundamentals["planck"]
        length = self.struct.length
        mass_sub = self.mass_sub
        Population = self.Population
        TE = self.TE

        R_forw = np.zeros((N, N))
        R_back = np.zeros((N, N))
        for i in range(0, N):
            for f in range(0, N):
                if i != f:
                    tau_ort = 1 / (1 / self.tau_pure - (self.W[i][i] + self.W[f][f]) / 2)

                    delta_if = -(eigs[i] - eigs[f] + self.U * length / 10 ** 7) / planck * 1.60218e-19
                    R_forw[i][f] = 2 * self.findOmega(i, f) ** 2 * tau_ort / (
                                1 + delta_if ** 2 * tau_ort ** 2) * qclSolver.sigma_b(delta_if, Population[i],
                                                                            mass_sub[i], TE)
                    R_back[f][i] = 2 * self.findOmega(i, f) ** 2 * tau_ort / (
                                1 + delta_if ** 2 * tau_ort ** 2) * qclSolver.sigma_b(-delta_if, Population[f],mass_sub[f], TE)

        for i in range(0, N):
            R_forw[i][i] = -R_forw[i][:].sum()
            R_back[i][i] = -R_back[i][:].sum()

        return R_forw, R_back

    Chi = lambda x, TE: np.heaviside(-x, 0) + np.heaviside(x, 0) * np.exp(-np.abs(x) / qclSolver.k_bol / TE)

    def sigma_b(delta_ab, N_b, m_b, TE):
        planck = qclSolver.fundamentals["planck"]
        Beta = 1 / qclSolver.k_bol / TE
        D_e = m_b / np.pi / planck ** 2
        mu_b = brentq(lambda mu_t: N_b - D_e / Beta * np.logaddexp(0, Beta * mu_t),
                      -1 / 6.24150965 / (10 ** 13),
                      1 / 6.24150965 / (10 ** 13),
                      xtol=1e-30)

        return np.log(1 + np.exp(Beta * mu_b) * qclSolver.Chi(planck * delta_ab, TE)) / np.logaddexp(0, Beta * mu_b)

    # ============================================
    # ============ LIGHT INTERACTION =============
    # ============================================

    def gain_if(self, i, f, omega):

        eps_0 = qclSolver.fundamentals["eps0"]
        c = qclSolver.fundamentals["c"]
        z = self.grid / 10 ** 9
        el = qclSolver.fundamentals["e-charge"]
        planck = qclSolver.fundamentals["planck"]
        k_bol = qclSolver.fundamentals["k-bol"]
        TE = self.TE

        m_i = self.mass_sub[i]
        m_f = self.mass_sub[f]
        n_i = self.Population[i]
        n_f = self.Population[f]

        delta_E = (self.eigs[i] - self.eigs[f]) * 1.60218e-19
        delta = (self.eigs[i] - self.eigs[f]) * 1.60218e-19 - planck * omega
        gamma = -planck * (self.W[f, f]) / 2

        Beta = 1 / k_bol / self.TE
        D_ei = m_i / np.pi / planck ** 2
        D_ef = m_f / np.pi / planck ** 2
        mu_i = brentq(lambda mu_t: n_i - D_ei / Beta * np.logaddexp(0, Beta * mu_t),
                      -1 / 6.24150965 / (10 ** 9),
                      1 / 6.24150965 / (10 ** 9),
                      xtol=1e-30)
        mu_f = brentq(lambda mu_t: n_f - D_ef / Beta * np.logaddexp(0, Beta * mu_t),
                      -1 / 6.24150965 / (10 ** 9),
                      1 / 6.24150965 / (10 ** 9),
                      xtol=1e-30)

        psi_i = self.psi[:, i]
        psi_f = self.psi[:, f]
        z_aw = (z[1] - z[0]) * (psi_i * z * psi_f).sum() * 100

        koeff = el ** 2 * z_aw ** 2 * delta_E ** 2 / (
                    eps_0 * c * planck ** 2 * omega * self.refr * self.struct.length / 10 ** 7)

        deltaN = D_ei / Beta * np.log(
            (1 + np.exp(Beta * mu_i)) / (1 + qclSolver.Chi(delta, TE) * np.exp(Beta * mu_f))) + D_ef / Beta * np.log(
            (1 + qclSolver.Chi(-delta, TE) * np.exp(Beta * mu_i)) / (1 + np.exp(Beta * mu_f)))
        deltaN_ci = D_ei / Beta * np.log(1 + np.exp(Beta * mu_i)) + D_ef / Beta * np.log(
            (1 + qclSolver.Chi(-delta, TE) * np.exp(Beta * mu_i)))
        deltaN_cf = -D_ei / Beta * np.log(1 + qclSolver.Chi(delta, TE) * np.exp(Beta * mu_f)) - D_ef / Beta * np.log(
            1 + np.exp(Beta * mu_f))

        g = koeff * gamma / (delta ** 2 + 4 * gamma ** 2) * deltaN
        g_ci = koeff * gamma / (delta ** 2 + 4 * gamma ** 2) * deltaN_ci / n_i
        g_cf = koeff * gamma / (delta ** 2 + 4 * gamma ** 2) * deltaN_cf / n_f

        return g, g_ci, g_cf

    def tot_gain(self, omega):
        N = len(self.eigs)
        gain = 0.
        for i in range(0, N):
            for f in range(0, N):
                if i != f:
                    gain += self.gain_if(i, f, omega)[0]

        return gain

    def Build_G(self, omega):
        N = len(self.eigs)
        G = np.zeros((N, N), dtype=float)
        for i in range(0, N):
            for f in range(0, N):
                if i > f:
                    g_if = self.gain_if(i, f, omega)
                    g_fi = self.gain_if(f, i, omega)
                    G[f][i] += g_if[1] - g_fi[2]
                    G[i][f] += -g_if[2] + g_fi[1]
                    G[i][i] += -g_if[1] + g_fi[2]
                    G[f][f] += g_if[2] - g_fi[1]

        return G

    def tot_gain_optical(self, omega, S, iterations=3):

        N_carr_tot = self.N_carr / 10 ** 4
        W = self.W

        for i in range(0, iterations):
            R_1, R_2 = self.Build_R()
            G = self.Build_G(omega)
            G = np.array(G * S, dtype=float)

            Population = np.linalg.eig((W + R_1 + R_2 + G).transpose())

            ind = np.argmin(np.abs(Population[0].real))
            Population = Population[1][:, ind]
            Population = Population.real / (Population.real.sum()) * N_carr_tot
            self.Population = Population

        return self.tot_gain(omega)

    def optPower(self, lam):
        c = qclSolver.fundamentals["c"]
        planck = qclSolver.fundamentals["planck"]
        Omega = 2 * np.pi * c / lam
        S = brentq(lambda s: self.tot_gain_optical(Omega, s) - self.alpha_m - self.alpha_w, 0, 10 ** 34,xtol=0.01)

        self.P = Omega * planck * S * self.periods * self.dim_w * self.alpha_m

    # ============================================
    # ================= OUTPUT ===================
    # ============================================

    def plotWF(self, saveFile = True):

        plt.plot(self.grid, self.potential)
        plt.xlabel('z, nm')
        plt.ylabel('E, eV')
        for i in range(0, len(self.eigs)):
            plt.plot(self.grid[:], (self.psi[:, i] / 10 ** 4) ** 2 / 20 + self.eigs[i])
        if saveFile:
            plt.savefig('./WaveFunc.png', format='png', dpi=1000)
        plt.show()

    def plotGainSP(self, lam_min, lam_max, saveFile = True):

        c = qclSolver.fundamentals["c"]
        lam = np.linspace(lam_min, lam_max, 1000) / 10 ** (6)

        omega_r = 2 * np.pi * c / lam
        gain_r = self.tot_gain(omega_r)

        plt.figure(figsize=(10, 5))
        plt.plot(lam * 10 ** 6, gain_r)
        plt.plot(lam * 10 ** 6, np.zeros_like(gain_r) + self.alpha_m + self.alpha_w, 'r--')
        plt.grid(True)
        if saveFile:
            plt.savefig('./Gain.png', format='png', dpi=1000)
        plt.show()

    def plotPopulation(self, saveFile = True):

        print(self.Population)
        plt.plot(self.Population)
        if saveFile:
            plt.savefig('./Population.png', format='png', dpi=1000)
        plt.show()

    def generateParOutput(self):

        print(self.eigs)
        if not self.evaluate_W :
            print('States Populations:',self.Population)
            print('Current density:', round(self.J_d / 1000, 3), 'kA/cm^2')
            if self.P > 0 :
                print('Optical Power:', self.P / 1000, 'mWt')

    def generatePlotOutput(self, lam_min = None, lam_max = None, saveFile = True):
        self.plotWF(saveFile=saveFile)
        if not self.evaluate_W:
            self.plotPopulation(saveFile=saveFile)
            if lam_max != None :
                self.plotGainSP(lam_min=lam_min, lam_max=lam_max, saveFile=saveFile)


