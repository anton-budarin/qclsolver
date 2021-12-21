import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.linalg import solve_banded, block_diag
from multiprocessing import Pool
from itertools import product
from functools import partial


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

    def __init__(self, struct, interval=2, step=0.05, istep=0.2, side=5., TE=400., TL=293.):

        if not (isinstance(struct, list)):
            struct = [struct]

        for i in range(0, len(struct)):
            struct[i].length = np.array([struct[i].layers[ind].width for ind in range(0, len(struct[i].layers))]).sum()

        self.istep = istep
        self.step = step
        self.struct = struct
        self.side = side

        self.TL = TL
        self.TE = TE

        self.index = []
        self.grid = []
        self.meff = []
        self.Valloy = []
        self.Perm = []
        self.lattconst = []
        self.comp = []
        self.Ep = []
        self.Population = []

        self.U = 0
        self.potential = []

        self.shifts = side * np.ones(len(struct))
        self.ends = np.zeros_like(self.shifts)
        self.N_carr = 0
        self.struct_end = []

        self.eigs = []
        self.psi = []
        self.dop = []

        # chunk division

        for i in range(0, len(struct)):

            # index interpolation

            self.shifts[i] = self.shifts[i] - struct[i].layers[0].width

            if i == 0:
                index, last, end = qclSolver.layerExtrap(struct[i], side)
                self.grid.append(np.arange(0, last, step))
            else:
                index, last, end = qclSolver.layerExtrap(struct[i], side, z_start=end)
                self.grid.append(np.arange(self.grid[i - 1][-1] - 2 * side // step * step, last, step))
            self.struct_end.append(end)
            self.index.append(index)

            # parameter grid filling

            self.meff.append(np.array([(struct[i].layers[int(self.index[i](z))].material.params['meff'])
                                       for z in self.grid[i]]))

            self.Valloy.append(np.array([(struct[i].layers[int(self.index[i](z))].material.params['meff'])
                                         for z in self.grid[i]]))

            self.Perm.append(np.array([(self.struct[i].layers[int(self.index[i](z))].material.params['eps0'])
                                       for z in self.grid[i]]) * qclSolver.fundamentals["eps0"])

            self.lattconst.append(np.array([(self.struct[i].layers[int(self.index[i](z))].material.params['lattconst'])
                                            for z in self.grid[i]]))

            self.comp.append(np.array([self.struct[i].layers[int(self.index[i](z))].material.x for z in self.grid[i]]))

            self.comp[i][self.comp is None] = 0

            self.Ep.append(np.array([(self.struct[i].layers[int(self.index[i](z))].material.params['Ep'])
                                     for z in self.grid[i]]))

            # doping processing

            self.N_carr += struct[i].getSheetDop() * (10 ** 4)

            dop = 0.
            if i == 0:
                shift = self.shifts[0]
                for j in range(0, len(struct[i].dopings)):
                    dop += np.piecewise(
                        self.grid[i], [self.grid[i]-shift < struct[i].dopings[j][0],
                                    self.grid[i]-shift >= struct[i].dopings[j][0],
                                    self.grid[i]-shift >= struct[i].dopings[j][1]],
                        [0, struct[i].dopings[j][2], 0]
                    )
            else:
                shift = self.struct_end[i-1]
                for j in range(0, len(struct[i].dopings)):
                    dop += np.piecewise(
                        self.grid[i], [self.grid[i]-shift < struct[i].dopings[j][0],
                                    self.grid[i]-shift >= struct[i].dopings[j][0],
                                    self.grid[i]-shift >= struct[i].dopings[j][1]],
                        [0, struct[i].dopings[j][2], 0]
                    )

            self.dop.append(dop)

        # various parameters

        self.tau_pure = 0.5 * 10 ** (-13)  # pure dephasing time

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

        step = self.step / 10 ** 9

        for i in range(0, len(self.struct)):

            m = self.meff[i] * qclSolver.fundamentals["m0"]
            Ep = self.Ep[i]

            Energy = np.arange(np.amin(self.potential[i]), np.amax(self.potential[i]), resolution)
            boundary = lambda E: np.dot(qclSolver.buildTM(E, self.potential[i], Ep, m, step)[:, :, -1], [1, -1]).sum()

            val = []
            eig = []
            psi = []

            old_settings = np.seterr(all='ignore')

            for E in Energy:
                val.append(boundary(E).real)

            for j in range(0, np.size(Energy) - 1):
                if val[j] * val[j + 1] < 0:
                    eig.append(brentq(lambda E: boundary(E).real, Energy[j], Energy[j + 1], xtol=1e-20))

            for E in eig:
                matArray = qclSolver.buildTM(E, self.potential[i], Ep, m, step)
                psi_tmp = np.sum(np.matmul(np.transpose(matArray, (2, 0, 1)), [1, -1]), axis=1).real
                nrm = ((psi_tmp ** 2).sum() * step)
                psi_tmp = psi_tmp / np.sqrt(nrm)

                psi.append(np.append(0., psi_tmp))

            np.seterr(**old_settings)

            self.eigs.append(np.array(eig)[::-1])
            self.psi.append((np.array(psi)[::-1][:]).transpose())

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
    # =================== MISC ===================
    # ============================================

    def layerExtrap(struct, side=5., z_start=0.):
        shift = side - struct.layers[0].width
        length = struct.length + shift
        z = np.array([struct.layerPos(i) + shift for i in range(0, struct.Nl)] + [length] + [length + side])
        n = np.arange(0, struct.Nl + 2, 1)
        z[0] = -0.01-struct.layers[0].width
        n[-2:] = 0
        if z_start != 0.:
            z = z + z_start - shift
        return interp1d(z, n, kind='previous'), z[-1], z[-2]

    def setPotential(self, U, hart=0.):
        self.U = U
        self.potential = []
        for i in range(0, len(self.struct)):
            self.potential.append(np.array(
                [(self.struct[i].layers[int(self.index[i](z))].material.params['Ec']) for z in self.grid[i]])
                                  - U * self.grid[i] / 10 ** 7)
            if not np.all(hart == 0.):
                front = np.argwhere(self.unified_grid <= self.grid[i][0])[-1][-1]
                back = np.argwhere(self.unified_grid <= self.grid[i][-1])[-1][-1]
                self.potential[-1] += hart[front:back+1]

    def setBGDoping(self, BGDoping):
        for chunk in range(0, len(self.struct)):
            self.dop[chunk][self.dop == 0] += BGDoping

            self.alpha_w *= self.step / 10 ** 9 * self.dop[chunk].sum() * (10 ** 6) / self.N_carr
            self.N_carr = self.step / 10 ** 9 * self.dop[chunk].sum() * (10 ** 6)

    def selectPsi(self, deloc_ext=1):
        self.mass_sub = []
        step = self.step / 10 ** 9
        U = self.U / 10 ** 7
        eigs_list = self.eigs
        psi_list = self.psi
        psi_out = []
        eigs_out = []

        for j in range(0, len(self.struct)):

            eigs = eigs_list[j]
            psi = psi_list[j]
            potential = self.potential[j]
            ind = np.zeros(0, dtype=int)
            Ep = self.Ep[j]

            for i in range(0, len(eigs)):
                if eigs[i] > potential[-1]:
                    left = (psi[np.nonzero(potential[0] - U * self.grid[j] - eigs[i] > 0), i] ** 2).sum()
                    right = (psi[np.nonzero(eigs[i] - potential[0] + U * self.grid[j] > 0), i] ** 2).sum()
                    if left < deloc_ext * right:
                        ind = np.append(ind, i)

            eigs = np.delete(eigs, ind)
            psi = np.delete(psi, ind, 1)

            mass_sub = np.zeros_like(eigs)

            for i in range(0, len(eigs)):
                mass_sub[i] = step * (
                        self.meff[j] * (1 + (eigs[i] - potential) / self.meff[j] / Ep) * (psi[:, i] ** 2)).sum()

            eigs_out.append(eigs)
            psi_out.append(psi)
            self.mass_sub.append(mass_sub * qclSolver.fundamentals["m0"])
        self.eigs = eigs_out
        self.psi = psi_out

    def unite_chunks(self):

        if len(self.struct) == 1:
            self.unified_grid = self.grid[0]
            self.unified_potential = self.potential[0]
            self.unified_dop = self.dop[0]
            self.unified_perm = self.Perm[0]
            self.unified_psi = self.psi[0]
            self.unified_eigs = self.eigs[0]
            self.unified_mass_sub = self.mass_sub[0]

            return None

        front, back = [], []

        front.append(0)
        back.append(np.argwhere(self.grid[0] <= self.struct_end[0])[-1][-1])

        for i in range(1, len(self.struct)-1):
            back.append(np.argwhere(self.grid[i] <= self.struct_end[i])[-1][-1])
            front.append(np.argwhere(self.grid[i] <= self.struct_end[i - 1])[-1][-1])

        back.append(len(self.grid[-1]))
        front.append(np.argwhere(self.grid[-1] <= self.struct_end[-2])[-1][-1])

        unified_potential = np.zeros(0)
        unified_dop = np.zeros(0)
        unified_grid = np.zeros(0)
        unified_perm = np.zeros(0)

        unified_eigs = np.zeros(0)
        unified_mass_sub = np.zeros(0)

        unified_psi = np.array(self.psi)

        for i in range(0, len(self.struct)):
            for k in range(0, len(self.struct)):
                if k < i:
                    unified_psi[i] = np.append(
                        np.zeros([len(self.grid[k][front[k]:back[k]]), len(self.eigs[i])]), unified_psi[i], axis=0)
                elif k > i:
                    unified_psi[i] = np.append(
                        unified_psi[i], np.zeros([len(self.grid[k][front[k]:back[k]]), len(self.eigs[i])]), axis=0)

        for i in range(0, len(self.struct)):

            unified_potential = np.append(unified_potential, self.potential[i][front[i]:back[i]])
            unified_dop = np.append(unified_dop, self.dop[i][front[i]:back[i]])
            unified_grid = np.append(unified_grid, self.grid[i][front[i]:back[i]])
            unified_perm = np.append(unified_perm, self.Perm[i][front[i]:back[i]])

            unified_eigs = np.append(unified_eigs, self.eigs[i])
            unified_mass_sub = np.append(unified_mass_sub, self.mass_sub[i])

            unified_psi[i] = unified_psi[i][front[i]:back[i]-len(self.grid[i])-1, :]

        self.unified_grid = unified_grid[:-1]
        self.unified_potential = unified_potential[:-1]
        self.unified_dop = unified_dop[:-1]
        self.unified_perm = unified_perm[:-1]
        self.unified_psi = np.hstack(unified_psi)
        self. unified_eigs = unified_eigs
        self.unified_mass_sub = unified_mass_sub

    # ============================================
    # ============ GENERAL SOLVERS ===============
    # ============================================

    def solvePoisson(self):
        h = self.step / 10 ** 9
        side = self.side / 10 ** 9
        z = self.unified_grid / 10 ** 9
        el = qclSolver.fundamentals["e-charge"]
        N = len(z)
        mass_sub = self.unified_mass_sub
        Perm = self.unified_perm
        eigs = self.unified_eigs
        psi = self.unified_psi

        N_car_tot = self.N_carr

        front = np.argwhere(z <= side)[-1][-1] - 1
        back = np.argwhere(z <= (self.struct_end[-1] + self.struct[0].layers[0].width) / 10 ** 9)[-1][-1]

        if self.TPop:
            mu = brentq(lambda mu_t: (N_car_tot - qclSolver.TDistr(mass_sub, eigs, mu_t, self.TE, self.TL).sum()),
                        -1,
                        1,
                        xtol=1e-30)

            Population = qclSolver.TDistr(mass_sub, eigs, mu, self.TE, self.TL)

            self.Population = []
            end = 0
            for i in range(0, len(self.struct)):
                self.Population.append(Population[end:end+len(self.eigs[i])])
                end += len(self.eigs[i])
        else:
            Population = np.concatenate(self.Population)

        f_carr = psi ** 2 * Population[np.newaxis, :]
        f_carr = np.sum(f_carr, axis=1)

        Ro_n = -el * (self.unified_dop[front:back] * (10 ** 6) - f_carr[front:back])

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
        self.unite_chunks()
        if self.N_carr != 0:
            if iteration > 0:
                for i in range(0, iteration):
                    self.setPotential(self.U, hart=self.solvePoisson())
                    self.eigTM(Resolution)

    def RESolve(self, r_iter=9, ncpu=4):
        # add integration precision
        el = qclSolver.fundamentals["e-charge"]
        if self.evaluate_W:
            self.Build_W(ncpu=ncpu)
            self.evaluate_W = False

        for i in range(0, len(self.struct)):
            if i == 0:
                W = self.W[i]
            else:
                W = block_diag(W, self.W[i])

        for i in range(0, r_iter):
            R_1, R_2 = self.Build_R()

            matrix = (W + R_1 + R_2).T
            population = np.linalg.eig(matrix)

            ind = np.argmin(np.abs(population[0]))

            population = population[1][:, ind]
            population = population.real / (population.real.sum()) * self.N_carr / 10 ** 4

            self.Population = []
            end = 0
            for j in range(0, len(self.struct)):
                self.Population.append(population[end:end + len(self.eigs[j])])
                end += len(self.eigs[j])


        self.TPop = False

        R_1, R_2 = self.Build_R_local(len(self.struct) - 1, 0)
        self.J_d = -el * ((R_1 * self.Population[len(self.struct) - 1][:, np.newaxis]).sum()
                          - (R_2 * self.Population[0][:, np.newaxis]).sum())

    # ============================================
    # =============== SCATTERING =================
    # ============================================

    def w_lo_ph(self, k_i, init, fin, chunk, pi_pts=150):
        istep = self.istep
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
        E_lo = self.struct[chunk].layers[0].material.params['ELO'] * 1.60218e-19
        perm0 = self.struct[chunk].layers[0].material.params['eps0']
        perm_inf = self.struct[chunk].layers[0].material.params['epsinf']
        k_bol = qclSolver.fundamentals["k-bol"]

        m_i = self.mass_sub[chunk][init]
        m_f = self.mass_sub[chunk][fin]
        E_f = self.eigs[chunk][fin] * 1.60218e-19
        E_i = self.eigs[chunk][init] * 1.60218e-19

        qs_s = el ** 2 * (h / skip * self.dop[chunk].sum() * 10 ** 6 / (
                self.struct[chunk].length / 10 ** 9)) / eps0 / perm0 / k_bol / self.TE  # perm0 should be avareged

        k_f_a = m_f / m_i * k_i + 2 * m_f * (E_i - E_f + E_lo) / planck ** 2
        k_f_e = m_f / m_i * k_i + 2 * m_f * (E_i - E_f - E_lo) / planck ** 2

        Psi_i = self.psi[chunk][::skip, init]
        Psi_f = self.psi[chunk][::skip, fin]
        z = self.grid[chunk][::skip] / 10 ** 9

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

    def w_ad(self, k_i, init, fin, chunk):

        istep = self.istep
        if istep > self.step:
            skip = np.around(istep / self.step).astype(int)
        else:
            skip = 1

        planck = qclSolver.fundamentals["planck"]
        h = skip * self.step / 10 ** 9
        m_i = self.mass_sub[chunk][init]
        m_f = self.mass_sub[chunk][fin]
        E_f = self.eigs[chunk][fin] * 1.60218e-19
        E_i = self.eigs[chunk][init] * 1.60218e-19

        if E_i + k_i * planck ** 2 / 2 / m_i - E_f < 0:
            return 0

        Psi_i = self.psi[chunk][::skip, init]
        Psi_f = self.psi[chunk][::skip, fin]
        Valloy = self.Valloy[chunk][::skip] * 1.60218e-19
        a = self.lattconst[chunk][::skip] / 10 ** 10
        x = self.comp[chunk][::skip]

        w = h * m_f / 8 / planck ** 3 * (a ** 3 * Valloy ** 2 * x * (1 - x) * Psi_i ** 2 * Psi_f ** 2).sum()

        return w

    def w_dop(self, k_i, init, fin, chunk, pi_pts=150):
        istep = self.istep
        if istep > self.step:
            skip = np.around(istep / self.step).astype(int)
        else:
            skip = 1

        h = skip * self.step / 10 ** 9
        m_i = self.mass_sub[chunk][init]
        m_f = self.mass_sub[chunk][fin]
        E_f = self.eigs[chunk][fin] * 1.60218e-19
        E_i = self.eigs[chunk][init] * 1.60218e-19

        el = qclSolver.fundamentals["e-charge"]
        planck = qclSolver.fundamentals["planck"]
        eps0 = qclSolver.fundamentals["eps0"]

        k_f = m_f / m_i * k_i + 2 * m_f * (E_i - E_f) / planck ** 2

        if k_f < 0:
            return 0

        Psi_i = self.psi[chunk][::skip, init]
        Psi_f = self.psi[chunk][::skip, fin]
        z = self.grid[chunk][::skip] / 10 ** 9
        dop = self.dop[chunk] * 10 ** 6  # skip?

        perm0 = self.struct[chunk].layers[1].material.params['eps0']
        Perm = self.Perm[chunk]

        theta = np.linspace(0., np.pi, pi_pts)
        th_step = (theta[1] - theta[0])

        k_bol = qclSolver.fundamentals["k-bol"]

        qs_s = el ** 2 * (h / skip * dop.sum() / (
                self.struct[chunk].length / 10 ** 9)) / eps0 / perm0 / k_bol / self.TE  # perm0 should be averaged

        q = np.sqrt(k_f + k_i - 2 * np.sqrt(k_f) * np.sqrt(k_i) * np.cos(theta) + qs_s)
        I = 0

        for i in range(0, pi_pts):
            I += h ** 2 / skip / q[i] ** 2 * th_step * (dop[dop > 0] / (Perm[dop > 0]) ** 2 * (
                np.sum((Psi_i * Psi_f) * np.exp(-q[i] * np.abs(z - self.grid[chunk][dop > 0, np.newaxis] / 10 ** 9)),
                       axis=1)) ** 2).sum()

        w = m_f * el ** 4 / 4 / np.pi / planck ** 3 * eps0 * I
        return w

    def w_ifr(self, k_i, init, fin, chunk, pi_pts=300, kappa=1.5):

        m_i = self.mass_sub[chunk][init]
        m_f = self.mass_sub[chunk][fin]
        E_f = self.eigs[chunk][fin] * 1.60218e-19
        E_i = self.eigs[chunk][init] * 1.60218e-19
        planck = qclSolver.fundamentals["planck"]

        k_f = m_f / m_i * k_i + 2 * m_f * (E_i - E_f) / planck / planck

        if k_f < 0:
            return 0

        h = self.step / 10 ** 9
        Delta = self.struct[chunk].eta / 10 ** 9  # нм -> м
        Lambda = self.struct[chunk].lam / 10 ** 9  # нм -> м
        potential = self.potential[chunk] * 1.60218e-19
        kappa /= 10 ** 9

        theta = np.linspace(0., np.pi, pi_pts)
        th_step = (theta[1] - theta[0])

        q_sq = k_f + k_i - 2 * np.sqrt(k_f) * np.sqrt(k_i) * np.cos(theta)

        Psi_i = self.psi[chunk][:, init]
        Psi_f = self.psi[chunk][:, fin]

        z_int = []
        n_int = []
        for i in range(1, len(self.struct[chunk].layers) + 1):
            z_int = np.append(z_int, self.struct[chunk].layerPos(i))
        z_int = z_int - z_int[0] + self.side
        if chunk != 0:
            z_int = z_int - self.side + self.struct_end[chunk-1]

        for i in range(0, len(z_int)):
            n_int = np.append(n_int, np.argwhere(self.grid[chunk] < z_int[i])[-1])
        n_int = n_int.astype(int)

        c_mat = np.exp(-h * np.abs(n_int - n_int[:, np.newaxis]) / kappa)
        F_line = -(potential[n_int + 1] - potential[n_int]) * Psi_i[n_int] * Psi_f[n_int]
        H_step_f = np.piecewise(q_sq, [q_sq < 0, q_sq >= 0], [0, 1])

        I = (c_mat * (F_line * F_line[:, np.newaxis])).sum()
        I *= th_step * (np.exp(-Lambda ** 2 / 4 * q_sq) * H_step_f).sum()
        w = m_f / planck ** 3 * Delta ** 2 * Lambda ** 2 * I

        return w

    def w_m(self, i, f, chunk, E_pts=50):

        # add integration precision

        m = self.mass_sub[chunk][i]
        Emax = self.potential[chunk][0] * 1.60218e-19
        E_i = self.eigs[chunk][i] * 1.60218e-19
        planck = qclSolver.fundamentals["planck"]
        k_bol = qclSolver.fundamentals["k-bol"]
        TE = self.TE

        kGrid = np.linspace(0, np.sqrt(2 * m * (Emax - E_i) / planck ** 2), E_pts)
        step = kGrid[1] - kGrid[0]

        sigma = planck ** 2 / m / k_bol / TE * step

        w_1 = lambda ks: self.w_lo_ph(ks, i, f, chunk)
        w_2 = lambda ks: self.w_ad(ks, i, f, chunk)
        w_3 = lambda ks: self.w_dop(ks, i, f, chunk)
        w_4 = lambda ks: self.w_ifr(ks, i, f, chunk)

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
        # add integration precision
        W_list = []
        old_settings = np.seterr(all='ignore')

        if ncpu > 1:
            for chunk in range(0, len(self.struct)):

                with Pool(processes=ncpu) as pool:
                    W = pool.starmap(partial(self.w_m,chunk=chunk), product(range(0, len(self.eigs[chunk])), repeat=2))
                W = np.array(W).reshape(len(self.eigs[chunk]), len(self.eigs[chunk]))

                for i in range(0, len(self.eigs[chunk])):
                    W[i][i] = 0.
                    W[i][i] = -W[i][:].sum()

                W_list.append(W)

        else:
            for chunk in range(0, len(self.struct)):
                W = np.zeros((len(self.eigs[chunk]), len(self.eigs[chunk])))
                for i in range(0, len(self.eigs[chunk])):
                    for f in range(0, len(self.eigs[chunk])):
                        if i != f:
                            W[i][f] = self.w_m(i, f, chunk)

                for i in range(0, len(self.eigs[chunk])):
                    W[i][i] = -W[i][:].sum()
                W_list.append(W)

        np.seterr(**old_settings)

        self.W = W_list


    # ============================================
    # =============== TUNNELLING =================
    # ============================================

    def findOmega(self, i, f, i_chunk, f_chunk):

        if f_chunk - i_chunk != 1:
            if not (f_chunk == 0 and i_chunk == len(self.struct)-1):
                return 0, 0

        planck = qclSolver.fundamentals["planck"]
        m0 = qclSolver.fundamentals["m0"]

        E_i = self.eigs[i_chunk][i]
        E_f = self.eigs[f_chunk][f]
        Ep_i = self.Ep[i_chunk]
        Ep_f = self.Ep[f_chunk]
        meff_i = self.meff[i_chunk]
        meff_f = self.meff[f_chunk]
        v_i = self.potential[i_chunk]
        v_f = self.potential[f_chunk]

        ends = np.append(self.shifts[0], self.struct_end)

        back = np.argwhere(self.grid[i_chunk] <= ends[i_chunk+1])[-1][-1] + 1
        front = np.argwhere(self.grid[f_chunk] <= ends[f_chunk])[-1][-1] + 2
        e_shift = - v_f[front-1] + v_i[back]

        v_f = v_f + e_shift

        Ep_coupl = np.append(Ep_i[:back], Ep_f[front:])
        m_coupl = np.append(meff_i[:back], meff_f[front:]) * m0
        v_coupl = np.append(v_i[:back], v_f[front:]) * 1.60218e-19

        h = self.step / 10 ** 9
        n_ap_i = len(v_coupl) - len(self.psi[i_chunk][:, i])
        n_ap_f = len(v_coupl) - len(self.psi[f_chunk][:, f])

        psi_l = np.append(self.psi[i_chunk][:, i], np.zeros(n_ap_i))
        psi_r = np.append(np.zeros(n_ap_f), self.psi[f_chunk][:, f])

        der_l = (np.append(psi_l, 0.) - np.append(0., psi_l))[1:] / 2 / h
        der_r = (np.append(psi_r, 0.) - np.append(0., psi_r))[1:] / 2 / h

        m_l = m_coupl * (1 + (E_i - v_coupl / 1.60218e-19) / (m_coupl / m0 * Ep_coupl))
        m_r = m_coupl * (1 + (E_f + e_shift - v_coupl / 1.60218e-19) / (m_coupl / m0 * Ep_coupl))

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

        return np.sqrt(np.abs(h_lr * h_rl)) / planck, e_shift

    def Build_R_local(self, i_chunk, f_chunk):

        eigs_i = self.eigs[i_chunk]
        eigs_f = self.eigs[f_chunk]
        N = len(eigs_i)
        M = len(eigs_f)

        planck = qclSolver.fundamentals["planck"]
        mass_sub_i = self.mass_sub[i_chunk]
        mass_sub_f = self.mass_sub[f_chunk]
        Population = self.Population
        TE = self.TE

        R_forw = np.zeros((N, M))
        R_back = np.zeros((M, N))

        for i in range(0, N):
            for f in range(0, M):

                tau_ort = 1 / (1 / self.tau_pure - (self.W[i_chunk][i][i] + self.W[f_chunk][f][f]) / 2)
                omega, e_shift = self.findOmega(i, f, i_chunk, f_chunk)

                delta_if = -(eigs_i[i] - eigs_f[f] - e_shift) / planck * 1.60218e-19

                R_forw[i][f] = 2 * omega ** 2 * tau_ort / (1 + delta_if ** 2 * tau_ort ** 2) \
                               * qclSolver.sigma_b(delta_if, Population[i_chunk][i], mass_sub_i[i], TE)
                R_back[f][i] = 2 * omega ** 2 * tau_ort / (1 + delta_if ** 2 * tau_ort ** 2) \
                               * qclSolver.sigma_b(-delta_if, Population[f_chunk][f], mass_sub_f[f], TE)

        return R_forw, R_back

    def Build_R(self):
        R_f = []
        R_b = []
        for f_chunk in range(0, len(self.struct)):
            rf_list, rb_list = [], []
            for i_chunk in range(0, len(self.struct)):
                if (f_chunk - i_chunk != 1) or (i_chunk == len(self.struct)-1 and f_chunk == 0):

                    rf_list.append(np.zeros((len(self.eigs[i_chunk]), len(self.eigs[f_chunk]))))
                    rb_list.append(np.zeros((len(self.eigs[f_chunk]), len(self.eigs[i_chunk]))))
                else:
                    r_f, r_b = self.Build_R_local(i_chunk, f_chunk)
                    rf_list.append(r_f)
                    rb_list.append(r_b)
            R_f.append(np.vstack(rf_list))
            R_b.append(np.hstack(rb_list))
        R_f = np.hstack(R_f)
        R_b = np.vstack(R_b)
        R_cf = np.zeros_like(R_f)
        R_cb = np.zeros_like(R_b)

        r_f, r_b = self.Build_R_local(len(self.struct)-1, 0)
        R_cf[len(R_cf[:, 0]) - len(r_f[:, 0]):, :len(r_f[0, :])] = r_f
        R_cb[:len(r_b[:, 0]), len(R_cb[:, 0]) - len(r_b[0, :]):] = r_b

        R_f += R_cf
        R_b += R_cb

        for i in range(0, len(R_f[0, :])):
            R_f[i][i] = -R_f[i][:].sum()
            R_b[i][i] = -R_b[i][:].sum()

        return R_f, R_b

    Chi = lambda x, TE: np.heaviside(-x, 0) + np.heaviside(x, 0) * np.exp(
        -np.abs(x) / qclSolver.fundamentals["k-bol"] / TE)

    def sigma_b(delta_ab, N_b, m_b, TE):
        planck = qclSolver.fundamentals["planck"]
        k_bol = qclSolver.fundamentals["k-bol"]
        Beta = 1 / k_bol / TE
        D_e = m_b / np.pi / planck ** 2
        mu_b = brentq(lambda mu_t: N_b - D_e / Beta * np.logaddexp(0, Beta * mu_t),
                      -1 / 6.24150965 / 10**12,
                      1 / 6.24150965 / 10**12,
                      xtol=1e-30)

        return np.log(1 + np.exp(Beta * mu_b) * qclSolver.Chi(planck * delta_ab, TE)) / np.logaddexp(0, Beta * mu_b)

    # ============================================
    # ============ LIGHT INTERACTION =============
    # ============================================

    def gain_if(self, i, f, chunk, omega):

        eps_0 = qclSolver.fundamentals["eps0"]
        c = qclSolver.fundamentals["c"]
        z = self.grid[chunk] / 10 ** 9
        el = qclSolver.fundamentals["e-charge"]
        planck = qclSolver.fundamentals["planck"]
        k_bol = qclSolver.fundamentals["k-bol"]
        TE = self.TE

        m_i = self.mass_sub[chunk][i]
        m_f = self.mass_sub[chunk][f]
        n_i = self.Population[chunk][i]
        n_f = self.Population[chunk][f]

        delta_E = (self.eigs[chunk][i] - self.eigs[chunk][f]) * 1.60218e-19
        delta = (self.eigs[chunk][i] - self.eigs[chunk][f]) * 1.60218e-19 - planck * omega
        gamma = -planck * (self.W[chunk][f, f]) / 2

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

        psi_i = self.psi[chunk][:, i]
        psi_f = self.psi[chunk][:, f]
        z_aw = (z[1] - z[0]) * (psi_i * z * psi_f).sum() * 100

        koeff = el ** 2 * z_aw ** 2 * delta_E ** 2 / (
                eps_0 * c * planck ** 2 * omega * self.refr * self.struct[chunk].length / 10 ** 7)

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

    def tot_gain(self, chunk, omega):
        N = len(self.eigs[chunk])
        gain = 0.
        for i in range(0, N):
            for f in range(0, N):
                if i != f:
                    gain += self.gain_if(i, f, chunk, omega)[0]

        return gain

    def Build_G_local(self, chunk, omega):
        N = len(self.eigs[chunk])
        G = np.zeros((N, N), dtype=float)
        for i in range(0, N):
            for f in range(0, N):
                if i > f:
                    g_if = self.gain_if(i, f, chunk, omega)
                    g_fi = self.gain_if(f, i, chunk, omega)
                    G[i][f] += g_if[1] - g_fi[2]
                    G[f][i] += -g_if[2] + g_fi[1]
                    G[i][i] += -g_if[1] + g_fi[2]
                    G[f][f] += g_if[2] - g_fi[1]


        return G

    def Build_G(self, omega):

        for i in range(0,len(self.struct)):
            if i == 0:
                g = self.Build_G_local(i, omega)
            else:
                g = block_diag(g, self.Build_G_local(i, omega))

        return g

    def tot_gain_optical(self, omega, S, r_iter=4):

        N_carr_tot = self.N_carr / 10 ** 4

        pop_temp = self.Population

        for i in range(0, len(self.struct)):
            if i == 0:
                W = self.W[i]
            else:
                W = block_diag(W, self.W[i])

        for i in range(0, r_iter):

            R_1, R_2 = self.Build_R()
            G = self.Build_G(omega)
            G = np.array(G * S, dtype=float)

            Population = np.linalg.eig((W + R_1 + R_2 + G).transpose())

            ind = np.argmin(np.abs(Population[0]))
            Population = Population[1][:, ind]
            Population = Population.real / (Population.real.sum()) * N_carr_tot

            self.Population = []
            end = 0
            for j in range(0, len(self.struct)):
                self.Population.append(Population[end:end + len(self.eigs[j])])
                end += len(self.eigs[j])

        gain = 0.
        for i in range(0, len(self.struct)):
            gain += self.tot_gain(i, omega)

        self.pop_g = self.Population
        self.Population = pop_temp

        return gain

    def findMaxGain(self, lam_min, lam_max):

        c = qclSolver.fundamentals["c"]
        lam = np.linspace(lam_min, lam_max, 1000) / 10 ** (6)

        omega_r = 2 * np.pi * c / lam
        gain_r = 0.
        for i in range(0, len(self.struct)):
            gain_r += self.tot_gain(i, omega_r)
        return lam[np.argmax(gain_r)], np.amax(gain_r)

    def optPower(self, lam, r_iter=4):
        c = qclSolver.fundamentals["c"]
        planck = qclSolver.fundamentals["planck"]
        el = qclSolver.fundamentals["e-charge"]

        Omega = 2 * np.pi * c / lam
        S = brentq(lambda s: self.tot_gain_optical(Omega, s, r_iter=r_iter) - self.alpha_m - self.alpha_w,
                   0, 10 ** 35,
                   xtol=0.01, maxiter=200)

        self.P = Omega * planck * S * self.periods * self.dim_w * self.alpha_m

        R_1, R_2 = self.Build_R_local(len(self.struct) - 1, 0)
        self.J_opt = -el * ((R_1 * self.pop_g[len(self.struct) - 1][:, np.newaxis]).sum()
                            - (R_2 * self.pop_g[0][:, np.newaxis]).sum())


    # ============================================
    # ================= OUTPUT ===================
    # ============================================

    def plotWF(self, saveFile=True):

        plt.figure(figsize=(10, 8))
        plt.plot(self.unified_grid, self.unified_potential)
        plt.xlabel('z, nm')
        plt.ylabel('E, eV')
        for i in range(0, len(self.unified_eigs)):
            grid = self.unified_grid[:]
            psi = (self.unified_psi[:, i] / 10 ** 4) ** 2 / 20
            grid = np.delete(grid, np.argwhere(psi == 0.))
            psi = np.delete(psi, np.argwhere(psi == 0.))
            plt.plot(grid, psi + self.unified_eigs[i])
        if saveFile:
            plt.savefig('./WaveFunc.png', format='png', dpi=1000)
        plt.show()

    def plotGainSP(self, lam_min, lam_max, saveFile=True):

        c = qclSolver.fundamentals["c"]
        lam = np.linspace(lam_min, lam_max, 1000) / 10 ** (6)

        omega_r = 2 * np.pi * c / lam
        gain_r = 0.

        for i in range(0, len(self.struct)):
            gain_r += self.tot_gain(i, omega_r)

        plt.figure(figsize=(10, 5))
        plt.plot(lam * 10 ** 6, gain_r)
        plt.plot(lam * 10 ** 6, np.zeros_like(gain_r) + self.alpha_m + self.alpha_w, 'r--')
        plt.grid(True)
        if saveFile:
            plt.savefig('./Gain.png', format='png', dpi=1000)
        plt.show()

    def plotPopulation(self, saveFile=True):

        print(self.Population)
        plt.plot(self.Population)
        if saveFile:
            plt.savefig('./Population.png', format='png', dpi=1000)
        plt.show()

    def generateParOutput(self):

        print(self.eigs)
        if not self.evaluate_W:
            print('State populations:', self.Population)
            print('Current density:', round(self.J_d / 1000, 3), 'kA/cm^2')
            if self.P > 0:
                print('Optical Current Density:', self.J_opt / 1000, 'kA/cm^2')
                print('Optical Power:', self.P / 1000, 'mWt')

    def generatePlotOutput(self, lam_min=None, lam_max=None, saveFile=True):
        self.plotWF(saveFile=saveFile)
        if not self.evaluate_W:
            self.plotPopulation(saveFile=saveFile)
            if lam_max != None:
                self.plotGainSP(lam_min=lam_min, lam_max=lam_max, saveFile=saveFile)

    # ============================================
    # ================== MISC ====================
    # ============================================

    def findLifeTimes(self, ncpu=1):
        if self.evaluate_W:
            self.Build_W(ncpu=ncpu)
            self.evaluate_W = False

        return np.diag(-1 / self.W)
