import maraboupy
from convertsinglenetwork import single_model
from maraboupy import MarabouCore, MarabouUtils
from maraboupy import Marabou
import os
from datetime import datetime
import numpy as np

# import Marabou.maraboupy import Marabou
# from Marabou.maraboupy import MarabouCore
# import sys
# sys.path.append("/barrett/scratch/udayanm/Marabou")
# from training_exp_mask import LyapunovNetworkV, TwoDimDocking

# from maraboupy import MarabouCore
import maraboupy
print(maraboupy.__file__)


class Queries:

    def __init__(self, combined_model, cert_model):

        # self controller name
        self.combined_model = combined_model
        self.cert_model = cert_model

    def check_descent_safe(self, unsafe_threshold_pos, docking_threshold_pos, l_inf_D, input=[[-3, -2], [-3, -2]]):
        # if input[0][0] <= 0.2876 <= input[0][1] and input[1][0] <= -0.0806 <= input[1][1]:
        #     print("Find in", input)

        start = datetime.now()
        useMILP = True

        options = Marabou.createOptions(
            verbosity=0, numWorkers=16, solveWithMILP=useMILP, snc=False)  # remove tighteningStrategy
        network = Marabou.read_onnx(self.combined_model)

        # print(network.inputVars)
        # print(network.outputVars)

        th_0, thdot_0, th_1, thdot_1, th_t, thdot_t, sin_lower, sin_upper, u, newthdot_lower, newthdot_upper,  init_val, out_val = self.run_unroll(
            network)

        network.setLowerBound(th_0, input[0][0])
        network.setUpperBound(th_0, input[0][1])
        network.setLowerBound(thdot_0, input[1][0])
        network.setUpperBound(thdot_0, input[1][1])

        # network.setLowerBound(th_0, 0.2876)
        # network.setUpperBound(th_0, 0.2876)
        # network.setLowerBound(thdot_0, -0.0806)
        # network.setUpperBound(thdot_0, -0.0806)

        # network.addEquality([th_t, th_1], [1, -1], 0, False)
        # network.addEquality([thdot_t, thdot_1], [1, -1], 0, False)
        # ----------- modify
        # l-inf
        # l_inf_D = 0
        delta_th = network.getNewVariable()
        delta_thdot = network.getNewVariable()
        network.addEquality([delta_th, th_t, th_1], [1, -1, 1], 0, False)
        network.addEquality([delta_thdot, thdot_t, thdot_1], [
            1, -1, 1], 0, False)
        network.setUpperBound(delta_th, l_inf_D)
        network.setUpperBound(delta_thdot, l_inf_D)
        network.setLowerBound(delta_th, -l_inf_D)
        network.setLowerBound(delta_thdot, -l_inf_D)
        # network.addInequality([th_t, th_1], [1, -1], l_inf_D)
        # network.addInequality([thdot_t, thdot_1], [1, -1], l_inf_D)
        # network.addInequality([th_t, th_1], [-1, 1], l_inf_D)
        # network.addInequality([thdot_t, thdot_1], [-1, 1], l_inf_D)
        # network.addEquality([th_1, th_t], [1, -1], l_inf_D, False)
        # network.addEquality([thdot_1, thdot_t], [1, -1], l_inf_D, False)
        # modify -------------
        # l-1
        # d_x = network.getNewVariable()
        # d_y = network.getNewVariable()
        # d_vx = network.getNewVariable()
        # d_vy = network.getNewVariable()
        # d_x_abs = network.getNewVariable()
        # d_y_abs = network.getNewVariable()
        # d_vx_abs = network.getNewVariable()
        # d_vy_abs = network.getNewVariable()
        # network.addEquality([d_x, th_t, th_1], [1, -1, 1], 0, False)
        # network.addEquality([d_y, thdot_t, thdot_1], [1, -1, 1], 0, False)
        # network.addAbsConstraint(d_x, d_x_abs)  # d_x = |th_t - th_1|
        # network.addAbsConstraint(d_y, d_y_abs)  # d_y = |thdot_t - thdot_1|
        # network.addInequality(
        #     [d_x_abs, d_y_abs, d_vx_abs, d_vy_abs], [1, 1, 1, 1], 0.015)

        # force that when cert <=1
        network.setUpperBound(init_val, 1)

        # checking descending condition.
        e1 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e1.addAddend(1.0, init_val)
        e1.addAddend(-1.0, out_val)
        # e1.setScalar(0.0000001)
        e1.setScalar(0)

        # called with input preprocessing ensuring we are outside the docking region + not in unsafe region
        # enforce that we do not enter unsafe region otherwise.

        # checking that we are outside unsafe region on next step
        # out of space
        # margin = max(1e-3, l_inf_D)
        margin = l_inf_D
        e10 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e10.addAddend(1.0, th_1)
        e10.setScalar(unsafe_threshold_pos+margin)

        e11 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e11.addAddend(1.0, thdot_1)
        e11.setScalar(unsafe_threshold_pos+margin)
        e12 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e12.addAddend(1.0, th_1)
        e12.setScalar(-unsafe_threshold_pos-margin)

        e13 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e13.addAddend(1.0, thdot_1)
        e13.setScalar(-unsafe_threshold_pos-margin)

        # in the unsafe region
        e30 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e30.addAddend(1.0, thdot_1)
        e30.setScalar(0.0)

        e31 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e31.addAddend(1.0, thdot_1)
        e31.setScalar(-0.7)

        e32 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e32.addAddend(1.0, th_1)
        e32.setScalar(-0.7)

        e33 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e33.addAddend(1.0, th_1)
        e33.setScalar(-0.6)

        e40 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e40.addAddend(1.0, thdot_1)
        e40.setScalar(0.0)

        e41 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e41.addAddend(1.0, thdot_1)
        e41.setScalar(0.7)

        e42 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e42.addAddend(1.0, th_1)
        e42.setScalar(0.7)

        e43 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e43.addAddend(1.0, th_1)
        e43.setScalar(0.6)

        # maps to either not decreasing or in unsafe region
        constraints = [[e1], [e10], [e11], [e12], [e13], [
            e30, e31, e32, e33], [e40, e41, e42, e43]]
        # counterexamples should be outside of the goal region
        e20 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e20.addAddend(1.0, thdot_t)
        e20.setScalar(-0.2-1e-3)

        e21 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e21.addAddend(1.0, th_t)
        e21.setScalar(-0.2-1e-3)

        e22 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e22.addAddend(1.0, thdot_t)
        e22.setScalar(0.2+1e-3)

        e23 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e23.addAddend(1.0, th_t)
        e23.setScalar(0.2+1e-3)

        e24 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e24.addAddend(1.0, thdot_0)
        e24.setScalar(-0.2-1e-3)

        e25 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e25.addAddend(1.0, th_0)
        e25.setScalar(-0.2-1e-3)

        e26 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e26.addAddend(1.0, thdot_0)
        e26.setScalar(0.2+1e-3)

        e27 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e27.addAddend(1.0, th_0)
        e27.setScalar(0.2+1e-3)

        # in counterexample, init state should be outside of the unsafe region
        e50 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e50.addAddend(1.0, th_0)
        e50.setScalar(unsafe_threshold_pos-1e-5)

        e51 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e51.addAddend(1.0, thdot_0)
        e51.setScalar(unsafe_threshold_pos-1e-5)

        e52 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e52.addAddend(1.0, th_0)
        e52.setScalar(-unsafe_threshold_pos+1e-5)

        e53 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e53.addAddend(1.0, thdot_0)
        e53.setScalar(-unsafe_threshold_pos+1e-5)

        # network.addDisjunctionConstraint(([[e50, e51, e52, e53]]))

        e54 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e54.addAddend(1.0, thdot_0)
        e54.setScalar(0.0)

        e55 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e55.addAddend(1.0, thdot_0)
        e55.setScalar(-0.7)

        e56 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e56.addAddend(1.0, th_0)
        e56.setScalar(-0.7)

        e57 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e57.addAddend(1.0, th_0)
        e57.setScalar(-0.6)

        # network.addDisjunctionConstraint(([e54], [e55], [e56], [e57]))

        e60 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e60.addAddend(1.0, thdot_0)
        e60.setScalar(0.0)

        e61 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e61.addAddend(1.0, thdot_0)
        e61.setScalar(0.7)

        e62 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e62.addAddend(1.0, th_0)
        e62.setScalar(0.7)

        e63 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e63.addAddend(1.0, th_0)
        e63.setScalar(0.6)
        # network.addDisjunctionConstraint(([e60], [e61], [e62], [e63]))

        network.addDisjunctionConstraint([[e21], [e23], [e20], [e22]])
        network.addDisjunctionConstraint([[e25], [e27], [e24], [e26]])
        # e20 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        # e20.addAddend(1.0, thdot_t)
        # e20.setScalar(-0.2-1e-3)

        # e21 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        # e21.addAddend(1.0, th_t)
        # e21.setScalar(-0.2-1e-3)

        # e22 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        # e22.addAddend(1.0, thdot_t)
        # e22.setScalar(0.2+1e-3)
        # e23 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        # e23.addAddend(1.0, th_t)
        # e23.setScalar(0.2+1e-3)

        # e24 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        # e24.addAddend(1.0, thdot_0)
        # e24.setScalar(-0.2-1e-3)

        # e25 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        # e25.addAddend(1.0, th_0)
        # e25.setScalar(-0.2-1e-3)

        # e26 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        # e26.addAddend(1.0, thdot_0)
        # e26.setScalar(0.2+1e-3)

        # e27 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        # e27.addAddend(1.0, th_0)
        # e27.setScalar(0.2+1e-3)

        # network.addDisjunctionConstraint([[e21], [e23], [e20], [e22]])
        # network.addDisjunctionConstraint([[e25], [e27], [e24], [e26]])
        # force that we end up outside the unsafe region(not in unsafe region)
        network.addDisjunctionConstraint(constraints)
        # Ipq = network.getInputQuery()
        # Ipq.dump()
        # exit()
        exitCode, vals, stats = network.solve(options=options, verbose=True)
        # print("index", th_1, thdot_1)
        end = datetime.now()
        diff = end - start

        # if (diff.seconds > 120):
        #     network.saveQuery(str(input) + "_descent.ipq")

        if exitCode == "sat":
            print("Sat time taken: ", diff.seconds, "seconds")
            print("theta1, thetadot1", vals[th_1], vals[thdot_1])
            print("l_inf_D", l_inf_D)
            print("Satisfying assignment ending at",
                  vals[th_0], vals[thdot_0],
                  vals[th_t], vals[thdot_t],  vals[newthdot_lower], vals[newthdot_upper], vals[init_val], vals[out_val])
            # if (abs(vals[th_t]) <= unsafe_threshold_pos and abs(vals[thdot_t]) <= unsafe_threshold_pos and vals[init_val] - vals[out_val] >= 0.0000001 and vals[init_val] <= 1):
            #     print(unsafe_threshold_pos)
            # with open('sperious.txt','a') as f:
            #     f.write(str(input) + ' ' + str(vals[x_0]) + ' ' + str(vals[y_0]) + ' ' + str(vals[v_x_0]) + ' ' + str(vals[v_y_0]) + ' ' + str(vals[th_t]) + ' ' + str(vals[thdot_t]) + ' ' + str(vals[v_th_t]) + ' ' + str(vals[v_thdot_t]) + ' ' + str(vals[init_val]) + ' ' + str(vals[out_val]) + '\n')
            return [vals[th_0], vals[thdot_0]]
        if exitCode == "unsat":
            print("Inductive proof completed in", diff.seconds, "seconds")
            return [1]
        else:
            print("Failed in", diff.seconds, "seconds")
            return [-1]

    def check_safe_region(self, docking_threshold_pos, input=[[-3, -2], [-3, -2], [-0.5, -0.3], [-0.5, -0.3]]):
        start = datetime.now()
        useMILP = True

        options = Marabou.createOptions(
            verbosity=0, numWorkers=16, solveWithMILP=useMILP, snc=False)
        network = Marabou.read_onnx(self.cert_model)

        inp_state = network.inputVars[0][0]
        lyap_value = network.outputVars[0][0]

        th_0 = inp_state[0]
        thdot_0 = inp_state[1]

        init_val = lyap_value[0]

        network.setLowerBound(th_0, input[0][0])
        network.setUpperBound(th_0, input[0][1])
        network.setLowerBound(thdot_0, input[1][0])
        network.setUpperBound(thdot_0, input[1][1])

        # check if for safe range if its the case that init val >= 1
        network.setLowerBound(init_val, 1.01)
        # network.addInequality([init_val], [-1], -1.01)

        exitCode, vals, stats = network.solve(options=options, verbose=True)

        end = datetime.now()
        diff = end - start

        if (diff.seconds > 120):
            network.saveQuery(str(input) + "_safe" + ".ipq")

        if exitCode == "sat":
            print("Sat time taken: ", diff.seconds, "seconds")
            print("Satisfying assignment ending at",
                  vals[th_0], vals[thdot_0], vals[init_val])
            return [vals[th_0], vals[thdot_0]]
        if exitCode == "unsat":
            print("Inductive proof completed in", diff.seconds, "seconds")
            return [1]
        else:
            print("Failed in", diff.seconds, "seconds")
            return [-1]

    def check_local_lip(self, input=[[-3, -2], [-3, -2], [-0.5, -0.3], [-0.5, -0.3]]):
        return [1]
        start = datetime.now()
        useMILP = True

        options = Marabou.createOptions(
            verbosity=0, numWorkers=16, solveWithMILP=useMILP, snc=False)
        network = Marabou.read_onnx(self.combined_model)

        inp_state = network.inputVars[0][0]
        lyap_value = network.outputVars[0][0]
        inp_state_noise = network.inputVars[1][0]
        lyap_value_noise = network.outputVars[1][0]

        first_output = network.outputVars[0][0]
        lyap_value = network.outputVars[1][0]
        lyap_value_noise = network.outputVars[2][0]

        th_0, thdot_0, th_0_noise, thdot_0_noise = inp_state[0], inp_state[1], \
            inp_state_noise[0], inp_state_noise[1]
        u, state_val, noise_val = first_output[0], lyap_value[0], lyap_value_noise[0]

        epsilon = 0.01
        L = 0.6

        network.setLowerBound(th_0, input[0][0])
        network.setUpperBound(th_0, input[0][1])
        network.setLowerBound(thdot_0, input[1][0])
        network.setUpperBound(thdot_0, input[1][1])

        network.setLowerBound(th_0_noise, input[0][0])
        network.setUpperBound(th_0_noise, input[0][1])
        network.setLowerBound(thdot_0_noise, input[1][0])
        network.setUpperBound(thdot_0_noise, input[1][1])

        th_0_diff = network.getNewVariable()
        thdot_0_diff = network.getNewVariable()
        th_0_diff_abs = network.getNewVariable()
        thdot_0_diff_abs = network.getNewVariable()
        state_diff_max = network.getNewVariable()
        val_diff = network.getNewVariable()
        val_diff_abs = network.getNewVariable()

        network.addEquality([th_0, th_0_noise, th_0_diff],
                            [1, -1, 1], 0, False)
        network.addEquality([thdot_0, thdot_0_noise, thdot_0_diff], [
                            1, -1, 1], 0, False)

        network.addAbsConstraint(th_0_diff, th_0_diff_abs)
        network.addAbsConstraint(thdot_0_diff, thdot_0_diff_abs)

        network.addMaxConstraint(
            set([th_0_diff_abs, thdot_0_diff_abs]), state_diff_max)
        network.setUpperBound(state_diff_max, epsilon)

        network.addEquality([state_val, noise_val, val_diff], [
                            1, -1, 1], 0, False)
        network.addAbsConstraint(val_diff, val_diff_abs)
        network.setLowerBound(state_diff_max, 1e-3)
        network.setLowerBound(val_diff_abs, 1e-3)
        network.addInequality([state_diff_max, val_diff_abs], [L, -1], 0)
        # e22 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        # e22.addAddend(L, state_diff_max)
        # e22.addAddend(-1, val_diff)
        # e22.setScalar(0)
        # e23 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        # e23.addAddend(L, state_diff_max)
        # e23.addAddend(1, val_diff)
        # e23.setScalar(0)
        # network.addDisjunctionConstraint([[e22], [e23]])
        network.setUpperBound(state_val, 1)

        exitCode, vals, stats = network.solve(options=options, verbose=True)

        end = datetime.now()
        diff = end - start

        if (diff.seconds > 120):
            network.saveQuery(str(input) + "_safe" + ".ipq")

        if exitCode == "sat":
            print("Sat time taken: ", diff.seconds, "seconds")
            print("Satisfying assignment ending at",
                  vals[th_0], vals[thdot_0], vals[state_diff_max], vals[val_diff], vals[val_diff_abs], vals[th_0_diff], vals[th_0_diff_abs])
            return [vals[th_0], vals[thdot_0]]
        if exitCode == "unsat":
            print("Inductive proof completed in", diff.seconds, "seconds")
            return [1]
        else:
            print("Failed in", diff.seconds, "seconds")
            return [-1]

    def clamp(self, network, var, lb, ub):
        """
        F_y_clip = network.getNewVariable()
        network.addEquality([aux1, F_y], [1, -1], -1 *
                            clip_lb, False)  # aux1 = F_y-clip_lb
        aux2 = network.getNewVariable()
        network.addRelu(aux1, aux2)  # aux2 = relu(aux1)
        aux3 = network.getNewVariable()
        network.addEquality([aux3, aux2], [1, -1], clip_lb,
                            False)  # aux3 = clip_lb + aux2
        aux4 = network.getNewVariable()
        network.addEquality([aux4, aux3], [-1, -1], -
                            clip_ub, False)  # aux4 = clip_ub - aux3
        aux5 = network.getNewVariable()
        network.addRelu(aux4, aux5)  # aux5 = relu(aux4)
        # F_y_clip = clip_ub - aux5
        network.addEquality([F_y_clip, aux5], [-1, -1], -clip_ub, False)
        """
        aux1 = network.getNewVariable()
        aux2 = network.getNewVariable()
        aux3 = network.getNewVariable()
        aux4 = network.getNewVariable()
        aux5 = network.getNewVariable()
        var_clipped = network.getNewVariable()

        network.addEquality([aux1, var], [1, -1], -lb, False)
        network.addRelu(aux1, aux2)
        network.addEquality([aux3, aux2], [1, -1], lb, False)
        network.addEquality([aux4, aux3], [-1, -1], -ub, False)
        network.addRelu(aux4, aux5)
        network.addEquality([var_clipped, aux5], [-1, -1], -ub, False)
        return var_clipped

    def scale_clip(self, network, var, minval, maxval, scale=1.0):
        # First clamp var to [minval, maxval]
        clamped = self.clamp(network, var, minval, maxval)

        # Then scale
        scaled = network.getNewVariable()
        network.addEquality([scaled, clamped], [1, -scale], 0, False)
        return scaled

    def encode_piecewise_sin_bounds(self, network, th):
        # Each segment: (coefficient, slope, breakpoint1, breakpoint2) for upper and lower
        def add_segment(network, th, s):
            k, slope1, bp1, slope2, bp2 = s
            assert slope2 is None or bp2 is not None, "Both slope2 and bp2 must be provided or both must be None"
            if slope2 is None and bp2 is None:
                # delta = th - bp1
                # relu_in = slope1 * (th - bp1)
                # relu_out = max(0, slope1*(th - bp1))
                delta = network.getNewVariable()
                network.addEquality([delta, th], [1, -1], -bp1, False)
                relu_in = network.getNewVariable()
                network.addEquality([relu_in, delta], [1, -slope1], 0, False)
                relu_out = network.getNewVariable()
                network.addRelu(relu_in, relu_out)
            else:
                # delta1 = th - bp1
                # t1 = slope1 * (th - bp1)
                # t2 = slope2 * (th - bp2)
                # min_t1_t2 = t1 + t2 - max(t1, t2)
                # relu_out = max(0, min(t1, t2))
                delta1 = network.getNewVariable()
                network.addEquality([delta1, th], [1, -1], -bp1)
                t1 = network.getNewVariable()
                network.addEquality([t1, delta1], [1, -slope1], 0)

                delta2 = network.getNewVariable()
                network.addEquality([delta2, th], [1, -1], -bp2)
                t2 = network.getNewVariable()
                network.addEquality([t2, delta2], [1, -slope2], 0)

                # min(a, b) = a + b - max(a, b)
                neg_t1 = network.getNewVariable()
                network.addEquality([neg_t1, t1], [1, 1], 0, False)
                neg_t2 = network.getNewVariable()
                network.addEquality([neg_t2, t2], [1, 1], 0, False)
                neg_max_t1_t2 = network.getNewVariable()
                network.addMaxConstraint(set([neg_t1, neg_t2]), neg_max_t1_t2)
                min_t1_t2 = network.getNewVariable()
                network.addEquality(
                    [neg_max_t1_t2, min_t1_t2], [1, 1], 0, False)
                # max_t1_t2 = network.getNewVariable()
                # network.addMaxConstraint(set([t1, t2]), max_t1_t2)
                # min_t1_t2 = network.getNewVariable()
                # network.addEquality(
                #     [min_t1_t2, t1, t2, max_t1_t2], [1, -1, -1, 1], 0)
                relu_out = network.getNewVariable()
                network.addRelu(min_t1_t2, relu_out)

            scaled = network.getNewVariable()
            network.addEquality([scaled, relu_out], [1, -k], 0)
            return scaled

        # Define segments for v_11 (upper bound of sin(th))
        """
        v_11 == -0.6442048028839462 * max(0, -7.116027437521648 * (th - -0.5594721551062094)) + 
        -0.5307260203042105 * max(0, min(7.116027437521648 * (th - -0.7), -6.579114308657342 * (th - -0.4074760119955604))) + 
        -0.3962803744602469 * max(0, min(6.579114308657342 * (th - -0.5594721551062094), -5.791890862570436 * (th - -0.2348208249887733))) + 
        -0.2326558471540191 * max(0, min(5.791890862570436 * (th - -0.4074760119955604), -4.2585660792555755 * (th - 0.0))) + 
        1.2884353744753822e-5 * max(0, min(4.2585660792555755 * (th - -0.2348208249887733), -4.669885975875376 * (th - 0.21413799077022405))) + 
        0.21415087512293396 * max(0, min(4.669885975875376 * (th - 0.0), -4.645705852508769 * (th - 0.429390534032295))) + 
        0.41835643466886857 * max(0, min(4.645705852508769 * (th - 0.21413799077022405), -5.39290695464878 * (th - 0.6148192848728751))) + 
        0.5790807671191807 * max(0, min(5.39290695464878 * (th - 0.429390534032295), -11.739746473219743 * (th - 0.7))) + 
        0.6442305715914358 * max(0, 11.739746473219743 * (th - 0.6148192848728751))
        v_12 == -0.6442305715914358 * max(0, -11.739746471639581 * (th - -0.6148192848614098)) + -0.579080767110056 * max(0, min(11.739746471639581 * (th - -0.7), -5.392906954380035 * (th - -0.4293905340115893))) + -0.4183564346504053 * max(0, min(5.392906954380035 * (th - -0.6148192848614098), -4.645705852733989 * (th - -0.21413799075995357))) + -0.21415087511238962 * max(0, min(4.645705852733989 * (th - -0.4293905340115893), -4.669885976099353 * (th - 0.0))) + -1.2884353744753822e-5 * max(0, min(4.669885976099353 * (th - -0.21413799075995357), -4.258566079569611 * (th - 0.2348208249714571))) + 0.23265584713717813 * max(0, min(4.258566079569611 * (th - 0.0), -5.791890862774022 * (th - 0.4074760119721753))) + 0.3962803744387765 * max(0, min(5.791890862774022 * (th - 0.2348208249714571), -6.579114308316329 * (th - 0.5594721550907027))) + 0.530726020291068 * max(0, min(6.579114308316329 * (th - 0.4074760119721753), -7.116027436736422 * (th - 0.7))) + 0.6442048028839462 * max(0, 7.116027436736422 * (th - 0.5594721550907027))
        """
        upper_segments = [
            (-0.6442048028839462, -7.116027437521648, -
             0.5594721551062094, None, None),
            (-0.5307260203042105, 7.116027437521648, -
             0.7, -6.579114308657342, -0.4074760119955604),
            (-0.3962803744602469, 6.579114308657342, -
             0.5594721551062094, -5.791890862570436, -0.2348208249887733),
            (-0.2326558471540191, 5.791890862570436, -
             0.4074760119955604, -4.2585660792555755, 0.0),
            (1.2884353744753822e-5, 4.2585660792555755, -
             0.2348208249887733, -4.669885975875376, 0.21413799077022405),
            (0.21415087512293396, 4.669885975875376,
             0.0, -4.645705852508769, 0.429390534032295),
            (0.41835643466886857, 4.645705852508769,
             0.21413799077022405, -5.39290695464878, 0.6148192848728751),
            (0.5790807671191807, 5.39290695464878,
             0.429390534032295, -11.739746473219743, 0.7),
            (0.6442305715914358, 11.739746473219743,
             0.6148192848728751, None, None),
        ]
        # upper_segments = [
        #     (-0.6, -7.1, -0.6, None, None),
        #     (-0.5, 7.1, -0.7, -6.6, -0.4),
        #     (-0.4, 6.6, -0.6, -5.8, -0.2),
        #     (-0.2, 5.8, -0.4, -4.3, 0.0),
        #     (0.0, 4.3, -0.2, -4.7, 0.2),
        #     (0.2, 4.7, 0.0, -4.6, 0.4),
        #     (0.4, 4.6, 0.2, -5.4, 0.6),
        #     (0.6, 5.4, 0.4, -11.7, 0.7),
        #     (0.6, 11.7, 0.6, None, None),
        # ]

        # Define segments for v_12 (lower bound of sin(th))
        lower_segments = [
            (-0.6442305715914358, -11.739746471639581, -
             0.6148192848614098, None, None),
            (-0.579080767110056, 11.739746471639581, -
             0.7, -5.392906954380035, -0.4293905340115893),
            (-0.4183564346504053, 5.392906954380035, -
             0.6148192848614098, -4.645705852733989, -0.21413799075995357),
            (-0.21415087511238962, 4.645705852733989, -
             0.4293905340115893, -4.669885976099353, 0.0),
            (-1.2884353744753822e-5, 4.669885976099353, -
             0.21413799075995357, -4.258566079569611, 0.2348208249714571),
            (0.23265584713717813, 4.258566079569611,
             0.0, -5.791890862774022, 0.4074760119721753),
            (0.3962803744387765, 5.791890862774022,
             0.2348208249714571, -6.579114308316329, 0.5594721550907027),
            (0.530726020291068, 6.579114308316329,
             0.4074760119721753, -7.116027436736422, 0.7),
            (0.6442048028839462, 7.116027436736422, 0.5594721550907027, None, None),
        ]

        def apply_segments(segments):
            outputs = []
            for s in segments:
                term = add_segment(network, th, s)
                outputs.append(term)
            result = network.getNewVariable()
            network.addEquality([result] + outputs, [1] +
                                [-1]*len(outputs), 0, False)
            return result

        upper = apply_segments(upper_segments)
        lower = apply_segments(lower_segments)
        # upper = network.getNewVariable()
        # network.addEquality([upper, th], [1, -1], 0, False)
        # lower = network.getNewVariable()
        # network.addEquality([lower, upper, 1], [1, -1, 0.0001], 0, False)
        return upper, lower

    def run_unroll(self, network):
        # INITIALIZATION

        first_state = network.inputVars[0][0]
        second_state = network.inputVars[1][0]

        first_output = network.outputVars[0][0]
        second_output = network.outputVars[1][0]
        third_output = network.outputVars[2][0]
        th_0, thdot_0, th_1, thdot_1 = first_state[0], first_state[1], second_state[0], second_state[1]
        action, init_val, out_val = first_output[0], second_output[0], third_output[0]

        dt = 0.05
        g = 10
        m = 0.15
        l = 0.5
        b = 0.1

        # NN output: u = 2 * clip(action, -1, 1)
        u = self.scale_clip(network, action, -1, 1, scale=2.0)

        # Get sin(th) bounds via piecewise ReLU approximation (v_11 and v_12)
        sin_upper, sin_lower = self.encode_piecewise_sin_bounds(network, th_0)
        # sin_upper = network.getNewVariable()
        # network.addEquality([sin_upper, th_0], [1, -1], 0, False)
        # sin_lower = network.getNewVariable()
        # network.addEquality([sin_lower, th_0], [1, -0.5], 0, False)

        # Constants
        k1 = 1 - b
        k2 = 3 * g * 0.5 / (2 * l) * dt  # 0.75
        k3 = 3.0 / (m * l ** 2) * dt    # 12

        # Create linear expressions for lower and upper bounds of newthdot
        newthdot_lower = network.getNewVariable()
        network.addEquality([newthdot_lower, thdot_0, sin_lower, u],
                            [1, -k1, -k2, -k3], 0, False)

        newthdot_upper = network.getNewVariable()
        network.addEquality([newthdot_upper, thdot_0, sin_upper, u],
                            [1, -k1, -k2, -k3], 0, False)

        # Now declare the newthdot variable and bound it
        newthdot = network.getNewVariable()
        network.addInequality([newthdot, newthdot_upper], [1, -1], 0)
        network.addInequality([newthdot_lower, newthdot], [1, -1], 0)

        # Clip newthdot to max speed
        newthdot_clipped = self.clamp(network, newthdot, -5, 5)

        # Compute newth = th + newthdot_clipped * dt

        newth = network.getNewVariable()
        network.addEquality([newth, th_0, newthdot_clipped], [
                            1, -1, -dt], 0, False)

        # Clamp newth to [-0.7, 0.7]
        # th_t = self.clamp(network, newth, -0.7, 0.7)
        # thdot_t = self.clamp(network, newthdot_clipped, -0.7, 0.7)

        return th_0, thdot_0, th_1, thdot_1, newth, newthdot, sin_lower, sin_upper, u, newthdot_lower, newthdot_upper, init_val, out_val
        first_state = network.inputVars[0][0]
        second_state = network.inputVars[1][0]

        first_output = network.outputVars[0][0]
        second_output = network.outputVars[1][0]
        third_output = network.outputVars[2][0]

        clip_lb = -1
        clip_ub = 1

        # EXTRACT FINAL STATE
        x_0, y_0, v_x_0, v_y_0, x_1, y_1, v_x_1, v_y_1 = first_state[0], first_state[1], first_state[2], first_state[3], \
            second_state[0], second_state[1], second_state[2], \
            second_state[3]
        F_x, F_y, init_val, out_val = first_output[0], first_output[1], second_output[0], third_output[0]

        # handling F_x_clipping -- TODO: add method in Marabou for this
        aux1 = network.getNewVariable()
        F_x_clip = network.getNewVariable()
        network.addEquality([aux1, F_x], [1, -1], -1 *
                            clip_lb, False)  # aux1 = F_x-clip_lb
        aux2 = network.getNewVariable()
        network.addRelu(aux1, aux2)  # aux2 = relu(aux1)
        aux3 = network.getNewVariable()
        network.addEquality([aux3, aux2], [1, -1], clip_lb,
                            False)  # aux3 = clip_lb + aux2
        aux4 = network.getNewVariable()
        network.addEquality([aux4, aux3], [-1, -1], -
                            clip_ub, False)  # aux4 = clip_ub - aux3
        aux5 = network.getNewVariable()
        network.addRelu(aux4, aux5)  # aux5 = relu(aux4)
        # F_x_clip = clip_ub - aux5
        network.addEquality([F_x_clip, aux5], [-1, -1], -clip_ub, False)

        # handling F_y_clipping
        aux1 = network.getNewVariable()
        F_y_clip = network.getNewVariable()
        network.addEquality([aux1, F_y], [1, -1], -1 *
                            clip_lb, False)  # aux1 = F_y-clip_lb
        aux2 = network.getNewVariable()
        network.addRelu(aux1, aux2)  # aux2 = relu(aux1)
        aux3 = network.getNewVariable()
        network.addEquality([aux3, aux2], [1, -1], clip_lb,
                            False)  # aux3 = clip_lb + aux2
        aux4 = network.getNewVariable()
        network.addEquality([aux4, aux3], [-1, -1], -
                            clip_ub, False)  # aux4 = clip_ub - aux3
        aux5 = network.getNewVariable()
        network.addRelu(aux4, aux5)  # aux5 = relu(aux4)
        # F_y_clip = clip_ub - aux5
        network.addEquality([F_y_clip, aux5], [-1, -1], -clip_ub, False)

        return x_0, y_0, v_x_0, v_y_0, x_1, y_1, v_x_1, v_y_1, x_t, y_t, v_x_t, v_y_t, init_val, out_val


def safe_descent_cond_check(PATH_TO_ONNX, PATH_TO_CERT, l_inf_D, limit_pos=5, safe_pos=4, docking_pos=0.35):
    vals = []
    failed_vals = []
    val_ranges = []

    queries = Queries(combined_model=PATH_TO_ONNX, cert_model=PATH_TO_CERT)
    n = 5
    excluding_space_r = np.linspace(docking_pos, limit_pos, n)
    excluding_space_l = np.linspace(-limit_pos, -docking_pos, n)
    default_space = np.linspace(-limit_pos, limit_pos, n)

    safe_space = np.linspace(-safe_pos, safe_pos, n)
    safe_space_r = np.linspace(docking_pos, safe_pos, n)
    safe_space_l = np.linspace(-safe_pos, -docking_pos, n)

    # m = 5
    # left_top_th = np.linspace(-0.7, -0.6, m)
    # left_top_thdot = np.linspace(0, 0.7, m)
    # right_bottom_th = np.linspace(0.6, 0.7, m)
    # right_bottom_thdot = np.linspace(-0.7, 0, m)
    # # check descent and lip for these spaces
    # for i in range(m-1):
    #     ans = queries.check_descent_safe(limit_pos, docking_pos, [[round(left_top_th[i],2),round(left_top_th[i+1],2)]])
    # ans = queries.check_descent_safe(limit_pos, docking_pos, [[np.float64(
    #     0.2), np.float64(0.32)], [np.float64(-0.1), np.float64(-0.005)]])
    # exit()
    for i in range(n-1):
        # inside x for docking region, but outside docking region for y
        ans = queries.check_descent_safe(limit_pos, docking_pos, l_inf_D, [
                                         [-docking_pos, docking_pos], [round(excluding_space_l[i], 2), round(excluding_space_l[i + 1], 2)]])
        if len(ans) == 2:
            vals.append(ans)
            val_ranges.append(
                [[-docking_pos, docking_pos], [round(excluding_space_l[i], 2), round(excluding_space_l[i + 1], 2)]])
            print("descent", ans, val_ranges[-1])
            return [vals, val_ranges, failed_vals]
        elif ans[0] == -1:
            failed_vals.append(ans)

        ans = queries.check_descent_safe(limit_pos, docking_pos, l_inf_D, [
                                         [-docking_pos, docking_pos], [round(excluding_space_r[i], 2), round(excluding_space_r[i + 1], 2)]])
        if len(ans) == 2:
            vals.append(ans)
            val_ranges.append(
                [[-docking_pos, docking_pos], [round(excluding_space_r[i], 2), round(excluding_space_r[i + 1], 2)]])
            return [vals, val_ranges, failed_vals]
        elif ans[0] == -1:
            failed_vals.append(ans)

        # check for lip
        ans = queries.check_local_lip(
            [[-docking_pos, docking_pos], [round(excluding_space_l[i], 2), round(excluding_space_l[i + 1], 2)]])
        if len(ans) == 2:
            vals.append(ans)
            val_ranges.append(
                [[-docking_pos, docking_pos], [round(excluding_space_l[i], 2), round(excluding_space_l[i + 1], 2)]])
            print("lip", ans, val_ranges[-1])
        elif ans[0] == -1:
            failed_vals.append(ans)

        ans = queries.check_local_lip(
            [[-docking_pos, docking_pos], [round(excluding_space_r[i], 2), round(excluding_space_r[i + 1], 2)]])

        if len(ans) == 2:
            vals.append(ans)
            val_ranges.append(
                [[-docking_pos, docking_pos], [round(excluding_space_r[i], 2), round(excluding_space_r[i + 1], 2)]])
            print("lip")
        elif ans[0] == -1:
            failed_vals.append(ans)

        for j in range(n-1):
            # outside x for docking region, all y
            ans = queries.check_descent_safe(limit_pos, docking_pos, l_inf_D, [[round(excluding_space_l[i], 2), round(
                excluding_space_l[i + 1], 2)], [round(default_space[j], 2), round(default_space[j + 1], 2)]])
            if len(ans) == 2:
                vals.append(ans)
                val_ranges.append([[round(excluding_space_l[i], 2), round(
                    excluding_space_l[i + 1], 2)], [round(default_space[j], 2), round(default_space[j + 1], 2)]])
                return [vals, val_ranges, failed_vals]
            elif ans[0] == -1:
                failed_vals.append(ans)

            ans = queries.check_descent_safe(limit_pos, docking_pos, l_inf_D, [[round(excluding_space_r[i], 2), round(
                excluding_space_r[i + 1], 2)], [round(default_space[j], 2), round(default_space[j + 1], 2)]])
            if len(ans) == 2:
                vals.append(ans)
                val_ranges.append([[round(excluding_space_r[i], 2), round(
                    excluding_space_r[i + 1], 2)], [round(default_space[j], 2), round(default_space[j + 1], 2)]])
                return [vals, val_ranges, failed_vals]
            elif ans[0] == -1:
                failed_vals.append(ans)

            # check for lip
            ans = queries.check_local_lip([[round(excluding_space_l[i], 2), round(
                excluding_space_l[i + 1], 2)], [round(default_space[j], 2), round(default_space[j + 1], 2)]])
            if len(ans) == 2:
                vals.append(ans)
                val_ranges.append([[round(excluding_space_l[i], 2), round(
                    excluding_space_l[i + 1], 2)], [round(default_space[j], 2), round(default_space[j + 1], 2)]])
                print("lip")
            elif ans[0] == -1:
                failed_vals.append(ans)

            ans = queries.check_local_lip([[round(excluding_space_r[i], 2), round(
                excluding_space_r[i + 1], 2)], [round(default_space[j], 2), round(default_space[j + 1], 2)]])
            if len(ans) == 2:
                vals.append(ans)
                val_ranges.append([[round(excluding_space_r[i], 2), round(
                    excluding_space_r[i + 1], 2)], [round(default_space[j], 2), round(default_space[j + 1], 2)]])
                print("lip")
            elif ans[0] == -1:
                failed_vals.append(ans)

    print("safe====================safe")
    for i in range(n-1):
        ans = queries.check_safe_region(docking_pos, [[-docking_pos, docking_pos], [
                                        round(safe_space_l[i], 2), round(safe_space_l[i + 1], 2)]])
        if len(ans) == 2:
            vals.append(ans)
            val_ranges.append([[-docking_pos, docking_pos], [round(safe_space_l[i],
                              2), round(safe_space_l[i + 1], 2)]])
            print("safe", ans, val_ranges[-1])
            return [vals, val_ranges, failed_vals]
        elif ans[0] == -1:
            failed_vals.append(ans)

        ans = queries.check_safe_region(docking_pos, [[-docking_pos, docking_pos], [
                                        round(safe_space_r[i], 2), round(safe_space_r[i + 1], 2)]])
        if len(ans) == 2:
            vals.append(ans)
            val_ranges.append([[-docking_pos, docking_pos], [round(safe_space_r[i],
                              2), round(safe_space_r[i + 1], 2)]])
            return [vals, val_ranges, failed_vals]

        elif ans[0] == -1:
            failed_vals.append(ans)

        for j in range(n-1):
            ans = queries.check_safe_region(docking_pos, [[round(safe_space_l[i], 2), round(
                safe_space_l[i + 1], 2)], [round(safe_space[j], 2), round(safe_space[j + 1], 2)]])
            if len(ans) == 2:
                vals.append(ans)
                val_ranges.append([[round(safe_space_l[i], 2), round(safe_space_l[i + 1], 2)], [
                                  round(safe_space[j], 2), round(safe_space[j + 1], 2)]])
                return [vals, val_ranges, failed_vals]
            elif ans[0] == -1:
                failed_vals.append(ans)

            ans = queries.check_safe_region(docking_pos, [[round(safe_space_r[i], 2), round(
                safe_space_r[i + 1], 2)], [round(safe_space[j], 2), round(safe_space[j + 1], 2)]])
            if len(ans) == 2:
                vals.append(ans)
                val_ranges.append([[round(safe_space_r[i], 2), round(safe_space_r[i + 1], 2)], [
                                  round(safe_space[j], 2), round(safe_space[j + 1], 2)]])
                return [vals, val_ranges, failed_vals]
            elif ans[0] == -1:
                failed_vals.append(ans)
    with open(file_name, 'a') as f:
        print(len(vals), file=f)
    return [vals, val_ranges, failed_vals]


if __name__ == "__main__":
    combined_model = "combined_epsl_5e-3_6/combined_0.onnx"
    cert_model = "models_epsl_5e-3_6/cert_0.onnx"
    queries = Queries(combined_model=combined_model,
                      cert_model=cert_model)
    # ans = queries.check_descent_safe(unsafe_threshold_pos=2, unsafe_threshold_vel=0.5, docking_threshold_pos=0.5, input=[
    #                                  [0.3500,   0.5100], [-2.0000,  -1.0000], [-0.5000,  -0.2500], [0.2500,   0.5000]])
    file_name = "query_results.txt"
    low = 0
    high = 0.01
    l_inf_D = (low+high)/2
    vals, val_ranges, failed_vals = safe_descent_cond_check(
        combined_model, cert_model, low, limit_pos=0.7, safe_pos=0.3, docking_pos=0.2)
    assert len(vals) == 0
    vals, val_ranges, failed_vals = safe_descent_cond_check(
        combined_model, cert_model, high, limit_pos=0.7, safe_pos=0.3, docking_pos=0.2)
    assert len(vals) > 0

    with open(file_name, 'a') as f:
        print("failed", high, file=f)
    if len(vals) > 0:
        while high - low > 0.0005:
            val, val_ranges, failed_vals = safe_descent_cond_check(combined_model, cert_model, l_inf_D, limit_pos=0.7,
                                                                   safe_pos=0.3, docking_pos=0.2)
            if len(val) > 0:
                high = l_inf_D
                with open(file_name, 'a') as f:
                    print("failed", l_inf_D, file=f)
            else:
                low = l_inf_D
                with open(file_name, 'a') as f:
                    print("success", l_inf_D, file=f)
            l_inf_D = (low+high)/2
        print("l_inf_D", l_inf_D)
