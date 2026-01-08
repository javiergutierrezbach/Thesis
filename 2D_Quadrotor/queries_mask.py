import os
from datetime import datetime
import numpy as np
import math

import sys
sys.path.append("../../Marabou")

# import Marabou.maraboupy import Marabou
# from Marabou.maraboupy import MarabouCore
from maraboupy import Marabou
from maraboupy import MarabouCore, MarabouUtils

from estimate_lip import encodeDifferences

from convertsinglenetwork import single_model
#from training_exp_mask import LyapunovNetworkV, TwoDimDocking

# from maraboupy import MarabouCore
class Queries:

    def __init__(self, combined_model, cert_model):

        # self controller name
        self.combined_model = combined_model
        self.cert_model = cert_model

    def check_descent_safe(self, unsafe_threshold_pos, unsafe_threshold_th, unsafe_threshold_vel, unsafe_threshold_omega, docking_threshold_pos, input=[[-3, -2], [-3, -2], [-0.5, -0.3], [-0.5, -0.3]]):
        start = datetime.now()
        useMILP = True

        options = Marabou.createOptions(verbosity=0, numWorkers=16, solveWithMILP=useMILP, snc=False) #remove tighteningStrategy
        network = Marabou.read_onnx(self.combined_model)

        # print(network.inputVars)
        # print(network.outputVars)

        px0, pz0, th0, vx0, vz0, omega0, px1, pz1, th1, vx1, vz1, omega1, action1, action2, init_val, out_val = self.run_unroll(
            network)
        


        # input bounds
        network.setLowerBound(px0, input[0][0])
        network.setUpperBound(px0, input[0][1])
        network.setLowerBound(pz0, input[1][0])
        network.setUpperBound(pz0, input[1][1])
        network.setLowerBound(th0, input[2][0])
        network.setUpperBound(th0, input[2][1])
        network.setLowerBound(vx0, input[3][0])
        network.setUpperBound(vx0, input[3][1])
        network.setLowerBound(vz0, input[4][0])
        network.setUpperBound(vz0, input[4][1])
        network.setLowerBound(omega0, input[5][0])
        network.setUpperBound(omega0, input[5][1])


        # force that when cert <=1
        network.setUpperBound(init_val, 1)

        #checking descending condition.
        e1 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e1.addAddend(1.0, init_val)
        e1.addAddend(-1.0, out_val)
        e1.setScalar(1e-7)

        #force that the cert value decreases
        #network.addEquation(e1)

        # called with input preprocessing ensuring we are outside the docking region + not in unsafe region
        # enforce that we do not enter unsafe region otherwise.

        # checking that we are outside unsafe region on next step
        # out of space
        e10 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e10.addAddend(1.0, px1)
        e10.setScalar(unsafe_threshold_pos)

        e11 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e11.addAddend(1.0, pz1)
        e11.setScalar(unsafe_threshold_pos)

        e12 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e12.addAddend(1.0, px1)
        e12.setScalar(-unsafe_threshold_pos)

        e13 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e13.addAddend(1.0, pz1)
        e13.setScalar(-unsafe_threshold_pos)

        e14 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e14.addAddend(1.0, th1)
        e14.setScalar(unsafe_threshold_th)

        e15 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e15.addAddend(1.0, th1)
        e15.setScalar(-unsafe_threshold_th)

        e16 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e16.addAddend(1.0, vx1)
        e16.setScalar(unsafe_threshold_vel)

        e17 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e17.addAddend(1.0, vz1)
        e17.setScalar(unsafe_threshold_vel)

        e18 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e18.addAddend(1.0, vx1)
        e18.setScalar(-unsafe_threshold_vel)

        e19 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e19.addAddend(1.0, vz1)
        e19.setScalar(-unsafe_threshold_vel)

        e20 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e20.addAddend(1.0, omega1)
        e20.setScalar(unsafe_threshold_omega)

        e21 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e21.addAddend(1.0, omega1)
        e21.setScalar(-unsafe_threshold_omega)


        # in the unsafe region

        # unsafe left
        e30 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e30.addAddend(1.0, vx1)
        e30.setScalar(0.0)

        e31 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e31.addAddend(1.0, vx1)
        e31.setScalar(-unsafe_threshold_vel)

        e32 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e32.addAddend(1.0, px1)
        e32.setScalar(-unsafe_threshold_pos)

        e33 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e33.addAddend(1.0, px1)
        e33.setScalar(-unsafe_threshold_pos+0.1)

        # unsafe right

        e40 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e40.addAddend(1.0, vx1)
        e40.setScalar(0.0)

        e41 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e41.addAddend(1.0, vx1)
        e41.setScalar(unsafe_threshold_vel)

        e42 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e42.addAddend(1.0, px1)
        e42.setScalar(unsafe_threshold_pos)

        e43 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e43.addAddend(1.0, px1)
        e43.setScalar(unsafe_threshold_pos-0.1)

        # unsafe bottom

        e50 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e50.addAddend(1.0, vz1)
        e50.setScalar(0.0)

        e51 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e51.addAddend(1.0, vz1)
        e51.setScalar(-unsafe_threshold_vel)

        e52 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e52.addAddend(1.0, pz1)
        e52.setScalar(-unsafe_threshold_pos)

        e53 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e53.addAddend(1.0, pz1)
        e53.setScalar(-unsafe_threshold_pos+0.1)

        # unsafe top

        e60 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e60.addAddend(1.0, vz1)
        e60.setScalar(0.0)

        e61 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e61.addAddend(1.0, vz1)
        e61.setScalar(unsafe_threshold_vel)

        e62 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e62.addAddend(1.0, pz1)
        e62.setScalar(unsafe_threshold_pos)

        e63 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e63.addAddend(1.0, pz1)
        e63.setScalar(unsafe_threshold_pos-0.1)

        # unsafe tilt left

        e70 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e70.addAddend(1.0, omega1)
        e70.setScalar(0.0)

        e71 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e71.addAddend(1.0, omega1)
        e71.setScalar(-unsafe_threshold_omega)

        e72 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e72.addAddend(1.0, th1)
        e72.setScalar(-unsafe_threshold_th)

        e73 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e73.addAddend(1.0, th1)
        e73.setScalar(-unsafe_threshold_th+0.1)

        # unsafe tilt right

        e80 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e80.addAddend(1.0, omega1)
        e80.setScalar(0.0)

        e81 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e81.addAddend(1.0, omega1)
        e81.setScalar(unsafe_threshold_omega)

        e82 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e82.addAddend(1.0, th1)
        e82.setScalar(unsafe_threshold_th)

        e83 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e83.addAddend(1.0, th1)
        e83.setScalar(unsafe_threshold_th-0.1)


        # maps to either not decreasing or in unsafe region
        constraints = [[e1], [e10], [e11], [e12], [e13], [e14], [e15], [e16], [e17], [e18], [e19], [e20], [e21], [
            e30, e31, e32, e33], [e40, e41, e42, e43], [e50, e51, e52, e53], [e60, e61, e62, e63], [e70, e71, e72, e73], [e80, e81, e82, e83]]

        #force that we end up outside the unsafe region
        #ipq = network.getMarabouQuery()


         # counterexamples should be outside of the goal region

         # initial state
        e90 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e90.addAddend(1.0, px0)
        e90.setScalar(-0.1-1e-3)

        e91 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e91.addAddend(1.0, pz0)
        e91.setScalar(-0.1-1e-3)

        e92 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e92.addAddend(1.0, px0)
        e92.setScalar(0.1+1e-3)

        e93 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e93.addAddend(1.0, pz0)
        e93.setScalar(0.1+1e-3)

        e94 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e94.addAddend(1.0, th0)
        e94.setScalar(-0.1-1e-3)

        e95 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e95.addAddend(1.0, th0)
        e95.setScalar(0.1+1e-3)

        e96 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e96.addAddend(1.0, vx0)
        e96.setScalar(-0.2-1e-3)

        e97 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e97.addAddend(1.0, vz0)
        e97.setScalar(-0.2-1e-3)

        e98 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e98.addAddend(1.0, vx0)
        e98.setScalar(0.2+1e-3)

        e99 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e99.addAddend(1.0, vz0)
        e99.setScalar(0.2+1e-3)

        e100 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e100.addAddend(1.0, omega0)
        e100.setScalar(-0.2-1e-3)

        e101 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e101.addAddend(1.0, omega0)
        e101.setScalar(0.2+1e-3)

        # next state
        e110 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e110.addAddend(1.0, px1)
        e110.setScalar(-0.1-1e-3)

        e111 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e111.addAddend(1.0, pz1)
        e111.setScalar(-0.1-1e-3)

        e112 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e112.addAddend(1.0, px1)
        e112.setScalar(0.1+1e-3)

        e113 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e113.addAddend(1.0, pz1)
        e113.setScalar(0.1+1e-3)

        e114 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e114.addAddend(1.0, th1)
        e114.setScalar(-0.1-1e-3)

        e115 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e115.addAddend(1.0, th1)
        e115.setScalar(0.1+1e-3)

        e116 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e116.addAddend(1.0, vx1)
        e116.setScalar(-0.2-1e-3)

        e117 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e117.addAddend(1.0, vz1)
        e117.setScalar(-0.2-1e-3)

        e118 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e118.addAddend(1.0, vx1)
        e118.setScalar(0.2+1e-3)

        e119 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e119.addAddend(1.0, vz1)
        e119.setScalar(0.2+1e-3)

        e120 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e120.addAddend(1.0, omega1)
        e120.setScalar(-0.2-1e-3)

        e121 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e121.addAddend(1.0, omega1)
        e121.setScalar(0.2+1e-3)


        network.addDisjunctionConstraint([[e90], [e91], [e92], [e93], [e94], [e95], [e96], [e97], [e98], [e99], [e100], [e101]])
        network.addDisjunctionConstraint([[e110], [e111], [e112], [e113], [e114], [e115], [e116], [e117], [e118], [e119], [e120], [e121]])
        # force that we end up outside the unsafe region(not in unsafe region)
        network.addDisjunctionConstraint(constraints)

        #network.saveQuery("test.ipq")

        exitCode, vals, stats = network.solve(options=options, verbose=True)

        end = datetime.now()
        diff = end - start

        #if (diff.seconds > 120):
         #   network.saveQuery(str(input) + "_descent.ipq")
        
        if exitCode == "sat":
            print("Sat time taken: ", diff.seconds, "seconds")
            # print("theta1, thetadot1", vals[th_1], vals[thdot_1])
            # print("action", vals[action])
            # print("sin_lower", vals[sin_lower])
            # print("sin_upper", vals[sin_upper])
            # print("newthdot_lower", vals[newthdot_lower])
            # print("newthdot_upper", vals[newthdot_upper])
            print("Satisfying assignment ending at",
                  vals[px0], vals[pz0], vals[th0], vals[vx0], vals[vz0], vals[omega0],
                  vals[px1], vals[pz1], vals[th1], vals[vx1], vals[vz1], vals[omega1], vals[init_val], vals[out_val])
            # if (abs(vals[th_t]) <= unsafe_threshold_pos and abs(vals[thdot_t]) <= unsafe_threshold_pos and vals[init_val] - vals[out_val] >= 0.0000001 and vals[init_val] <= 1):
            #     print(unsafe_threshold_pos)
            # with open('sperious.txt','a') as f:
            #     f.write(str(input) + ' ' + str(vals[x_0]) + ' ' + str(vals[y_0]) + ' ' + str(vals[v_x_0]) + ' ' + str(vals[v_y_0]) + ' ' + str(vals[th_t]) + ' ' + str(vals[thdot_t]) + ' ' + str(vals[v_th_t]) + ' ' + str(vals[v_thdot_t]) + ' ' + str(vals[init_val]) + ' ' + str(vals[out_val]) + '\n')
            return [vals[px0], vals[pz0], vals[th0], vals[vx0], vals[vz0], vals[omega0]]
        if exitCode == "unsat":
            print("Inductive proof completed in", diff.seconds, "seconds")
            return [1]
        else:
            print("Failed in", diff.seconds, "seconds")
            return [-1]

    def check_safe_region(self, docking_threshold_pos, input=[[-3, -2], [-3, -2], [-0.5, -0.3], [-0.5, -0.3]]):
        start = datetime.now()
        useMILP = True

        options = Marabou.createOptions(verbosity=0, numWorkers=16, solveWithMILP=useMILP, snc=False)
        network = Marabou.read_onnx(self.cert_model)

        '''
        x_0, y_0, v_x_0, v_y_0, x_1, y_1, v_x_1, v_y_1, x_t, y_t, v_x_t, v_y_t, init_val, out_val = self.run_unroll(
            network)
        '''

        inp_state = network.inputVars[0][0]
        lyap_value = network.outputVars[0][0]

        px0 = inp_state[0]
        pz0 = inp_state[1]
        th0 = inp_state[2]
        vx0 = inp_state[3]
        vz0 = inp_state[4]
        omega0 = inp_state[5]

        init_val = lyap_value[0]

        network.setLowerBound(px0, input[0][0])
        network.setUpperBound(px0, input[0][1])
        network.setLowerBound(pz0, input[1][0])
        network.setUpperBound(pz0, input[1][1])
        network.setLowerBound(th0, input[2][0])
        network.setUpperBound(th0, input[2][1])
        network.setLowerBound(vx0, input[3][0])
        network.setUpperBound(vx0, input[3][1])
        network.setLowerBound(vz0, input[4][0])
        network.setUpperBound(vz0, input[4][1])
        network.setLowerBound(omega0, input[5][0])
        network.setUpperBound(omega0, input[5][1])

        # check if for safe range if its the case that init val >= 1
        network.setLowerBound(init_val, 1)
        # network.addInequality([init_val], [-1], -1.01)

        exitCode, vals, stats = network.solve(options=options, verbose=True)

        end = datetime.now()
        diff = end - start

        if (diff.seconds > 120):
            network.saveQuery(str(input) + "_safe" + ".ipq")

        if exitCode == "sat":
            print("Sat time taken: ", diff.seconds, "seconds")
            print("Satisfying assignment ending at",
                  vals[px0], vals[pz0], vals[th0], vals[vx0], vals[vz0], vals[omega0], vals[init_val])
            return [vals[px0], vals[pz0], vals[th0], vals[vx0], vals[vz0], vals[omega0]]
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
    
    def encode_piecewise_sin_bounds(self, network, th):
        # Each segment: (coefficient, slope, breakpoint1, breakpoint2) for upper and lower
        def add_segments(network, th, s):
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
                term = add_segments(network, th, s)
                outputs.append(term)
            result = network.getNewVariable()
            network.addEquality([result] + outputs, [1] +
                                [-1]*len(outputs), 0, False)
            return result

        upper = apply_segments(upper_segments)
        lower = apply_segments(lower_segments)

        return upper, lower

    '''
    def check_unsafe(self, vel, input, timeout):
        start = time.perf_counter()

        #first do unrolls with initialized network
        #solvewithMILP var
        useMILP = False

        options = Marabou.createOptions(verbosity=0, timeoutInSeconds=timeout, numWorkers=10, solveWithMILP=useMILP)

        network = Marabou.read_onnx(self.networkName)

        x_0, y_0, v_x_0, v_y_0, x_t, y_t, v_x_t, v_y_t = self.run_unrolls(network, 1)

        #now run the query
        x_lb = input[0][0]
        x_ub = input[0][1]
        y_lb = input[1][0]
        y_ub = input[1][1]

        v_x_lb = vel[0]
        v_x_ub = vel[1]
        v_y_lb = vel[0]
        v_y_ub = vel[1]

        # INPUT BOUNDING #
        network.setLowerBound(x_0, x_lb)
        network.setUpperBound(x_0, x_ub)
        network.setLowerBound(y_0, y_lb)
        network.setUpperBound(y_0, y_ub)
        network.setLowerBound(v_x_0, v_x_lb)
        network.setUpperBound(v_x_0, v_x_ub)
        network.setLowerBound(v_y_0, v_y_lb)
        network.setUpperBound(v_y_0, v_y_ub)

        e1 = MarabouCore.Equation(MarabouCore.Equation.LE)
        e1.addAddend(1.0, v_x_t)
        e1.setScalar(vel[0])

        e2 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e2.addAddend(1.0, v_x_t)
        e2.setScalar(vel[1])

        e3 = MarabouCore.Equation(MarabouCore.Equation.LE)
        e3.addAddend(1.0, v_y_t)
        e3.setScalar(vel[0])

        e4 = MarabouCore.Equation(MarabouCore.Equation.GE)
        e4.addAddend(1.0, v_y_t)
        e4.setScalar(vel[1])

        network.addDisjunctionConstraint([[e1],[e2],[e3],[e4]])

        print("Attempting proof for input bounds",
              [[x_lb, x_ub], [y_lb, y_ub], [v_x_lb, v_x_ub], [v_y_lb, v_y_ub]])

        exitCode, vals, stats = network.solve(options=options, verbose=True)

        if exitCode == "sat":
            print("Satisfying assignment ending at", vals[x_t], vals[y_t], vals[v_x_t], vals[v_y_t])
            return 0
        if exitCode == "unsat":
            print("Inductive proof completed in", time.perf_counter() - start, "seconds")
            return 1
        else:
            print("Failed in", time.perf_counter() - start, "seconds")
            return -1
    '''

    def encodeDifferences(network, inp, out):
        if len(inp) != len(out):
            raise ValueError("inp and out must have the same length")

        outvars = []
        for xi, yi in zip(inp, out):
            d = network.getNewVariable()
            # d = xi - yi  ==>  d - xi + yi = 0
            network.addEquality(vars=[d, xi, yi], coeffs=[1.0, -1.0, 1.0], scalar=0.0)

            absd = network.getNewVariable()
            network.addAbsConstraint(d, absd)
            outvars.append(absd)

        return outvars


    def run_unroll(self, network):
        # INITIALIZATION

        first_state = network.inputVars[0][0]
        second_state = network.inputVars[1][0]
        dyn_input = network.inputVars[2][0]

        first_output = network.outputVars[0][0]
        second_output = network.outputVars[1][0]
        third_output = network.outputVars[2][0]
        dyn_output = network.outputVars[3][0]


        px0, pz0, th0, vx0, vz0, omega0 = first_state[0], first_state[1], first_state[2], first_state[3], first_state[4], first_state[5]
        px1, pz1, th1, vx1, vz1, omega1 = second_state[0], second_state[1], second_state[2], second_state[3], second_state[4], second_state[5]

        u1, u2, init_val, out_val = first_output[0], first_output[1], second_output[0], third_output[0]
        px1_d, pz1_d, th1_d, vx1_d, vz1_d, omega1_d = dyn_output[0], dyn_output[1], dyn_output[2], dyn_output[3], dyn_output[4], dyn_output[5]

        px0_d, pz0_d, th0_d, vx0_d, vz0_d, omega0_d, u1_d, u2_d = dyn_input[0], dyn_input[1], dyn_input[2], dyn_input[3], dyn_input[4], dyn_input[5], dyn_input[6], dyn_input[7]


        # NN output: u = 2 * clip(action, -1, 1)
        action1 = self.scale_clip(network, u1, 0, 8, scale=1.0)
        action2 = self.scale_clip(network, u2, 0, 8, scale=1.0)

        # Get sin(th) bounds via piecewise ReLU approximation (v_11 and v_12)
        #sin_upper, sin_lower = self.encode_piecewise_sin_bounds(network, th_0)
        # sin_upper = network.getNewVariable()
        # network.addEquality([sin_upper, th_0], [1, -1], 0, False)
        # sin_lower = network.getNewVariable()
        # network.addEquality([sin_lower, th_0], [1, -0.5], 0, False)


        # Create linear expressions for lower and upper bounds of newthdot
        # newthdot_lower = network.getNewVariable()

        # newthdot_upper = network.getNewVariable()


        # Now declare the newthdot variable and bound it
        # newthdot = network.getNewVariable()

        # Clip newthdot to max speed

        # Compute newth = th + newthdot_clipped * dt

        newth = network.getNewVariable()

        
        # Encode that the inputs for the dynamic are the same as the first state and the output forces of the first state
        network.addEquality([px0, px0_d], [1, -1], 0, False)
        network.addEquality([pz0, pz0_d], [1, -1], 0, False)
        network.addEquality([th0, th0_d], [1, -1], 0, False)
        network.addEquality([vx0, vx0_d], [1, -1], 0, False)
        network.addEquality([vz0, vz0_d], [1, -1], 0, False)
        network.addEquality([omega0, omega0_d], [1, -1], 0, False)
        network.addEquality([action1, u1_d], [1, -1], 0, False)
        network.addEquality([action2, u2_d], [1, -1], 0, False)

        # Encode robustness, possible error, no robustness if perturbation is 0

        d_px, d_pz, d_th, d_vx, d_vz, d_omega = encodeDifferences(network, [px1, pz1, th1, vx1, vz1, omega1], [px1_d, pz1_d, th1_d, vx1_d, vz1_d, omega1_d])

        max_perturbation = 0

        network.setUpperBound(d_px, max_perturbation)
        network.setUpperBound(d_pz, max_perturbation)
        network.setUpperBound(d_th, max_perturbation)
        network.setUpperBound(d_vx, max_perturbation)
        network.setUpperBound(d_vz, max_perturbation)
        network.setUpperBound(d_omega, max_perturbation)


        # Clamp newth to [-0.7, 0.7]
        # th_t = self.clamp(network, newth, -0.7, 0.7)
        # thdot_t = self.clamp(network, newthdot_clipped, -0.7, 0.7)

        return px0, pz0, th0, vx0, vz0, omega0, px1, pz1, th1, vx1, vz1, omega1, action1, action2, init_val, out_val

def safe_descent_cond_check(PATH_TO_ONNX, PATH_TO_CERT, limit_pos=1.2, safe_pos=1.0, docking_pos=0.1, docking_th=0.1, limit_th = 0.6*math.pi, safe_th=0.45*math.pi-0.1, limit_vel=1.5, limit_omega=0.9):
    vals = []
    failed_vals = []
    val_ranges = []

    queries = Queries(combined_model=PATH_TO_ONNX, cert_model=PATH_TO_CERT)
    a = 5

    excluding_space_r = np.linspace(docking_pos, limit_pos, a)
    excluding_space_l = np.linspace(-limit_pos, -docking_pos, a)

    excluding_space_r_th = np.linspace(docking_th, limit_th, a)
    excluding_space_l_th = np.linspace(-limit_th, -docking_th, a)

    default_space = np.linspace(-limit_pos, limit_pos, a)
    default_space_th = np.linspace(-limit_th, limit_th, a)

    velocity_space = np.linspace(-limit_vel, limit_vel, a)
    omega_space = np.linspace(-limit_omega, limit_omega, a)


    safe_space = np.linspace(-safe_pos, safe_pos, a)
    safe_space_r = np.linspace(docking_pos, safe_pos, a)
    safe_space_l = np.linspace(-safe_pos, -docking_pos, a)

    safe_space_th = np.linspace(-safe_th, safe_th, a)
    safe_space_r_th = np.linspace(docking_th, safe_th, a)
    safe_space_l_th = np.linspace(-safe_th, -docking_th, a)

    # m = 5
    # left_top_th = np.linspace(-0.7, -0.6, m)
    # left_top_thdot = np.linspace(0, 0.7, m)
    # right_bottom_th = np.linspace(0.6, 0.7, m)
    # right_bottom_thdot = np.linspace(-0.7, 0, m)
    # # check descent and lip for these spaces
    # for i in range(m-1):
    #     ans = queries.check_descent_safe(limit_pos, docking_pos, [[round(left_top_th[i],2),round(left_top_th[i+1],2)]])
    # ans = queries.check_descent_safe(limit_pos, docking_pos, [[np.float64(
    #     -0.32), np.float64(-0.2)], [np.float64(0.0), np.float64(0.35)]])
    # exit()
    for i in range(a-1):
        for j in range(a-1):
            for k in range(a-1):
                for l in range(a-1):
                    #inside x and z for docking region, but outside docking region for th
                    ans = queries.check_descent_safe(limit_pos, limit_th, limit_vel, limit_omega, docking_pos, [[-docking_pos, docking_pos], [-docking_pos, docking_pos], [round(excluding_space_l_th[i], 2), round(excluding_space_l_th[i + 1], 2)],
                                                        [round(velocity_space[j], 2), round(velocity_space[j + 1], 2)], [round(velocity_space[k], 2), round(velocity_space[k + 1], 2)], [round(omega_space[l], 2), round(omega_space[l + 1], 2)]])
                    if len(ans) == 6:
                        vals.append(ans)
                        val_ranges.append([[-docking_pos, docking_pos], [-docking_pos, docking_pos], [round(excluding_space_l_th[i], 2), round(excluding_space_l_th[i + 1], 2)],
                                                        [round(velocity_space[j], 2), round(velocity_space[j + 1], 2)], [round(velocity_space[k], 2), round(velocity_space[k + 1], 2)], [round(omega_space[l], 2), round(omega_space[l + 1], 2)]])
                    elif ans[0] == -1:
                        failed_vals.append(ans)

                    ans = queries.check_descent_safe(limit_pos, limit_th, limit_vel, limit_omega, docking_pos, [[-docking_pos, docking_pos], [-docking_pos, docking_pos], [round(excluding_space_r_th[i], 2), round(excluding_space_r_th[i + 1], 2)],
                                                        [round(velocity_space[j], 2), round(velocity_space[j + 1], 2)], [round(velocity_space[k], 2), round(velocity_space[k + 1], 2)], [round(omega_space[l], 2), round(omega_space[l + 1], 2)]])

                    if len(ans) == 6:
                        vals.append(ans)
                        val_ranges.append([[-docking_pos, docking_pos], [-docking_pos, docking_pos], [round(excluding_space_r_th[i], 2), round(excluding_space_r_th[i + 1], 2)],
                                                        [round(velocity_space[j], 2), round(velocity_space[j + 1], 2)], [round(velocity_space[k], 2), round(velocity_space[k + 1], 2)], [round(omega_space[l], 2), round(omega_space[l + 1], 2)]])
                    elif ans[0] == -1:
                        failed_vals.append(ans)

                    #inside x and th for docking region, but outside docking region for z

                    ans = queries.check_descent_safe(limit_pos, limit_th, limit_vel, limit_omega, docking_pos, [[-docking_pos, docking_pos], [round(excluding_space_l[i], 2), round(excluding_space_l[i + 1], 2)], [-docking_pos, docking_pos],
                                                        [round(velocity_space[j], 2), round(velocity_space[j + 1], 2)], [round(velocity_space[k], 2), round(velocity_space[k + 1], 2)], [round(omega_space[l], 2), round(omega_space[l + 1], 2)]])
                    if len(ans) == 6:
                        vals.append(ans)
                        val_ranges.append([[-docking_pos, docking_pos], [round(excluding_space_l[i], 2), round(excluding_space_l[i + 1], 2)], [-docking_pos, docking_pos],
                                                        [round(velocity_space[j], 2), round(velocity_space[j + 1], 2)], [round(velocity_space[k], 2), round(velocity_space[k + 1], 2)], [round(omega_space[l], 2), round(omega_space[l + 1], 2)]])
                    elif ans[0] == -1:
                        failed_vals.append(ans)

                    ans = queries.check_descent_safe(limit_pos, limit_th, limit_vel, limit_omega, docking_pos, [[-docking_pos, docking_pos], [round(excluding_space_r[i], 2), round(excluding_space_r[i + 1], 2)], [-docking_pos, docking_pos],
                                                        [round(velocity_space[j], 2), round(velocity_space[j + 1], 2)], [round(velocity_space[k], 2), round(velocity_space[k + 1], 2)], [round(omega_space[l], 2), round(omega_space[l + 1], 2)]])

                    if len(ans) == 6:
                        vals.append(ans)
                        val_ranges.append([[-docking_pos, docking_pos], [round(excluding_space_r[i], 2), round(excluding_space_r[i + 1], 2)], [-docking_pos, docking_pos],
                                                        [round(velocity_space[j], 2), round(velocity_space[j + 1], 2)], [round(velocity_space[k], 2), round(velocity_space[k + 1], 2)], [round(omega_space[l], 2), round(omega_space[l + 1], 2)]])
                    elif ans[0] == -1:
                        failed_vals.append(ans)
                    
                    # inside x for docking region, but outside docking region for z and th

                    for m in range(a-1):
                        # both z and th left
                        ans = queries.check_descent_safe(limit_pos, limit_th, limit_vel, limit_omega, docking_pos, [[-docking_pos, docking_pos], [round(excluding_space_l[i], 2), round(excluding_space_l[i + 1], 2)], [round(excluding_space_l_th[j], 2), round(excluding_space_l_th[j + 1], 2)],
                                                            [round(velocity_space[k], 2), round(velocity_space[k + 1], 2)], [round(velocity_space[l], 2), round(velocity_space[l + 1], 2)], [round(omega_space[m], 2), round(omega_space[m + 1], 2)]])
                        if len(ans) == 6:
                            vals.append(ans)
                            val_ranges.append([[-docking_pos, docking_pos], [round(excluding_space_l[i], 2), round(excluding_space_l[i + 1], 2)], [round(excluding_space_l_th[j], 2), round(excluding_space_l_th[j + 1], 2)],
                                                            [round(velocity_space[k], 2), round(velocity_space[k + 1], 2)], [round(velocity_space[l], 2), round(velocity_space[l + 1], 2)], [round(omega_space[m], 2), round(omega_space[m + 1], 2)]])
                        elif ans[0] == -1:
                            failed_vals.append(ans)
                        
                        # both z and th right
                        ans = queries.check_descent_safe(limit_pos, limit_th, limit_vel, limit_omega, docking_pos, [[-docking_pos, docking_pos], [round(excluding_space_r[i], 2), round(excluding_space_r[i + 1], 2)], [round(excluding_space_r_th[j], 2), round(excluding_space_r_th[j + 1], 2)],
                                                            [round(velocity_space[k], 2), round(velocity_space[k + 1], 2)], [round(velocity_space[l], 2), round(velocity_space[l + 1], 2)], [round(omega_space[m], 2), round(omega_space[m + 1], 2)]])
                        if len(ans) == 6:
                            vals.append(ans)
                            val_ranges.append([[-docking_pos, docking_pos], [round(excluding_space_r[i], 2), round(excluding_space_r[i + 1], 2)], [round(excluding_space_r_th[j], 2), round(excluding_space_r_th[j + 1], 2)],
                                                            [round(velocity_space[k], 2), round(velocity_space[k + 1], 2)], [round(velocity_space[l], 2), round(velocity_space[l + 1], 2)], [round(omega_space[m], 2), round(omega_space[m + 1], 2)]])
                        elif ans[0] == -1:
                            failed_vals.append(ans)

                        # z left and th right
                        ans = queries.check_descent_safe(limit_pos, limit_th, limit_vel, limit_omega, docking_pos, [[-docking_pos, docking_pos], [round(excluding_space_l[i], 2), round(excluding_space_l[i + 1], 2)], [round(excluding_space_r_th[j], 2), round(excluding_space_r_th[j + 1], 2)],
                                                            [round(velocity_space[k], 2), round(velocity_space[k + 1], 2)], [round(velocity_space[l], 2), round(velocity_space[l + 1], 2)], [round(omega_space[m], 2), round(omega_space[m + 1], 2)]])
                        if len(ans) == 6:
                            vals.append(ans)
                            val_ranges.append([[-docking_pos, docking_pos], [round(excluding_space_l[i], 2), round(excluding_space_l[i + 1], 2)], [round(excluding_space_r_th[j], 2), round(excluding_space_r_th[j + 1], 2)],
                                                            [round(velocity_space[k], 2), round(velocity_space[k + 1], 2)], [round(velocity_space[l], 2), round(velocity_space[l + 1], 2)], [round(omega_space[m], 2), round(omega_space[m + 1], 2)]])
                        elif ans[0] == -1:
                            failed_vals.append(ans)

                        # z right and th left
                        ans = queries.check_descent_safe(limit_pos, limit_th, limit_vel, limit_omega, docking_pos, [[-docking_pos, docking_pos], [round(excluding_space_r[i], 2), round(excluding_space_r[i + 1], 2)], [round(excluding_space_l_th[j], 2), round(excluding_space_l_th[j + 1], 2)],
                                                            [round(velocity_space[k], 2), round(velocity_space[k + 1], 2)], [round(velocity_space[l], 2), round(velocity_space[l + 1], 2)], [round(omega_space[m], 2), round(omega_space[m + 1], 2)]])
                        if len(ans) == 6:
                            vals.append(ans)
                            val_ranges.append([[-docking_pos, docking_pos], [round(excluding_space_r[i], 2), round(excluding_space_r[i + 1], 2)], [round(excluding_space_l_th[j], 2), round(excluding_space_l_th[j + 1], 2)],
                                                            [round(velocity_space[k], 2), round(velocity_space[k + 1], 2)], [round(velocity_space[l], 2), round(velocity_space[l + 1], 2)], [round(omega_space[m], 2), round(omega_space[m + 1], 2)]])
                        elif ans[0] == -1:
                            failed_vals.append(ans)

  
                    # outside x for docking region, all z and th
                
                    for m in range(a-1):
                        for n in range(a-1):
                        
                            ans = queries.check_descent_safe(limit_pos, limit_th, limit_vel, limit_omega, docking_pos, [[round(excluding_space_l[i], 2), round(excluding_space_l[i + 1], 2)], [round(default_space[j], 2), round(default_space[j + 1], 2)], [round(default_space_th[k], 2), round(default_space_th[k + 1], 2)],
                                                                [round(velocity_space[l], 2), round(velocity_space[l + 1], 2)], [round(velocity_space[m], 2), round(velocity_space[m + 1], 2)], [round(omega_space[n], 2), round(omega_space[n + 1], 2)]])
                            if len(ans) == 6:
                                vals.append(ans)
                                val_ranges.append([[round(excluding_space_l[i], 2), round(excluding_space_l[i + 1], 2)], [round(default_space[j], 2), round(default_space[j + 1], 2)], [round(default_space_th[k], 2), round(default_space_th[k + 1], 2)],
                                                                [round(velocity_space[l], 2), round(velocity_space[l + 1], 2)], [round(velocity_space[m], 2), round(velocity_space[m + 1], 2)], [round(omega_space[n], 2), round(omega_space[n + 1], 2)]])

                            elif ans[0] == -1:
                                failed_vals.append(ans)

                            ans = queries.check_descent_safe(limit_pos, limit_th, limit_vel, limit_omega, docking_pos, [[round(excluding_space_r[i], 2), round(excluding_space_r[i + 1], 2)], [round(default_space[j], 2), round(default_space[j + 1], 2)], [round(default_space_th[k], 2), round(default_space_th[k + 1], 2)],
                                                                [round(velocity_space[l], 2), round(velocity_space[l + 1], 2)], [round(velocity_space[m], 2), round(velocity_space[m + 1], 2)], [round(omega_space[n], 2), round(omega_space[n + 1], 2)]])
                            if len(ans) == 6:
                                vals.append(ans)
                                val_ranges.append([[round(excluding_space_l[i], 2), round(excluding_space_l[i + 1], 2)], [round(default_space[j], 2), round(default_space[j + 1], 2)], [round(default_space_th[k], 2), round(default_space_th[k + 1], 2)],
                                                                [round(velocity_space[l], 2), round(velocity_space[l + 1], 2)], [round(velocity_space[m], 2), round(velocity_space[m + 1], 2)], [round(omega_space[n], 2), round(omega_space[n + 1], 2)]])
                            elif ans[0] == -1:
                                failed_vals.append(ans)

        # # check for lip
        # ans = queries.check_local_lip(
        #     [[-docking_pos, docking_pos], [round(excluding_space_l[i], 2), round(excluding_space_l[i + 1], 2)]])
        # if len(ans) == 6:
        #     vals.append(ans)
        #     val_ranges.append(
        #         [[-docking_pos, docking_pos], [round(excluding_space_l[i], 2), round(excluding_space_l[i + 1], 2)]])
        #     print("lip", ans, val_ranges[-1])
        # elif ans[0] == -1:
        #     failed_vals.append(ans)

        # ans = queries.check_local_lip(
        #     [[-docking_pos, docking_pos], [round(excluding_space_r[i], 2), round(excluding_space_r[i + 1], 2)]])

        # if len(ans) == 6:
        #     vals.append(ans)
        #     val_ranges.append(
        #         [[-docking_pos, docking_pos], [round(excluding_space_r[i], 2), round(excluding_space_r[i + 1], 2)]])
        #     print("lip")
        # elif ans[0] == -1:
        #     failed_vals.append(ans)

            # check for lip
            # ans = queries.check_local_lip([[round(excluding_space_l[i], 2), round(
            #     excluding_space_l[i + 1], 2)], [round(default_space[j], 2), round(default_space[j + 1], 2)]])
            # if len(ans) == 2:
            #     vals.append(ans)
            #     val_ranges.append([[round(excluding_space_l[i], 2), round(
            #         excluding_space_l[i + 1], 2)], [round(default_space[j], 2), round(default_space[j + 1], 2)]])
            #     print("lip")
            # elif ans[0] == -1:
            #     failed_vals.append(ans)

            # ans = queries.check_local_lip([[round(excluding_space_r[i], 2), round(
            #     excluding_space_r[i + 1], 2)], [round(default_space[j], 2), round(default_space[j + 1], 2)]])
            # if len(ans) == 2:
            #     vals.append(ans)
            #     val_ranges.append([[round(excluding_space_r[i], 2), round(
            #         excluding_space_r[i + 1], 2)], [round(default_space[j], 2), round(default_space[j + 1], 2)]])
            #     print("lip")
            # elif ans[0] == -1:
            #     failed_vals.append(ans)

    print("safe====================safe")
    for i in range(a-1):
        # x and z in docking, th safe
        ans = queries.check_safe_region(docking_pos, [[-docking_pos, docking_pos], [-docking_pos, docking_pos], [round(safe_space_l_th[i], 2), round(safe_space_l_th[i + 1], 2)], [0,0], [0,0], [0,0]])
        if len(ans) == 6:
            vals.append(ans)
            val_ranges.append([[-docking_pos, docking_pos], [-docking_pos, docking_pos], [round(safe_space_l_th[i], 2), round(safe_space_l_th[i + 1], 2)], [0,0], [0,0], [0,0]])
        elif ans[0] == -1:
            failed_vals.append(ans)

        ans = queries.check_safe_region(docking_pos, [[-docking_pos, docking_pos], [-docking_pos, docking_pos], [round(safe_space_r_th[i], 2), round(safe_space_r_th[i + 1], 2)], [0,0], [0,0], [0,0]])
        if len(ans) == 4:
            vals.append(ans)
            val_ranges.append([[-docking_pos, docking_pos], [-docking_pos, docking_pos], [round(safe_space_r_th[i], 2), round(safe_space_r_th[i + 1], 2)], [0,0], [0,0], [0,0]])
        elif ans[0] == -1:
            failed_vals.append(ans)

        # x and th in docking, z safe
        ans = queries.check_safe_region(docking_pos, [[-docking_pos, docking_pos], [round(safe_space_l[i], 2), round(safe_space_l[i + 1], 2)], [-docking_pos, docking_pos], [0,0], [0,0], [0,0]])
        if len(ans) == 6:
            vals.append(ans)
            val_ranges.append([[-docking_pos, docking_pos], [round(safe_space_l[i], 2), round(safe_space_l[i + 1], 2)], [-docking_pos, docking_pos], [0,0], [0,0], [0,0]])
        elif ans[0] == -1:
            failed_vals.append(ans)

        ans = queries.check_safe_region(docking_pos, [[-docking_pos, docking_pos], [round(safe_space_r[i], 2), round(safe_space_r[i + 1], 2)], [-docking_pos, docking_pos], [0,0], [0,0], [0,0]])
        if len(ans) == 4:
            vals.append(ans)
            val_ranges.append([[-docking_pos, docking_pos], [round(safe_space_r[i], 2), round(safe_space_r[i + 1], 2)], [-docking_pos, docking_pos], [0,0], [0,0], [0,0]])
        elif ans[0] == -1:
            failed_vals.append(ans)

        # x in docking, z and th safe

        for j in range(a-1):

            # z and th left
            ans = queries.check_safe_region(docking_pos, [[-docking_pos, docking_pos], [round(safe_space_l[i], 2), round(safe_space_l[i + 1], 2)], [round(safe_space_l_th[i], 2), round(safe_space_l_th[i + 1], 2)], [0,0], [0,0], [0,0]])
            if len(ans) == 6:
                vals.append(ans)
                val_ranges.append([[-docking_pos, docking_pos], [round(safe_space_l[i], 2), round(safe_space_l[i + 1], 2)], [round(safe_space_l_th[i], 2), round(safe_space_l_th[i + 1], 2)], [0,0], [0,0], [0,0]])
            elif ans[0] == -1:
                failed_vals.append(ans)

            # z and th right
            ans = queries.check_safe_region(docking_pos, [[-docking_pos, docking_pos], [round(safe_space_r[i], 2), round(safe_space_r[i + 1], 2)], [round(safe_space_r_th[i], 2), round(safe_space_r_th[i + 1], 2)], [0,0], [0,0], [0,0]])
            if len(ans) == 6:
                vals.append(ans)
                val_ranges.append([[-docking_pos, docking_pos], [round(safe_space_r[i], 2), round(safe_space_r[i + 1], 2)], [round(safe_space_r_th[i], 2), round(safe_space_r_th[i + 1], 2)], [0,0], [0,0], [0,0]])
            elif ans[0] == -1:
                failed_vals.append(ans)

            # z left th right
            ans = queries.check_safe_region(docking_pos, [[-docking_pos, docking_pos], [round(safe_space_l[i], 2), round(safe_space_l[i + 1], 2)], [round(safe_space_r_th[i], 2), round(safe_space_r_th[i + 1], 2)], [0,0], [0,0], [0,0]])
            if len(ans) == 6:
                vals.append(ans)
                val_ranges.append([[-docking_pos, docking_pos], [round(safe_space_l[i], 2), round(safe_space_l[i + 1], 2)], [round(safe_space_r_th[i], 2), round(safe_space_r_th[i + 1], 2)], [0,0], [0,0], [0,0]])
            elif ans[0] == -1:
                failed_vals.append(ans)

            # z right th left
            ans = queries.check_safe_region(docking_pos, [[-docking_pos, docking_pos], [round(safe_space_r[i], 2), round(safe_space_r[i + 1], 2)], [round(safe_space_l_th[i], 2), round(safe_space_l_th[i + 1], 2)], [0,0], [0,0], [0,0]])
            if len(ans) == 6:
                vals.append(ans)
                val_ranges.append([[-docking_pos, docking_pos], [round(safe_space_r[i], 2), round(safe_space_r[i + 1], 2)], [round(safe_space_l_th[i], 2), round(safe_space_l_th[i + 1], 2)], [0,0], [0,0], [0,0]])
            elif ans[0] == -1:
                failed_vals.append(ans)

        # x outside in safe, z and th anywhere

        for j in range(a-1):
            for k in range(a-1):
                ans = queries.check_safe_region(docking_pos, [[round(safe_space_l[i], 2), round(safe_space_l[i + 1], 2)], [round(safe_space[j], 2), round(safe_space[j + 1], 2)], [round(safe_space_th[k], 2), round(safe_space_th[k + 1], 2)], [0,0], [0,0], [0,0]])
                if len(ans) == 6:
                    vals.append(ans)
                    val_ranges.append([[round(safe_space_l[i], 2), round(safe_space_l[i + 1], 2)], [round(safe_space[j], 2), round(safe_space[j + 1], 2)], [round(safe_space_th[k], 2), round(safe_space_th[k + 1], 2)], [0,0], [0,0], [0,0]])
                elif ans[0] == -1:
                    failed_vals.append(ans)

                ans = queries.check_safe_region(docking_pos, [[round(safe_space_r[i], 2), round(safe_space_r[i + 1], 2)], [round(safe_space[j], 2), round(safe_space[j + 1], 2)], [round(safe_space_th[k], 2), round(safe_space_th[k + 1], 2)], [0,0], [0,0], [0,0]])
                if len(ans) == 6:
                    vals.append(ans)
                    val_ranges.append([[round(safe_space_r[i], 2), round(safe_space_r[i + 1], 2)], [round(safe_space[j], 2), round(safe_space[j + 1], 2)], [round(safe_space_th[k], 2), round(safe_space_th[k + 1], 2)], [0,0], [0,0], [0,0]])
                elif ans[0] == -1:
                    failed_vals.append(ans)

    with open('query_results.txt', 'a') as f:
        print(len(vals), file=f)

    return [vals, val_ranges, failed_vals]

if __name__ == "__main__":
    pos_limit = 0.7
    safe_pos = 0.3

    cert_file = "models/cert_0_veri.pt"
    controller_file = "controllers/controller_0_veri.pt"

    comb_file = "combined/combined_0_veri.onnx"
    model_onnx_file = "models/cert_0_veri.onnx"

    ret, ret_ranges, failed = safe_descent_cond_check(comb_file, model_onnx_file, safe_pos = safe_pos, limit_pos = pos_limit, docking_pos = 0.2)

    print(ret)
    print("Number of counterexamples: ", len(ret))
    print(failed)


