import os
from datetime import datetime
import numpy as np

import sys
sys.path.append("../../Marabou")

# import Marabou.maraboupy import Marabou
# from Marabou.maraboupy import MarabouCore
from maraboupy import Marabou
from maraboupy import MarabouCore, MarabouUtils

from convertsinglenetwork import single_model
#from training_exp_mask import LyapunovNetworkV, TwoDimDocking

# from maraboupy import MarabouCore
class Queries:

    def __init__(self, combined_model, cert_model):

        # self controller name
        self.combined_model = combined_model
        self.cert_model = cert_model

        # fixed m,n,t
        self.m = 12
        self.n = 0.001027
        self.t = 1

        # Matrix encoding of system dynamics
        self.coeffs_x_t = [
            4 - 3 * np.cos(self.n * self.t),
            0,
            1 / self.n * np.sin(self.n * self.t),
            2 / self.n - 2 / self.n * np.cos(self.n * self.t),
            (1 - np.cos(self.n * self.t)) / (self.m * self.n ** 2),
            2 * self.t / (self.m * self.n) - 2 * np.sin(self.n * self.t) / (self.m * self.n ** 2),
            -1
        ]
        self.coeffs_y_t = [
            -6 * self.n * self.t + 6 * np.sin(self.n * self.t),
            1,
            -2 / self.n + 2 / self.n * np.cos(self.n * self.t),
            -3 * self.t + 4 / self.n * np.sin(self.n * self.t),
            (-2 * self.t) / (self.m * self.n) + (2 * np.sin(self.n * self.t)) / (self.m * self.n ** 2),
            4 / (self.m * self.n ** 2) - (3 * self.t ** 2) / (2 * self.m) - (4 * np.cos(self.n * self.t)) / (
                        self.m * self.n ** 2),
            -1
        ]
        self.coeffs_v_x_t = [
            3 * self.n * np.sin(self.n * self.t),
            0,
            np.cos(self.n * self.t),
            2 * np.sin(self.n * self.t),
            np.sin(self.n * self.t) / (self.m * self.n),
            2 / (self.m * self.n) - (2 * np.cos(self.n * self.t)) / (self.m * self.n),
            -1
        ]
        self.coeffs_v_y_t = [
            -6 * self.n + 6 * self.n * np.cos(self.n * self.t),
            0,
            -2 * np.sin(self.n * self.t),
            -3 + 4 * np.cos(self.n * self.t),
            (2 * np.cos(self.n * self.t) - 2) / (self.m * self.n),
            (-3 * self.t) / (self.m) + (4 * np.sin(self.n * self.t)) / (self.m * self.n),
            -1
        ]

    def check_descent_safe(self, unsafe_threshold_pos, unsafe_threshold_vel, docking_threshold_pos, input=[[-3, -2], [-3, -2], [-0.5, -0.3], [-0.5, -0.3]]):
        start = datetime.now()
        useMILP = True

        options = Marabou.createOptions(verbosity=0, numWorkers=16, solveWithMILP=useMILP, snc=False) #remove tighteningStrategy
        network = Marabou.read_onnx(self.combined_model)

        # print(network.inputVars)
        # print(network.outputVars)

        x_0, y_0, v_x_0, v_y_0, x_1, y_1, v_x_1, v_y_1, x_t, y_t, v_x_t, v_y_t, init_val, out_val = self.run_unroll(
            network)


        network.setLowerBound(x_0, input[0][0])
        network.setUpperBound(x_0, input[0][1])
        network.setLowerBound(y_0, input[1][0])
        network.setUpperBound(y_0, input[1][1])
        network.setLowerBound(v_x_0, input[2][0])
        network.setUpperBound(v_x_0, input[2][1])
        network.setLowerBound(v_y_0, input[3][0])
        network.setUpperBound(v_y_0, input[3][1])

        network.addEquality([x_t, x_1], [1, -1], 0, False)
        network.addEquality([y_t, y_1], [1, -1], 0, False)
        network.addEquality([v_x_t, v_x_1], [1, -1], 0, False)
        network.addEquality([v_y_t, v_y_1], [1, -1], 0, False)

        #force that when cert <=1
        network.setUpperBound(init_val, 1)

        #checking descending condition.
        e1 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e1.addAddend(1.0, init_val)
        e1.addAddend(-1.0, out_val)
        e1.setScalar(0.0000001)

        #force that the cert value decreases
        #network.addEquation(e1)

        #called with input preprocessing ensuring we are outside the docking region + not in unsafe region
        #enforce that we do not enter unsafe region otherwise.

        #checking that we are outside unsafe region on next step
        e6 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e6.addAddend(1.0, v_x_t)
        e6.setScalar(unsafe_threshold_vel)

        e7 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e7.addAddend(1.0, v_y_t)
        e7.setScalar(unsafe_threshold_vel)

        e8 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e8.addAddend(1.0, v_x_t)
        e8.setScalar(-unsafe_threshold_vel)

        e9 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e9.addAddend(1.0, v_y_t)
        e9.setScalar(-unsafe_threshold_vel)

        e10 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e10.addAddend(1.0, x_t)
        e10.setScalar(unsafe_threshold_pos)

        e11 = MarabouUtils.Equation(MarabouCore.Equation.GE)
        e11.addAddend(1.0, y_t)
        e11.setScalar(unsafe_threshold_pos)

        e12 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e12.addAddend(1.0, x_t)
        e12.setScalar(-unsafe_threshold_pos)

        e13 = MarabouUtils.Equation(MarabouCore.Equation.LE)
        e13.addAddend(1.0, y_t)
        e13.setScalar(-unsafe_threshold_pos)

        #maps to either not decreasing or in unsafe region
        constraints = [[e1], [e6],[e7],[e8],[e9],[e10],[e11],[e12],[e13]]
        #force that we end up outside the unsafe region
        #ipq = network.getMarabouQuery()
        network.addDisjunctionConstraint(constraints)

        exitCode, vals, stats = network.solve(options=options, verbose=True)

        end = datetime.now()
        diff = end - start

        if (diff.seconds > 120):
            network.saveQuery(str(input) + "_descent.ipq")

        if exitCode == "sat":
            print("Sat time taken: ", diff.seconds, "seconds")
            print("Satisfying assignment ending at", vals[x_t], vals[y_t], vals[v_x_t], vals[v_y_t], vals[out_val])
            return [vals[x_0],vals[y_0],vals[v_x_0],vals[v_y_0]]
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

        x_0 = inp_state[0]
        y_0 = inp_state[1]
        v_x_0 = inp_state[2]
        v_y_0 = inp_state[3]

        init_val = lyap_value[0]

        network.setLowerBound(x_0, input[0][0])
        network.setUpperBound(x_0, input[0][1])
        network.setLowerBound(y_0, input[1][0])
        network.setUpperBound(y_0, input[1][1])
        network.setLowerBound(v_x_0, input[2][0])
        network.setUpperBound(v_x_0, input[2][1])
        network.setLowerBound(v_y_0, input[3][0])
        network.setUpperBound(v_y_0, input[3][1])

        '''
        network.addEquality([x_t, x_1], [1, -1], 0, False)
        network.addEquality([y_t, y_1], [1, -1], 0, False)
        network.addEquality([v_x_t, v_x_1], [1, -1], 0, False)
        network.addEquality([v_y_t, v_y_1], [1, -1], 0, False)
        '''

        #check if for safe range if its the case that init val >= 1
        network.setLowerBound(init_val, 1)

        exitCode, vals, stats = network.solve(options=options, verbose=True)

        end = datetime.now()
        diff = end - start

        if (diff.seconds > 120):
            network.saveQuery(str(input) + "_safe" + ".ipq")

        if exitCode == "sat":
            print("Sat time taken: ", diff.seconds, "seconds")
            print("Satisfying assignment ending at", vals[x_0], vals[y_0], vals[v_x_0], vals[v_y_0], vals[init_val])
            return [vals[x_0],vals[y_0],vals[v_x_0],vals[v_y_0]]
        if exitCode == "unsat":
            print("Inductive proof completed in", diff.seconds, "seconds")
            return [1]
        else:
            print("Failed in", diff.seconds, "seconds")
            return [-1]

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

    def run_unroll(self, network):
        # INITIALIZATION
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
        network.addEquality([aux1, F_x], [1, -1], -1 * clip_lb, False)  # aux1 = F_x-clip_lb
        aux2 = network.getNewVariable()
        network.addRelu(aux1, aux2)  # aux2 = relu(aux1)
        aux3 = network.getNewVariable()
        network.addEquality([aux3, aux2], [1, -1], clip_lb, False)  # aux3 = clip_lb + aux2
        aux4 = network.getNewVariable()
        network.addEquality([aux4, aux3], [-1, -1], -clip_ub, False)  # aux4 = clip_ub - aux3
        aux5 = network.getNewVariable()
        network.addRelu(aux4, aux5)  # aux5 = relu(aux4)
        network.addEquality([F_x_clip, aux5], [-1, -1], -clip_ub, False)  # F_x_clip = clip_ub - aux5

        # handling F_y_clipping
        aux1 = network.getNewVariable()
        F_y_clip = network.getNewVariable()
        network.addEquality([aux1, F_y], [1, -1], -1 * clip_lb, False)  # aux1 = F_y-clip_lb
        aux2 = network.getNewVariable()
        network.addRelu(aux1, aux2)  # aux2 = relu(aux1)
        aux3 = network.getNewVariable()
        network.addEquality([aux3, aux2], [1, -1], clip_lb, False)  # aux3 = clip_lb + aux2
        aux4 = network.getNewVariable()
        network.addEquality([aux4, aux3], [-1, -1], -clip_ub, False)  # aux4 = clip_ub - aux3
        aux5 = network.getNewVariable()
        network.addRelu(aux4, aux5)  # aux5 = relu(aux4)
        network.addEquality([F_y_clip, aux5], [-1, -1], -clip_ub, False)  # F_y_clip = clip_ub - aux5

        x_t = network.getNewVariable()
        y_t = network.getNewVariable()
        v_x_t = network.getNewVariable()
        v_y_t = network.getNewVariable()

        # System dynamics for final step
        vars_x_t = [x_0, y_0, v_x_0, v_y_0, F_x_clip, F_y_clip, x_t]
        network.addEquality(vars_x_t, self.coeffs_x_t, 0, False)

        vars_y_t = [x_0, y_0, v_x_0, v_y_0, F_x_clip, F_y_clip, y_t]
        network.addEquality(vars_y_t, self.coeffs_y_t, 0, False)

        vars_v_x_t = [x_0, y_0, v_x_0, v_y_0, F_x_clip, F_y_clip, v_x_t]
        network.addEquality(vars_v_x_t, self.coeffs_v_x_t, 0, False)

        vars_v_y_t = [x_0, y_0, v_x_0, v_y_0, F_x_clip, F_y_clip, v_y_t]
        network.addEquality(vars_v_y_t, self.coeffs_v_y_t, 0, False)

        return x_0, y_0, v_x_0, v_y_0, x_1, y_1, v_x_1, v_y_1, x_t, y_t, v_x_t, v_y_t, init_val, out_val

def safe_descent_cond_check(PATH_TO_ONNX, PATH_TO_CERT, limit_pos = 5, safe_pos = 4, docking_pos = 0.35, vel_limit = 0.5):
    vals = []
    failed_vals = []
    val_ranges = []

    queries = Queries(combined_model=PATH_TO_ONNX, cert_model=PATH_TO_CERT)

    excluding_space_r = np.linspace(docking_pos, limit_pos, 5)
    excluding_space_l = np.linspace(-limit_pos, -docking_pos, 5)
    default_space = np.linspace(-limit_pos, limit_pos, 5)
    velocity_space = np.linspace(-vel_limit, vel_limit, 5)

    safe_space = np.linspace(-safe_pos, safe_pos, 5)
    safe_space_r = np.linspace(docking_pos, safe_pos, 5)
    safe_space_l = np.linspace(-safe_pos, -docking_pos, 5)

    for i in range(4):
        for j in range(4):
            for k in range(4):
                #inside x for docking region, but outside docking region for y
                ans = queries.check_descent_safe(limit_pos, vel_limit, docking_pos, [[-docking_pos, docking_pos], [round(excluding_space_l[i], 2), round(excluding_space_l[i + 1], 2)],
                                                    [round(velocity_space[j], 2), round(velocity_space[j + 1], 2)], [round(velocity_space[k], 2), round(velocity_space[k + 1], 2)]])
                if len(ans) == 4:
                    vals.append(ans)
                    val_ranges.append([[-docking_pos, docking_pos], [round(excluding_space_l[i], 2), round(excluding_space_l[i + 1], 2)], [round(velocity_space[j], 2), round(velocity_space[j + 1], 2)], [round(velocity_space[k], 2), round(velocity_space[k + 1], 2)]])
                elif ans[0] == -1:
                    failed_vals.append(ans)

                ans = queries.check_descent_safe(limit_pos, vel_limit, docking_pos, [[-docking_pos, docking_pos], [round(excluding_space_r[i], 2), round(excluding_space_r[i + 1], 2)],
                                                    [round(velocity_space[j], 2), round(velocity_space[j + 1], 2)], [round(velocity_space[k], 2), round(velocity_space[k + 1], 2)]])

                if len(ans) == 4:
                    vals.append(ans)
                    val_ranges.append([[-docking_pos, docking_pos], [round(excluding_space_r[i], 2), round(excluding_space_r[i + 1], 2)], [round(velocity_space[j], 2), round(velocity_space[j + 1], 2)], [round(velocity_space[k], 2), round(velocity_space[k + 1], 2)]])
                elif ans[0] == -1:
                    failed_vals.append(ans)

                for l in range(4):
                    #outside x for docking region, all y
                    ans = queries.check_descent_safe(limit_pos, vel_limit, docking_pos, [[round(excluding_space_l[i], 2), round(excluding_space_l[i + 1], 2)], [round(default_space[j], 2), round(default_space[j + 1], 2)],
                                                        [round(velocity_space[k], 2), round(velocity_space[k + 1], 2)], [round(velocity_space[l], 2), round(velocity_space[l + 1], 2)]])
                    if len(ans) == 4:
                        vals.append(ans)
                        val_ranges.append([[round(excluding_space_l[i], 2), round(excluding_space_l[i + 1], 2)], [round(default_space[j], 2), round(default_space[j + 1], 2)], [round(velocity_space[k], 2), round(velocity_space[k + 1], 2)], [round(velocity_space[l], 2), round(velocity_space[l + 1], 2)]])

                    elif ans[0] == -1:
                        failed_vals.append(ans)

                    ans = queries.check_descent_safe(limit_pos, vel_limit, docking_pos, [[round(excluding_space_r[i], 2), round(excluding_space_r[i + 1], 2)], [round(default_space[j], 2), round(default_space[j + 1], 2)],
                                                        [round(velocity_space[k], 2), round(velocity_space[k + 1], 2)], [round(velocity_space[l], 2), round(velocity_space[l + 1], 2)]])
                    if len(ans) == 4:
                        vals.append(ans)
                        val_ranges.append([[round(excluding_space_r[i], 2), round(excluding_space_r[i + 1], 2)], [round(default_space[j], 2), round(default_space[j + 1], 2)], [round(velocity_space[k], 2), round(velocity_space[k + 1], 2)], [round(velocity_space[l], 2), round(velocity_space[l + 1], 2)]])
                    elif ans[0] == -1:
                        failed_vals.append(ans)

    for i in range(4):
        ans = queries.check_safe_region(docking_pos, [[-docking_pos, docking_pos], [round(safe_space_l[i], 2), round(safe_space_l[i + 1], 2)], [0,0], [0,0]])
        if len(ans) == 4:
            vals.append(ans)
            val_ranges.append([[-docking_pos, docking_pos], [round(safe_space_l[i], 2), round(safe_space_l[i + 1], 2)], [0,0], [0,0]])
        elif ans[0] == -1:
            failed_vals.append(ans)

        ans = queries.check_safe_region(docking_pos, [[-docking_pos, docking_pos], [round(safe_space_r[i], 2), round(safe_space_r[i + 1], 2)], [0,0], [0,0]])
        if len(ans) == 4:
            vals.append(ans)
            val_ranges.append([[-docking_pos, docking_pos], [round(safe_space_r[i], 2), round(safe_space_r[i + 1], 2)], [0,0], [0,0]])
        elif ans[0] == -1:
            failed_vals.append(ans)

        for j in range(4):
            ans = queries.check_safe_region(docking_pos, [[round(safe_space_l[i], 2), round(safe_space_l[i + 1], 2)], [round(safe_space[j], 2), round(safe_space[j + 1], 2)], [0,0], [0,0]])
            if len(ans) == 4:
                vals.append(ans)
                val_ranges.append([[round(safe_space_l[i], 2), round(safe_space_l[i + 1], 2)], [round(safe_space[j], 2), round(safe_space[j + 1], 2)], [0,0], [0,0]])
            elif ans[0] == -1:
                failed_vals.append(ans)

            ans = queries.check_safe_region(docking_pos, [[round(safe_space_r[i], 2), round(safe_space_r[i + 1], 2)], [round(safe_space[j], 2), round(safe_space[j + 1], 2)], [0,0], [0,0]])
            if len(ans) == 4:
                vals.append(ans)
                val_ranges.append([[round(safe_space_r[i], 2), round(safe_space_r[i + 1], 2)], [round(safe_space[j], 2), round(safe_space[j + 1], 2)], [0,0], [0,0]])
            elif ans[0] == -1:
                failed_vals.append(ans)

    return [vals, val_ranges, failed_vals]

if __name__ == "__main__":
    
    safe_descent_cond_check("combined/combined_0.onnx", "models/cert_0.onnx", limit_pos = 5, safe_pos = 4, docking_pos = 0.35, vel_limit = 0.5)
    print("done")

