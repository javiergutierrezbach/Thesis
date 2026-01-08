import torch 
import onnx
import numpy as np
from datetime import datetime

import sys
sys.path.append("../../Marabou")

# import Marabou.maraboupy import Marabou
# from Marabou.maraboupy import MarabouCore
from maraboupy import Marabou
from maraboupy import MarabouCore, MarabouUtils

from training_exp_mask import Dynamic

import torch
import torch.nn as nn
import torch.onnx


def onnx_dyn_controller(dyn_file, controller_file, file_2):
    x = torch.randn(1,6, requires_grad=True, dtype=torch.float32)
    y = torch.randn(1,6, requires_grad=True, dtype=torch.float32)
    z = torch.randn(1,4, requires_grad=True, dtype=torch.float32)
    w = torch.randn(1,4, requires_grad=True, dtype=torch.float32)

    dyn = Dynamic(dyn_file).float()
    dyn2 = Dynamic(dyn_file).float()
    controller = torch.load(controller_file).float()
    controller2 = torch.load(controller_file).float()
    
    class OnnxDynController(nn.Module):
        def __init__(self, dynA, dynB, controllerA, controllerB):
            super().__init__()
            self.dynA = dynA.nn
            self.dynB = dynB.nn
            self.controllerA = controllerA
            self.controllerB = controllerB

        def forward(self, x, y, z, w):
            outputA = self.dynA(x)
            outputB = self.dynB(y)
            outputC = self.controllerA.linear_relu_stack(z)
            outputD = self.controllerB.linear_relu_stack(w)
            return outputA, outputB, outputC, outputD
    
    model = OnnxDynController(dyn, dyn2, controller, controller2)

    output = model(x, y, z, w)
    print(x, y, z, w, output)
        
    torch.onnx.export(model, (x,y,z,w), file_2, export_params=True,opset_version=10,do_constant_folding=True,input_names = ['x','y','z','w'],output_names = ['outA','outB','outC','outD'])

def onnx_controller(controller_file, file_2):
    x = torch.randn(1,4, requires_grad=True, dtype=torch.float64)
    y = torch.randn(1,4, requires_grad=True, dtype=torch.float64)

    controller = torch.load(controller_file)
    controller2 = torch.load(controller_file)
    
    class OnnxController(nn.Module):
        def __init__(self, controllerA, controllerB):
            super().__init__()

            self.controllerA = controllerA
            self.controllerB = controllerB

        def forward(self, x, y):
            outputA = self.controllerA.linear_relu_stack(x)
            outputB = self.controllerB.linear_relu_stack(y)
            return outputA, outputB
    
    model = OnnxController(controller, controller2)

    output = model(x, y)
    print(x, y, output)
        
    torch.onnx.export(model, (x,y), file_2, export_params=True,opset_version=10,do_constant_folding=True,input_names = ['x','y'],output_names = ['outA','outB'])


def test_lip_network(onnx_path, pivot):

    start = datetime.now()
    useMILP = True
    options = Marabou.createOptions(verbosity=0, numWorkers=16, solveWithMILP=useMILP, snc=False)
    network = Marabou.read_onnx(onnx_path)

    # get two different inputs for dynamic
    x, y, vx, vy, fx, fy = network.inputVars[0][0]
    x1, y1, vx1, vy1, fx1, fy1 = network.inputVars[1][0]
    outx, outy, outvx, outvy = network.outputVars[0][0]
    outx1, outy1, outvx1, outvy1 = network.outputVars[1][0]

    #controller inputs and outputs
    cx, cy, cvx, cvy = network.inputVars[2][0]
    cfx, cfy = network.outputVars[2][0]
    cx1, cy1, cvx1, cvy1 = network.inputVars[3][0]
    cfx1, cfy1 = network.outputVars[3][0]

    ranges = [[-2.2,2.2], [-0.55,0.55]]


    # set up bounds of state space
    network.setLowerBound(x, ranges[0][0])
    network.setLowerBound(y, ranges[0][0])
    network.setLowerBound(vx, ranges[1][0])
    network.setLowerBound(vy, ranges[1][0])
    # network.setLowerBound(fx, -1.0)
    # network.setLowerBound(fy, -1.0)

    network.setUpperBound(x, ranges[0][1])
    network.setUpperBound(y, ranges[0][1])
    network.setUpperBound(vx, ranges[1][1])
    network.setUpperBound(vy, ranges[1][1])
    # network.setUpperBound(fx, 1.0)
    # network.setUpperBound(fy, 1.0)

    network.setLowerBound(x1, ranges[0][0])
    network.setLowerBound(y1, ranges[0][0])
    network.setLowerBound(vx1, ranges[1][0])
    network.setLowerBound(vy1, ranges[1][0])
    # network.setLowerBound(fx1, -1.0)
    # network.setLowerBound(fy1, -1.0)

    network.setUpperBound(x1, ranges[0][1])
    network.setUpperBound(y1, ranges[0][1])
    network.setUpperBound(vx1, ranges[1][1])
    network.setUpperBound(vy1, ranges[1][1])
    # network.setUpperBound(fx1, 1.0)
    # network.setUpperBound(fy1, 1.0)

    # clip forces

    clip_lb = -1
    clip_ub = 1

    # fx_clip = cfx
    # fy_clip = cfy
    # fx1_clip = cfx1
    # fy1_clip = cfy1

    fx_clip = clipForce(network, cfx, clip_lb, clip_ub)
    fy_clip = clipForce(network, cfy, clip_lb, clip_ub)

    fx1_clip = clipForce(network, cfx1, clip_lb, clip_ub)
    fy1_clip = clipForce(network, cfy1, clip_lb, clip_ub)

    # equate controller inputs outputs to dynamic 
    network.addEquality([x, cx], [1, -1], 0, False)
    network.addEquality([y, cy], [1, -1], 0, False)
    network.addEquality([vx, cvx], [1, -1], 0, False)
    network.addEquality([vy, cvy], [1, -1], 0, False)

    network.addEquality([fx_clip, fx], [1, -1], 0, False)
    network.addEquality([fy_clip, fy], [1, -1], 0, False)

    network.addEquality([x1, cx1], [1, -1], 0, False)
    network.addEquality([y1, cy1], [1, -1], 0, False)
    network.addEquality([vx1, cvx1], [1, -1], 0, False)
    network.addEquality([vy1, cvy1], [1, -1], 0, False)

    network.addEquality([fx1_clip, fx1], [1, -1], 0, False)
    network.addEquality([fy1_clip, fy1], [1, -1], 0, False)


    # for v in ( cfx1, cfy1):
    #     network.setLowerBound(v, -1.0)
    #     network.setUpperBound(v,  1.0)

    dx, dy, dvx, dvy, dfx, dfy = encodeDifferences(network, [x, y, vx, vy, fx, fy], [x1, y1, vx1, vy1, fx1, fy1])
    #dx, dy, dvx, dvy = encodeDifferences(network, [x, y, vx, vy], [x1, y1, vx1, vy1])

    delta = network.getNewVariable()
    network.addMaxConstraint({dx, dy, dvx, dvy, dfx, dfy}, delta)

    # make sure the points are not equal by adding a lower bound on max norm
    tol = 1e-4
    network.setLowerBound(delta, tol)

    dout_x, dout_y, dout_vx, dout_vy = encodeDifferences(network, [outx, outy, outvx, outvy], [outx1, outy1, outvx1, outvy1])

    epsilon = network.getNewVariable()
    network.addMaxConstraint({dout_x, dout_y, dout_vx, dout_vy}, epsilon)

    print_vars = [x, y, vx, vy, cfx, cfy, fx, fy, x1, y1, vx1, vy1, cfx1, cfy1, fx1, fy1, dx, dy, dvx, dvy, dfx, dfy, outx, outy, outvx, outvy, outx1, outy1, outvx1, outvy1, dout_x, dout_y, dout_vx, dout_vy, delta, epsilon]

    query_path = "temp_query.query"
    network.saveQuery(query_path)

    lip = binarySearch(query_path, options, delta, epsilon, pivot, 10, None, print_vars)

    end = datetime.now()
    diff = end - start

    print("Lipschitz: ", lip)
    print("Constant found in:", diff.seconds, "seconds." )

def clipForce(network, var, lb, ub):

    aux1 = network.getNewVariable()
    var_clip = network.getNewVariable()
    network.addEquality([aux1, var], [1, -1], -1 * lb, False)  # aux1 = F_x-clip_lb
    aux2 = network.getNewVariable()
    network.addRelu(aux1, aux2)  # aux2 = relu(aux1)
    aux3 = network.getNewVariable()
    network.addEquality([aux3, aux2], [1, -1], lb, False)  # aux3 = clip_lb + aux2
    aux4 = network.getNewVariable()
    network.addEquality([aux4, aux3], [-1, -1], -ub, False)  # aux4 = clip_ub - aux3
    aux5 = network.getNewVariable()
    network.addRelu(aux4, aux5)  # aux5 = relu(aux4)
    network.addEquality([var_clip, aux5], [-1, -1], -ub, False)  # F_x_clip = clip_ub - aux5

    return var_clip
    
def binarySearch(query_path, options, delta, epsilon, pivot, factor, prev, print_vars):

    # solve the lipschitz condition
    iq = MarabouCore.InputQuery()
    MarabouCore.loadQuery(query_path, iq)

    le = MarabouCore.Equation(MarabouCore.Equation.LE)
    le.addAddend(-1.0, epsilon) 
    le.addAddend(pivot, delta)    
    le.setScalar(1e-9)              
    iq.addEquation(le)

    exitCode, vals, stats = MarabouCore.solve(iq, options)

    if factor < 1e-8:
        return pivot + (2*factor)

    if exitCode == "sat":
        print("Sat")

        if prev == "unsat":
            shift = factor / 2.0
        else: 
            shift = factor

        print(pivot)
        print(shift)

        print(print_vars)

        if print_vars:
            print("vals: \n")
            for var in print_vars:
                print(vals[var], ", ")
        

        return binarySearch(query_path, options, delta, epsilon, pivot + shift, shift, "sat", print_vars)
    
    if exitCode == "unsat":

        print("Unsat: Inductive proof completed")

        if prev == "sat":
            shift = factor / 2.0
        else: 
            shift = factor
        print(pivot)
        print(shift)

        return binarySearch(query_path, options, delta, epsilon, pivot - shift, shift, "unsat", print_vars)
    else:
        print("Failed")
        return [-1]


	
	
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

def encodeDifferencesTrue(ipq, inp, out):
    if len(inp) != len(out):
        raise ValueError("inp and out must have the same length")

    outvars = []
    for xi, yi in zip(inp, out):
        d = new_var(ipq)
        # d = xi - yi  ==>  d - xi + yi = 0
        eq = MarabouCore.Equation(MarabouCore.Equation.EQ)
        eq.addAddend(1.0, d)
        eq.addAddend(-1.0, xi)
        eq.addAddend(1.0, yi)
        eq.setScalar(0.0)
        ipq.addEquation(eq)

        absd = new_var(ipq)
        MarabouCore.addAbsConstraint(ipq, d, absd)
        outvars.append(absd)

    return outvars

def new_var(ipq):
        vid = ipq.getNumberOfVariables()
        ipq.setNumberOfVariables(vid + 1)
        return vid

def test_lip_true(pivot):

    start = datetime.now()
    useMILP = True
    options = Marabou.createOptions(verbosity=0, numWorkers=16, solveWithMILP=useMILP, snc=False)

    ipq = MarabouCore.InputQuery()
    ipq.setNumberOfVariables(20)

    # get two different inputs for dynamic

    x = 0
    y = 1
    vx = 2
    vy = 3
    fx = 4
    fy = 5

    out_x = 6
    out_y = 7
    out_vx = 8
    out_vy = 9

    x1 = 10
    y1 = 11
    vx1 = 12
    vy1 = 13
    fx1 = 14 
    fy1 = 15

    out_x1 = 16
    out_y1 = 17
    out_vx1 = 18
    out_vy1 = 19

    ranges = [[-2.2,2.2], [-0.55,0.55]]

    # set up bounds of state space
    ipq.setLowerBound(x, ranges[0][0])
    ipq.setLowerBound(y, ranges[0][0])
    ipq.setLowerBound(vx, ranges[1][0])
    ipq.setLowerBound(vy, ranges[1][0])
    ipq.setLowerBound(fx, -1.0)
    ipq.setLowerBound(fy, -1.0)

    ipq.setUpperBound(x, ranges[0][1])
    ipq.setUpperBound(y, ranges[0][1])
    ipq.setUpperBound(vx, ranges[1][1])
    ipq.setUpperBound(vy, ranges[1][1])
    ipq.setUpperBound(fx, 1.0)
    ipq.setUpperBound(fy, 1.0)

    ipq.setLowerBound(x1, ranges[0][0])
    ipq.setLowerBound(y1, ranges[0][0])
    ipq.setLowerBound(vx1, ranges[1][0])
    ipq.setLowerBound(vy1, ranges[1][0])
    ipq.setLowerBound(fx1, -1.0)
    ipq.setLowerBound(fy1, -1.0)

    ipq.setUpperBound(x1, ranges[0][1])
    ipq.setUpperBound(y1, ranges[0][1])
    ipq.setUpperBound(vx1, ranges[1][1])
    ipq.setUpperBound(vy1, ranges[1][1])
    ipq.setUpperBound(fx1, 1.0)
    ipq.setUpperBound(fy1, 1.0)

    encodeDynamicTrue(ipq, [x, y, vx, vy, fx, fy], [out_x, out_y, out_vx, out_vy])
    encodeDynamicTrue(ipq, [x1, y1, vx1, vy1, fx1, fy1], [out_x1, out_y1, out_vx1, out_vy1])

    dx, dy, dvx, dvy, dfx, dfy = encodeDifferencesTrue(ipq, [x, y, vx, vy, fx, fy], [x1, y1, vx1, vy1, fx1, fy1])

    # ends up being 32
    delta = new_var(ipq)
    MarabouCore.addMaxConstraint(ipq, {dx, dy, dvx, dvy, dfx, dfy}, delta)

    # make sure the points are not equal by adding a lower bound on max norm
    tol = 1e-4
    ipq.setLowerBound(delta, tol)

    dout_x, dout_y, dout_vx, dout_vy = encodeDifferencesTrue(ipq, [out_x, out_y, out_vx, out_vy], [out_x1, out_y1, out_vx1, out_vy1])

    # ends up being last or 41
    epsilon = new_var(ipq)
    MarabouCore.addMaxConstraint(ipq, {dout_x, dout_y, dout_vx, dout_vy}, epsilon)

    query_path = "temp_query_true.query"
    MarabouCore.saveQuery(ipq, query_path)

    lip = binarySearch(query_path, options, delta, epsilon, pivot, 1, None, None)

    end = datetime.now()
    diff = end - start

    print("Lipschitz: ", lip)
    print("Constant found in:", diff.seconds, "seconds." )

def test_lip_true_controller(onnx_path, pivot):

    start = datetime.now()
    useMILP = True
    options = Marabou.createOptions(verbosity=0, numWorkers=16, solveWithMILP=useMILP, snc=False)
    network = Marabou.read_onnx(onnx_path)

    #controller inputs and outputs
    cx, cy, cvx, cvy = network.inputVars[0][0]
    cfx, cfy = network.outputVars[0][0]
    cx1, cy1, cvx1, cvy1 = network.inputVars[1][0]
    cfx1, cfy1 = network.outputVars[1][0]

    clip_lb = -1
    clip_ub = 1

    x      = network.getNewVariable()
    y      = network.getNewVariable()
    vx     = network.getNewVariable()
    vy     = network.getNewVariable()
    fx     = network.getNewVariable()
    fy     = network.getNewVariable()

    x1      = network.getNewVariable()
    y1      = network.getNewVariable()
    vx1     = network.getNewVariable()
    vy1     = network.getNewVariable()
    fx1    = network.getNewVariable()
    fy1     = network.getNewVariable()

     # equate controller inputs outputs to dynamic 
    network.addEquality([x, cx], [1, -1], 0, False)
    network.addEquality([y, cy], [1, -1], 0, False)
    network.addEquality([vx, cvx], [1, -1], 0, False)
    network.addEquality([vy, cvy], [1, -1], 0, False)
    network.addEquality([fx, cfx], [1, -1], 0, False)
    network.addEquality([fy, cfy], [1, -1], 0, False)

    network.addEquality([x1, cx1], [1, -1], 0, False)
    network.addEquality([y1, cy1], [1, -1], 0, False)
    network.addEquality([vx1, cvx1], [1, -1], 0, False)
    network.addEquality([vy1, cvy1], [1, -1], 0, False)
    network.addEquality([fx1, cfx1], [1, -1], 0, False)
    network.addEquality([fy1, cfy1], [1, -1], 0, False)

    fx_clip = clipForce(network, fx, clip_lb, clip_ub)
    fy_clip = clipForce(network, fy, clip_lb, clip_ub)

    fx1_clip = clipForce(network, fx1, clip_lb, clip_ub)
    fy1_clip = clipForce(network, fy1, clip_lb, clip_ub)

    out_x = network.getNewVariable()
    out_y = network.getNewVariable()
    out_vx = network.getNewVariable()
    out_vy = network.getNewVariable()

    out_x1 = network.getNewVariable()
    out_y1 = network.getNewVariable()
    out_vx1 = network.getNewVariable()
    out_vy1 = network.getNewVariable()

    ranges = [[-2.2,2.2], [-0.55,0.55]]

    # set up bounds of state space
    network.setLowerBound(x, ranges[0][0])
    network.setLowerBound(y, ranges[0][0])
    network.setLowerBound(vx, ranges[1][0])
    network.setLowerBound(vy, ranges[1][0])
    # network.setLowerBound(fx, -1.0)
    # network.setLowerBound(fy, -1.0)

    network.setUpperBound(x, ranges[0][1])
    network.setUpperBound(y, ranges[0][1])
    network.setUpperBound(vx, ranges[1][1])
    network.setUpperBound(vy, ranges[1][1])
    # network.setUpperBound(fx, 1.0)
    # network.setUpperBound(fy, 1.0)

    network.setLowerBound(x1, ranges[0][0])
    network.setLowerBound(y1, ranges[0][0])
    network.setLowerBound(vx1, ranges[1][0])
    network.setLowerBound(vy1, ranges[1][0])
    # network.setLowerBound(fx1, -1.0)
    # network.setLowerBound(fy1, -1.0)

    network.setUpperBound(x1, ranges[0][1])
    network.setUpperBound(y1, ranges[0][1])
    network.setUpperBound(vx1, ranges[1][1])
    network.setUpperBound(vy1, ranges[1][1])
    # network.setUpperBound(fx1, 1.0)
    # network.setUpperBound(fy1, 1.0)


    encodeDynamicTrueController(network, [x, y, vx, vy, fx_clip, fy_clip], [out_x, out_y, out_vx, out_vy])
    encodeDynamicTrueController(network, [x1, y1, vx1, vy1, fx1_clip, fy1_clip], [out_x1, out_y1, out_vx1, out_vy1])

    dx, dy, dvx, dvy, dfx, dfy = encodeDifferences(network, [x, y, vx, vy, fx_clip, fy_clip], [x1, y1, vx1, vy1, fx1_clip, fy1_clip])

    delta = network.getNewVariable()
    network.addMaxConstraint({dx, dy, dvx, dvy, dfx, dfy}, delta)

    # make sure the points are not equal by adding a lower bound on max norm
    tol = 1e-4
    network.setLowerBound(delta, tol)

    dout_x, dout_y, dout_vx, dout_vy = encodeDifferences(network, [out_x, out_y, out_vx, out_vy], [out_x1, out_y1, out_vx1, out_vy1])

    epsilon = network.getNewVariable()
    network.addMaxConstraint({dout_x, dout_y, dout_vx, dout_vy}, epsilon)

    query_path = "temp_query_true_controller.query"
    network.saveQuery(query_path)

    lip = binarySearch(query_path, options, delta, epsilon, pivot, 1, None, None)

    end = datetime.now()
    diff = end - start

    print("Lipschitz: ", lip)
    print("Constant found in:", diff.seconds, "seconds." )

def encodeDynamicTrueController(network, inp, out):

    m = 12.0
    n = 0.001027
    t = 1.0

    x, y, vx, vy, fx, fy = inp
    x_out, y_out, vx_out, vy_out = out

    coeff_x = np.array([
            4 - 3 * np.cos(n * t),
            0,
            1 / n * np.sin(n * t),
            2 / n - 2 / n * np.cos(n * t),
            (1 - np.cos(n * t)) / (m * n ** 2),
            2 * t / (m * n) - 2 * np.sin(n * t) / (m * n ** 2),
            -1.0
        ], dtype=float).tolist()
    coeff_y = np.array([
            -6 * n * t + 6 * np.sin(n * t),
            1,
            -2 / n + 2 / n * np.cos(n * t),
            -3 * t + 4 / n * np.sin(n * t),
            (-2 * t) / (m * n) + (2 * np.sin(n * t)) / (m * n ** 2),
            4 / (m * n ** 2) - (3 * t ** 2) / (2 * m) - (4 * np.cos(n * t)) / (m * n ** 2),
            -1.0
        ], dtype=float).tolist()
    coeff_vx = np.array([
            3 * n * np.sin(n * t),
            0,
            np.cos(n * t),
            2 * np.sin(n * t),
            np.sin(n * t) / (m * n),
            2 / (m * n) - (2 * np.cos(n * t)) / (m * n),
            -1.0
        ], dtype=float).tolist()
    coeff_vy = np.array([
            -6 * n + 6 * n * np.cos(n * t),
            0,
            -2 * np.sin(n * t),
            -3 + 4 * np.cos(n * t),
            (2 * np.cos(n * t) - 2) / (m * n),
            (-3 * t) / (m) + (4 * np.sin(n * t)) / (m * n),
            -1.0
        ], dtype=float).tolist()
    
    network.addEquality([x, y, vx, vy, fx, fy, x_out], coeff_x, 0.0)
    network.addEquality([x, y, vx, vy, fx, fy, y_out], coeff_y, 0.0)
    network.addEquality([x, y, vx, vy, fx, fy, vx_out], coeff_vx, 0.0)
    network.addEquality([x, y, vx, vy, fx, fy, vy_out], coeff_vy, 0.0)


def encodeDynamicTrue(ipq, inp, out):

    m = 12.0
    n = 0.001027
    t = 1.0

    x, y, vx, vy, fx, fy = inp
    x_out, y_out, vx_out, vy_out = out

    cx = [
            4 - 3 * np.cos(n * t),
            0,
            1 / n * np.sin(n * t),
            2 / n - 2 / n * np.cos(n * t),
            (1 - np.cos(n * t)) / (m * n ** 2),
            2 * t / (m * n) - 2 * np.sin(n * t) / (m * n ** 2),
        ]
    cy = [
            -6 * n * t + 6 * np.sin(n * t),
            1,
            -2 / n + 2 / n * np.cos(n * t),
            -3 * t + 4 / n * np.sin(n * t),
            (-2 * t) / (m * n) + (2 * np.sin(n * t)) / (m * n ** 2),
            4 / (m * n ** 2) - (3 * t ** 2) / (2 * m) - (4 * np.cos(n * t)) / (m * n ** 2),
        ]
    cvx = [
            3 * n * np.sin(n * t),
            0,
            np.cos(n * t),
            2 * np.sin(n * t),
            np.sin(n * t) / (m * n),
            2 / (m * n) - (2 * np.cos(n * t)) / (m * n),
        ]
    cvy = [
            -6 * n + 6 * n * np.cos(n * t),
            0,
            -2 * np.sin(n * t),
            -3 + 4 * np.cos(n * t),
            (2 * np.cos(n * t) - 2) / (m * n),
            (-3 * t) / (m) + (4 * np.sin(n * t)) / (m * n),
        ]
    
    add_eq_coeffs(ipq, x_out, [(cx[0], x), (cx[1], y), (cx[2], vx), (cx[3], vy), (cx[4], fx), (cx[5], fy)])
    add_eq_coeffs(ipq, y_out, [(cy[0], x), (cy[1], y), (cy[2], vx), (cy[3], vy), (cy[4], fx), (cy[5], fy)])
    add_eq_coeffs(ipq, vx_out, [(cvx[0], x), (cvx[1], y), (cvx[2], vx), (cvx[3], vy), (cvx[4], fx), (cvx[5], fy)])
    add_eq_coeffs(ipq, vy_out, [(cvy[0], x), (cvy[1], y), (cvy[2], vx), (cvy[3], vy), (cvy[4], fx), (cvy[5], fy)])


def add_eq_coeffs(ipq, lhs_var, terms, scalar=0.0):
        # Enforce: lhs_var = sum_i (coef_i * var_i) + scalar
        eq = MarabouCore.Equation(MarabouCore.Equation.EQ)
        eq.addAddend(1.0, lhs_var)
        for coef, var in terms:
            if coef != 0.0:
                eq.addAddend(-coef, var)   # move to LHS
        eq.setScalar(scalar)
        ipq.addEquation(eq)
    
def add_eq(ipq, x, y):
        
        eq = MarabouCore.Equation(MarabouCore.Equation.EQ)
        eq.addAddend(1.0, x)
        eq.addAddend(-1.0, y)
        eq.setScalar(0)
        ipq.addEquation(eq)
    


if __name__ == "__main__":
    
    dynamic_path = "dynamics/dynamic__lip_0.pt"
    controller_path = "controllers/controller_lip_5_verified.pt"
    onnx_path = dynamic_path.replace(".pt", ".onnx")
    onnx_path_controller = controller_path.replace(".pt", ".onnx")
    onnx_dyn_controller(dynamic_path, controller_path, onnx_path)
    onnx_controller(controller_path, onnx_path_controller)
    test_lip_network(onnx_path, 3)
    #test_lip_true(3)
    #test_lip_true_controller(onnx_path_controller, 3)
