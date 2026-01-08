from estimate_lip import onnx_dyn_controller
import numpy as np
import onnx
import onnxruntime as ort

dynamic_path = "dynamics/dynamic__lip_0.pt"
controller_path = "controllers/controller_lip_5_verified.pt"
onnx_path = dynamic_path.replace(".pt", ".onnx")

onnx_dyn_controller(dynamic_path, controller_path, onnx_path)

# Create a session (CPU or GPU)
sess = ort.InferenceSession(
    onnx_path,
    providers=["CPUExecutionProvider"]  # or ["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# See inputs/outputs
for i in sess.get_inputs():
    print("INPUT:", i.name, i.shape, i.type)
for o in sess.get_outputs():
    print("OUTPUT:", o.name, o.shape, o.type)

state = [ -1.7178055118430084 , -2.1999,  -0.18742085341411882,  -0.18834608644612033]
state1 = [ -1.7178055118430084,  -2.2, -0.18745192929749818,  -0.18844608644612032]

# last two inputs are for the controller, we just care to get the forces from the states here

x = np.array([[ 0.0, 0.0,  0.0,  0.0, 0.0,  0.0]], dtype=np.float32)
y = np.array([[ 0.0,  0.0, 0.0,  0.0, 0.0,  0.0]], dtype=np.float32)
z = np.array([[ state[0],  state[1],  state[2],  state[3]]], dtype=np.float32)
w = np.array([[ state1[0],  state1[1], state1[2],  state1[3]]], dtype=np.float32)


feeds = {
    "x": x,
    "y": y,
    "z": z,
    "w": w,
}

outputs = sess.run(None, feeds)

# clip the forces

f = np.clip(outputs[2], -1.0, 1.0)
f1 = np.clip(outputs[3], -1.0, 1.0)

print(outputs[2])
print(outputs[3])
print(f)
print(f1)
print(f[0])

# first two inputs are for the dynamic, use the states and the forces from them

x = np.array([[ state[0], state[1],  state[2],  state[3], f[0][0],  f[0][1]]], dtype=np.float32)
y = np.array([[ state1[0],  state1[1], state1[2],  state1[3], f1[0][0],  f1[0][1]]], dtype=np.float32)
z = np.array([[ 0.0,  0.0, 0.0,  0.0]], dtype=np.float32)
w = np.array([[ 0.0,  0.0, 0.0,  0.0]], dtype=np.float32)

feeds = {
    "x": x,
    "y": y,
    "z": z,
    "w": w,
}

outputs = sess.run(None, feeds)

# calculate norms and lipschitz

delta = np.max(np.abs(y - x))
epsilon = np.max(np.abs(outputs[0][0] - outputs[1][0]))

print(delta, epsilon)

lipschitz = epsilon / delta

print(lipschitz)
