import torch
import torch.nn as nn
import torch.onnx

from training_exp_mask import LyapunovNetworkV, Dynamic
from attempt_conversion import LearnedController
from evaluate_dynamic import stat_eval

def combined_model(file_1,file_2,file_3, output_file): 
    V = torch.load(file_1)
    # V.linear_relu_stack = V.linear_relu_stack.float()
    # V = V.to(torch.float32)
    #V = V.to(device="cpu")

    controller = torch.load(file_2)
    # controller.linear_relu_stack = controller.linear_relu_stack.float()
    # controller = controller.to(torch.float32)
    #controller = controller.to(device="cpu")

    dynamic = Dynamic(file_3)
    # dynamic.nn = dynamic.nn.float()
    # dynamic = dynamic.to(torch.float32)
    # torch.save(dynamic.nn.state_dict(), "dynamic32_test.pt")

    # mse, mae = stat_eval("dynamic32_test.pt", "dynamic_data/train_0.pt")
    # print("combined dynamic32 mse: ", mse)

    # Create the combined network with stacked A and two copies of B
    class CombinedNetwork(nn.Module):
        def __init__(self, model_A, model_B, model_C):
            super(CombinedNetwork, self).__init__()
            self.model_A = model_A
            self.model_B1 = model_B
            self.model_B2 = model_B
            self.model_C = model_C

        def forward(self, x, y, z):

            input_A = x
            input_B1 = x
            input_B2 = y
            input_C = z

            output_A = self.model_A.net(input_A)
            output_B1 = self.model_B1.linear_relu_stack(input_B1)
            output_B2 = self.model_B2.linear_relu_stack(input_B2)
            output_C = self.model_C.nn(input_C)

            
            return output_A, output_B1, output_B2, output_C


    # Create an instance of the combined network
    combined_network = CombinedNetwork(controller,V, dynamic)

    # Test the combined network
    input_data_one = torch.randn(1, 6, dtype=torch.float32)  
    input_data_two = torch.randn(1, 6, dtype=torch.float32)
    input_data_three = torch.randn(1, 8, dtype=torch.float32)

    input_data_one = torch.tensor([[1,1,1,1,1,1]], dtype=torch.float32)
    input_data_two = torch.tensor([[1,1,1,1,1,1]], dtype=torch.float32)
    input_data_three = torch.tensor([[1,1,1,1,1,1,1,1]], dtype=torch.float32)

    print(input_data_one)
    print(input_data_two)
    output_one, output_two, output_twoB, output_three = combined_network(input_data_one,input_data_two, input_data_three)
    print(output_one)
    print(output_two)
    print(output_twoB)
    print(output_three)

    x = torch.randn(1,6, dtype=torch.float32,requires_grad=True)
    y = torch.randn(1,6, dtype=torch.float32,requires_grad=True)
    z = torch.randn(1,8, dtype=torch.float32,requires_grad=True)
 
    torch.onnx.export(combined_network,(x,y,z),output_file,export_params=True,opset_version=10,do_constant_folding=True,input_names = ['input_1','input_2', 'input_3'],output_names = ['output_1','output_2A','output_2B', 'output_3'])

def combined_model_real(file_1,file_2, output_file): 
    V = torch.load(file_1)
    # V.linear_relu_stack = V.linear_relu_stack.float()
    # V = V.to(torch.float32)
    #V = V.to(device="cpu")

    controller = torch.load(file_2)
    # controller.linear_relu_stack = controller.linear_relu_stack.float()
    # controller = controller.to(torch.float32)
    #controller = controller.to(device="cpu")

    # mse, mae = stat_eval("dynamic32_test.pt", "dynamic_data/train_0.pt")
    # print("combined dynamic32 mse: ", mse)

    # Create the combined network with stacked A and two copies of B
    class CombinedNetworkReal(nn.Module):
        def __init__(self, model_A, model_B):
            super(CombinedNetworkReal, self).__init__()
            self.model_A = model_A
            self.model_B1 = model_B
            self.model_B2 = model_B

        def forward(self, x, y):

            input_A = x
            input_B1 = x
            input_B2 = y

            output_A = self.model_A.net(input_A)
            output_B1 = self.model_B1.linear_relu_stack(input_B1)
            output_B2 = self.model_B2.linear_relu_stack(input_B2)

            
            return output_A, output_B1, output_B2


    # Create an instance of the combined network
    combined_network = CombinedNetworkReal(controller,V)

    # Test the combined network
    input_data_one = torch.randn(1, 2, dtype=torch.float32)  
    input_data_two = torch.randn(1, 2, dtype=torch.float32)

    input_data_one = torch.tensor([[1,1]], dtype=torch.float32)
    input_data_two = torch.tensor([[1,1]], dtype=torch.float32)

    print(input_data_one)
    print(input_data_two)
    output_one, output_two, output_twoB = combined_network(input_data_one,input_data_two)
    print(output_one)
    print(output_two)
    print(output_twoB)

    x = torch.randn(1,2, dtype=torch.float32,requires_grad=True)
    y = torch.randn(1,2, dtype=torch.float32,requires_grad=True)
 
    torch.onnx.export(combined_network,(x,y),output_file,export_params=True,opset_version=10,do_constant_folding=True,input_names = ['input_1','input_2'],output_names = ['output_1','output_2A','output_2B'])

if __name__ == "__main__":
    pass