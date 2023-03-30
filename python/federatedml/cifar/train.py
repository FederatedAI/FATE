from types import SimpleNamespace
from federatedml.model_base import ModelBase
from .param import CifarParam
import torch.utils.data
import torchvision
import deepspeed
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F


class CifarTrainer(ModelBase):
    def __init__(self):
        super(CifarTrainer, self).__init__()
        self.model_param = CifarParam()
        self.args = SimpleNamespace(
            with_cuda=False,
            use_ema=False,
            batch_size=32,
            epochs=30,
            local_rank=-1,
            log_interval=2000,
            moe=False,
            ep_world_size=1,
            num_experts=[
                1,
            ],
            mlp_type="standard",
            top_k=1,
            min_capacity=0,
            noisy_gate_policy=None,
            moe_param_group=False,
        )

    def _init_model(self, param: CifarParam):
        self.ds_config = param.config
        self.data_root = param.data_root
        deepspeed.init_distributed()


    def fit(self, train_input, validate_input=None):
        trainset, testset, classes = load_data(self.data_root)
        net = build_net(
            self.args.moe,
            self.args.num_experts,
            self.args.ep_world_size,
            self.args.mlp_type,
            self.args.top_k,
            self.args.min_capacity,
            self.args.noisy_gate_policy,
        )

        parameters = filter(lambda p: p.requires_grad, net.parameters())
        if self.args.moe_param_group:
            from deepspeed.moe.utils import (
                split_params_into_different_moe_groups_for_optimizer,
            )

            parameters = {
                "params": [p for p in net.parameters()],
                "name": "parameters",
            }
            parameters = split_params_into_different_moe_groups_for_optimizer(parameters)


        model_engine, _, trainloader, _ = deepspeed.initialize(
            model=net,
            model_parameters=parameters,
            training_data=trainset,
            config=self.ds_config,
        )

        fp16 = model_engine.fp16_enabled()
        print(f"fp16={fp16}")

        criterion = nn.CrossEntropyLoss()

        for epoch in range(2):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(model_engine.local_rank), data[1].to(
                    model_engine.local_rank
                )
                if fp16:
                    inputs = inputs.half()
                outputs = model_engine(inputs)
                loss = criterion(outputs, labels)

                model_engine.backward(loss)
                model_engine.step()

                # print statistics
                running_loss += loss.item()
                if i % self.args.log_interval == (
                    self.args.log_interval - 1
                ):  # print every log_interval mini-batches
                    print(
                        "[%d, %5d] loss: %.3f"
                        % (epoch + 1, i + 1, running_loss / self.args.log_interval)
                    )
                    running_loss = 0.0

        print("Finished Training")

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=4, shuffle=False, num_workers=2
        )
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                if fp16:
                    images = images.half()
                outputs = net(images.to(model_engine.local_rank))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (
                    (predicted == labels.to(model_engine.local_rank)).sum().item()
                )

        print(
            "Accuracy of the network on the 10000 test images: %d %%"
            % (100 * correct / total)
        )

        ########################################################################
        # That looks way better than chance, which is 10% accuracy (randomly picking
        # a class out of 10 classes).
        # Seems like the network learnt something.
        #
        # Hmmm, what are the classes that performed well, and the classes that did
        # not perform well:

        class_correct = list(0.0 for i in range(10))
        class_total = list(0.0 for i in range(10))
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                if fp16:
                    images = images.half()
                outputs = net(images.to(model_engine.local_rank))
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels.to(model_engine.local_rank)).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        for i in range(10):
            print(
                "Accuracy of %5s : %2d %%"
                % (classes[i], 100 * class_correct[i] / class_total[i])
            )

def load_data(root):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    if torch.distributed.get_rank() != 0:
        # might be downloading cifar data, let rank 0 download first
        torch.distributed.barrier()

    trainset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform
    )

    if torch.distributed.get_rank() == 0:
        # cifar data is downloaded, indicate other ranks can proceed
        torch.distributed.barrier()
    testset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform
    )
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    return trainset, testset, classes

def build_net(
    moe, num_experts, ep_world_size, mlp_type, top_k, min_capacity, noisy_gate_policy
):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            if moe:
                fc3 = nn.Linear(84, 84)
                self.moe_layer_list = []
                for n_e in num_experts:
                    # create moe layers based on the number of experts
                    self.moe_layer_list.append(
                        deepspeed.moe.layer.MoE(
                            hidden_size=84,
                            expert=fc3,
                            num_experts=n_e,
                            ep_size=ep_world_size,
                            use_residual=mlp_type == "residual",
                            k=top_k,
                            min_capacity=min_capacity,
                            noisy_gate_policy=noisy_gate_policy,
                        )
                    )
                self.moe_layer_list = nn.ModuleList(self.moe_layer_list)
                self.fc4 = nn.Linear(84, 10)
            else:
                self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            if moe:
                for layer in self.moe_layer_list:
                    x, _, _ = layer(x)
                x = self.fc4(x)
            else:
                x = self.fc3(x)
            return x

    return Net()
