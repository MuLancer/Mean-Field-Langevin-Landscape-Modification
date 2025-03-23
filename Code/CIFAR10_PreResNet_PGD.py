import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim import SGD, Adagrad
from torch.utils.data import DataLoader

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# Define PreResNet110 Model
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )
        
    def forward(self, x):
        out = self.conv1(self.bn1(x))
        out = torch.relu(out)
        out = self.conv2(self.bn2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class PreResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.linear = nn.Linear(64 * block.expansion, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.bn(out)
        out = torch.relu(out)
        out = torch.nn.functional.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# Load CIFAR-10 Dataset
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def get_dataloaders(batch_size=128):
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader

# Define PGD Optimizer
class PGDOptimizer(optim.Optimizer):
    def __init__(self, params, lr=0.01, tau=1.0, step_size=0.08, annealed=True, num_particles=10):
        defaults = dict(lr=lr, tau=tau, step_size=step_size, annealed=annealed, num_particles=num_particles)
        super(PGDOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            tau = group['tau']
            step_size = group['step_size']
            annealed = group['annealed']
            num_particles = group['num_particles']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                
                if p not in self.state:
                    self.state[p] = {}
                if 'step' not in self.state[p]:
                    self.state[p]['step'] = 0
                
                if annealed:
                    tau_t = tau * (self.state[p]['step'] + 1) ** -1
                else:
                    tau_t = tau
                
                particle_updates = torch.zeros_like(p.data)
                for _ in range(num_particles):
                    noise = torch.sqrt(torch.tensor(2 * step_size * tau_t, dtype=torch.float32)) * torch.randn_like(p.data)
                    particle_updates += -step_size * grad + noise
                
                p.data.add_(particle_updates / num_particles)  # Average over particles
                self.state[p]['step'] += 1

        return loss

    
# Define PGD with Landscape Modification Optimizer
class PGD_LMOptimizer(optim.Optimizer):
    def __init__(self, params, lr=0.01, tau=1.0, step_size=0.08, annealed=True, alpha_init=1.0, decay_factor=1.0, num_particles=10):
        defaults = dict(lr=lr, tau=tau, step_size=step_size, annealed=annealed, alpha_init=alpha_init, decay_factor=decay_factor, num_particles=num_particles)
        super(PGD_LMOptimizer, self).__init__(params, defaults)
        self.current_loss = 1.0  # Initialize loss tracking

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            self.current_loss = loss.item()  # Update loss tracking

        for group in self.param_groups:
            step_size = group['step_size']
            tau = group['tau']
            annealed = group['annealed']
            alpha_init = group['alpha_init']
            decay_factor = group['decay_factor']
            num_particles = group['num_particles']
            
            # Maintain a running minimum of loss
            self.running_min_loss = min(self.running_min_loss, self.current_loss) if hasattr(self, 'running_min_loss') else self.current_loss
            c = self.running_min_loss  # Use running minimum loss as modification factor

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                
                if p not in self.state:
                    self.state[p] = {}
                if 'step' not in self.state[p]:
                    self.state[p]['step'] = 0
                
                iteration = self.state[p]['step']
                alpha = alpha_init * (decay_factor ** iteration)
                mod_factor = alpha * torch.exp(-decay_factor * max(0, (self.current_loss - c))) + 1
                grad /= mod_factor

                # Compute noise and updates using multiple particles
                particle_updates = torch.zeros_like(p.data)
                for _ in range(num_particles):
                    noise = torch.sqrt(torch.tensor(2 * step_size * tau, dtype=torch.float32)) * torch.randn_like(p.data)
                    particle_updates += -step_size * grad + noise
                
                p.data.add_(particle_updates / num_particles)  # Average over particles
                
                self.state[p]['step'] += 1  # Increment step count

        return loss


# Train Model

def train(model, trainloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, labels in trainloader:
        
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(trainloader)

# Evaluate Model

def evaluate(model, testloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return total_loss / len(testloader), accuracy


# Main Function

def main():
    model = PreResNet(BasicBlock, [18, 18, 18]).to(device)
    trainloader, testloader = get_dataloaders()
    criterion = nn.CrossEntropyLoss()
    
    optimizers = {
        #'SGD': SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4),
        #'Adagrad': Adagrad(model.parameters(), lr=0.01),
        'PGD': PGDOptimizer(model.parameters(), lr=0.01, tau=1.0, step_size=0.08, annealed=True),
        'PGD_LM': PGD_LMOptimizer(model.parameters(), lr=0.01, tau=1.0, step_size=0.08, annealed=True)
    }
    
    epochs = 2
    for opt_name, optimizer in optimizers.items():
        print(f"Training with {opt_name} optimizer")
        for epoch in range(epochs):
            train_loss = train(model, trainloader, optimizer, criterion, device)
            test_loss, test_acc = evaluate(model, testloader, criterion, device)
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
