import sys

# from generals.simpleConv import SimpleConvGeneral
# from generals.extraSimpleConv import ExtraSimpleConv
# from generals.conv2deep import Conv2Deep
# from generals.conv3deep2 import Conv3Deep2
# from generals.conv4deep3 import Conv4Deep3
from generals.conv4deep3sigmoid import Conv4Deep3
from training.dataset import JsonDataset, ExtraSimplerDataset, MountainDataset, GeneralsAndMountainsDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch import save, load, argmax
from numpy import mean
import tqdm

# path = 'src/generals/weights/extra-simple-conv2d-0.weights'
save_path = 'src/generals/weights/simple-conv4deep2sigmoid.weights'

load_path = 'src/generals/weights/simple-conv4deep2sigmoid.weights'

training_files = [f'training-data/{prefix}/paired-data' for prefix in  sys.argv[1].split(',')]

dataset = GeneralsAndMountainsDataset(training_files)
batch_size = 128
dataloader = DataLoader(dataset, batch_size=batch_size)

# general = SimpleConvGeneral(15, 15)
# general = ExtraSimpleConv()
general = Conv4Deep3("mps")
# general.load_state_dict(load(load_path))
general.to("mps")

optimizer = optim.Adam(general.parameters(), 1e-3)
objective = nn.CrossEntropyLoss()

save_and_report_interval = 10

prev_loss = 0
losses = []
num_epochs = 10001
with tqdm.tqdm(total=num_epochs) as pbar:
    for epoch in range(num_epochs):
        for x, y_truth in dataloader:
            x, y_truth = x.to("mps"), y_truth.to("mps")
            iteration_batch_size = y_truth.shape[0]
            optimizer.zero_grad()
            y_hat = general(x)
            loss = objective(y_hat, y_truth)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        if epoch % save_and_report_interval == 0:
            avg_loss = mean(losses)
            difference = abs(prev_loss - avg_loss)
            print(f'{epoch} average loss {avg_loss} for difference of {difference}')
            if difference < 0.000000000000001:
                break
            prev_loss = avg_loss
            losses = []
            save(general.state_dict(), save_path)
        pbar.update(1)


save(general.state_dict(), save_path)

