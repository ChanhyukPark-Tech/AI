import argparse
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from model import RobustModel


class ImageDataset(Dataset):
    """ Image shape: 28x28x3 """

    def __init__(self, root_dir, fmt=':06d', extension='.png'):
        self.root_dir = root_dir
        self.fmtstr = '{' + fmt + '}' + extension

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.fmtstr.format(idx)
        img_path = os.path.join(self.root_dir, img_name)
        img = plt.imread(img_path)
        data = torch.from_numpy(img)
        return data


def inference(data_loader, model):
    """ model inference """

    model.eval()
    preds = []

    with torch.no_grad():
        for X in data_loader:
            y_hat = model(X)
            y_hat.argmax()

            _, predicted = torch.max(y_hat, 1)
            preds.extend(map(lambda t: t.item(), predicted))

    return preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2022 DL Term Project #1')
    parser.add_argument('--load-model', default='model.pt', help="Model's state_dict")
    parser.add_argument('--dataset', default='./test/', help='image dataset directory')
    parser.add_argument('--batch-size', default=16, help='test loader batch size')

    args = parser.parse_args()

    # instantiate model
    model = RobustModel()
    model.load_state_dict(torch.load(args.load_model))

    # load dataset in test image folder
    test_data = ImageDataset(args.dataset)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)

    # write model inference
    preds = inference(test_loader, model)
    answer = [7,2,1,0,4,1,4,9,5,9,0,6,9,0,1,5,9,7,3,4,9,6,6,5,4,0,7,4]

    at = 0

    for i in range(28):
        if preds[i] == answer[i]:
            at = at + 1

    
    print("project accurcay = " , at / 28 )

    with open('result.txt', 'w') as f:
        f.writelines('\n'.join(map(str, preds)))
