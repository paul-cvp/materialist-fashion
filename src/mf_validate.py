from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from src.mf_metrics import fbeta_score


##################################################
#### Validate function
def validate(epoch, valid_loader, model, loss_func, mlb):
    ## Volatile variables do not save intermediate results and build graphs for backprop, achieving massive memory savings.

    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    ids = []

    print("Starting Validation")
    for batch_idx, (data, target, id) in enumerate(tqdm(valid_loader)):
        true_labels.append(target.cpu().numpy())
        ids.append(id.cpu().numpy())

        data, target = data.cuda(async=True), target.cuda(async=True).float()
        data, target = Variable(data, volatile=True).cuda(), Variable(target, volatile=True).cuda()

        pred = model(data)
        predictions.append(F.sigmoid(pred).data.cpu().numpy())

        total_loss += loss_func(pred, target).data[0]

    avg_loss = total_loss / len(valid_loader)

    predictions = np.vstack(predictions)
    true_labels = np.vstack(true_labels)
    ids = np.vstack(ids)

    score, threshold = fbeta_score(true_labels, predictions)
    for id, predicted, gt in zip(ids, predictions, true_labels):
        print('ID: {0} =>\r\nPredicted: {1} \r\nGround truth: {2}'.format(str(id),
                                                                          ', '.join(str(k) for k in predicted),
                                                                          ', '.join(str(k) for k in gt)))

    print("===> Validation - Avg. loss: {:.4f}\tF2 Score: {:.4f}".format(avg_loss, score))
    return score, avg_loss, threshold