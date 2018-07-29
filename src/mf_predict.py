from torch.autograd import Variable
import numpy as np
import os
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm


##################################################
#### Prediction function
def predict(test_loader, model):
    model.eval()
    predictions = []

    print("Starting Prediction")
    for batch_idx, (data, _) in enumerate(tqdm(test_loader)):
        data = data.cuda(async=True)
        data = Variable(data, volatile=True)

        pred = F.sigmoid(model(data))
        predictions.append(pred.data.cpu().numpy())

    predictions = np.vstack(predictions)

    print("===> Raw predictions done. Here is a snippet")
    print(predictions)
    return predictions


def output(predictions, threshold, X_test, mlb, dir_path, run_name, accuracy):
    raw_pred_path = os.path.join(dir_path, run_name + '-raw-pred-' + str(accuracy) + '.csv')
    np.savetxt(raw_pred_path, predictions, delimiter=";")
    print("Raw predictions saved to {}".format(raw_pred_path))

    predictions = predictions > threshold

    result = pd.DataFrame({
        'image_id': X_test.X,
        'label_id': mlb.inverse_transform(predictions)
    })
    result['tags'] = result['tags'].apply(lambda tags: " ".join(tags))

    print("===> Final predictions done. Here is a snippet")
    print(result)

    result_path = os.path.join(dir_path, run_name + '-pred-' + str(accuracy) + '.csv')
    result.to_csv(result_path, index=False)
    print("Final predictions saved to {}".format(result_path))