import json
import matplotlib.pyplot as plt
from skimage import io
import os.path
import operator
#from main import image_vec

# filename = "../validation.json"
# image_folder = '/media/spike/Scoob/materialist_fashion_val/'

filename = "../train.json"
image_folder = '/media/spike/Scoob/materialist_fashion_data/'

data = []
imageId_bin = dict()
label_bin = dict()
negative_label_bin = dict()

def create_bins():
    print('[i] Loading json please wait')
    with open(filename, 'r') as f:
        data = json.load(f)
        for annotation in data['annotations']:
            imageId = int(annotation['imageId'])
            imageId_bin[imageId] = []
            for l in annotation['labelId']:
                imageId_bin[imageId].append(int(l))
                if int(l) in label_bin.keys():
                    label_bin[int(l)].append(int(imageId))
                else:
                    negative_label_bin[int(l)] = []
                    label_bin[int(l)] = []
                    label_bin[int(l)].append(int(imageId))

    print('[i] Finished loading json')
    print('[i] No of unique keys {}, {}'.format(len(label_bin), label_bin.keys()))

# create and save positive label candidates
save_positive = False
if save_positive:
    with open('../positive_labels.json', 'w') as fp:
        json.dump(label_bin, fp)

# create and save negative label candidates
save_negative = False
if save_negative:
    for key in imageId_bin:
        for j in negative_label_bin.keys():
            if j not in imageId_bin[key]:
                negative_label_bin[j].append(key)
    with open('../negative_labels.json', 'w') as fp:
        json.dump(negative_label_bin, fp)

show_vector = False
# if show_vector:
#     vector_transform = image_vec.ImageVector()
def show_image_and_label():
    key = -1
    try:
        print('[i] Type image ID to view or e to exit')
        run = True
        while run:
            id = input("[?] ID: ")
            if id != 'e':
                if id.__contains__('k'):
                    v1, v2 = id.strip().split('k')
                    id = v1.strip()
                    key = int(v2.strip())
                idx = int(id)
                fig = plt.figure()
                ax = plt.subplot(1, 1, 1)
                ax.set_title('ID: {} {}'.format(idx, imageId_bin[idx]))
                ax.axis('off')
                image_path = os.path.abspath(os.path.join(image_folder, id + '.jpg'))
                image = io.imread(image_path)
                print("[i] Size : {}x{}".format(image.shape[0], image.shape[1]))
                ax.imshow(image)
                plt.show()
                if key > 0:
                    print('[i] Images with label {} : {}'.format(key, label_bin[key]))
                # if show_vector:
                #     print(vector_transform.get_vector(image_path))

            else:
                run = False
    except Exception as e:
        print('[X] Error: {}'.format(e))
    print('[i] Stopped')

def explore_positive_bins():
    with open('../positive_labels.json', 'r') as f:
        sum = 0
        data = json.load(f)
        exists_bin = [False]*1014544
        counts = dict()
        for item in range(1, 229):
            item = str(item)

            count = 0
            for image_id in data[item]:
                count += 1
                index = image_id-1
                if not exists_bin[index]:
                    exists_bin[index] = True
                    sum += 1
            counts[item] = count

        sorted_counts = sorted(counts.items(), key=lambda x:x[1], reverse=True)
        for item in sorted_counts:
            print('Label: {0} |#| Count: {1}'.format(item[0], item[1]))
        print('{0} == {1}'.format(sum, str(1014544)))

if __name__ == "__main__":
    # explore_positive_bins()
    create_bins()
    show_image_and_label()