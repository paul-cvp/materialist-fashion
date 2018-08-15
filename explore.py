import json
import matplotlib.pyplot as plt
import sys
from skimage import io
import os.path
import operator


class Explore:
    def __init__(self, json_file, image_folder):
        # data = []
        self.imageId_bin = dict()
        self.label_bin = dict()
        self.negative_label_bin = dict()
        self.json_file = json_file
        self.image_folder = image_folder
        print('[i] Loading json please wait')
        with open(self.json_file, 'r') as f:
            data = json.load(f)
            for annotation in data['annotations']:
                image_id = int(annotation['imageId'])
                self.imageId_bin[image_id] = []
                for l in annotation['labelId']:
                    self.imageId_bin[image_id].append(int(l))
                    if int(l) in self.label_bin.keys():
                        self.label_bin[int(l)].append(int(image_id))
                    else:
                        self.negative_label_bin[int(l)] = []
                        self.label_bin[int(l)] = []
                        self.label_bin[int(l)].append(int(image_id))

        print('[i] Finished loading json')
        print('[i] No of unique keys {}, {}'.format(len(self.label_bin), self.label_bin.keys()))

    # def create_and_save(self):
    #     # create and save positive label candidates
    #     save_positive = False
    #     if save_positive:
    #         with open('../positive_labels.json', 'w') as fp:
    #             json.dump(self.label_bin, fp)
    #
    #     # create and save negative label candidates
    #     save_negative = False
    #     if save_negative:
    #         for key in self.imageId_bin:
    #             for j in self.negative_label_bin.keys():
    #                 if j not in self.imageId_bin[key]:
    #                     self.negative_label_bin[j].append(key)
    #         with open('../negative_labels.json', 'w') as fp:
    #             json.dump(self.negative_label_bin, fp)

    def show_image_and_label(self):
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
                    ax.set_title('ID: {} {}'.format(idx, self.imageId_bin[idx]))
                    ax.axis('off')
                    image_path = os.path.abspath(os.path.join(self.image_folder, id + '.jpg'))
                    image = io.imread(image_path)
                    print("[i] Size : {}x{}".format(image.shape[0], image.shape[1]))
                    ax.imshow(image)
                    plt.show()
                    if key > 0:
                        print('[i] Images with label {} : {}'.format(key, self.label_bin[key]))
                else:
                    run = False
        except Exception as e:
            print('[X] Error: {}'.format(e))
        print('[i] Stopped')

    def compare_filtered_and_source(self):
        all_matches = True
        #largest_x = -1
        #largest_y = -1
        try:
            filtered_image_id_bin = dict()
            json_result_file_name = input("[?] Filtered json destination: ")  # 'train_filtered.json'
            with open(json_result_file_name, 'r') as f:
                data = json.load(f)
                for annotation in data['annotations']:
                    image_id = int(annotation['imageId'])
                    filtered_image_id_bin[image_id] = []
                    for l in annotation['labelId']:
                        filtered_image_id_bin[image_id].append(int(l))
            for idx in filtered_image_id_bin.keys():
                #image_path = os.path.abspath(os.path.join(self.image_folder, '{}.jpg'.format(idx)))
                #image = io.imread(image_path)
                label_matches = set(self.imageId_bin[idx]) == set(filtered_image_id_bin[idx])
                #largest_x = image.shape[0] if image.shape[0]>largest_x else largest_x
                #largest_y = image.shape[1] if image.shape[1]>largest_y else largest_y
                print('[i] ID: {} Matches: {}'.format(idx, label_matches))
                all_matches &= label_matches
        except Exception as e:
            print('[X] Error: {}'.format(e))
        print('[i] Done with compare_filtered_and_source. Matches: {}'.format(all_matches))

    @staticmethod
    def explore_positive_bins():
        with open('../positive_labels.json', 'r') as f:
            sum = 0
            data = json.load(f)
            exists_bin = [False] * 1014544
            counts = dict()
            for item in range(1, 229):
                item = str(item)

                count = 0
                for image_id in data[item]:
                    count += 1
                    index = image_id - 1
                    if not exists_bin[index]:
                        exists_bin[index] = True
                        sum += 1
                counts[item] = count

            sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            for item in sorted_counts:
                print('Label: {0} |#| Count: {1}'.format(item[0], item[1]))
            print('{0} == {1}'.format(sum, str(1014544)))

    def filter_json(self):
        json_result_file_name = input("[?] Filtered json destination: ")  # 'train_filtered.json'
        processed_image_ids = [int(f.split('.')[0]) for f in os.listdir(self.image_folder)]
        print(len(processed_image_ids))
        processed_image_ids = sorted(processed_image_ids)
        filtered_json = {}
        filtered_json['annotations'] = [{'imageId': id, 'labelId': self.imageId_bin[id]} for id in processed_image_ids]
        print(len(filtered_json))
        with open('{}'.format(json_result_file_name), 'w') as fp:
            json.dump(filtered_json, fp)

    def add_folder_to_json(self):
        json_result_file_name = input("[?] With folder json: ")  # 'train_filtered.json'
        filtered_json = {'annotations': []}
        with open(self.json_file, 'r') as f:
            data = json.load(f)
            for annotation in data['annotations']:
                image_id = int(annotation['imageId'])
                filtered_json['annotations'].append({'imageId': '{}/{}.jpg'.format(self.image_folder, image_id),
                                                     'labelId': self.imageId_bin[image_id]})
        with open('{}'.format(json_result_file_name), 'w') as fp:
            json.dump(filtered_json, fp)


if __name__ == "__main__":
    # filename = "../validation.json"
    # image_folder = '~materialist/val_filtered/'
    json_file, image_folder = sys.argv[1:]
    explore = Explore(json_file, image_folder)
    run = True
    while run:
        user_input = input("[?] Command: ")
        if user_input.startswith('-a'):
            explore.add_folder_to_json()
        elif user_input.startswith('-f'):
            explore.filter_json()
        elif user_input.startswith('-c'):
            explore.compare_filtered_and_source()
        elif user_input.startswith('-s'):
            explore.show_image_and_label()
        elif user_input.startswith('-e'):
            run = False
        elif user_input.startswith('-h'):
            print(' -a = add_folder_to_json \r\n -f = filter_json \r\n -c= compare_filtered_and_source \r\n -s = show_images_and_source \r\n -e = exit'
                  '\r\n -h = help')
        else:
            print(' -a = add_folder_to_json \r\n -f = filter_json \r\n -c= compare_filtered_and_source \r\n -s = show_images_and_source \r\n -e = exit'
                  '\r\n -h = help')
    print("[i] Done.")
