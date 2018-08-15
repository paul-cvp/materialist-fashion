from imageai.Detection import ObjectDetection
from imageai.Prediction import ImagePrediction
import os
from PIL import Image
import shutil
from scipy.misc import imread
from scipy.linalg import norm
from scipy import sum, average

class Util:

    def __init__(self, image_folder):
        super().__init__()
        self.img1 = Util.to_grayscale(imread(os.path.join(self.execution_path, 'bad_image.jpg')).astype(float))
        self.execution_path = '/home/spike/Projects/object_detection/' #os.getcwd()
        self.image_folder = image_folder
        self.temp_path = os.path.join(self.execution_path, 'tmp')

        self.detector = ObjectDetection()
        self.detector.setModelTypeAsRetinaNet()
        self.detector.setModelPath(os.path.join(self.execution_path, "resnet50_coco_best_v2.0.1.h5"))
        self.detector.loadModel()
        self.custom_objects = self.detector.CustomObjects(person=True, backpack=True, umbrella=True, handbag=True, tie=True, suitcase=True)

        self.prediction = ImagePrediction()
        self.prediction.setModelTypeAsResNet()
        self.prediction.setModelPath(os.path.join(self.execution_path, 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'))
        self.prediction.loadModel()

    @staticmethod
    def to_grayscale(arr):
        "If arr is a color image (3D array), convert it to grayscale (2D array)."
        if len(arr.shape) == 3:
            return average(arr, -1)  # average over the last axis (color channels)
        else:
            return arr

    @staticmethod
    def normalize(arr):
        rng = arr.max()-arr.min()
        if rng == 0:
            rng = 1
        amin = arr.min()
        return (arr-amin)*255/rng

    @staticmethod
    def compare_images(img1, img2):
        # normalize to compensate for exposure difference, this may be unnecessary
        # consider disabling it
        img1 = Util.normalize(img1)
        img2 = Util.normalize(img2)
        # calculate the difference and its norms
        diff = img1 - img2  # elementwise for scipy arrays
        m_norm = sum(abs(diff))  # Manhattan norm
        z_norm = norm(diff.ravel(), 0)  # Zero norm
        return (m_norm, z_norm)

    def is_image_bad(self, image_to_compare):
        img2 = Util.to_grayscale(imread(os.path.join(self.image_folder, image_to_compare)).astype(float))

        if self.img1.shape != img2.shape:
            return False

        # compare
        n_m, n_0 = Util.compare_images(self.img1, img2)

        if n_m<1:
            return True
        else:
            return False


    # def combine_images(individual_images_path, image_name, result_folder):
    #     if len(individual_images_path) > 0:
    #         images = map(Image.open, individual_images_path)
    #         widths, heights = zip(*(i.size for i in images))
    #
    #         total_width = sum(widths)
    #         max_height = max(heights)
    #
    #         new_im = Image.new('RGB', (total_width, max_height))
    #
    #         x_offset = 0
    #         for im_path in individual_images_path:
    #             im = Image.open(im_path)
    #             new_im.paste(im, (x_offset, 0))
    #             x_offset += im.size[0]
    #
    #         new_im.save(os.path.join(result_folder, image_name))
    #     else:
    #         shutil.copy(os.path.join(image_folder, image_name), os.path.join(result_folder, image_name))
    #
    #
    # def delete_temp_folder():
    #     shutil.rmtree(temp_path)
    #
    #
    # run = False
    # while run:
    #     id = input('[?] ID: ')
    #     if id.startswith('b'):
    #         total_bad_images = 0
    #         total_good_images = 0
    #         for image_path in os.listdir(save_result_folder):
    #             if is_image_bad(image_path):
    #                 os.remove(os.path.join(save_result_folder, image_path))
    #                 total_bad_images += 1
    #             else:
    #                 total_good_images += 1
    #         print("Bad Images: {}".format(total_bad_images))
    #         print("Good Images: {}".format(total_good_images))
    #
    #     elif id.startswith('t'):
    #         source_folder = '/home/spike/Projects/object_detection/tmp/detected1.jpg-objects'
    #         combine_images([os.path.join(source_folder, x) for x in os.listdir(source_folder)],
    #                        '1.jpg',
    #                        save_result_folder)
    #     elif '-' in id:
    #         range_min, range_max = [x.strip() for x in id.split('-')]
    #         print("range_min={} | range_max={}".format(range_min, range_max))
    #         for range_id in range(int(range_min), int(range_max)+1):
    #             try:
    #                 image_path = os.path.abspath(os.path.join(image_folder, str(range_id) + '.jpg'))
    #                 detections, objects_path = detector.detectCustomObjectsFromImage(custom_objects=custom_objects, input_image=image_path,
    #                                                                                  output_image_path=os.path.join(temp_path, 'detected' + str(range_id) + '.jpg'),
    #                                                                                  minimum_percentage_probability=50, extract_detected_objects=True)
    #                 combine_images(objects_path, str(range_id) + '.jpg', save_result_folder)
    #             except:
    #                 combine_images([], str(range_id) + '.jpg', save_result_folder)
    #     elif id != 'e':
    #         image_path = os.path.abspath(os.path.join(image_folder, id + '.jpg'))
    #         detections, objects_path = detector.detectCustomObjectsFromImage(custom_objects=custom_objects, input_image=image_path,
    #                                                                          output_image_path=os.path.join(temp_path, 'detected' + id + '.jpg'),
    #                                                                          minimum_percentage_probability=50, extract_detected_objects=True)
    #         combine_images(objects_path, id+'.jpg', save_result_folder)
    #         print("Detections: -----------------------------------------------------------------")
    #         for eachObject, eachObjectPath in zip(detections, objects_path):
    #             print(eachObject["name"] + " : " + str(eachObject["percentage_probability"]))
    #             print("Predictions: ----------------------------------------------------------------")
    #             preds, probs = prediction.predictImage(eachObjectPath, result_count=10)
    #             for eachPred, eachProb in zip(preds, probs):
    #                 if eachProb > 5.0:
    #                     print(str(eachPred) + " : " + str(eachProb))
    #     else:
    #         run = False
