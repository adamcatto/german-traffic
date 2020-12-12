import time
print('Hi Raj!\n')
    time.sleep(1)
    print('expect about 30 minutes of running time...')
    for i in range(3):
        print(3-i)
        time.sleep(1)
print('importing libraries...\n')
import os
import random
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
from skimage import feature
from scipy.cluster.vq import kmeans, vq
import sklearn
from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix
from collections import Counter
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from time import sleep
print('imports completed.\n')



def augment(img):
    IMG_SIZE = img.shape[0]
    inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = preprocessing.RandomRotation(factor=0.15)(inputs)
    x = preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1)(x)
    x = preprocessing.RandomFlip()(x)
    x = preprocessing.RandomContrast(factor=0.1)(x)
    return x


def load_images():
    """
    with open('train.csv') as f:
        train_paths = []
        for line in f.readlines():
            train_paths.append(line.split(',')[7])
    """
    num_classes = 43
    train_data = []
    train_labels = []

    train_data = np.load('np_train/train_images.npy')
    train_labels = np.load('np_train/train_labels.npy')
    print('total samples = ' + str(len(train_data)))
    if len(train_data) != 0:
        X_train, y_train, X_test, y_test = sklearn.model_selection.train_test_split(train_data, train_labels)
        return X_train, y_train, X_test, y_test
    for i in tqdm(range(num_classes)):
        image_list = [os.path.join('train/'+str(i)+'/', x) for x in os.listdir('train/' + str(i) + '/')]
        print(image_list[0])
        for img in image_list:
            try:
                img = np.array(Image.open(img).resize((64,64)))
                train_data.append(augment(img))
                train_labels.append(i)
            except:
                continue
                
            
            
    print(len(train_data))
    #np.save('np_train/train_images_augment.npy', train_data)
    #np.save('np_train/train_labels_augment.npy', train_labels)

    X_train, y_train, X_test, y_test = sklearn.model_selection.train_test_split(train_data, train_labels)
    #ohe = sklearn.preprocessing.OneHotEncoder()
    #ohe1 = sklearn.preprocessing.OneHotEncoder()
    #y_train = ohe.fit_transform(y_train)
    #y_test = ohe1.fit_transform(y_test)
    return X_train, y_train, X_test, y_test

def pad_img(img, max_height, max_width):
    img_height, img_width = img.shape
    top = (max_height - img_height)//2
    bottom = max_height - (img_height + top)
    left = (max_width - img_width)//2
    right = max_width - (img_width + left)
    image = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)
    return image
    



def compute_lbp(image: np.array, radius=1, neighborhood_size=8, method='uniform'):
    if len(image.shape) > 2:
        # convert to grayscale
        image = image[:, :, 0] * 0.3 + image[:, :, 1] * 0.59 + image[:, :, 2] * 0.11
    try:
        image_max, image_min = np.max(image), np.min(image)
        image = image - image_min / (image_max - image_min)
    except Warning:
        pass
    
    lbp = feature.local_binary_pattern(image, neighborhood_size, radius, method)
    histogram, edges = np.histogram(lbp.ravel(), bins = 250)
    #histogram /= np.sum(histogram)
    return histogram


def reduce_dimension(features, y):
    lda = LinearDiscriminantAnalysis()
    l = lda.fit_transform(features, y)
    return l


def extract_features(image_list, k=2): ########change back to 128
    # Extract features – SIFT
    key_points_list = []
    descriptions = []
    #sift = cv2.SIFT()
    brisk = cv2.BRISK_create(16)
    print('extracting features...')
    for i in tqdm(range(len(image_list))):
        img = image_list[i]
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        key_points, description = brisk.detectAndCompute(grayscale_img, None)
        if description is None:
            description = np.zeros((1,64))
        descriptions.append(description)

    print('stacking features...')
    stacked_descriptions = descriptions[0]
    stacked_descriptions = np.concatenate(descriptions)
    stacked_descriptions = stacked_descriptions.astype(float)

    
    print('features stacked. clustering on k=' + str(k) + ' visual words...')
    code_book, var = kmeans(stacked_descriptions, k, 15)

    print('clustering complete. computing features...')
    features = np.zeros((len(image_list), k), "float32")
    for i in tqdm(range(len(image_list))):
        words, dist = vq(descriptions[i], code_book)
        for w in words:
            features[i][w] += 1
    print('features computed.')
    return features, code_book


def extract_features_reduced(image_list, y, k=2): ########change back to 128
    
    key_points_list = []
    descriptions = []
    
    brisk = cv2.BRISK_create(16)
    print('extracting features...')
    for i in tqdm(range(len(image_list))):
        img = image_list[i]
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        key_points, description = brisk.detectAndCompute(grayscale_img, None)
        if description is None:
            description = np.zeros((1,64))
        descriptions.append(description)

    descriptions = reduce_dimension(descriptions, y)
    print('stacking features...')
    stacked_descriptions = descriptions[0]
    stacked_descriptions = np.concatenate(descriptions)
    stacked_descriptions = stacked_descriptions.astype(float)
    
    print('features stacked. reducing dimension...')
    stacked_descriptions = reduce_dimension(stacked_descriptions, y)

    print('features stacked. clustering on k=' + str(k) + ' visual words...')
    code_book, var = kmeans(stacked_descriptions, k, 15)

    print('clustering complete. computing features...')
    features = np.zeros((len(image_list), k), "float32")
    for i in tqdm(range(len(image_list))):
        words, dist = vq(descriptions[i], code_book)
        for w in words:
            features[i][w] += 1
    print('features computed.')
    return features, code_book


def extract_features_test(image_list, code_book, y, k=2, reduced=False): ########change back to 128
    brisk = cv2.BRISK_create(16)
    descriptions = []
    key_points_list = []
    for i in tqdm(range(len(image_list))):
        img = image_list[i]
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        key_points, description = brisk.detectAndCompute(grayscale_img, None)
        if description is None:
            description = np.zeros((1,64))
        descriptions.append(description)

    if reduced:
        lda = LinearDiscriminantAnalysis()
        descriptions = lda.fit_transform(descriptions, y)
    
    features = np.zeros((len(image_list), k), "float32")
    for i in tqdm(range(len(image_list))):
        words, dist = vq(descriptions[i], code_book)
        for w in words:
            features[i][w] += 1
    return features



def compute_lbps_image_set(image_set):
    print('computing LBPs for each image...\n')
    lbps = []
    for x in tqdm(image_set):
        lbp = compute_lbp(x)
        lbps.append(lbp)
    lbps = np.array(lbps)
    print('\nlocal binary patterns computed.\n')
    #print('sample LBP: ' + lbps[-1])
    return lbps


def stack_patterns(pattern_list):
    print('shape: ' + str(pattern_list.shape) + ' of ' + str(pattern_list) + ' of type ' + str(type(pattern_list)))
    print('stacking LBPs...\n')
    print(pattern_list[0])
    s = np.concatenate(pattern_list)
    print('LBPs stacked, ready to cluster.\n')
    print('LBP stack shape: ' + str(s.shape))
    return s


def generate_bow(pattern_list, stacked_patterns, k=2):
    print('clustering on k=' + str(k) + ' visual words...')
    voc, var = kmeans(stacked_patterns, k, 5)
    print('clustering complete.')
    print(voc, var, voc.shape, var.shape)
    print('LBP shape: ' + str(pattern_list[1].shape))
    print('voc shape: ' + str(voc.shape))
    features = np.zeros((len(pattern_list), k), "float32")
    print('generating features...')
    for i in tqdm(range(len(pattern_list))):
        words, dist = vq(pattern_list[i], voc)
        for w in words:
            features[i][w] += 1
    print('features generated.')
    print('feature shape: ' + str(features.shape))
    return features


def train_svm(x_train, y_train):
    classifier = LinearSVC(max_iter=1000000, kernel='rbf')
    print('training SVM...')
    print('x_train shape: ' + str(x_train.shape))
    #print('y_train shape: ' + str(y_train.shape))
    classifier.fit(x_train, y_train)
    print('SVM trained.')
    return classifier


def save_confusion_matrix(classifier, X_test, y_test, class_names):
    disp = plot_confusion_matrix(classifier, X_test, y_test)
    disp.ax_.set_title('confusion matrix SVM')
    print(disp.confusion_matrix)
    plt.savefig('confusion_matrix_svm.png')


def test_svm(classifier, x_test, y_test):
    print('testing SVM...')
    predictions = classifier.predict(x_test)
    print('SVM predictions made.')
    correct_predictions = [i for i in range(len(y_test)) if y_test[i] == predictions[i]]
    print('Predicted correctly ' + str(100 * len(correct_predictions) / len(predictions)) + ' percent of the time')
    save_confusion_matrix(classifier, x_test, y_test, set(y_test))
    return predictions, correct_predictions

# code adapted from https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
def train_efficient_net(x_train, y_train, num_classes):
    IMG_SIZE = x_train[0].shape[0]
    NUM_CLASSES = 43
    inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    #x = augment()(inputs)
    x=inputs
    model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild output layer for classification
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = tf.keras.layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    # Compile model
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
  
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    
    model.fit(x_train, y_train, epochs=16, callbacks=[
        tf.keras.callbacks.ModelCheckpoint('fine_tuned_efficient_net.h5', verbose=1, save_best_model=True)
    ])

    return model


def test_efficient_net(model, x_test, y_test):
    predictions = np.argmax(model.predict(x_test), axis=-1)
    accurates = [i for i in range(len(predictions)) if predictions[i] == y_test[i]]
    accuracy = len(accurates) / len(predictions)
    return accuracy



def main():
    ####################
    ##### training #####
    ####################

    print('loading images and labels...')
    x_train, x_test, y_train, y_test = load_images()
    print('loaded data.\nshape of images:')
    print(set(x.shape for x in x_train))
 
    features, code_book = extract_features(x_train)
    
    print('reducing dimensionality')
    reduced_features = extract_features_reduced(x_train, y_train)

    print('training SVM with RBF kernel on original features...')
    rbf_classifier = train_svm(features, y_train)

    print('training SVM on reduced dimension features...')
    classifier_reduced = train_svm(reduced_features, y_train)

    print('training logistic regression on features...')
    log_reg = LogisticRegression(max_iter = 1000000)
    log_reg.fit(features, y_train)

    print('training multinomial naive bayes on features...')
    nb = MultinomialNB()
    nb.fit(features, y_train)
    
    print('training complete.\n')
    
    print('loading fine-tuned efficient-net CNN...')
    en_model = tf.keras.models.load_model('fine_tuned_efficient_net.h5')
    #print('training efficientNet...')
    #model = train_efficient_net(x_train, y_train, num_classes=43)

    ####################
    ##### testing ######
    ####################

    print('beginning testing...')
    
    features_test = extract_features_test(x_test, code_book, y_test)
    reduced_features_test = extract_features_test(features_test, y_test, reduced=True)

    accuracy_results = {
        'RBF kernel SVM': None,
        'reduced dimension SVM': None,
        'Logistic regression': None,
        'Naive Bayes': None,
        'Fine-tuned efficient-net': None
    }
    
   
    print('testing SVM RBF kernel on regular test features...')
    print('regular dimension accuracy: ')
    predictions, correct_predictions = test_svm(rbf_classifier, features_test, y_test)
    accuracy_results['RBF kernel SVM'] = len(correct_predictions) / len(predictions)

    print('testing SVM on reduced test features...')
    print('reduced dimension accuracy: ')
    predictions, correct_predictions = test_svm(classifier_reduced, reduced_features_test, y_test)
    accuracy_results['reduced dimension SVM'] = len(correct_predictions) / len(predictions)

    print('testing logistic regression on regular test features...')
    print('score: ')
    s = log_reg.score(features_test, y_test)
    print(s)
    accuracy_results['Logistic regression'] = s

    print('testing naive bayes classifier on regular test features...')
    print('score: ')
    s = nb.score(features_test, y_test)
    print(s)
    accuracy_results['Naive Bayes'] = s
    
    print('architecture of CNN: ')
    print(en_model.summary())
    print('testing fine-tuned efficient-net CNN...')
    accuracy = test_efficient_net(en_model, x_test, y_test)
    print('accuracy of efficient net fine tuned: ' + str(accuracy))
    accuracy_results['Fine-tuned efficient-net'] = accuracy

    print('results:')
    print(accuracy_results)


if __name__ == '__main__':
    main()

