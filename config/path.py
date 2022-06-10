import os

main_path = os.getcwd()


train_data_folder = os.path.abspath(os.path.join(main_path, 'data', "Fashion-MNIST", 'fashion_train_jpg_60000'))
train_label_file = os.path.abspath(os.path.join(main_path, 'data', "Fashion-MNIST", 'fashion_train_jpg_60000.txt'))
test_data_folder = os.path.abspath(os.path.join(main_path, 'data', "Fashion-MNIST", 'fashion_test_jpg_10000'))
test_label_file = os.path.abspath(os.path.join(main_path, 'data', "Fashion-MNIST", 'fashion_test_jpg_10000.txt'))


data_folder_path = os.path.abspath(os.path.join(main_path, "data"))
image_folder_path = os.path.abspath(os.path.join(main_path, "data", "images"))
label_file_path = os.path.abspath(os.path.join(main_path, "data", "labels.csv"))

model_folder_path = os.path.abspath(os.path.join(main_path, "model"))

pre_trained_resnet34 = os.path.abspath(os.path.join(main_path, "model", "pre_trained", "resnet34-b627a593.pth"))
pre_trained_resnet50 = os.path.abspath(os.path.join(main_path, "model", "pre_trained", "resnet50-0676ba61.pth"))
pre_trained_resnet101 = os.path.abspath(os.path.join(main_path, "model", "pre_trained", "resnet101-5d3b4d8f.pth"))
pre_trained_resnet101_2 = os.path.abspath(os.path.join(main_path, "model", "pre_trained", "resnext101_20.pth"))

train_log_path = os.path.abspath(os.path.join(main_path, "model", "train.log"))
