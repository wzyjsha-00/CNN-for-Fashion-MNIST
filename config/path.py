import os

main_path = os.getcwd()


train_data_folder = os.path.abspath(os.path.join(main_path, 'data', "Fashion-MNIST", 'fashion_train_jpg_60000'))
train_label_file = os.path.abspath(os.path.join(main_path, 'data', "Fashion-MNIST", 'fashion_train_jpg_60000.txt'))
test_data_folder = os.path.abspath(os.path.join(main_path, 'data', "Fashion-MNIST", 'fashion_test_jpg_10000'))
test_label_file = os.path.abspath(os.path.join(main_path, 'data', "Fashion-MNIST", 'fashion_test_jpg_10000.txt'))

model_folder_path = os.path.abspath(os.path.join(main_path, "model"))

train_log_path = os.path.abspath(os.path.join(main_path, "model", "train.log"))
