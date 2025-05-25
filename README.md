# SoilClassification_annam

SoilClassification_annam
🧠 What's the goal?

Building a soil classification system using images. Based on a picture of soil, the model will classify it into one of four types:

               Alluvial soil
               Black Soil
               Clay soil
               Red soil
Soil_Classification_Image
![image](https://github.com/user-attachments/assets/88bdac9b-88f1-4884-883f-7a58edc268d1)


🚀 1. Setup and Imports Importing necessary libraries like:

PyTorch (for model, training, etc.)

Torchvision (for image models like ResNet)

sklearn (for F1 score calculation)

Pandas & PIL (for reading image names and image loading)

📁 2. Custom Dataset: SoilDataset

This is a custom PyTorch Dataset that:

Loads images from a folder

Matches them with their labels from a .csv file

Applies image transformations

File Organizing:

                      soil_classification-2025/
                      ├── train/
                      │   ├── img_1.jpg
                      │   ├── img_2.jpg
                      │   └── ...
                      ├── test/
                      │   ├── img_101.jpg
                      │   ├── img_102.jpg
                      │   └── ...
                      ├── train_labels.csv
                      └── test_ids.csv
🖼️ 3. Image Transformations

You apply data augmentation on the training images to help the model generalize better:

Testing images use only basic resizing and cropping

📊 4. Load and Prepare Data

You read the CSV file containing training labels and prepare your custom dataset and DataLoader:

⚖️ 5. Handle Class Imbalance

If some soil types appear more than others, the model might be biased. You fix this by giving more weight to the minority classes:

These weights are passed to the loss function (CrossEntropyLoss) to balance the learning.

🧠 6. Model: ResNet50

You load a pretrained ResNet-50 model and modify its last layer to output 4 classes:

This helps the model start with good image features learned from big datasets like ImageNet.

🛠️ 7. Loss Function & Optimizer

You use:

CrossEntropyLoss with your class weights

Adam optimizer (a good general-purpose optimizer)

🔁 8. Training Loop with F1 Score

Define a function to train the model for a few epochs.

Each epoch:

Loops through batches of training data

Runs forward and backward passes

Tracks predictions and computes macro F1-score (which treats all classes equally)

This helps you monitor how well the model is learning beyond just accuracy.

🔮 9. Prediction on Test Set

You define a predict() function to make predictions on unseen test images.

For each image:

Load it and apply the same test transforms

Run it through the model

Pick the class with the highest score

It returns the soil type name like "Red soil", not the numeric label.

📝 10. Generating Submission

Load test image IDs from a CSV

Predict their labels using your trained model

Save results in a format required for submission

📊 Model Highlights

🧠 Transfer Learning with ResNet-50

🧪 Data Augmentation for better generalization

⚖️ Class Weights to handle imbalanced dataset

📈 F1 Score Tracking to monitor performance

📁 File Structure

File/Folder	Purpose
train.py	Training loop using ResNet-50
predict.py	Generate predictions for test set
train_labels.csv	Labels for training images
test_ids.csv	Filenames for test images
sample_submission.csv	Output submission file
🙌 Acknowledgements

PyTorch for deep learning

Torchvision for pretrained models

scikit-learn for F1 score

PIL & Pandas for image and data handling
