# Why 

I wanted to use transfer learning to solve some real problem. In this repo, let's do that. 

# Objective

!! got the idea from Kaggle. 

We have a small dataset of rural and urban images. We want to classify them as rural and urban image. But the dataset is very small though. 

This means that for proper generalization, we have to have a model that have lots of trainable parameters. but if we have lots of trainable parameters then they will definitely overfit the data thereby making it impossible to make a good classifier here in this case. 

Here, we definitely can use transfer learning. We will use some models that already have some learned feature representations. This will make sure that we are capturing some important features in the image. Will this be a great solution ? It looks like it will be a great solution as the output representation will have information about various features.
We are going to use Resnet for this which has been trained on imagenet originally. 

# Structure of the repo- 

1. load_model.py - file for loading the model. 
2. load_data.py - file for loading the train_dataloader and the test dataloader. 
3. notebook for tranfer learning.ipynb - the notebook where all the models have been trained. 

# Observations and Learnings 

1. The model that initializes on the trained model and tunes all the layers performs the best on training as well as 
test data. This is obvious , a lot of feature reporesentations are already present in it and gets tuned. Giving a really high accuracy on test dataset. If we trained the model for longer durations of time though, it beleive the model will still not generalize well on unseen data. 
2. The Model that initializes with trained resnet weights and only last fully connected layer is tuned, here the accuracy is 100 % , and i think it will also generalize very well as it has very detailed feature representation from the original resnet architecture. 
3. The model that initializes at random weights on resnet, performs horribly bad. The reason is simple, it has no good feature represeantion at beginning and a lot of parameters to train, therefore it overfits very quickly and gives 0 accuracy over test set. 

# Overall comment

1. For tasks, such as this , the test dataset should contain a larger variety of images, which gives a better idea , about how the model is going to perform if the data is completely unseen. 

# Other implementation learnings 

1. Some scripting techniques. 
2. dataloaders. 
3. Continous validation. 
4. visuals of learning curves. 
