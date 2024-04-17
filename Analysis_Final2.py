#!/usr/bin/env python
# coding: utf-8

# # Multi-Class Text Classification using ANN & BERT Algorithms

# In[1]:


# Importing the required libraries
from req_library_loading import *


# In[2]:


# Data Loading
goemotions_df = pd.read_csv("D:/ML/ml_project/full_dataset/goemotions_fd.csv")
print("Shape of the Dataframe before:",goemotions_df.shape)
goemotions_df = goemotions_df.head(100000)
print("Shape of the Dataframe before:",goemotions_df.shape)


# In[3]:


# import os

# folder_path = 'D:\\Tarun\\ml_project\\embedded_dataset\\'
# file_name = "xtmp_embedded.csv"
# file_path = os.path.join(folder_path, file_name)
# batch_size = 100

# counter=0
# records_processed=0

# # loop through the dataframe in batches
# for i in range(0, len(goemotions_df), batch_size):
#     batch_df = goemotions_df.iloc[i:i+batch_size]
    
#     x_tmp=batch_df[['text']]
#     y_tmp=batch_df[['admiration','amusement','anger','annoyance',
#                     'approval','caring','confusion','curiosity','desire',
#                     'disappointment','disapproval','disgust','embarrassment',
#                     'excitement','fear','gratitude','grief','joy','love',
#                     'nervousness','optimism','pride','realization','relief',
#                     'remorse','sadness','surprise','neutral']]

#     # Shorten the 28 Emotions into 7 Segments
#     y_tmp=pd.DataFrame(y_tmp.apply(shorten_emotions, axis=1).tolist(), columns=ekman_mapping.keys())
    
#     # Applying the Text Preprocessing Pipeline on the Text Column in Train, Validation and Test Datasets seperately
#     x_tmp['text'] = x_tmp['text'].apply(text_preprocessing_pipeline)
#     xtmp_embedded=get_embeddings(x_tmp['text'])
#     # print("Shape of X_Train_Embedded:",xtrain_embedded.shape)
    
#     counter+=1
#     records_processed+=batch_size
#     print("Batch", counter, " ==>", records_processed, "rows processed")
    
#     if os.path.exists(file_path):
#         # print("The file exists in the folder.")
#         # append data frame to CSV file
#         xtmp_embedded.to_csv(folder_path+"xtmp_embedded.csv", mode='a', index=False, header=False)
#         y_tmp.to_csv(folder_path+"y_shortened.csv", mode='a', index=False, header=False)
#     else:
#         # print("The file does not exist in the folder.")
#         xtmp_embedded.to_csv(folder_path+"xtmp_embedded.csv", index=False)
#         y_tmp.to_csv(folder_path+"y_shortened.csv", index=False)
#     # print("\n")


# In[ ]:





# In[17]:


# Importing the required libraries
from req_library_loading import *


# In[63]:


x_tmp=pd.read_csv("D:/Tarun/ml_project/embedded_dataset/xtmp_embedded.csv")
y_tmp=pd.read_csv("D:/Tarun/ml_project/embedded_dataset/y_shortened.csv")

# x_tmp=x_tmp.head(25000)
# y_tmp=y_tmp.head(25000)


# In[64]:


# Count the number of ones in each column
ones_per_column = np.sum(y_tmp, axis=0)

# Plot a bar chart of the results
plt.bar(range(y_tmp.shape[1]), ones_per_column)
plt.xticks(range(y_tmp.shape[1]), y_tmp.columns)
plt.ylabel('Counts')
plt.title('Count of Emotions across dataset')
# plt.savefig('y_tmp.png')
plt.show()


# In[65]:


print(x_tmp.shape)
print(y_tmp.shape)


# In[66]:


# Train, Validation and Test Split
tmp_op = train_val_test_split(X=x_tmp, Y=y_tmp, tt_ratio=0.1, tv_ratio=0.1)
xtrain_embedded, y_train, xval_embedded, y_val, xtest_embedded, y_test = tmp_op['x_train'], tmp_op['y_train'], \
                                                 tmp_op['x_val'], tmp_op['y_val'], \
                                                 tmp_op['x_test'], tmp_op['y_test']

print("Shape of X_Train:",xtrain_embedded.shape)
print("Shape of Y_Train:",y_train.shape)
print("=======================================================")
print("Shape of X_Val:",xval_embedded.shape)
print("Shape of Y_Val:",y_val.shape)
print("=======================================================")
print("Shape of X_Test:",xtest_embedded.shape)
print("Shape of Y_Test:",y_test.shape)

# del goemotions_df


# ---
# # Model Building

# In[51]:


# ANN
n_inputs, n_classes = xtrain_embedded.shape[1], y_train.shape[1]

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Create the model
basic_model = Sequential()
basic_model.add(Dense(512, input_dim=n_inputs, activation='relu'))
basic_model.add(Dense(256, activation='relu'))
basic_model.add(Dropout(0.5))
basic_model.add(Dense(64, activation='relu'))
basic_model.add(Dense(n_classes, activation='sigmoid'))

# Compile the model
basic_model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Fit data to model
basicann_history = basic_model.fit(xtrain_embedded, y_train,
                batch_size=64,
                epochs=15,
                verbose=1,
                validation_split=0.2)

# Generate generalization metrics
score = basic_model.evaluate(xval_embedded, y_val, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')


# In[52]:


# Plot the training and validation loss graphs
plt.plot(basicann_history.history['loss'], label='Training Loss')
plt.plot(basicann_history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[8]:


# Generate generalization metrics
train_score = basic_model.evaluate(xtrain_embedded, y_train, verbose=0)
print(f'Train loss: {train_score[0]} / Train accuracy: {train_score[1]}')

val_score = basic_model.evaluate(xval_embedded, y_val, verbose=0)
print(f'Test loss: {val_score[0]} / Test accuracy: {val_score[1]}')


# In[9]:


def objective(trial):
    keras.backend.clear_session()

    # Model Design
    n_layers = trial.suggest_int('n_layers', 1, 10) #optimum number of hidden layers
    model = keras.Sequential()
    for i in range(n_layers):

        #optimum number of hidden nodes
        num_hidden = trial.suggest_int(f'n_units_l{i}', 32, xtrain_embedded.shape[1], log=True)
        
        #optimum activation function
        model.add(keras.layers.Dense(num_hidden, input_shape=(xtrain_embedded.shape[1],),
                                     activation=trial.suggest_categorical(f'activation{i}', ['relu','relu']),
                                     kernel_initializer=trial.suggest_categorical(f'kernel_initializer{i}', ['LecunUniform','he_uniform','glorot_uniform']),
                                     # bias_initializer=trial.suggest_categorical(f'bias_initializer{i}', ['Ones','Orthogonal','variance_scaling','truncated_normal'])
                                    ))
        #optimum dropout value
        model.add(keras.layers.Dropout(rate = trial.suggest_float(f'dropout{i}', 0.0, 0.7))) 
    
    model.add(keras.layers.Dense(7,activation=tf.keras.activations.sigmoid)) #output Layer
    
    # Other Components
    val_ds = (xval_embedded, y_val)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=1,min_lr=1e-05,verbose=0)
    early_stoping = EarlyStopping(monitor="val_loss",min_delta=0,patience=5,verbose=0,mode="auto", baseline=None,restore_best_weights=True)
    model.compile(loss='binary_crossentropy',metrics='categorical_crossentropy', optimizer='Adam')
    
    #optimum batch size
    history = model.fit(xtrain_embedded, y_train, validation_data=val_ds, epochs=20, callbacks=[reduce_lr,early_stoping],
                        verbose=0, batch_size=trial.suggest_int('size', 8, 1024))
    return min(history.history['val_loss'])


# In[10]:


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50, timeout=1200)
print("Number of finished trials: {}".format(len(study.trials)))
print("Best trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))


# In[11]:


# Hyper Parameter Optimization
optuna.visualization.plot_optimization_history(study)


# In[12]:


print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# In[53]:


from keras.models import Sequential

# Build the Final Best model of ANN
bestann_model=auto_construct(best_params=trial.params, n_inputs=n_inputs, n_classes=n_classes)
# bestann_model.summary()

# Compile the model
bestann_model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Fit data to model
bestann_history = bestann_model.fit(xtrain_embedded, y_train,
                batch_size=250,
                epochs=50,
                verbose=0,
                validation_split=0.2)


# In[53]:


# Generate Evaluation metrics
bestann_model_train_score = bestann_model.evaluate(xtrain_embedded, y_train, verbose=0)
print(f'Train loss: {bestann_model_train_score[0]} / Train accuracy: {bestann_model_train_score[1]}')

bestann_model_val_score = bestann_model.evaluate(xval_embedded, y_val, verbose=0)
print(f'Valdation loss: {bestann_model_val_score[0]} / Validation accuracy: {bestann_model_val_score[1]}')


# In[68]:


bestann_model_test_score = bestann_model.evaluate(xtest_embedded, y_test, verbose=0)
print(f'Test loss: {bestann_model_test_score[0]} / Test accuracy: {bestann_model_test_score[1]}')


# In[54]:


# Plot the training and validation loss graphs
plt.plot(bestann_history.history['loss'], label='Training Loss')
plt.plot(bestann_history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[14]:


bestann_model.summary()


# In[15]:


# Graphs


# In[16]:


bestann_model_train_probs = bestann_model.predict(xtrain_embedded, verbose=0)
bestann_model_val_probs = bestann_model.predict(xval_embedded, verbose=0)

train_threshold=best_threshold_val(y_true=y_train, y_pred_probs=bestann_model_train_probs)
val_threshold=best_threshold_val(y_true=y_val, y_pred_probs=bestann_model_val_probs)

print("Train threshold :", train_threshold)
print("Validation threshold :", val_threshold)
print("Average Threshold :", round(np.mean([train_threshold, val_threshold]), 3))


# In[69]:


threshold=round(np.mean([train_threshold, val_threshold]), 3)
print("Optimal Threshold Value :", threshold)

bestann_model_train_predictions = (bestann_model_train_probs >= threshold).astype(int)
bestann_model_val_predictions = (bestann_model_val_probs >= threshold).astype(int)

bestann_model_test_probs = bestann_model.predict(xtest_embedded, verbose=0)
bestann_model_test_predictions = (bestann_model_test_probs >= threshold).astype(int)

metrics(ytrain_true=y_train, ytrain_pred=bestann_model_train_predictions, ytest_true=y_val, ytest_pred=bestann_model_val_predictions, 
        target_names=y_train.columns, test_included=True, yTest_true=y_test, yTest_pred=bestann_model_test_predictions)


# In[26]:


bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

def get_embeddings_demo(sentences):
    preprocessed_text = bert_preprocess(sentences)
    return bert_encoder(preprocessed_text)['pooled_output']

def emotion_prediction_demo(text, model, threshold):
    for sent in text:
        preproc_text = text_preprocessing_pipeline(sent)
        embeds = get_embeddings_demo([preproc_text])
        probs = model.predict(embeds, verbose=0)
        predictions = (probs >= threshold).astype(int)
        print(sent, "->", extract_emotions_from_probs(predictions))


# In[27]:


text=['That game hurt.',
      " >sexuality shouldn’t be a grouping category It makes you different from othet ppl so imo it fits the definition of grouping "]

emotion_prediction_demo(text, model=bestann_model, threshold=threshold)    


# ---
# # BERT MODEL

# In[71]:


# Data Loading
goemotions_df = pd.read_csv("D:/Tarun/ml_project/full_dataset/goemotions_fd.csv")
print("Shape of the Dataframe before:",goemotions_df.shape)
goemotions_df = goemotions_df.head(10000)
print("Shape of the Dataframe before:",goemotions_df.shape)

x_tmp=goemotions_df[['text']]
y_tmp=goemotions_df[['admiration','amusement','anger','annoyance',
                     'approval','caring','confusion','curiosity','desire',
                     'disappointment','disapproval','disgust','embarrassment',
                     'excitement','fear','gratitude','grief','joy','love',
                     'nervousness','optimism','pride','realization','relief',
                     'remorse','sadness','surprise','neutral']]

# Shorten the 28 Emotions into 7 Segments
y_tmp=pd.DataFrame(y_tmp.apply(shorten_emotions, axis=1).tolist(), columns=ekman_mapping.keys())

# Train, Validation and Test Split
tmp_op = train_val_test_split(X=x_tmp, Y=y_tmp, tt_ratio=0.1, tv_ratio=0.1)
x_train, y_train, x_val, y_val, x_test, y_test = tmp_op['x_train'], tmp_op['y_train'], \
                                                 tmp_op['x_val'], tmp_op['y_val'], \
                                                 tmp_op['x_test'], tmp_op['y_test']

print("Shape of X_Train:",x_train.shape)
print("Shape of Y_Train:",y_train.shape)
print("=======================================================")
print("Shape of X_Val:",x_val.shape)
print("Shape of Y_Val:",y_val.shape)
print("=======================================================")
print("Shape of X_Test:",x_test.shape)
print("Shape of Y_Test:",y_test.shape)

del goemotions_df # Deleting the huge dataset for memory optimization

# Applying the Text Preprocessing Pipeline on the Text Column in Train, Validation and Test Datasets seperately
x_train['text'] = x_train['text'].apply(text_preprocessing_pipeline)
x_val['text'] = x_val['text'].apply(text_preprocessing_pipeline)
x_test['text'] = x_test['text'].apply(text_preprocessing_pipeline)


# In[56]:


n_inputs, n_classes = xtrain_embedded.shape[1], y_train.shape[1]

# Defining the BERT Model for the given data
i = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
x = bert_preprocess(i)
x = bert_encoder(x)
x = tf.keras.layers.Dropout(0.2, name="dropout")(x['pooled_output'])
x = tf.keras.layers.Dense(n_classes, activation='sigmoid', name="output")(x)

basic_bert_model = tf.keras.Model(i, x)

# METRICS = [tf.keras.metrics.CategoricalAccuracy(name="accuracy")]

earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", 
                                                      patience = 3,
                                                      restore_best_weights = True)

basic_bert_model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

basic_bert_history = basic_bert_model.fit(x_train, y_train, epochs = 20, batch_size=256, verbose=1, validation_data = (x_val, y_val), callbacks = [earlystop_callback])

# Generate generalization metrics
basic_bert_score = basic_bert_model.evaluate(x_val, y_val, verbose=0)
print(f'Test loss: {basic_bert_score[0]} / Test accuracy: {basic_bert_score[1]}')


# In[57]:


# Plot the training and validation loss graphs
plt.plot(basic_bert_history.history['loss'], label='Training Loss')
plt.plot(basic_bert_history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[32]:


# Generate Evaluation metrics
basic_bert_train_score = basic_bert_model.evaluate(x_train, y_train, verbose=0)
print(f'Train loss: {basic_bert_train_score[0]} / Train accuracy: {basic_bert_train_score[1]}')

basic_bert_val_score = basic_bert_model.evaluate(x_val, y_val, verbose=0)
print(f'Valdation loss: {basic_bert_val_score[0]} / Validation accuracy: {basic_bert_val_score[1]}')


# In[33]:


from tensorflow.keras.optimizers import Adam
# bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
# bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

# Fine-Tuning the BERT Model
def bert_objective(trial):
    keras.backend.clear_session()
    n_layers = trial.suggest_int('n_layers', 1, 10) #optimum number of hidden layers

    # Model Design - BERT
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessed_text = bert_preprocess(text_input)
    outputs = bert_encoder(preprocessed_text)
    link = tf.keras.layers.Dropout(0.1)(outputs['pooled_output'])

    for i in range(n_layers):
        
        #optimum number of hidden neurons
        num_hidden = trial.suggest_int(f'n_units_l{i}', 32, 768, log=True)
        
        #optimum layer config
        link = keras.layers.Dense(num_hidden, activation=trial.suggest_categorical(f'activation{i}', ['relu']),
                                  kernel_initializer=trial.suggest_categorical(f'kernel_initializer{i}', ['LecunUniform','he_uniform','glorot_uniform']),
                                  # bias_initializer=trial.suggest_categorical(f'bias_initializer{i}', ['Ones','Orthogonal','variance_scaling','truncated_normal'])
                                  )(link)
        #optimum dropout value
        link = keras.layers.Dropout(rate = trial.suggest_float(f'dropout{i}', 0.0, 0.7))(link)
    
    oplayer = tf.keras.layers.Dense(n_classes, activation='sigmoid', name="output")(link) #output Layer
    
    # Defining the BERT Model Blueprint
    bert_finetuned_model = tf.keras.Model(inputs=[text_input], outputs = [oplayer])
    
    # Other Components
    val_ds_bert = (x_val, y_val)
    reduce_lr_bert = ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=1,min_lr=1e-05,verbose=0)
    early_stoping_bert = EarlyStopping(monitor="val_loss",min_delta=0,patience=5,verbose=0,mode="auto", baseline=None,restore_best_weights=True)
    METRICS_bert = [tf.keras.metrics.BinaryAccuracy(name='accuracy'), 
                    tf.keras.metrics.Precision(name='precision'), 
                    tf.keras.metrics.Recall(name='recall')]
    # optimizer_for_bert = Adam(learning_rate=5e-05, epsilon=1e-08, decay=0.01, clipnorm=1.0)
    
    bert_finetuned_model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=METRICS_bert)
    
    history_bert = bert_finetuned_model.fit(x_train, y_train, validation_data=val_ds_bert, epochs=15, 
                                            callbacks=[reduce_lr_bert, early_stoping_bert],
                                            verbose=0, batch_size=trial.suggest_int('size', 8, 2048))
    
    return min(history_bert.history['val_loss'])


# In[34]:


bert_study = optuna.create_study(direction="minimize")
bert_study.optimize(bert_objective, n_trials=5, timeout=1200)
print("Number of finished trials: {}".format(len(bert_study.trials)))
print("Best trial:")
bert_trial = study.best_trial
print("  Value: {}".format(bert_trial.value))


# In[58]:


# Hyper Parameter Optimization Visualization
optuna.visualization.plot_optimization_history(bert_study)


# In[36]:


print("  Params: ")
for key, value in bert_trial.params.items():
    print("    {}: {}".format(key, value))


# In[37]:


bert_finetuned_model = auto_construct_bert(best_params=bert_trial.params, n_classes=n_classes)
bert_finetuned_model.summary()


# In[39]:


from tensorflow.keras.optimizers import Adam

reduce_lr_bert = ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=1,min_lr=1e-05,verbose=0)
early_stoping_bert = EarlyStopping(monitor="val_loss",min_delta=0,patience=5,verbose=0,mode="auto", baseline=None,restore_best_weights=True)
METRICS_bert = [tf.keras.metrics.BinaryAccuracy(name='accuracy'), 
                tf.keras.metrics.Precision(name='precision'), 
                tf.keras.metrics.Recall(name='recall')]
optimizer_for_bert = Adam(learning_rate=5e-05, epsilon=1e-08, decay=0.01, clipnorm=1.0)
val_ds_bert = (x_val, y_val)

bert_finetuned_model.compile(optimizer=optimizer_for_bert, loss='binary_crossentropy', metrics=METRICS_bert)

history_bert = bert_finetuned_model.fit(x_train, y_train, validation_data=val_ds_bert, epochs=15, 
                                        callbacks=[reduce_lr_bert, early_stoping_bert],
                                        verbose=0, batch_size=trial.suggest_int('size', 8, 2048))


# In[44]:


# Plot the training and validation loss graphs
plt.plot(history_bert.history['loss'], label='Training Loss')
plt.plot(history_bert.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[40]:


# Generate Evaluation metrics
bert_finetuned_model_trainscore = bert_finetuned_model.evaluate(x_train, y_train, verbose=0)
print(f'Train loss: {bert_finetuned_model_trainscore[0]} / Train accuracy: {bert_finetuned_model_trainscore[1]}')

bert_finetuned_model_valscore = bert_finetuned_model.evaluate(x_val, y_val, verbose=0)
print(f'Valdation loss: {bert_finetuned_model_valscore[0]} / Validation accuracy: {bert_finetuned_model_valscore[1]}')


# In[60]:


bert_finetuned_model_test_score = bert_finetuned_model.evaluate(x_test, y_test, verbose=0)
print(f'Test loss: {bert_finetuned_model_test_score[0]} / Test accuracy: {bert_finetuned_model_test_score[1]}')


# In[41]:


bert_finetuned_model_train_probs = bert_finetuned_model.predict(x_train, verbose=0)
bert_finetuned_model_val_probs = bert_finetuned_model.predict(x_val, verbose=0)

bert_finetuned_model_train_threshold=best_threshold_val(y_true=y_train, y_pred_probs=bert_finetuned_model_train_probs)
bert_finetuned_model_val_threshold=best_threshold_val(y_true=y_val, y_pred_probs=bert_finetuned_model_val_probs)

print("Train threshold :", bert_finetuned_model_train_threshold)
print("Validation threshold :", bert_finetuned_model_val_threshold)
print("Average Threshold :", round(np.mean([bert_finetuned_model_train_threshold, bert_finetuned_model_val_threshold]), 3))


# In[72]:


finedtunedbert_threshold=0.1
# finedtunedbert_threshold=round(np.mean([bert_finetuned_model_train_threshold, bert_finetuned_model_val_threshold]), 3)
# print("Optimal Threshold Value for Fine-tuned BERT Model:", finedtunedbert_threshold)

bestbert_model_train_predictions = (bert_finetuned_model_train_probs >= finedtunedbert_threshold).astype(int)
bestbert_model_val_predictions = (bert_finetuned_model_val_probs >= finedtunedbert_threshold).astype(int)

bert_finetuned_model_test_probs = bert_finetuned_model.predict(x_test, verbose=0)
bert_finetuned_model_test_predictions = (bert_finetuned_model_test_probs >= threshold).astype(int)

metrics(ytrain_true=y_train, ytrain_pred=bestbert_model_train_predictions, ytest_true=y_val, ytest_pred=bestbert_model_val_predictions, 
        target_names=y_train.columns, test_included=True, yTest_true=y_test, yTest_pred=bert_finetuned_model_test_predictions)


# In[73]:


tf.saved_model.save(basic_model, 'basicann_model')
tf.saved_model.save(bestann_model, 'bestann_model')
tf.saved_model.save(basic_bert_model, 'basic_bert_model')
tf.saved_model.save(bert_finetuned_model, 'bert_finetuned_model')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[29]:


y_predicted = model.predict(x_test)
y_predicted = y_predicted.flatten()

import numpy as np

y_predicted = np.where(y_predicted > 0.5, 1, 0)
y_predicted


# In[32]:


sample_dataset = [
 'You can win a lot of money, register in the link below',
 'You have an iPhone 10, spin the image below to claim your prize and it will be delivered in your door step',
 'You have an offer, the company will give you 50% off on every item purchased.',
 "Hey Bravin, don't be late for the meeting tomorrow will start lot exactly 10:30 am",
 "See you monday, we have alot to talk about the future of this company ."
]


# In[33]:


model.predict(sample_dataset)


# In[ ]:




