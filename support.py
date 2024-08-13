import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_confusion_matrix(label_dict, cm, output_dir):
    # Ticklabels
    ticklabels = label_dict.values()
    
    # Plot
    plt.figure(figsize=(12,9))
    
    # Set font scale
    sns.set(font_scale=1.25)  # Increase the font scale
    
    # Confusion matrix
    sns.heatmap(cm, annot=True, fmt='.3g', xticklabels=ticklabels, yticklabels=ticklabels, cmap=plt.cm.Blues, cbar_kws={"shrink": .75}, annot_kws={"size": 15})  # Adjust annotation size
    plt.title('Confusion Matrix', fontsize=16)  # Increase title font size
    plt.xlabel('Predicted', fontsize=14)  # Increase x-label font size
    plt.ylabel('Actual', fontsize=14)  # Increase y-label font size
    plt.xticks(fontsize=14)  # Increase x-tick labels font size
    plt.yticks(fontsize=14)  # Increase y-tick labels font size
    
    # Save the figure
    plt.savefig(f'{output_dir}/Confusion_Matrix_{str(output_dir).split("/")[-1]}.png')
    plt.figure()
    # plt.clf()