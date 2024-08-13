import matplotlib.pyplot as plt
import seaborn as sns

# Default Theme
sns.set_theme(style="white", font_scale=1.2)  # Simpler font scale, default theme
common_figsize = (12, 8)  # Standard figure size
title_fontsize = 16
label_fontsize = 14
tick_fontsize = 12
legend_fontsize = 12

# Default color palette
palette = sns.color_palette("pastel")

def plot_confusion_matrix(label_dict, cm, model_name, output_dir):
    ticklabels = list(label_dict.values())
    
    plt.figure(figsize=common_figsize)
    sns.heatmap(cm, annot=True, fmt='.3g', 
                xticklabels=ticklabels, 
                yticklabels=ticklabels, 
                cmap='Blues', 
                cbar_kws={"shrink": .8})
    plt.title(f'Confusion Matrix ({model_name})', fontsize=title_fontsize)
    plt.xlabel('Predicted', fontsize=label_fontsize)
    plt.ylabel('Actual', fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    
    output_file = f'{output_dir}/{model_name}_confusion_matrix.png'
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

def plot_acc_loss_curve(epoch, history, model_name, output_dir):
    plt.figure(figsize=common_figsize)
    # Plot accuracy 
    plt.plot(epoch, history['train_acc_savings'], label="Train Accuracy", color=palette[0])
    plt.plot(epoch, history['val_acc_savings'], label="Validation Accuracy", color=palette[1])
    plt.title(f"Accuracy Plotting ({model_name})", fontsize=title_fontsize)
    plt.xlabel("Epoch", fontsize=label_fontsize)
    plt.ylabel("Accuracy", fontsize=label_fontsize)
    plt.legend(fontsize=legend_fontsize)   
    plt.savefig(f"{output_dir}/{model_name}_accuracy.png", bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=common_figsize)
    # Plot loss
    plt.plot(epoch, history['train_loss_savings'], label="Train Loss", color=palette[0])
    plt.plot(epoch, history['val_loss_savings'], label="Validation Loss", color=palette[1])
    plt.title(f"Loss Plotting ({model_name})", fontsize=title_fontsize)
    plt.xlabel("Epoch", fontsize=label_fontsize)
    plt.ylabel("Loss", fontsize=label_fontsize)
    plt.legend(fontsize=legend_fontsize)
    plt.savefig(f"{output_dir}/{model_name}_loss.png", bbox_inches='tight')
    plt.close()
