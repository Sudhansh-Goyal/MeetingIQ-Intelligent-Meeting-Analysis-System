import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import pandas as pd
from pathlib import Path

# --- TOPIC SEGMENTATION XAI ---

def plot_segmentation_coherence(file_id, coherence, depths, threshold, utterances, output_dir):
    """
    Visualises TextTiling coherence and depth scores to show why boundaries were chosen.
    Valleys in coherence (high depth) = Topic boundaries.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot Coherence
    plt.subplot(2, 1, 1)
    plt.plot(coherence, color='blue', label='Coherence (Cosine Similarity)')
    plt.title(f"XAI: Topic Coherence Map - {file_id}")
    plt.ylabel("Coherence")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot Depth Scores
    plt.subplot(2, 1, 2)
    plt.bar(range(len(depths)), depths, color='orange', alpha=0.7, label='Depth Scores')
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.3f})')
    plt.xlabel("Utterance Gap Index")
    plt.ylabel("Depth")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    save_path = Path(output_dir) / f"{file_id}_coherence_xai.png"
    plt.savefig(save_path)
    plt.close()
    print(f"  [XAI] Saved Segmentation Report: {save_path}")


def plot_lda_insights(file_id, lda_model, vectorizer, output_dir):
    """
    Visualises top keywords for each LDA topic assigned to segments.
    """
    feature_names = vectorizer.get_feature_names_out()
    n_topics = lda_model.n_components
    
    fig, axes = plt.subplots(1, n_topics, figsize=(15, 6), sharey=True)
    fig.suptitle(f"XAI: LDA Topic Word Distributions - {file_id}")

    for i in range(n_topics):
        topic_weights = lda_model.components_[i]
        top_indices = topic_weights.argsort()[-10:][::-1]
        top_words = [feature_names[j] for j in top_indices]
        top_weights = topic_weights[top_indices]
        
        sns.barplot(x=top_weights, y=top_words, ax=axes[i], palette='viridis')
        axes[i].set_title(f"Topic {i}")
        axes[i].set_xlabel("Weight")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = Path(output_dir) / f"{file_id}_lda_xai.png"
    plt.savefig(save_path)
    plt.close()
    print(f"  [XAI] Saved LDA Topic Report: {save_path}")


def plot_bilstm_saliency(file_id, model, embeddings, utterances, output_dir):
    """
    Computes Gradient Saliency (SHAP-inspired) to show which utterances 
    most influenced the BiLSTM boundary predictions.
    """
    model.train() # cuDNN RNN backward requires training mode
    device = next(model.parameters()).device
    x = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0).to(device)
    x.requires_grad = True
    
    # Forward pass
    probs = model(x) # (1, T-1)
    
    # Target the max probability gap (the most 'confident' boundary)
    if probs.numel() == 0: 
        model.eval()
        return
    
    target_idx = torch.argmax(probs)
    boundary_score = probs[0, target_idx]
    
    # Backward pass to get gradients
    model.zero_grad()
    boundary_score.backward()
    
    # Saliency = Norm of gradients per utterance
    saliency = x.grad.detach().norm(dim=2).squeeze(0).cpu().numpy()
    model.eval()
    
    # Visualise the importance of utterances leading up to that boundary
    plt.figure(figsize=(12, 6))
    top_n = min(len(utterances), 20)
    indices = range(max(0, target_idx-top_n//2), min(len(utterances), target_idx+top_n//2+1))
    
    display_texts = [f"{i}: {utterances[i]['text'][:40]}..." for i in indices]
    display_saliency = [saliency[i] for i in indices]
    
    sns.barplot(x=display_saliency, y=display_texts, palette='magma')
    plt.title(f"XAI: BiLSTM Saliency (Influence on Boundary at Gap {target_idx})\nTarget Utterance: '{utterances[target_idx]['text'][:60]}...'")
    plt.xlabel("Influence Score (Gradient Magnitude)")
    plt.ylabel("Transcription Context")
    
    plt.tight_layout()
    save_path = Path(output_dir) / f"{file_id}_bilstm_saliency_xai.png"
    plt.savefig(save_path)
    plt.close()
    print(f"  [XAI] Saved BiLSTM Saliency Report: {save_path}")


# --- TO-DO EXTRACTION XAI ---

def generate_todo_xai(file_id, df_todos, filter_logs, output_dir):
    """
    Saves a report explaining why specific utterances were selected as To-Dos.
    """
    report_path = Path(output_dir) / f"{file_id}_todo_insights.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Meeting AI To-Do XAI Report: {file_id}\n")
        f.write("="*50 + "\n\n")
        
        if df_todos.empty:
            f.write("No tasks were extracted for this meeting.\n")
        else:
            f.write(f"Total Tasks Identified: {len(df_todos)}\n\n")
            for idx, row in df_todos.iterrows():
                f.write(f"Task {idx+1}: {row['task']}\n")
                f.write(f"  Source: '{row['raw_text']}'\n")
                
                # Match against logs to find why it was accepted
                match = next((l for l in filter_logs if l['text'] == row['raw_text']), None)
                if match:
                    f.write(f"  XAI Insight: {match['reason']}\n")
                f.write("-" * 20 + "\n")
                
    print(f"  [XAI] Saved To-Do Insights: {report_path}")


# --- INTEGRATION HELPER ---

def generate_reports(file_id, data, output_root="Outputs/XAI"):
    """
    Main entry point for XAI generation.
    """
    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Segmentation / TextTiling
    if 'coherence' in data and 'depths' in data:
        plot_segmentation_coherence(
            file_id, data['coherence'], data['depths'], 
            data['threshold'], data.get('utterances', []), out_dir
        )
    
    # 2. LDA
    if 'lda' in data and 'vec' in data:
        plot_lda_insights(file_id, data['lda'], data['vec'], out_dir)
        
    # 3. BiLSTM Saliency (NEW)
    if 'model' in data and 'emb' in data:
        plot_bilstm_saliency(file_id, data['model'], data['emb'], data['utterances'], out_dir)

    # 4. To-Do
    if 'todos' in data and 'filter_logs' in data:
        generate_todo_xai(file_id, data['todos'], data['filter_logs'], out_dir)
