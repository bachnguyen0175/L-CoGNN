import numpy as np
import torch
from .logreg import LogReg
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from torch.nn.functional import softmax
import scipy.sparse as sp
from sklearn.preprocessing import normalize


##################################################
# This section of code adapted from pcy1302/DMGI #
##################################################

def evaluate(embeds, ratio, idx_train, idx_val, idx_test, label, nb_classes, device, dataset, lr, wd
             , isTest=True):
    hid_units = embeds.shape[1]
    xent = nn.CrossEntropyLoss()

    train_embs = embeds[idx_train]
    val_embs = embeds[idx_val]
    test_embs = embeds[idx_test]

    train_lbls = torch.argmax(label[idx_train], dim=-1)
    val_lbls = torch.argmax(label[idx_val], dim=-1)
    test_lbls = torch.argmax(label[idx_test], dim=-1)
    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []
    auc_score_list = []

    for _ in range(1):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=lr, weight_decay=wd)
        log.to(device)

        val_accs = []
        test_accs = []
        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []

        logits_list = []
        for iter_ in range(20):
            # train
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

            # val
            logits = log(val_embs)
            preds = torch.argmax(logits, dim=1)

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
            val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            # test
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)

            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')

            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)
            logits_list.append(logits)

        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])
        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])
        macro_f1s_val.append(val_macro_f1s[max_iter])

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])

        # auc
        best_logits = logits_list[max_iter]
        best_proba = softmax(best_logits, dim=1)
        auc_score_list.append(roc_auc_score(y_true=test_lbls.detach().cpu().numpy(),
                                            y_score=best_proba.detach().cpu().numpy(),
                                            multi_class='ovr'
                                            ))

    if isTest:
        print("\t[Classification] Macro-F1_mean: {:.4f} var: {:.4f}  Micro-F1_mean: {:.4f} var: {:.4f} auc {:.4f}"
              .format(np.mean(macro_f1s),
                      np.std(macro_f1s),
                      np.mean(micro_f1s),
                      np.std(micro_f1s),
                      np.mean(auc_score_list),
                      np.std(auc_score_list)
                      )
              )
    else:
        return np.mean(macro_f1s_val), np.mean(macro_f1s)

    f = open("result_"+dataset+str(ratio)+".txt", "a")
    f.write(str(np.mean(macro_f1s))+"\t"+str(np.mean(micro_f1s))+"\t"+str(np.mean(auc_score_list))+"\n")
    f.close()


def evaluate_node_classification(embeds, idx_train, idx_val, idx_test, label, nb_classes, device, lr, wd):
    """
    Evaluate node classification performance

    Args:
        embeds: Node embeddings
        idx_train: Training node indices
        idx_val: Validation node indices
        idx_test: Test node indices
        label: Node labels
        nb_classes: Number of classes
        device: Device to run on
        lr: Learning rate for classifier
        wd: Weight decay for classifier

    Returns:
        accuracy, macro_f1, micro_f1: Performance metrics
    """
    hid_units = embeds.shape[1]
    xent = nn.CrossEntropyLoss()

    train_embs = embeds[idx_train]
    val_embs = embeds[idx_val]
    test_embs = embeds[idx_test]

    train_lbls = torch.argmax(label[idx_train], dim=-1)
    val_lbls = torch.argmax(label[idx_val], dim=-1)
    test_lbls = torch.argmax(label[idx_test], dim=-1)

    log = LogReg(hid_units, nb_classes)
    opt = torch.optim.Adam(log.parameters(), lr=lr, weight_decay=wd)
    log.to(device)

    val_accs = []
    test_accs = []
    val_micro_f1s = []
    test_micro_f1s = []
    val_macro_f1s = []
    test_macro_f1s = []
    logits_list = []

    for iter_ in range(10000):
        # train
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        train_lbls = train_lbls.to(logits.device)  # Move train_lbls to the same device as logits

        loss = xent(logits, train_lbls)

        loss.backward()
        opt.step()

        # val
        logits = log(val_embs)

        # Move tensors to the same device
        logits = logits.to(device)
        preds = torch.argmax(logits, dim=1).to(device)
        val_lbls = val_lbls.to(device)

        val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
        val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
        val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')

        val_accs.append(val_acc.item())
        val_macro_f1s.append(val_f1_macro)
        val_micro_f1s.append(val_f1_micro)

        # test
        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        test_lbls = test_lbls.to(preds.device)

        test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
        test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')

        test_accs.append(test_acc.item())
        test_macro_f1s.append(test_f1_macro)
        test_micro_f1s.append(test_f1_micro)
        logits_list.append(logits)

    max_iter = val_accs.index(max(val_accs))
    acc = test_accs[max_iter]

    max_iter = val_macro_f1s.index(max(val_macro_f1s))
    macro_f1 = test_macro_f1s[max_iter]

    max_iter = val_micro_f1s.index(max(val_micro_f1s))
    micro_f1 = test_micro_f1s[max_iter]

    return acc, macro_f1, micro_f1


def evaluate_link_prediction(embeddings, test_edges, test_edges_false, device='cpu'):
    """
    Evaluate link prediction performance using AUC and AP metrics

    Args:
        embeddings: Node embeddings [num_nodes, embedding_dim]
        test_edges: Positive test edges [num_pos_edges, 2]
        test_edges_false: Negative test edges [num_neg_edges, 2]
        device: Device to run evaluation on

    Returns:
        auc_score: Area Under Curve score
        ap_score: Average Precision score
        hits_at_k: Dictionary with Hits@K scores
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Move to CPU for sklearn compatibility
    if torch.is_tensor(embeddings):
        embeddings = embeddings.cpu().numpy()
    if torch.is_tensor(test_edges):
        test_edges = test_edges.cpu().numpy()
    if torch.is_tensor(test_edges_false):
        test_edges_false = test_edges_false.cpu().numpy()

    # Calculate edge scores using dot product
    def get_edge_score(edges, embeddings):
        scores = []
        for edge in edges:
            node1, node2 = edge[0], edge[1]
            # Dot product similarity
            score = np.dot(embeddings[node1], embeddings[node2])
            scores.append(score)
        return np.array(scores)

    # Get scores for positive and negative edges
    pos_scores = get_edge_score(test_edges, embeddings)
    neg_scores = get_edge_score(test_edges_false, embeddings)

    # Create labels
    pos_labels = np.ones(len(pos_scores))
    neg_labels = np.zeros(len(neg_scores))

    # Combine scores and labels
    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate([pos_labels, neg_labels])

    # Calculate AUC and AP
    auc_score = roc_auc_score(all_labels, all_scores)
    ap_score = average_precision_score(all_labels, all_scores)

    # Calculate Hits@K
    hits_at_k = {}
    for k in [10, 50, 100]:
        # Sort all edges by score (descending)
        edge_list = []
        for i, edge in enumerate(test_edges):
            edge_list.append((pos_scores[i], 1, tuple(edge)))  # (score, label, edge)
        for i, edge in enumerate(test_edges_false):
            edge_list.append((neg_scores[i], 0, tuple(edge)))  # (score, label, edge)

        # Sort by score (descending)
        edge_list.sort(key=lambda x: x[0], reverse=True)

        # Calculate Hits@K
        if len(edge_list) >= k:
            top_k_edges = edge_list[:k]
            hits = sum([1 for score, label, edge in top_k_edges if label == 1])
            hits_at_k[f'hits_at_{k}'] = hits / min(k, len(test_edges))
        else:
            hits_at_k[f'hits_at_{k}'] = 0.0

    return auc_score, ap_score, hits_at_k


def generate_negative_edges(pos_edges, num_nodes, num_neg_edges=None):
    """
    Generate negative edges for link prediction evaluation

    Args:
        pos_edges: Positive edges [num_edges, 2]
        num_nodes: Total number of nodes
        num_neg_edges: Number of negative edges to generate (default: same as positive)

    Returns:
        neg_edges: Negative edges [num_neg_edges, 2]
    """
    if num_neg_edges is None:
        num_neg_edges = len(pos_edges)

    # Convert positive edges to set for fast lookup
    pos_edge_set = set()
    for edge in pos_edges:
        pos_edge_set.add((min(edge[0], edge[1]), max(edge[0], edge[1])))

    neg_edges = []
    while len(neg_edges) < num_neg_edges:
        # Sample random node pair
        node1 = np.random.randint(0, num_nodes)
        node2 = np.random.randint(0, num_nodes)

        if node1 != node2:  # No self-loops
            edge = (min(node1, node2), max(node1, node2))
            if edge not in pos_edge_set:
                neg_edges.append([node1, node2])
                pos_edge_set.add(edge)  # Avoid duplicates

    return np.array(neg_edges)


def split_edges_for_link_prediction(edges, test_ratio=0.1, val_ratio=0.05):
    """
    Split edges into train/val/test sets for link prediction

    Args:
        edges: All edges [num_edges, 2]
        test_ratio: Ratio of edges for testing
        val_ratio: Ratio of edges for validation

    Returns:
        train_edges, val_edges, test_edges: Split edge sets
    """
    if torch.is_tensor(edges):
        edges = edges.cpu().numpy()

    num_edges = len(edges)
    num_test = int(num_edges * test_ratio)
    num_val = int(num_edges * val_ratio)
    num_train = num_edges - num_test - num_val

    # Shuffle edges
    np.random.shuffle(edges)

    train_edges = edges[:num_train]
    val_edges = edges[num_train:num_train + num_val]
    test_edges = edges[num_train + num_val:]

    return train_edges, val_edges, test_edges


def evaluate_node_clustering(embeddings, true_labels, n_clusters, device='cpu', n_runs=10):
    """
    Evaluate node clustering performance using NMI, ARI, and Modularity

    Args:
        embeddings: Node embeddings [num_nodes, embedding_dim]
        true_labels: Ground truth cluster labels [num_nodes]
        n_clusters: Number of clusters to use
        device: Device to run evaluation on
        n_runs: Number of runs for averaging (K-means is non-deterministic)

    Returns:
        nmi_score: Normalized Mutual Information score
        ari_score: Adjusted Rand Index score
        modularity_score: Modularity score
        cluster_accuracy: Clustering accuracy (best label matching)
    """
    # Move to CPU for sklearn compatibility
    if torch.is_tensor(embeddings):
        embeddings = embeddings.cpu().numpy()
    if torch.is_tensor(true_labels):
        true_labels = true_labels.cpu().numpy()

    # If true_labels are one-hot encoded, convert to class indices
    if len(true_labels.shape) > 1:
        true_labels = np.argmax(true_labels, axis=1)

    nmi_scores = []
    ari_scores = []
    cluster_accuracies = []

    for run in range(n_runs):
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=run, n_init=10)
        predicted_labels = kmeans.fit_predict(embeddings)

        # Calculate NMI and ARI
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)
        ari = adjusted_rand_score(true_labels, predicted_labels)

        nmi_scores.append(nmi)
        ari_scores.append(ari)

        # Calculate clustering accuracy using Hungarian algorithm (best label matching)
        cluster_acc = cluster_accuracy(true_labels, predicted_labels, n_clusters)
        cluster_accuracies.append(cluster_acc)

    # Calculate modularity (using the best clustering result)
    best_run = np.argmax(nmi_scores)
    kmeans_best = KMeans(n_clusters=n_clusters, random_state=best_run, n_init=10)
    best_predicted_labels = kmeans_best.fit_predict(embeddings)

    # For modularity, we need the adjacency matrix - we'll estimate it from embeddings
    modularity_score = calculate_modularity_from_embeddings(embeddings, best_predicted_labels)

    return {
        'nmi': np.mean(nmi_scores),
        'nmi_std': np.std(nmi_scores),
        'ari': np.mean(ari_scores),
        'ari_std': np.std(ari_scores),
        'accuracy': np.mean(cluster_accuracies),
        'accuracy_std': np.std(cluster_accuracies),
        'modularity': modularity_score
    }


def cluster_accuracy(true_labels, predicted_labels, n_clusters):
    """
    Calculate clustering accuracy using Hungarian algorithm for best label matching

    Args:
        true_labels: Ground truth labels
        predicted_labels: Predicted cluster labels
        n_clusters: Number of clusters

    Returns:
        accuracy: Best possible accuracy after optimal label matching
    """
    from scipy.optimize import linear_sum_assignment

    # Create confusion matrix
    confusion_matrix = np.zeros((n_clusters, n_clusters))
    for i in range(len(true_labels)):
        confusion_matrix[true_labels[i], predicted_labels[i]] += 1

    # Use Hungarian algorithm to find optimal label matching
    row_indices, col_indices = linear_sum_assignment(-confusion_matrix)

    # Calculate accuracy with optimal matching
    total_correct = confusion_matrix[row_indices, col_indices].sum()
    accuracy = total_correct / len(true_labels)

    return accuracy


def calculate_modularity_from_embeddings(embeddings, cluster_labels, threshold=0.5):
    """
    Calculate modularity score using embedding similarity as edge weights

    Args:
        embeddings: Node embeddings [num_nodes, embedding_dim]
        cluster_labels: Predicted cluster labels
        threshold: Similarity threshold for considering edges

    Returns:
        modularity: Modularity score
    """
    n_nodes = len(embeddings)

    # Calculate pairwise cosine similarities
    embeddings_norm = normalize(embeddings, norm='l2', axis=1)
    similarity_matrix = np.dot(embeddings_norm, embeddings_norm.T)

    # Create adjacency matrix based on similarity threshold
    adj_matrix = (similarity_matrix > threshold).astype(float)
    np.fill_diagonal(adj_matrix, 0)  # Remove self-loops

    # Calculate modularity
    m = np.sum(adj_matrix) / 2  # Total number of edges
    if m == 0:
        return 0.0

    modularity = 0.0
    unique_clusters = np.unique(cluster_labels)

    for cluster in unique_clusters:
        cluster_nodes = np.where(cluster_labels == cluster)[0]

        # Internal edges within cluster
        internal_edges = 0
        for i in cluster_nodes:
            for j in cluster_nodes:
                if i < j:  # Avoid double counting
                    internal_edges += adj_matrix[i, j]

        # Expected internal edges
        cluster_degree = np.sum(adj_matrix[cluster_nodes, :])
        expected_internal = (cluster_degree ** 2) / (4 * m)

        modularity += (internal_edges - expected_internal) / m

    return modularity


def evaluate_all_downstream_tasks(embeddings, labels, edges=None, num_nodes=None, device='cpu'):
    """
    Comprehensive evaluation on all three downstream tasks

    Args:
        embeddings: Node embeddings [num_nodes, embedding_dim]
        labels: Node labels for classification and clustering
        edges: Edge list for link prediction [num_edges, 2] (optional)
        num_nodes: Total number of nodes (required if edges provided)
        device: Device to run evaluation on

    Returns:
        results: Dictionary containing results from all three tasks
    """
    results = {}

    # Move to appropriate device
    if torch.is_tensor(embeddings):
        embeddings = embeddings.to(device)
    if torch.is_tensor(labels):
        labels = labels.to(device)

    print("=" * 60)
    print("COMPREHENSIVE DOWNSTREAM TASK EVALUATION")
    print("=" * 60)

    # 1. Node Classification Evaluation
    print("\n1. Node Classification...")
    try:
        # Create train/val/test splits (you may want to use your existing splits)
        num_nodes_total = embeddings.shape[0]
        indices = np.random.permutation(num_nodes_total)

        train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2
        train_size = int(train_ratio * num_nodes_total)
        val_size = int(val_ratio * num_nodes_total)

        train_idx = torch.tensor(indices[:train_size]).to(device)
        val_idx = torch.tensor(indices[train_size:train_size + val_size]).to(device)
        test_idx = torch.tensor(indices[train_size + val_size:]).to(device)

        # Get number of classes
        if len(labels.shape) > 1:
            nb_classes = labels.shape[1]
        else:
            nb_classes = len(torch.unique(labels))

        accuracy, macro_f1, micro_f1 = evaluate_node_classification(
            embeddings, train_idx, val_idx, test_idx, labels, nb_classes, device, 0.01, 0
        )

        results['node_classification'] = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1
        }
        print(f"   ✓ Accuracy: {accuracy:.4f}")
        print(f"   ✓ Macro-F1: {macro_f1:.4f}")
        print(f"   ✓ Micro-F1: {micro_f1:.4f}")

    except Exception as e:
        print(f"   ✗ Node classification failed: {e}")
        results['node_classification'] = {'error': str(e)}

    # 2. Link Prediction Evaluation
    print("\n2. Link Prediction...")
    if edges is not None and num_nodes is not None:
        try:
            # Split edges for link prediction
            train_edges, val_edges, test_edges = split_edges_for_link_prediction(edges)

            # Generate negative edges
            test_neg_edges = generate_negative_edges(test_edges, num_nodes)

            auc_score, ap_score, hits_at_k = evaluate_link_prediction(
                embeddings, test_edges, test_neg_edges, device
            )

            results['link_prediction'] = {
                'auc': auc_score,
                'ap': ap_score,
                'hits_at_10': hits_at_k.get('hits_at_10', 0),
                'hits_at_50': hits_at_k.get('hits_at_50', 0),
                'hits_at_100': hits_at_k.get('hits_at_100', 0)
            }
            print(f"   ✓ AUC: {auc_score:.4f}")
            print(f"   ✓ AP: {ap_score:.4f}")
            print(f"   ✓ Hits@10: {hits_at_k.get('hits_at_10', 0):.4f}")
            print(f"   ✓ Hits@50: {hits_at_k.get('hits_at_50', 0):.4f}")
            print(f"   ✓ Hits@100: {hits_at_k.get('hits_at_100', 0):.4f}")

        except Exception as e:
            print(f"   ✗ Link prediction failed: {e}")
            results['link_prediction'] = {'error': str(e)}
    else:
        print("   ⚠ Skipped (no edges provided)")
        results['link_prediction'] = {'skipped': 'no edges provided'}

    # 3. Node Clustering Evaluation
    print("\n3. Node Clustering...")
    try:
        # Determine number of clusters
        if len(labels.shape) > 1:
            n_clusters = labels.shape[1]
        else:
            n_clusters = len(torch.unique(labels))

        clustering_results = evaluate_node_clustering(
            embeddings, labels, n_clusters, device
        )

        results['node_clustering'] = clustering_results
        print(f"   ✓ NMI: {clustering_results['nmi']:.4f} ± {clustering_results['nmi_std']:.4f}")
        print(f"   ✓ ARI: {clustering_results['ari']:.4f} ± {clustering_results['ari_std']:.4f}")
        print(f"   ✓ Accuracy: {clustering_results['accuracy']:.4f} ± {clustering_results['accuracy_std']:.4f}")
        print(f"   ✓ Modularity: {clustering_results['modularity']:.4f}")

    except Exception as e:
        print(f"   ✗ Node clustering failed: {e}")
        results['node_clustering'] = {'error': str(e)}

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    return results
