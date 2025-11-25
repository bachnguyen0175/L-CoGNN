\section{Methodology}

\subsection{Problem Definition}
We define a Heterogeneous Information Network (HIN) as a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{A}, \mathcal{R})$, where $\mathcal{V}$ and $\mathcal{E}$ denote the sets of nodes and edges, respectively. The mapping functions $\phi: \mathcal{V} \rightarrow \mathcal{A}$ and $\psi: \mathcal{E} \rightarrow \mathcal{R}$ associate each object with a node type $A \in \mathcal{A}$ and each link with a relation type $R \in \mathcal{R}$, where $|\mathcal{A}| + |\mathcal{R}| > 2$.
The goal of \textbf{ReGraD} is to learn a lightweight student encoder $f_S: \mathcal{V} \rightarrow \mathbb{R}^{d'}$ that maps nodes to low-dimensional embeddings $Z_S$, such that $f_S$ retains the semantic expressiveness of a heavy teacher model $f_T$ while exhibiting superior robustness to structural noise.

\subsection{The ReGraD Framework}
ReGraD operates on a \textbf{multi-view distillation paradigm}, consisting of three core components: (1) A \textbf{Main Teacher} that learns stable semantic-topological representations from the original graph; (2) An \textbf{Augmentation Teacher} that generates diverse structural views via a heterogeneous augmentation pipeline; and (3) A \textbf{Student Network} that distills knowledge from both teachers through a unified relational objective.

\subsection{Heterogeneous Augmentation Module}
To overcome the limitations of static graph learning, we introduce a dedicated \textbf{Heterogeneous Augmentation Module}. Unlike generic edge dropout methods which may destroy semantic semantics, our module employs a \textbf{Structure-Aware Meta-Path Connection} mechanism. This mechanism injects higher-order structural information directly into the feature space, creating "virtual connections" that are robust to local topology perturbations.

Formally, let $H^{(0)} \in \mathbb{R}^{N \times d}$ be the initial node features. For a specific meta-path $\mathcal{P}$, let $A_{\mathcal{P}}$ be its adjacency matrix. We first project the features into a low-rank latent space to capture essential semantic properties efficiently:
\begin{equation}
    H_{proj} = H^{(0)} W_{down} W_{up}
\end{equation}
where $W_{down} \in \mathbb{R}^{d \times k}$ and $W_{up} \in \mathbb{R}^{k \times d}$ form a low-rank bottleneck ($k \ll d$).

We then propagate these projected features along the meta-path structure to generate the augmented signal $H_{aug}^{\mathcal{P}}$:
\begin{equation}
    H_{aug}^{\mathcal{P}} = (A_{\mathcal{P}} \cdot H_{proj}) \odot \sigma(W_{gate})
\end{equation}
Here, $\sigma(\cdot)$ denotes the activation function, and $W_{gate}$ is a learnable gating parameter (embedding) that adaptively controls the flow of augmented information.

Finally, the augmented feature view $\tilde{H}$ is constructed by fusing the original features with the meta-path signal via a weighted residual connection:
\begin{equation}
    \tilde{H} = (1 + \alpha) H^{(0)} + (1 - \alpha) \cdot H_{aug}^{\mathcal{P}}
\end{equation}
where $\alpha \in [0, 1]$ is a hyperparameter controlling the balance between preserving original features and injecting structural augmentation. This module allows the Augmentation Teacher to learn from a structurally enriched view, providing a robust counterpoint to the Main Teacher's view.

\subsection{Student Network with Guidance Injection}
The Student Network is a compressed version of the teacher ($d' < d$). To further enhance its robustness, we introduce a \textbf{Guidance Injection} mechanism during the forward pass. The student receives structural guidance $Z_{guide}$ from the Augmentation Teacher and fuses it with its own features $Z_S$:
\begin{equation}
    Z_{fused} = (1 - \gamma) Z_S + \gamma \cdot (Z'_{guide} \odot \sigma(W_{gate}[Z_S \| Z'_{guide}]))
\end{equation}
where $Z'_{guide} = \text{Proj}(Z_{guide})$ aligns dimensions, and $\gamma$ is a learned fusion weight modulated by the teacher's attention importance. This allows the student to dynamically leverage the teacher's structural insights during inference.

\subsection{Dual-Teacher Knowledge Distillation}
The core of ReGraD lies in its ability to distill knowledge from two complementary sources: the \textbf{Main Teacher} (High Precision) and the \textbf{Augmentation Teacher} (High Robustness). We propose a \textbf{Complementary Fusion Distillation} strategy to integrate these signals dynamically.

\subsubsection{Knowledge Alignment}
First, we align the student's representation space with the teacher's space using a learnable projection head $W_{align} \in \mathbb{R}^{d' \times d}$:
\begin{equation}
    \tilde{Z}_S = f_{align}(Z_S) = \text{ReLU}(Z_S W_{align}^{(1)}) W_{align}^{(2)}
\end{equation}

\subsubsection{Adaptive Fusion Mechanism}
To determine which teacher to trust for a given node $v$, we compute an \textbf{Agreement Score} $s_{agree}(v)$ between the Main Teacher's embedding $z_T(v)$ and the Augmentation Teacher's embedding $z_{Aug}(v)$:
\begin{equation}
    s_{agree}(v) = \cos(z_T(v), z_{Aug}(v)) = \frac{z_T(v) \cdot z_{Aug}(v)}{\|z_T(v)\| \|z_{Aug}(v)\|}
\end{equation}
A high agreement score indicates that the structural perturbation did not affect the node's representation, implying the local structure is stable. A low score suggests structural ambiguity.

We then compute a dynamic fusion weight $\beta(v)$ using a learned gating network:
\begin{equation}
    \beta(v) = \sigma(\text{MLP}([z_T(v) \| z_{Aug}(v)]))
\end{equation}
This gate learns to balance the two teachers. The fused target representation $Z_{target}$ is defined as:
\begin{equation}
    Z_{target}(v) = (1 - \beta(v)) \cdot z_T(v) + \beta(v) \cdot z_{Aug}(v)
\end{equation}
The distillation loss minimizes the distance between the aligned student embedding and this fused target:
\begin{equation}
    \mathcal{L}_{fusion} = \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} \| \tilde{z}_S(v) - Z_{target}(v) \|^2_2
\end{equation}

\subsection{Unified Optimization Objective}
In addition to the fusion loss, we incorporate a standard Knowledge Distillation loss $\mathcal{L}_{KD}$ targeting the Main Teacher to ensure semantic fidelity, and a Link Reconstruction loss $\mathcal{L}_{rec}$ to preserve global topology.

The \textbf{Link Reconstruction Loss} maximizes the likelihood of observed edges while minimizing that of sampled negative edges:
\begin{equation}
    \mathcal{L}_{rec} = - \sum_{(u,v) \in \mathcal{E}} \log \sigma(z_u^S \cdot z_v^S) - \sum_{(u,v') \in \mathcal{E}^-} \log (1 - \sigma(z_u^S \cdot z_{v'}^S))
\end{equation}

To prevent mode collapse where the Augmentation Teacher simply mimics the Main Teacher, we add a \textbf{Diversity Regularization} term:
\begin{equation}
    \mathcal{L}_{div} = \max(0, \bar{s}_{agree} - \tau_{div})
\end{equation}
where $\tau_{div}$ (set to 0.7) is a threshold ensuring sufficient diversity between the two views.

The final objective function is a weighted sum of these components:
\begin{equation}
    \mathcal{L}_{total} = \mathcal{L}_{KD} + \lambda_1 \mathcal{L}_{fusion} + \lambda_2 \mathcal{L}_{rec} + \lambda_3 \mathcal{L}_{div}
\end{equation}
This unified objective ensures the student learns a representation that is semantically rich, structurally robust, and topologically accurate.