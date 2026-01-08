# Raft-LM Risk Preference & Training Chapters

Based on the provided sources, here are the chapters, papers, and conceptual frameworks that directly support risk preference formulation and training methodologies for the Raft-LM project.

## 1. Conceptual Risk Frameworks for Preference Definition

### From "The Ten Lenses of Convex Risk Mapping"

#### Category 1: Structural / Ruin Risk
**Why selected for Raft-LM:** Establishes the fundamental training preference: any reasoning path or trade that threatens survival (reaches an unrecoverable state) must be penalized to zero, regardless of potential upside. This is the "master rule" for aligning the LLM's decision-making.

#### Category 5: Asymmetry & Convexity
**Why selected for Raft-LM:** Provides the mathematical expression of the desired training philosophy. This category trains the model to actively seek convex payoffs (CI > 1), avoid concave traps (CI < 1), and evaluate risk using metrics like the Omega ratio and Sortino ratio.

#### Category 8: Behavioral & Perception
**Why selected for Raft-LM:** Trains the LLM to distinguish objective risk from distorted human market perception. The goal is to develop a "calm psychologist" capability that can detect when participants are irrationally scared or complacent.

### From "Asymmetric Constructivism: The Path of the Convexity Hunter"

#### Title: The Path: Seeking the Formulation of Risk Preference
**Why selected for Raft-LM:** Describes the core paradigm shift from standard risk management (Sharpe Ratio) to Convexity (Asymmetry) as the AI's guiding principle. This is the intellectual foundation for Raft-LM's objective function.

#### Title: The Preference System: The "Convexity Score"
**Why selected for Raft-LM:** Defines the measurable system (Vitality Score) for training the model. It shifts the focus from predicting the probability of success to assessing the "Quality of Uncertainty," a key learning objective.

#### Title: The "Religion" of the Algorithm
**Why selected for Raft-LM:** Serves as the doctrinal basis for the training process, explicitly stating the core preference: "I do not seek comfort; I seek asymmetry."

---

## 2. Relevant Academic Papers & Chapters for Training Methodologies

### Paper: "What is the Alignment Objective of GRPO?"

#### Chapter/Title: Aggregation of Preferences
**Why selected for Raft-LM:** Explains how Group Relative Policy Optimization (GRPO) aligns reward maximization with a reference model's preference using a reverse KL divergence. This informs the design of Raft-LM's risk-aware preference learning algorithms.

### Paper: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL"

#### Chapter 2.2.2: Reward Modeling
**Why selected for Raft-LM:** Describes the core training mechanism that incentivizes specific reasoning behaviors (self-verification, reflection) using Accuracy Rewards and Format Rewards. This methodology is directly applicable to structuring Raft-LM's training to prefer risk-aware reasoning paths.

### Paper: "Riemannian Metric Learning: Closer to You than You imagine"

#### Chapter 5.1: Statistical Guarantees for Nonlinear Metric Learning
**Why selected for Raft-LM:** Links metric learning to preference systems and "pluralistic alignment" in LLM fine-tuning. This supports Raft-LM's goal of learning a risk-aware metric that accounts for latent data distributions and user-defined risk preferences.

### Paper: "Geometry and convergence of natural policy gradient methods"

#### Chapter 3.1: Definition and general properties of natural gradients
**Why selected for Raft-LM:** Explains how natural gradients identify the "best direction" for improvement in model space, which is critical for training Raft-LM in a manner invariant to re-parameterization and aligned with risk geometry.