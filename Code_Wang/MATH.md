# Evaluation functions

## hdp_to_lda
sticks: mean of stick-breaking variables
alpha: expected topic proportions, multiplied
beta: roughly speaking, the mean word probabilities of each topic (since it's 
    each topic's Dirichlet parameters normalized by the sum over all words). The 
    additional self.m_W + self.m_eta is probably to avoid underflow in division.
    
-> This is closer to the true HDP than LDA. Essentially the document's G_d conditioned 
on the corpus G_0 is exactly a DP rather than the truncated DP.

## lda_e_step(doc, lda_alpha, lda_beta)

## lda_e_step_split(doc, lda_alpha, lda_beta)

# Mapping between code's notation and SVI paper's notation

## Instance variables

### Easy
m_W: vocabulary size
m_D: number of documents
m_T: corpus-level cap on number of topics
m_K: document-level cap on number of topics
m_gamma: Beta(1,m_gamma) of corpus-level stick-breaking
m_alpha: Beta(1,m_alpha) of document-level stick-breaking

### Trickier
m_var_sticks: variational parameters of corpus-level stick-breaking
m_varphi_ss: amount of change to apply to the first coordinate of m_var_sticks


# Computing the ``near equivalent'' LDA of an HDP? 