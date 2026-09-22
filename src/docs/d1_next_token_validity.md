# Diagnostic 1: Next-Token Validity

Suppose the previous moves are D3, C3, and C4. This sequence is the only information given to the sequence model. 
Independently, the same move sequence is replayed using an Othello simulator starting from the standard initial board. 
The simulator reconstructs the true board state, determines whose turn it is, and calculates all moves that are legally possible from that state.
For example, suppose the legal-move set is {C5, E3}. The same prefix, D3 C3 C4, is then given to the sequence model, which produces probabilities for possible next tokens and proposes its most likely next move. 
Suppose the move that actually occurred in the recorded game was C5, but the model proposes E3. The model's prediction is still counted as valid because E3 is also contained in the legal-move set {C5, E3}. 
The model does not need to predict the exact move that occurred in the dataset; it only needs to propose a move that is legal from the reconstructed board state. 
If instead the model proposes a move such as C3, which is not in the legal set, the prediction is counted as invalid. Diagnostic 1 repeats this comparison across many prefixes and measures how often the model's proposed next token is a legally valid move.
