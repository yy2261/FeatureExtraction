# Distributed representation generation

1. Download source code from 200+ projects (getProjs.py).
2. Extract AST feature using Eclipse JDT.
3. Method name tokenize, stem.
4. Word embedding.

# Defect Prediction

1. Download source and defect information.
2. Extract AST feature using Eclipse JDT.
	- method invocation (method name last name)
	- class ins creation (class name)
	- control flow nodes (if, while, try/catch, throw, no return)

3. Select feature files which source code is included in csv files (selectFile.py).
4. Label feature files with defect information (fileLabel.py).
5. Generate training/test csv by transforming features into word vectors (genSamples.py).
6. Build LSTM model and evaluation.
