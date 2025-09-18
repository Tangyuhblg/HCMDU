# HCMDU
An Undersampling Method for Software Defect Prediction Based on Hilbert Curve Mapping Distance

Evaluation_indexs.py是评估模型预测结果的评价指标值。

data是实验使用的公开软件缺陷数据集。

Result是源代码提取向量表示的实验结果文件。
amq-5.0.0-java_clean_code.csv：包括数据预处理后的源代码文件。amq-5.0.0-code_vectors.csv：源代码提取的向量表示文件。amq-5.0.0-imbalanced_data.csv：平衡后的训练集文件。amq-5.0.0-predict_defect_code.csv：包含预测结果、文件名以及对应的缺陷代码行号的文件。

source_code包含源代码文件Dataset。

源代码预处理方法文件extract_csv.py，提取源代码向量表示文件extract_vectors.py。

HCMDU.py是本作品提出的欠采样方法文件。

main.py是本作品实现软件缺陷预测欠采样工具的执行文件。
