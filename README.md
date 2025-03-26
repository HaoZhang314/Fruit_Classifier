# Fruit_Classifier
Using machine learning neural networks to identify fruit species and rot

### data/preprocessing.py 文件总结
data/preprocessing.py 是水果分类器项目中的数据预处理模块，主要负责处理原始水果图像数据集并生成结构化的数据索引。以下是该文件的主要功能和结构：

### 主要功能
#### 数据集索引创建
从原始数据集目录遍历并提取所有图像文件
根据目录结构解析水果类型和腐烂状态信息
生成结构化的CSV文件，包含图像路径、水果类型、腐烂状态和数据集划分信息
#### 数据集统计分析
统计数据集中各类别样本的分布情况
输出数据集的基本信息，如总样本数和各类别样本数量
### 关键函数
#### create_dataset_csv
接收数据集目录和输出CSV文件路径作为参数
遍历训练集和测试集目录，提取图像文件信息
解析目录名称获取水果类型和腐烂状态
生成CSV文件并输出数据集统计信息
### 数据集结构假设
数据集分为训练集（train）和测试集（test）两个子目录
每个类别目录的命名格式为：[状态][水果类型]，例如：freshapples, rottenbanana
图像文件格式支持：.png、.jpg、.jpeg
### 工作流程
加载项目配置，获取数据集路径和输出CSV文件路径
遍历数据集目录，收集所有图像文件的信息
根据目录结构解析水果类型和腐烂状态
将收集到的数据保存为CSV文件
输出数据集统计信息，包括样本总数和各类别的分布情况
这个模块为后续的数据加载和模型训练提供了结构化的数据索引，使得数据的访问和处理更加方便和高效。通过这个预处理步骤，原始的图像数据被组织成易于处理的格式，便于后续的数据加载器模块使用。

## data/loaders.py文件总结
data/loaders.py 是一个水果分类器项目中的数据加载模块，主要负责处理水果图像数据集的加载和预处理。这个文件实现了以下功能：

### 数据集类（FruitDataset）
继承自PyTorch的Dataset类
加载图像和对应的标签（水果类型和腐烂状态）
支持数据集划分（训练集/测试集）
处理图像加载和标签转换

### 数据加载器类（FruitDataLoader）
封装了PyTorch的DataLoader创建过程
提供了获取训练集、验证集和测试集的方法
处理数据集的划分、变换应用和批次加载

### 工具函数
- get_transforms：获取数据预处理和增强的变换
- get_data_loaders：根据配置创建数据加载器，是模块的主要入口函数

## models/model.py文件的总结：

水果分类模型文件总结
这个文件实现了一个多任务学习模型，用于同时对水果类型和腐烂状态进行分类。整个文件的结构清晰，主要包含以下几个核心组件：

### 1. FruitClassifier 类
多任务分类模型，继承自PyTorch的nn.Module
支持多种骨干网络（EfficientNet和ResNet系列）作为特征提取器
包含两个分类头：一个用于水果类型分类，一个用于腐烂状态分类
主要方法：
init：初始化模型，设置骨干网络和分类头
_create_backbone：创建特征提取器，支持多种预训练网络
forward：前向传播，提取特征并进行双任务预测
predict：预测函数，返回类别索引而非logits

### 2. create_model 函数
工厂函数，用于创建FruitClassifier实例
封装了模型创建过程，使外部代码更简洁
支持配置不同的骨干网络和预训练选项

### 3. MultiTaskLoss 类
多任务联合损失函数，用于同时优化两个分类任务
可以通过权重调整对不同任务的关注度
使用交叉熵损失函数计算各个任务的损失
返回加权总损失和各个子任务损失的字典

### 4. calculate_metrics 函数
计算模型评价指标，包括：
水果类型分类准确率
腐烂状态分类准确率
联合准确率（两个任务都正确的比例）
可以扩展实现精确率、召回率和F1分数等指标

### 5. 主函数部分
用于测试模型的功能和性能
测试不同骨干网络的模型创建
测试模型的前向传播、损失计算、预测和评价指标计算
提供了一个完整的模型使用流程示例

## evaluation.py文件总结
这个文件是水果分类器项目中的模型评估模块，主要负责评估训练好的模型性能并生成详细的评估报告。

### 文件结构和主要功能
文件头部：包含模块说明、导入必要的库和项目其他模块
主要函数：
- evaluate_model：评估模型性能，计算准确率、精确率、召回率和F1分数等指标
- plot_confusion_matrix：绘制混淆矩阵并保存为图像文件
- generate_classification_report：生成分类报告，包括各类别的精确率、召回率和F1分数
- evaluate_and_report：高级封装函数，调用上述函数进行全面评估并生成报告
- visualize_error_samples：可视化错误预测的样本，帮助分析模型失败的情况
- parse_args：解析命令行参数
- load_model：加载模型和检查点
- get_class_names：获取类别名称
- main：主函数，整合上述功能
### 评估指标：
- 准确率(Accuracy)：正确预测的样本比例
- 精确率(Precision)：在预测为正类的样本中，真正为正类的比例
- 召回率(Recall)：在所有真正为正类的样本中，被正确预测为正类的比例
- F1分数：精确率和召回率的调和平均值
### 可视化功能：
- 混淆矩阵：展示模型预测结果与真实标签之间的关系
- 错误样本可视化：显示模型预测错误的样本图像及其预测结果和真实标签
### 输出内容：
- 评估指标摘要（JSON格式）
- 分类报告（文本格式）
- 混淆矩阵图像
- 错误样本可视化图像
- 错误分析报告