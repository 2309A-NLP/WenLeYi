---
license: Apache License 2.0
task: financial_analysis
dataset_type: Dataset
---
# 股票与金融数据集说明文档

## 一、数据集概述
本数据集包含多份与股票、指数及金融数据相关的CSV文件，涵盖股票关联指数、实时交易、金融数据等内容，适用于：
- 股票分析
- 金融市场研究
- 量化策略开发

为金融领域从业者和研究者提供基础数据支撑。

## 二、文件列表及说明

| 文件名                                | 内容说明                                                                 |
|---------------------------------------|--------------------------------------------------------------------------|
| index_related_to_stock.csv            | 存储指数与股票关联数据，可用于分析指数成分股关联、指数对股票的影响等场景 |
| real_time_trading_data_with.csv       | 包含股票实时交易数据（交易时间、价格、成交量等实时行情信息）             |
| stock_list.csv                        | 股票列表基础信息（股票代码、名称、所属板块等）                           |
| stock_related_to_index.csv            | 记录股票与指数关联细节，辅助研究股票在指数体系中的角色与表现             |
| hs_index_list.csv                     | 恒生指数列表数据（成分股、基本参数等）                                   |
| hs_index_realtime.csv                 | 恒生指数实时数据（实时走势、变动情况）                                   |
| index_industry_concept_tree.csv       | 指数行业概念分类体系，助力分析行业与指数关联、概念板块影响               |
| Financial_Data_20230330_20230630.csv  | 2023年3月30日-2023年6月30日期间的金融数据（金融趋势回溯分析）            |

## 三、数据格式与字段
- **格式**：CSV（逗号分隔）
- **编码**：建议UTF-8
- **字段示例**：
  - `stock_related_to_index.csv`：股票代码、指数代码、关联权重
  - `real_time_trading_data_with.csv`：交易时间戳、股票代码、成交价、成交量

## 四、数据来源与更新
- **来源**：合法合规金融数据采集渠道（金融数据供应商、公开市场数据接口等）
- **更新**：文件修改时间为最后更新时间（如`stock_related_to_index.csv`最后更新为2025/7/7）

## 五、使用方法import pandas as pd
### （一）环境准备
```bash
pip install pandas
file_path = "stock_related_to_index.csv"  
try:
    data = pd.read_csv(file_path)
    print("数据读取成功，数据预览：")
    print(data.head())  # 打印前5行数据
except FileNotFoundError:
    print(f"文件 {file_path} 未找到，请检查路径")
except Exception as e:
    print(f"数据读取出错：{e}")

### （一）环境准备
```bash
pip install pandas